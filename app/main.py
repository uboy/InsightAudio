import logging
import os
import uuid
from typing import Dict, List, Optional

from fastapi import FastAPI, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from app import config_manager, file_utils, models as model_registry, summarizer, transcriber

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory="app/templates")
app = FastAPI(title="InsightAudio")


def _init_dirs():
    cfg = config_manager.get_config()
    results_dir = os.path.abspath(cfg.get("RESULTS_DIR", "/tmp"))
    log_dir = os.path.abspath(cfg.get("LOG_DIR", os.path.join(BASE_DIR, "..", "logs")))
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "requests"), exist_ok=True)
    return results_dir, log_dir


RESULTS_DIR, LOG_DIR = _init_dirs()
REQUEST_LOG_DIR = os.path.join(LOG_DIR, "requests")


def _configure_logging():
    logger = logging.getLogger("insightaudio")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    file_handler = logging.FileHandler(os.path.join(LOG_DIR, "server.log"), encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


ROOT_LOGGER = _configure_logging()


class RequestLogBufferHandler(logging.Handler):
    def __init__(self, level=logging.INFO):
        super().__init__(level=level)
        self.records: List[str] = []
        self.setFormatter(logging.Formatter("%Y-%m-%d %H:%M:%S - %(message)s"))

    def emit(self, record):
        self.records.append(self.format(record))

    def dump(self) -> List[str]:
        return self.records


def _get_config_view() -> Dict:
    cfg = config_manager.get_config()
    return {
        "MODEL_DIR": cfg.get("MODEL_DIR"),
        "CONFIG_DIR": cfg.get("CONFIG_DIR", config_manager.CONFIG_DIR),
        "RESULTS_DIR": cfg.get("RESULTS_DIR"),
        "SUMMARY_BACKEND": cfg.get("SUMMARY_BACKEND", "ollama"),
        "OLLAMA_API_BASE": cfg.get("OLLAMA_API_BASE"),
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    cfg = config_manager.get_config()
    model_options = model_registry.list_all_models(summary_backend=cfg.get("SUMMARY_BACKEND", "ollama"))
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "model_options": model_options,
            "config": _get_config_view(),
        },
    )


def _current_results_dir() -> str:
    cfg = config_manager.get_config()
    path = os.path.abspath(cfg.get("RESULTS_DIR", RESULTS_DIR))
    os.makedirs(path, exist_ok=True)
    return path


def _create_request_logger(request_id: str):
    request_logger = logging.getLogger(f"insightaudio.request.{request_id}")
    request_logger.setLevel(logging.INFO)
    buffer_handler = RequestLogBufferHandler()
    log_file_path = os.path.join(REQUEST_LOG_DIR, f"{request_id}.log")
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    request_logger.addHandler(buffer_handler)
    request_logger.addHandler(file_handler)
    return request_logger, buffer_handler, file_handler, log_file_path


@app.post("/process")
async def process(
    request: Request,
    file: UploadFile,
    transcribe_model: str = Form(...),
    summarize_model: str = Form(...),
    summary_backend: str = Form("ollama"),
    custom_api_url: str = Form(""),
    custom_api_models: str = Form(""),
):
    request_id = uuid.uuid4().hex
    request_logger, buffer_handler, file_handler, log_file_path = _create_request_logger(request_id)
    files: List[str] = []
    try:
        results_dir = _current_results_dir()
        saved_path = file_utils.handle_upload(file, save_dir=results_dir)
        request_logger.info("Получен файл: %s", saved_path)

        transcript = transcriber.transcribe(saved_path, model=transcribe_model)
        request_logger.info("Транскрипция завершена (%s символов)", len(transcript))

        summary = summarizer.summarize(
            transcript,
            model=summarize_model,
            backend=summary_backend,
            custom_api_url=custom_api_url or None,
        )
        request_logger.info("Пересказ сформирован (%s символов)", len(summary))

        files = file_utils.save_results({"transcript": transcript, "summary": summary}, save_dir=results_dir)
        request_logger.info("Результаты сохранены: %s", files)

        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "files": files,
                "log_lines": buffer_handler.dump(),
                "request_id": request_id,
                "log_file": os.path.basename(log_file_path),
            },
        )
    except Exception as exc:  # pylint: disable=broad-except
        request_logger.exception("Ошибка обработки: %s", exc)
        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "files": files,
                "error_message": str(exc),
                "log_lines": buffer_handler.dump(),
                "request_id": request_id,
                "log_file": os.path.basename(log_file_path),
            },
            status_code=500,
        )
    finally:
        request_logger.removeHandler(buffer_handler)
        request_logger.removeHandler(file_handler)
        file_handler.close()


@app.get("/download/{filename}")
def download(filename: str):
    results_dir = _current_results_dir()
    filepath = os.path.join(results_dir, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Файл не найден")
    return FileResponse(filepath, media_type="application/octet-stream", filename=filename)


@app.get("/logs/request/{request_id}")
def download_request_log(request_id: str):
    log_path = os.path.join(REQUEST_LOG_DIR, f"{request_id}.log")
    if not os.path.exists(log_path):
        raise HTTPException(status_code=404, detail="Лог не найден")
    return FileResponse(log_path, media_type="text/plain", filename=f"request_{request_id}.log")


@app.get("/list_models")
def list_models(summary_backend: Optional[str] = Query(None), custom_models: Optional[str] = Query(None)):
    custom_list = []
    if custom_models:
        custom_list = [item.strip() for item in custom_models.split(",") if item.strip()]
    data = model_registry.list_all_models(summary_backend=summary_backend, custom_models=custom_list or None)
    return data


@app.post("/download_model")
def download_model(model_name: str = Form(...), model_type: str = Form(...), backend: Optional[str] = Form(None)):
    ROOT_LOGGER.info("Запрошено скачивание модели: %s (%s)", model_name, model_type)
    return model_registry.download_model(model_name, model_type, backend=backend)


@app.get("/config")
def get_config():
    return config_manager.get_config()


@app.post("/config")
def update_config(cfg: Dict):
    ROOT_LOGGER.info("Обновление конфига: %s", list(cfg.keys()))
    updated = config_manager.set_config(cfg)
    return {"status": "OK", "cfg": updated}


@app.get("/health")
def health():
    return {"status": "InsightAudio is alive"}
