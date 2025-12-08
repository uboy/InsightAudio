import asyncio
import json
import logging
import logging.handlers
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app import config_manager, file_utils, models as model_registry, summarizer, transcriber, translator
from app.db import SessionLocal, get_session
from app.db_models import Job, User
from app.job_service import build_user_job_dir, create_job, ensure_tables, get_or_create_user
from app.tasks import audio_job, doc_job
from app import settings_loader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory="app/templates")
app = FastAPI(title="InsightAudio")
app.mount("/static", StaticFiles(directory="app/static"), name="static")


def _init_dirs():
    cfg = config_manager.get_config()
    results_dir = os.path.abspath(cfg.get("RESULTS_DIR", "/tmp"))
    log_dir = os.path.abspath(cfg.get("LOG_DIR", os.path.join(BASE_DIR, "..", "logs")))
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "requests"), exist_ok=True)
    # Создаем директорию для кеша WAV файлов
    os.makedirs(os.path.join(results_dir, ".wav_cache"), exist_ok=True)
    return results_dir, log_dir


RESULTS_DIR, LOG_DIR = _init_dirs()
REQUEST_LOG_DIR = os.path.join(LOG_DIR, "requests")


def _configure_logging():
    logger = logging.getLogger("insightaudio")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    
    # Форматтер для всех логов
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Ротация по размеру файла (10MB, до 5 файлов)
    log_file = os.path.join(LOG_DIR, "server.log")
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    # Консольный вывод
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    # Настройка логирования для других модулей
    logging.getLogger("app").setLevel(logging.INFO)
    logging.getLogger("celery").setLevel(logging.INFO)
    
    return logger


ROOT_LOGGER = _configure_logging()


def _setup_whisper_cache():
    """Устанавливает XDG_CACHE_HOME для Whisper, чтобы модели сохранялись в models/.cache/"""
    from app.models import MODELS_STORAGE
    if MODELS_STORAGE:
        whisper_cache_home = os.path.join(MODELS_STORAGE, ".cache")
        os.makedirs(whisper_cache_home, exist_ok=True)
        # Устанавливаем глобально, чтобы Whisper всегда использовал правильный путь
        if "XDG_CACHE_HOME" not in os.environ:
            os.environ["XDG_CACHE_HOME"] = whisper_cache_home
            ROOT_LOGGER.info("Установлен XDG_CACHE_HOME=%s для сохранения моделей Whisper", whisper_cache_home)


_setup_whisper_cache()


@app.on_event("startup")
def _startup():
    ensure_tables()


def _cleanup_old_temp_files():
    """Очищает временные файлы старше 1 суток"""
    cfg = config_manager.get_config()
    results_dir = os.path.abspath(cfg.get("RESULTS_DIR", "/tmp"))
    now = datetime.now()
    deleted_count = 0
    total_size = 0
    cache_dirs = [
        os.path.join(results_dir, ".wav_cache"),
        os.path.join(results_dir, ".transcribe_cache"),
    ]

    def cleanup_dir(directory: str):
        nonlocal deleted_count, total_size
        if not os.path.exists(directory):
            return
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if not os.path.isfile(file_path):
                continue
            try:
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                if now - file_time >= timedelta(days=1):
                    file_size = os.path.getsize(file_path)
                    os.remove(file_path)
                    deleted_count += 1
                    total_size += file_size
                    ROOT_LOGGER.debug("Удален устаревший файл: %s (возраст: %s)", filename, now - file_time)
            except Exception as exc:
                ROOT_LOGGER.warning("Ошибка при удалении файла %s: %s", filename, exc)

    try:
        for directory in cache_dirs:
            cleanup_dir(directory)

        # Также очищаем другие временные файлы в results_dir (кроме кешей)
        for filename in os.listdir(results_dir):
            if filename in {".wav_cache", ".transcribe_cache"}:
                continue
            file_path = os.path.join(results_dir, filename)
            if os.path.isfile(file_path):
                try:
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if now - file_time >= timedelta(days=1):
                        file_size = os.path.getsize(file_path)
                        os.remove(file_path)
                        deleted_count += 1
                        total_size += file_size
                        ROOT_LOGGER.debug("Удален устаревший временный файл: %s", filename)
                except Exception as exc:
                    ROOT_LOGGER.warning("Ошибка при удалении временного файла %s: %s", filename, exc)

        if deleted_count > 0:
            ROOT_LOGGER.info(
                "Очищено %d устаревших временных файлов (освобождено ~%.2f МБ)",
                deleted_count,
                total_size / (1024 * 1024),
            )
    except Exception as exc:
        ROOT_LOGGER.warning("Ошибка при очистке временных файлов: %s", exc)


_cleanup_old_temp_files()


class RequestLogBufferHandler(logging.Handler):
    def __init__(self, level=logging.INFO):
        super().__init__(level=level)
        self.records: List[str] = []
        self.setFormatter(logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

    def emit(self, record):
        self.records.append(self.format(record))

    def dump(self) -> List[str]:
        return self.records


def _get_config_view() -> Dict:
    cfg = config_manager.get_config()
    prompts = []
    try:
        import json as _json
        from pathlib import Path
        # Пробуем несколько путей для поиска prompt_templates.json
        config_dir = cfg.get("CONFIG_DIR", config_manager.CONFIG_DIR)
        possible_paths = [
            Path(config_dir) / "prompt_templates.json",
            Path(BASE_DIR) / ".." / "config" / "prompt_templates.json",
            Path("config") / "prompt_templates.json",
        ]
        for tmpl_path in possible_paths:
            tmpl_path = tmpl_path.resolve()
            if tmpl_path.exists():
                prompts = _json.loads(tmpl_path.read_text(encoding="utf-8"))
                ROOT_LOGGER.info("Loaded %d prompt templates from %s", len(prompts), tmpl_path)
                break
        if not prompts:
            ROOT_LOGGER.warning("Prompt templates not found in any of: %s", [str(p) for p in possible_paths])
    except Exception as e:
        ROOT_LOGGER.error("Error loading prompt templates: %s", e, exc_info=True)
        prompts = []
    return {
        "MODEL_DIR": cfg.get("MODEL_DIR"),
        "CONFIG_DIR": cfg.get("CONFIG_DIR", config_manager.CONFIG_DIR),
        "RESULTS_DIR": cfg.get("RESULTS_DIR"),
        "SUMMARY_BACKEND": cfg.get("SUMMARY_BACKEND", "ollama"),
        "OLLAMA_API_BASE": cfg.get("OLLAMA_API_BASE"),
        "DEFAULT_TRANSLATE_MODEL": cfg.get("DEFAULT_TRANSLATE_MODEL", cfg.get("DEFAULT_SUMMARIZE_MODEL")),
        "TRANSLATION_TARGET_LANG": cfg.get("TRANSLATION_TARGET_LANG", "ru"),
        "DEFAULT_TRANSLATION_MODE": cfg.get("DEFAULT_TRANSLATION_MODE", "block"),
        "DEFAULT_PDF_REFLOW": cfg.get("DEFAULT_PDF_REFLOW", False),
        "DEFAULT_IMAGE_TRANSLATION_MODE": cfg.get("DEFAULT_IMAGE_TRANSLATION_MODE", "notes"),
        "ENABLE_SPEAKER_DIARIZATION": cfg.get("ENABLE_SPEAKER_DIARIZATION", True),
        "PROMPT_TEMPLATES": prompts,
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


def _cookie_name() -> str:
    cfg = config_manager.get_config()
    return cfg.get("SESSION_COOKIE_NAME", "insight_session")


@app.middleware("http")
async def assign_session_cookie(request: Request, call_next):
    cookie_name = _cookie_name()
    session_id = request.cookies.get(cookie_name)
    if not session_id:
        session_id = uuid.uuid4().hex
        request.state.new_session_id = session_id
    request.state.session_id = session_id
    response = await call_next(request)
    if getattr(request.state, "new_session_id", None):
        response.set_cookie(
            cookie_name,
            session_id,
            httponly=True,
            samesite="lax",
            max_age=90 * 24 * 3600,
        )
    return response


def _get_user(request: Request, db: Session) -> User:
    session_id = getattr(request.state, "session_id", None)
    if not session_id:
        session_id = uuid.uuid4().hex
    return get_or_create_user(db, session_id)


def _check_owner(job: Job, user: User):
    if not job or job.user_id != user.id:
        raise HTTPException(status_code=404, detail="Задача не найдена")


def _job_to_dict(job: Job) -> Dict:
    return {
        "id": job.id,
        "user_id": job.user_id,
        "job_type": job.job_type,
        "status": job.status,
        "stage": job.stage,
        "progress": job.progress,
        "eta_seconds": job.eta_seconds,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "params": job.params_json,
        "error_message": job.error_message,
        "error_traceback": job.error_traceback,
        "result_manifest": job.result_manifest_json,
    }


def _enforce_size_limit(path: str):
    cfg = config_manager.get_config()
    limit_mb = cfg.get("MAX_UPLOAD_MB")
    if limit_mb is None:
        return
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb > float(limit_mb):
        try:
            os.remove(path)
        except OSError:
            pass
        raise HTTPException(status_code=413, detail=f"Файл превышает лимит {limit_mb} MB")


@app.post("/api/jobs/audio")
async def create_audio_job(
    request: Request,
    file: UploadFile = File(...),
    params: str = Form("{}"),
    db: Session = Depends(get_session),
):
    user = _get_user(request, db)
    try:
        params_dict = json.loads(params or "{}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="params должен быть валидным JSON")

    job_id = uuid.uuid4().hex
    job_dir = build_user_job_dir(user.id, job_id)
    saved_path = file_utils.handle_upload(file, save_dir=job_dir)
    _enforce_size_limit(saved_path)
    params_dict.update(
        {
            "input_path": saved_path,
            "original_filename": file.filename,
            "asr_model": params_dict.get("asr_model") or params_dict.get("transcribe_model") or "whisper-tiny",
        }
    )
    job = create_job(db, user_id=user.id, job_type="audio", stage="upload_saved", params=params_dict, job_id=job_id)
    audio_job.delay(job.id, user.id, params_dict)
    return {"job_id": job.id}


@app.post("/api/jobs/doc_translate")
async def create_doc_job(
    request: Request,
    file: UploadFile = File(...),
    params: str = Form("{}"),
    db: Session = Depends(get_session),
):
    user = _get_user(request, db)
    try:
        params_dict = json.loads(params or "{}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="params должен быть валидным JSON")

    job_id = uuid.uuid4().hex
    job_dir = build_user_job_dir(user.id, job_id)
    saved_path = file_utils.handle_upload(file, save_dir=job_dir)
    _enforce_size_limit(saved_path)
    params_dict.update(
        {
            "input_path": saved_path,
            "original_filename": file.filename,
        }
    )
    job = create_job(db, user_id=user.id, job_type="doc_translate", stage="upload_saved", params=params_dict, job_id=job_id)
    doc_job.delay(job.id, user.id, params_dict)
    return {"job_id": job.id}


@app.get("/api/jobs")
def list_jobs(request: Request, limit: int = 50, db: Session = Depends(get_session)):
    user = _get_user(request, db)
    jobs = (
        db.execute(select(Job).where(Job.user_id == user.id).order_by(desc(Job.created_at)).limit(limit)).scalars().all()
    )
    return [_job_to_dict(job) for job in jobs]


@app.get("/api/jobs/{job_id}")
def get_job(request: Request, job_id: str, db: Session = Depends(get_session)):
    user = _get_user(request, db)
    job = db.get(Job, job_id)
    _check_owner(job, user)
    return _job_to_dict(job)


@app.get("/api/jobs/{job_id}/download/{asset_name}")
def download_asset(request: Request, job_id: str, asset_name: str, db: Session = Depends(get_session)):
    user = _get_user(request, db)
    job = db.get(Job, job_id)
    _check_owner(job, user)
    manifest = job.result_manifest_json or []
    names = {item.get("name") for item in manifest if isinstance(item, dict)}
    if asset_name not in names:
        raise HTTPException(status_code=404, detail="Файл не найден")
    job_dir = build_user_job_dir(user.id, job_id)
    filepath = os.path.join(job_dir, asset_name)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Файл не найден")
    return FileResponse(filepath, media_type="application/octet-stream", filename=asset_name)


def _is_terminal(job: Job) -> bool:
    return job.status in {"success", "failed", "canceled"}


@app.get("/api/jobs/{job_id}/events")
async def job_events(request: Request, job_id: str):
    # SSE stream with periodic polling of DB
    async def event_stream():
        while True:
            if await request.is_disconnected():
                break
            with SessionLocal() as session:
                user = _get_user(request, session)
                job = session.get(Job, job_id)
                _check_owner(job, user)
                payload = _job_to_dict(job)
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            if _is_terminal(job):
                break
            await asyncio.sleep(1)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/list_models")
def list_models(
    summary_backend: Optional[str] = Query(None),
    custom_models: Optional[str] = Query(None),
    translate_backend: Optional[str] = Query(None),
    translate_custom_models: Optional[str] = Query(None),
):
    custom_list = []
    if custom_models:
        custom_list = [item.strip() for item in custom_models.split(",") if item.strip()]
    translate_custom_list = []
    if translate_custom_models:
        translate_custom_list = [item.strip() for item in translate_custom_models.split(",") if item.strip()]
    data = model_registry.list_all_models(
        summary_backend=summary_backend,
        custom_models=custom_list or None,
        translate_backend=translate_backend,
        translate_custom_models=translate_custom_list or None,
    )
    return data


@app.post("/api/check_custom_api")
async def check_custom_api(
    api_url: str = Form(...),
    scope: str = Form("summarize"),  # summarize или translate
):
    """Проверяет подключение к пользовательскому API и получает список доступных моделей"""
    try:
        import requests
        # Пробуем разные варианты endpoints для получения списка моделей
        endpoints_to_try = [
            f"{api_url.rstrip('/')}/models",
            f"{api_url.rstrip('/')}/api/models",
            f"{api_url.rstrip('/')}/list",
            f"{api_url.rstrip('/')}/api/list",
            f"{api_url.rstrip('/')}/v1/models",
            f"{api_url.rstrip('/')}/api/v1/models",
            f"{api_url.rstrip('/')}/model/list",
            f"{api_url.rstrip('/')}/api/model/list",
        ]
        
        # Также пробуем POST запросы с разными форматами
        post_endpoints_to_try = [
            f"{api_url.rstrip('/')}/models",
            f"{api_url.rstrip('/')}/api/models",
            f"{api_url.rstrip('/')}/list",
        ]
        
        models = []
        error_msg = None
        
        def parse_models_response(data):
            """Парсит ответ API и извлекает список моделей"""
            parsed_models = []
            if isinstance(data, list):
                parsed_models = [{"name": str(m), "display_name": str(m)} for m in data]
            elif isinstance(data, dict):
                # Пробуем разные ключи
                for key in ["models", "data", "items", "results", "model_list"]:
                    if key in data:
                        models_list = data[key]
                        if isinstance(models_list, list):
                            parsed_models = [
                                {
                                    "name": m.get("name") or m.get("model") or m.get("id") or str(m),
                                    "display_name": m.get("display_name") or m.get("name") or m.get("model") or m.get("id") or str(m),
                                }
                                for m in models_list
                            ]
                            break
                # Если не нашли в стандартных ключах, пробуем найти массив в значениях
                if not parsed_models:
                    for value in data.values():
                        if isinstance(value, list) and value:
                            parsed_models = [
                                {
                                    "name": m.get("name") or m.get("model") or m.get("id") or str(m),
                                    "display_name": m.get("display_name") or m.get("name") or m.get("model") or m.get("id") or str(m),
                                }
                                for m in value
                            ]
                            break
            return parsed_models
        
        # Пробуем GET запросы
        for endpoint in endpoints_to_try:
            try:
                response = requests.get(endpoint, timeout=10)
                if response.status_code == 200:
                    try:
                        data = response.json()
                        parsed = parse_models_response(data)
                        if parsed:
                            models = parsed
                            break
                    except ValueError:
                        # Не JSON ответ, пробуем следующий endpoint
                        continue
            except requests.RequestException as e:
                error_msg = str(e)
                continue
        
        # Если GET не сработал, пробуем POST запросы
        if not models:
            for endpoint in post_endpoints_to_try:
                try:
                    # Пробуем разные форматы POST запросов
                    for payload in [{}, {"action": "list"}, {"method": "list"}, {"query": "models"}]:
                        response = requests.post(endpoint, json=payload, timeout=10)
                        if response.status_code == 200:
                            try:
                                data = response.json()
                                parsed = parse_models_response(data)
                                if parsed:
                                    models = parsed
                                    break
                            except ValueError:
                                continue
                    if models:
                        break
                except requests.RequestException:
                    continue
        
        # Если не получилось получить список моделей, проверяем хотя бы доступность API
        if not models:
            try:
                # Пробуем простой health check или ping
                test_response = requests.get(api_url.rstrip('/'), timeout=5)
                if test_response.status_code < 500:
                    # API доступен, но список моделей получить не удалось
                    return {
                        "status": "connected",
                        "models": [],
                        "message": "API доступен, но список моделей получить не удалось. Укажите модели вручную.",
                    }
            except requests.RequestException:
                pass
            
            return {
                "status": "error",
                "models": [],
                "message": error_msg or "Не удалось подключиться к API или получить список моделей",
            }
        
        return {
            "status": "success",
            "models": models,
            "message": f"Найдено моделей: {len(models)}",
        }
    except Exception as e:
        return {
            "status": "error",
            "models": [],
            "message": f"Ошибка: {str(e)}",
        }


@app.post("/download_model")
def download_model(model_name: str = Form(...), model_type: str = Form(...), backend: Optional[str] = Form(None)):
    ROOT_LOGGER.info("Запрошено скачивание модели: %s (%s)", model_name, model_type)
    return model_registry.download_model(model_name, model_type, backend=backend)


@app.post("/translate_document")
async def translate_document(
    request: Request,
    file: UploadFile = File(...),
    target_language: str = Form("ru"),
    translate_model: Optional[str] = Form(None),
    translate_backend: str = Form("ollama"),
    translate_custom_api_url: str = Form(""),
    translate_custom_api_models: str = Form(""),
    pdf_reflow: Optional[str] = Form("off"),
    image_translation_mode: str = Form("notes"),
    translation_mode: Optional[str] = Form(None),
):
    request_id = uuid.uuid4().hex
    cfg = config_manager.get_config()
    request_logger, buffer_handler, file_handler, log_file_path = _create_request_logger(request_id)
    try:
        results_dir = _current_results_dir()
        saved_path = file_utils.handle_upload(file, save_dir=results_dir)
        request_logger.info("Получен документ для перевода: %s", saved_path)
        translate_mode = translation_mode or cfg.get("DEFAULT_TRANSLATION_MODE", "block")
        if translate_mode not in {"block", "full"}:
            translate_mode = cfg.get("DEFAULT_TRANSLATION_MODE", "block")
        _, ext = os.path.splitext(saved_path)
        if ext.lower() == ".pdf" and translate_mode == "full":
            request_logger.info("PDF-регламент: режим единого запроса пока не поддерживается, переключаемся на блочный.")
            translate_mode = "block"
        output_path = translator.translate_document(
            saved_path,
            target_language,
            translate_model,
            backend=translate_backend,
            custom_api_url=translate_custom_api_url or None,
            pdf_reflow=(pdf_reflow == "on" or pdf_reflow is True),
            image_mode=image_translation_mode or cfg.get("DEFAULT_IMAGE_TRANSLATION_MODE", "notes"),
            translation_mode=translate_mode,
        )
        request_logger.info("Перевод завершён: %s", output_path)

        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "files": [],
                "translation_file": os.path.basename(output_path),
                "log_lines": buffer_handler.dump(),
                "request_id": request_id,
                "log_file": os.path.basename(log_file_path),
            },
        )
    except Exception as exc:  # pylint: disable=broad-except
        request_logger.exception("Ошибка перевода: %s", exc)
        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "files": [],
                "translation_file": None,
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
