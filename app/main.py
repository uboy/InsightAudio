import asyncio
import gzip
import json
import logging
import logging.handlers
import os
import shutil
import uuid
from datetime import datetime, timedelta, timezone
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
from app.job_service import build_user_job_dir, create_job, ensure_tables, get_or_create_user, update_job
from app.tasks import audio_job, doc_job
from app import settings_loader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory="app/templates")
app = FastAPI(title="InsightAudio")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
SERVER_STARTED_AT = datetime.utcnow()
# default extended to avoid false positives on long model loads; can override via env
STALE_RUNNING_MINUTES = int(os.getenv("INSIGHT_STALE_RUNNING_MINUTES", "60"))


def _init_dirs():
    cfg = config_manager.get_config()
    results_dir = os.path.abspath(cfg.get("RESULTS_DIR", "/tmp"))
    log_dir = os.path.abspath(cfg.get("LOG_DIR", os.path.join(BASE_DIR, "..", "logs")))
    # If running as non-root inside container, ensure dirs exist and writable
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "requests"), exist_ok=True)
    # Создаем директорию для кеша WAV файлов
    os.makedirs(os.path.join(results_dir, ".wav_cache"), exist_ok=True)
    return results_dir, log_dir


RESULTS_DIR, LOG_DIR = _init_dirs()
REQUEST_LOG_DIR = os.path.join(LOG_DIR, "requests")


def _configure_logging():
    cfg = config_manager.get_config()
    level_name = str(cfg.get("LOG_LEVEL", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    logger = logging.getLogger("insightaudio")
    logger.setLevel(level)
    
    # Форматтер для всех логов
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Ротация по размеру файла (10MB, до 5 файлов)
    log_file = os.path.join(LOG_DIR, "server.log")
    existing_file = next((h for h in logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)), None)
    if existing_file:
        existing_file.setLevel(level)
        existing_file.setFormatter(formatter)
        file_handler = existing_file
    else:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=int(cfg.get("LOG_FILE_MAX_MB", 5)) * 1024 * 1024,
            backupCount=int(cfg.get("LOG_BACKUP_COUNT", 5)),
            encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        # Сжимаем ротации в gzip
        def _rotator(source, dest):
            with open(source, "rb") as f_in, gzip.open(dest, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            try:
                os.remove(source)
            except OSError:
                pass
        file_handler.rotator = _rotator
        file_handler.namer = lambda name: f"{name}.gz"
        logger.addHandler(file_handler)
    
    # Консольный вывод
    existing_stream = next((h for h in logger.handlers if isinstance(h, logging.StreamHandler)), None)
    if existing_stream:
        existing_stream.setLevel(level)
        existing_stream.setFormatter(formatter)
    else:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(level)
        logger.addHandler(stream_handler)
    
    logger.propagate = True
    
    # Настройка логирования для других модулей
    logging.getLogger("app").setLevel(level)
    logging.getLogger("celery").setLevel(level)

    # Дублируем хендлеры на root, чтобы улавливать всё, даже если сторонние логеры не наследуют insightaudio
    root = logging.getLogger()
    root.setLevel(level)
    for h in logger.handlers:
        if h not in root.handlers:
            root.addHandler(h)

    logger.info("Logging configured level=%s file=%s", level_name, log_file)
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
    _reset_inflight_jobs()


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
    def _iso(dt: Optional[datetime]) -> Optional[str]:
        if not dt:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    return {
        "id": job.id,
        "user_id": job.user_id,
        "job_type": job.job_type,
        "status": job.status,
        "stage": job.stage,
        "progress": job.progress,
        "eta_seconds": job.eta_seconds,
        "created_at": _iso(job.created_at),
        "started_at": _iso(job.started_at),
        "updated_at": _iso(job.updated_at),
        "finished_at": _iso(job.finished_at),
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


def _parse_params_json(params_raw: str) -> Dict:
    try:
        return json.loads(params_raw or "{}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="params должен быть валидным JSON")


def _create_audio_job_record(user: User, upload_file: UploadFile, params_dict: Dict, db: Session) -> str:
    job_id = uuid.uuid4().hex
    job_dir = build_user_job_dir(user.id, job_id)
    saved_path = file_utils.handle_upload(upload_file, save_dir=job_dir)
    _enforce_size_limit(saved_path)
    job_params = dict(params_dict or {})
    job_params.update(
        {
            "input_path": saved_path,
            "original_filename": upload_file.filename,
            "asr_model": job_params.get("asr_model") or job_params.get("transcribe_model") or "whisper-tiny",
        }
    )
    job = create_job(db, user_id=user.id, job_type="audio", stage="upload_saved", params=job_params, job_id=job_id)
    audio_job.delay(job.id, user.id, job_params)
    return job.id


@app.post("/api/jobs/audio")
async def create_audio_job(
    request: Request,
    file: UploadFile = File(...),
    params: str = Form("{}"),
    db: Session = Depends(get_session),
):
    user = _get_user(request, db)
    params_dict = _parse_params_json(params)
    job_id = _create_audio_job_record(user, file, params_dict, db)
    return {"job_id": job_id}


@app.post("/api/jobs/audio/batch")
async def create_audio_jobs_batch(
    request: Request,
    files: List[UploadFile] = File(...),
    params: str = Form("{}"),
    db: Session = Depends(get_session),
):
    user = _get_user(request, db)
    params_dict = _parse_params_json(params)
    cfg = config_manager.get_config()
    max_files = int(cfg.get("MAX_AUDIO_BATCH_FILES", 5))
    upload_files = [f for f in files if f is not None]
    if not upload_files:
        raise HTTPException(status_code=400, detail="Не переданы файлы для обработки")
    if len(upload_files) > max_files:
        raise HTTPException(status_code=400, detail=f"Можно загрузить не более {max_files} файлов за один запрос")

    job_ids: List[str] = []
    for upload_file in upload_files:
        job_params = dict(params_dict or {})
        job_id = _create_audio_job_record(user, upload_file, job_params, db)
        job_ids.append(job_id)
    return {"job_ids": job_ids}


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
    _cleanup_stale_running_jobs(db)
    jobs = (
        db.execute(select(Job).where(Job.user_id == user.id).order_by(desc(Job.created_at)).limit(limit)).scalars().all()
    )
    return [_job_to_dict(job) for job in jobs]


@app.get("/api/jobs/{job_id}")
def get_job(request: Request, job_id: str, db: Session = Depends(get_session)):
    user = _get_user(request, db)
    _cleanup_stale_running_jobs(db)
    job = db.get(Job, job_id)
    _check_owner(job, user)
    return _job_to_dict(job)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(request: Request, job_id: str, db: Session = Depends(get_session)):
    user = _get_user(request, db)
    job = db.get(Job, job_id)
    _check_owner(job, user)
    if _is_terminal(job):
        return {"status": job.status}
    update_job(
        db,
        job_id,
        status="canceled",
        stage="canceled",
        finished_at=datetime.utcnow(),
        error_message="Задача отменена пользователем",
        eta_seconds=None,
    )
    return {"status": "canceled"}


@app.get("/api/jobs/{job_id}/download/{asset_name}")
def download_asset(request: Request, job_id: str, asset_name: str, db: Session = Depends(get_session)):
    """
    Безопасное скачивание файла из job.
    Проверки:
    1. Job принадлежит текущему user
    2. asset_name есть в manifest
    3. Файл реально существует внутри job_dir
    4. Защита от path traversal (только имя файла, без пути)
    """
    user = _get_user(request, db)
    job = db.get(Job, job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    
    # Проверка ownership
    if job.user_id != user.id:
        raise HTTPException(status_code=403, detail="Доступ запрещен")
    
    # Проверка manifest
    manifest = job.result_manifest_json or []
    if not isinstance(manifest, list):
        raise HTTPException(status_code=500, detail="Некорректный manifest")
    
    # Проверяем что asset_name есть в manifest
    manifest_item = None
    for item in manifest:
        if isinstance(item, dict) and item.get("name") == asset_name:
            manifest_item = item
            break
    
    if not manifest_item:
        raise HTTPException(status_code=404, detail="Файл не найден в manifest")
    
    # Защита от path traversal: проверяем что asset_name не содержит путь
    if os.path.sep in asset_name or os.path.altsep and os.path.altsep in asset_name:
        raise HTTPException(status_code=400, detail="Некорректное имя файла")
    
    # Проверяем что файл существует внутри job_dir
    job_dir = build_user_job_dir(user.id, job_id)
    filepath = os.path.join(job_dir, asset_name)
    
    # Дополнительная проверка: реальный путь файла должен быть внутри job_dir
    try:
        real_filepath = os.path.realpath(filepath)
        real_job_dir = os.path.realpath(job_dir)
        if not real_filepath.startswith(real_job_dir):
            raise HTTPException(status_code=403, detail="Доступ запрещен: файл вне job_dir")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при проверке пути: {str(e)}")
    
    if not os.path.exists(filepath) or not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail="Файл не найден")
    
    # Определяем media_type по расширению
    ext = os.path.splitext(asset_name)[1].lower()
    media_types = {
        ".txt": "text/plain",
        ".json": "application/json",
        ".md": "text/markdown",
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".mp4": "video/mp4",
    }
    media_type = media_types.get(ext, "application/octet-stream")
    
    return FileResponse(filepath, media_type=media_type, filename=asset_name)


def _is_terminal(job: Job) -> bool:
    return job.status in {"success", "failed", "canceled"}


def _cleanup_stale_running_jobs(session: Session):
    """
    Помечаем зависшие running-задачи как завершенные с ошибкой, если давно нет обновлений.
    Это помогает после перезапуска сервера, когда реальные воркеры уже не исполняют задачу.
    """
    cutoff = datetime.utcnow() - timedelta(minutes=STALE_RUNNING_MINUTES)
    stale_jobs = (
        session.execute(select(Job).where(Job.status == "running", Job.updated_at < cutoff))
        .scalars()
        .all()
    )
    for job in stale_jobs:
        update_job(
            session,
            job.id,
            status="failed",
            stage="stale_orphaned",
            error_message=f"Задача остановлена: нет обновлений > {STALE_RUNNING_MINUTES} мин (возможно, перезапуск сервера).",
            finished_at=datetime.utcnow(),
            eta_seconds=None,
        )


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


def _reset_inflight_jobs():
    """При старте помечаем зависшие незавершённые jobs как failed после рестарта."""
    with SessionLocal() as session:
        stuck = (
            session.execute(select(Job).where(Job.status.in_(["running", "queued", "pending"])))
            .scalars()
            .all()
        )
        for job in stuck:
            update_job(
                session,
                job.id,
                status="failed",
                stage="failed",
                progress=100,
                error_message="Сервис перезапущен: незавершённая задача остановлена",
                finished_at=datetime.utcnow(),
                eta_seconds=None,
            )
            ROOT_LOGGER.warning("Job %s marked failed on startup (previous status=%s)", job.id, job.status)


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
