import gzip
import json
import logging
import os
import shutil
import threading
import traceback
import time
from datetime import datetime
from typing import Dict, List, Optional

from app import config_manager, summarizer, transcriber, translator
from app.models import is_model_downloaded
from app.celery_app import celery_app
from app.job_service import (
    build_user_job_dir,
    ensure_tables,
    prune_expired_jobs,
    update_job,
)
from app.db import session_scope
from app.db_models import Job
import json
from app.db import SessionLocal
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout


def _configure_worker_logging() -> logging.Logger:
    """Ensure celery workers have the same logging config as the web app."""
    cfg = config_manager.get_config()
    level_name = str(cfg.get("LOG_LEVEL", "DEBUG")).upper()
    level = getattr(logging, level_name, logging.DEBUG)
    log_dir = os.path.abspath(cfg.get("LOG_DIR", "./logs"))
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "server.log")

    logger = logging.getLogger("insightaudio")
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
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
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)

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

    logging.getLogger("app").setLevel(level)
    logging.getLogger("celery").setLevel(level)

    # Поднимаем уровень и хендлеры на root, чтобы забирать все логи Celery/3rd party
    root = logging.getLogger()
    root.setLevel(level)
    for h in logger.handlers:
        if h not in root.handlers:
            root.addHandler(h)

    logger.info("Worker logging configured level=%s file=%s", level_name, log_file)
    return logger


LOGGER = _configure_worker_logging()

PRESETS = {
    "quality": {"beam_size": 5, "temperature": 0.0, "vad_filter": True, "loudnorm": True},
    "balanced": {"beam_size": 3, "temperature": 0.0, "vad_filter": True, "loudnorm": True},
    "fast": {"beam_size": 1, "temperature": 0.0, "vad_filter": True, "loudnorm": False},
}

STAGE_WEIGHTS = {
    "upload_saved": 5,
    "audio_preprocess": 10,
    "loading_model": 5,
    "transcribing": 45,
    "summarizing": 20,
    "reviewing": 5,
    "translating": 10,
    "saving": 5,
}


def _calc_progress(stage: str, inner_fraction: float = 0.0) -> int:
    passed = 0
    total = 0
    stages = [
        "upload_saved",
        "audio_preprocess",
        "loading_model",
        "transcribing",
        "summarizing",
        "reviewing",
        "translating",
        "saving",
        "done",
    ]
    for s in stages:
        weight = STAGE_WEIGHTS.get(s, 0)
        if s == stage:
            passed += weight * max(0.0, min(1.0, inner_fraction))
            total += weight
            break
        passed += weight
        total += weight
    if total == 0:
        return 0
    return int((passed / sum(STAGE_WEIGHTS.values())) * 100)


def _estimate_eta_transcribe(duration_sec: Optional[float], last_end: float, rtf: float = 0.5) -> Optional[int]:
    if not duration_sec or duration_sec <= 0:
        return None
    remaining = max(duration_sec - last_end, 0)
    rtf = max(rtf, 0.1)
    return int(remaining / rtf)


def _start_keepalive(job_id: str, stage: str, interval_sec: int = 30):
    """
    Periodically touches the job updated_at to avoid stale cleanup during long operations (e.g. model load).
    Uses a separate session per tick for thread safety.
    """
    stop_event = threading.Event()

    def _run():
        while not stop_event.wait(interval_sec):
            try:
                with SessionLocal() as sess:
                    update_job(sess, job_id, stage=stage)
            except Exception as exc:
                LOGGER.warning("Job %s: keepalive tick failed: %s", job_id, exc)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    def _stop():
        stop_event.set()
    return _stop


def _start_transcribe_heartbeat(job_id: str, duration_sec: Optional[float], stage: str = "transcribing", interval_sec: int = 15):
    """
    Periodically bumps progress based on elapsed/duration to show activity until real segments arrive.
    Returns (stop_fn, bump_fn) where bump_fn keeps max progress in sync with real updates.
    """
    stop_event = threading.Event()
    start_ts = time.time()
    max_progress = 0

    def bump(progress: int):
        nonlocal max_progress
        if progress > max_progress:
            max_progress = progress

    def _run():
        nonlocal max_progress
        while not stop_event.wait(interval_sec):
            if not duration_sec or duration_sec <= 0:
                continue
            elapsed = time.time() - start_ts
            inner = min(0.95, elapsed / duration_sec)  # don't reach 100% by heartbeat alone
            progress = max(max_progress, _calc_progress(stage, inner))
            try:
                with SessionLocal() as sess:
                    update_job(sess, job_id, stage=stage, progress=progress)
            except Exception as exc:
                LOGGER.warning("Job %s: transcribe heartbeat tick failed: %s", job_id, exc)
            max_progress = progress
            LOGGER.debug("Job %s: heartbeat tick progress=%d", job_id, progress)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    def _stop():
        stop_event.set()
    return _stop, bump


def _summarize_chunked(text: str, params: Dict, segments: List[Dict]) -> str:
    max_chunk = 5000
    overlap = 500
    if len(text) <= max_chunk:
        return summarizer.summarize(
            text,
            model=params.get("summary_model", params.get("summary_model_name", "llama3:8b")),
            backend=params.get("summary_backend", "ollama"),
            prompt=params.get("custom_prompt"),
            segments=segments,
            custom_api_url=params.get("summary_custom_api_url"),
        )
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chunk)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    partial_summaries = []
    for chunk in chunks:
        partial_summaries.append(
            summarizer.summarize(
                chunk,
                model=params.get("summary_model", params.get("summary_model_name", "llama3:8b")),
                backend=params.get("summary_backend", "ollama"),
                prompt=params.get("custom_prompt"),
                segments=segments,
                custom_api_url=params.get("summary_custom_api_url"),
            )
        )
    combined = "\n\n".join(partial_summaries)
    return summarizer.summarize(
        combined,
        model=params.get("summary_model", params.get("summary_model_name", "llama3:8b")),
        backend=params.get("summary_backend", "ollama"),
        prompt=params.get("custom_prompt") or "Собери общий пересказ по частям выше.",
        custom_api_url=params.get("summary_custom_api_url"),
    )


def _write_text(path: str, content: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content or "")
    return path


def _add_manifest_item(manifest: List[Dict], file_path: str, kind: str, base_dir: str):
    """Добавляет элемент в manifest с проверкой существования файла"""
    if not os.path.exists(file_path):
        return
    file_name = os.path.basename(file_path)
    # Проверяем что файл находится внутри base_dir (защита от path traversal)
    try:
        real_file_path = os.path.realpath(file_path)
        real_base_dir = os.path.realpath(base_dir)
        if not real_file_path.startswith(real_base_dir):
            LOGGER.warning("Файл %s находится вне job_dir %s, пропускаем", file_path, base_dir)
            return
    except Exception as e:
        LOGGER.warning("Ошибка при проверке пути файла %s: %s", file_path, e)
        return
    
    size_bytes = os.path.getsize(file_path)
    created_at = datetime.utcnow().isoformat()
    manifest.append({
        "name": file_name,
        "kind": kind,
        "size_bytes": size_bytes,
        "created_at": created_at,
    })


def _save_manifest(
    base_dir: str,
    transcript_text: str,
    summary_text: Optional[str],
    translation_path: Optional[str],
    segments: Optional[List[Dict]] = None,
    meta: Optional[Dict] = None,
    input_original_path: Optional[str] = None,
    audio_wav_path: Optional[str] = None,
) -> List[Dict]:
    """
    Сохраняет результаты задачи и создает manifest.
    Все файлы должны находиться внутри base_dir (job_dir).
    """
    manifest: List[Dict] = []
    
    # Сохраняем input_original если указан
    if input_original_path and os.path.exists(input_original_path):
        # Копируем оригинальный файл в job_dir
        original_name = os.path.basename(input_original_path)
        original_ext = os.path.splitext(original_name)[1]
        dest_original = os.path.join(base_dir, f"input_original{original_ext}")
        try:
            shutil.copy2(input_original_path, dest_original)
            _add_manifest_item(manifest, dest_original, "input", base_dir)
        except Exception as e:
            LOGGER.warning("Не удалось скопировать оригинальный файл: %s", e)
    
    # Сохраняем audio.wav если указан
    if audio_wav_path and os.path.exists(audio_wav_path):
        # Копируем WAV файл в job_dir если он не там уже
        wav_name = "audio.wav"
        dest_wav = os.path.join(base_dir, wav_name)
        try:
            real_wav = os.path.realpath(audio_wav_path)
            real_base = os.path.realpath(base_dir)
            if not real_wav.startswith(real_base):
                shutil.copy2(audio_wav_path, dest_wav)
            else:
                dest_wav = audio_wav_path
            _add_manifest_item(manifest, dest_wav, "wav", base_dir)
        except Exception as e:
            LOGGER.warning("Не удалось сохранить WAV файл: %s", e)
    
    # Сохраняем transcript.txt
    transcript_path = os.path.join(base_dir, "transcript.txt")
    _write_text(transcript_path, transcript_text)
    _add_manifest_item(manifest, transcript_path, "transcript_txt", base_dir)
    
    # Сохраняем transcript.json
    transcript_json_path = os.path.join(base_dir, "transcript.json")
    with open(transcript_json_path, "w", encoding="utf-8") as f:
        json.dump({"text": transcript_text, "segments": segments or []}, f, ensure_ascii=False, indent=2)
    _add_manifest_item(manifest, transcript_json_path, "transcript_json", base_dir)

    # Сохраняем summary.md
    if summary_text is not None:
        summary_path = os.path.join(base_dir, "summary.md")
        _write_text(summary_path, summary_text)
        _add_manifest_item(manifest, summary_path, "summary_md", base_dir)

    # Сохраняем translation
    if translation_path is not None and os.path.exists(translation_path):
        # Если translation_path не в job_dir, копируем его туда
        real_translation = os.path.realpath(translation_path)
        real_base = os.path.realpath(base_dir)
        if not real_translation.startswith(real_base):
            translation_name = os.path.basename(translation_path)
            dest_translation = os.path.join(base_dir, translation_name)
            try:
                shutil.copy2(translation_path, dest_translation)
                _add_manifest_item(manifest, dest_translation, "translation", base_dir)
            except Exception as e:
                LOGGER.warning("Не удалось скопировать файл перевода: %s", e)
        else:
            _add_manifest_item(manifest, translation_path, "translation", base_dir)

    # Сохраняем meta.json
    if meta is not None:
        meta_path = os.path.join(base_dir, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        _add_manifest_item(manifest, meta_path, "meta", base_dir)
    
    return manifest


@celery_app.task(name="app.tasks.audio_job", bind=True)
def audio_job(self, job_id: str, user_id: str, params: Dict) -> str:
    ensure_tables()
    with session_scope() as session:
        job = session.get(Job, job_id)
        if not job:
            return job_id
        job_dir = build_user_job_dir(user_id, job_id)
        try:
            update_job(session, job_id, status="running", stage="audio_preprocess", started_at=datetime.utcnow())
            audio_path = params["input_path"]
            preset = PRESETS.get(params.get("preset") or "balanced", PRESETS["balanced"])
            LOGGER.info(
                "Job %s: Starting audio preprocess. input=%s preset=%s loudnorm=%s",
                job_id,
                audio_path,
                params.get("preset") or "balanced",
                preset.get("loudnorm", False),
            )
            wav_path, duration_sec = transcriber.convert_to_wav(
                audio_path,
                model=params.get("asr_model", "whisper-tiny"),
                loudnorm=preset.get("loudnorm", False),
            )
            params["duration_sec"] = duration_sec
            job.params_json = params
            update_job(session, job_id, stage="audio_preprocess", progress=_calc_progress("audio_preprocess", 1.0))
            LOGGER.info(
                "Job %s: Audio converted to wav=%s, duration=%.2fs",
                job_id,
                wav_path,
                duration_sec or 0,
            )
            estimated_eta = _estimate_eta_transcribe(duration_sec, 0, params.get("rtf", 0.5) or 0.5)
            cfg = config_manager.get_config()
            load_interval = int(cfg.get("ASR_LOAD_HEARTBEAT_SEC", 20))
            transcribe_interval = int(cfg.get("ASR_TRANSCRIBE_HEARTBEAT_SEC", 20))
            update_job(session, job_id, stage="loading_model", progress=_calc_progress("loading_model", 0.05), eta_seconds=estimated_eta)
            LOGGER.info("Job %s: loading_model heartbeat interval=%ss transcribe heartbeat interval=%ss", job_id, load_interval, transcribe_interval)
            
            # Определяем движок ASR
            asr_engine = params.get("asr_engine", "auto")
            
            # Определяем vad_filter только для faster-whisper
            vad_filter_value = None
            if asr_engine == "faster_whisper" or (asr_engine == "auto" and params.get("asr_model", "").startswith("faster-whisper")):
                vad_filter_value = preset.get("vad_filter")
            elif asr_engine == "openai_whisper":
                # vad_filter не поддерживается в openai-whisper
                if preset.get("vad_filter") is not None:
                    LOGGER.warning("Job %s: vad_filter не поддерживается в openai-whisper, игнорируется", job_id)
                vad_filter_value = None
            LOGGER.info(
                "Job %s: Transcribing start model=%s engine=%s language=%s vad_filter=%s beam=%s temp=%s diarization=%s",
                job_id,
                params.get("asr_model", "whisper-tiny"),
                asr_engine,
                params.get("language"),
                vad_filter_value,
                preset.get("beam_size"),
                preset.get("temperature"),
                params.get("enable_diarization", True),
            )
            
            # Если модель faster-whisper не скачана, не ждём бесконечной загрузки
            asr_model = params.get("asr_model", "whisper-tiny")
            downloaded = is_model_downloaded(asr_model, "transcribe")
            LOGGER.info("Job %s: ASR model %s downloaded=%s", job_id, asr_model, downloaded)
            if asr_engine == "faster_whisper" and asr_model.startswith("faster-whisper") and not downloaded:
                msg = (
                    f"ASR model {asr_model} not downloaded. "
                    "Download it via UI/CLI or choose a smaller model."
                )
                LOGGER.error("Job %s: %s", job_id, msg)
                update_job(
                    session,
                    job_id,
                    status="failed",
                    stage="failed",
                    progress=100,
                    error_message=msg,
                    finished_at=datetime.utcnow(),
                )
                return job_id
            
            # Callback для обновления прогресса (для faster-whisper)
            def progress_callback(inner_progress: int, last_segment_end: float):
                """Обновляет прогресс транскрипции"""
                if duration_sec and duration_sec > 0:
                    inner_fraction = min(1.0, last_segment_end / duration_sec)
                    pct = _calc_progress("transcribing", inner_fraction)
                    eta = _estimate_eta_transcribe(duration_sec, last_segment_end, params.get("rtf", 0.5) or 0.5)
                    heartbeat_bump(pct)
                    try:
                        with SessionLocal() as progress_session:
                            update_job(progress_session, job_id, stage="transcribing", progress=pct, eta_seconds=eta)
                    except Exception as exc:
                        LOGGER.debug("Job %s: progress callback update failed: %s", job_id, exc)
                    LOGGER.debug("Job %s: progress callback inner=%.3f pct=%d eta=%s", job_id, inner_fraction, pct, eta)
            
            stop_keepalive = _start_keepalive(job_id, "loading_model", interval_sec=load_interval)
            stop_heartbeat, heartbeat_bump = _start_transcribe_heartbeat(job_id, duration_sec, interval_sec=transcribe_interval)
            timeout_sec = int(cfg.get("ASR_TIMEOUT_SECONDS", 1200))
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(
                transcriber.transcribe,
                wav_path,
                params.get("asr_model", "whisper-tiny"),
                asr_engine=asr_engine,
                language=params.get("language"),
                beam_size=preset.get("beam_size"),
                temperature=preset.get("temperature"),
                vad_filter=vad_filter_value,
                no_speech_threshold=params.get("no_speech_threshold"),
                enable_diarization=params.get("enable_diarization", True),
                task=params.get("task"),
                progress_callback=progress_callback if asr_engine in ("faster_whisper", "auto") else None,
            )
            try:
                transcription = future.result(timeout=timeout_sec)
            except FuturesTimeout:
                future.cancel()
                LOGGER.error("Job %s: Transcription timed out after %ss", job_id, timeout_sec)
                raise RuntimeError(f"ASR timeout after {timeout_sec}s (model load/inference took too long)")
            finally:
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
                try:
                    stop_keepalive()
                except Exception:
                    pass
                try:
                    stop_heartbeat()
                except Exception:
                    pass
            update_job(session, job_id, stage="transcribing", progress=_calc_progress("transcribing", 0.05), eta_seconds=estimated_eta)
            LOGGER.info(
                "Job %s: Transcription finished. duration=%.2fs segments=%d text_chars=%d",
                job_id,
                transcription.get("duration_sec") or duration_sec or 0,
                len(transcription.get("segments") or []),
                len(transcription.get("text") or ""),
            )
            update_job(session, job_id, stage="transcribing", progress=_calc_progress("transcribing", 1.0), eta_seconds=None)

            transcript_text = transcription.get("text", "")
            segments = transcription.get("segments", [])
            duration_sec = transcription.get("duration_sec") or params.get("duration_sec")
            if duration_sec and segments:
                last_end = max((seg.get("end") or 0 for seg in segments), default=0)
                eta = _estimate_eta_transcribe(duration_sec, last_end, params.get("rtf", 0.5) or 0.5)
                pct = _calc_progress("transcribing", inner_fraction=min(1.0, last_end / duration_sec))
                update_job(session, job_id, stage="transcribing", progress=pct, eta_seconds=eta)

            summary_text = None
            if params.get("enable_summary"):
                LOGGER.info("Job %s: Starting summarization with model %s", job_id, params.get("summary_model"))
                update_job(session, job_id, stage="summarizing", progress=_calc_progress("summarizing"))
                summary_text = _summarize_chunked(transcript_text, params, segments)
                LOGGER.info("Job %s: Summarization completed, length: %d chars", job_id, len(summary_text) if summary_text else 0)
                
                # Если указана модель для ревью, выполняем ревью пересказа
                review_model = params.get("review_model")
                if review_model and summary_text:
                    LOGGER.info("Job %s: Starting review with model %s", job_id, review_model)
                    update_job(session, job_id, stage="reviewing", progress=_calc_progress("reviewing"))
                    
                    # Используем промпт из параметров, конфига или стандартный промпт для ревью
                    review_prompt_template = params.get("review_prompt") or cfg.get("REVIEW_PROMPT", 
                        "Проверь и улучши следующий пересказ. Исправь ошибки, улучши структуру, добавь недостающие детали, если они важны. Сохрани формат Markdown.\n\nПересказ:\n")
                    
                    # Если промпт не содержит текст пересказа, добавляем его
                    if "{summary}" in review_prompt_template:
                        review_prompt = review_prompt_template.format(summary=summary_text)
                    elif "Пересказ:" in review_prompt_template or "пересказ:" in review_prompt_template.lower():
                        # Если уже есть метка "Пересказ:", добавляем текст после неё
                        review_prompt = review_prompt_template + summary_text
                    else:
                        # Если метки нет, добавляем её и текст
                        review_prompt = review_prompt_template + "\n\nПересказ:\n" + summary_text
                    
                    review_text = summarizer.summarize(
                        summary_text,
                        model=review_model,
                        backend=params.get("summary_backend", "ollama"),
                        prompt=review_prompt,
                        custom_api_url=params.get("summary_custom_api_url"),
                    )
                    if review_text and review_text.strip():
                        LOGGER.info("Job %s: Review completed, length: %d chars", job_id, len(review_text))
                        summary_text = review_text
                    else:
                        LOGGER.warning("Job %s: Review returned empty result", job_id)

            translation_path = None
            if params.get("enable_translation"):
                LOGGER.info("Job %s: Starting translation to %s", job_id, params.get("target_language", "en"))
                update_job(session, job_id, stage="translating", progress=_calc_progress("translating"))
                # Reuse translator to translate transcript text; output plain txt
                translation_text = translator._DocTranslator(  # pylint: disable=protected-access
                    backend=params.get("translate_backend", "ollama"),
                    model=params.get("translate_model", params.get("summary_model", "llama3:8b")),
                    target_lang=params.get("target_language", "en"),
                    max_ratio=1.35,
                )._call_translate_api("Переведи текст", transcript_text)  # noqa: SLF001
                translation_path = os.path.join(job_dir, f"translation_{params.get('target_language','en')}.txt")
                _write_text(translation_path, translation_text)
                LOGGER.info("Job %s: Translation completed, saved to %s", job_id, translation_path)

            LOGGER.info("Job %s: Saving results", job_id)
            update_job(session, job_id, stage="saving", progress=_calc_progress("saving"))
            meta = {
                "params": params, 
                "duration_sec": duration_sec, 
                "model": params.get("asr_model"),
                "asr_engine": params.get("asr_engine", "auto"),
            }
            manifest = _save_manifest(
                job_dir, 
                transcript_text, 
                summary_text, 
                translation_path, 
                segments=segments, 
                meta=meta,
                input_original_path=audio_path,
                audio_wav_path=wav_path,
            )
            LOGGER.info("Job %s: Results saved, manifest: %s", job_id, manifest)
            update_job(
                session,
                job_id,
                status="success",
                stage="done",
                progress=100,
                finished_at=datetime.utcnow(),
                result_manifest=manifest,
            )
            LOGGER.info("Job %s: Completed successfully", job_id)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.error("Job %s: Failed with error: %s", job_id, str(exc), exc_info=True)
            update_job(
                session,
                job_id,
                status="failed",
                stage="failed",
                progress=100,
                error_message=str(exc),
                error_traceback=traceback.format_exc(),
                finished_at=datetime.utcnow(),
            )
    return job_id


@celery_app.task(name="app.tasks.doc_job", bind=True)
def doc_job(self, job_id: str, user_id: str, params: Dict) -> str:
    ensure_tables()
    with session_scope() as session:
        job = session.get(Job, job_id)
        if not job:
            return job_id
        job_dir = build_user_job_dir(user_id, job_id)
        try:
            update_job(session, job_id, status="running", stage="extracting", started_at=datetime.utcnow(), progress=10)
            translated_path = translator.translate_document(
                params["input_path"],
                params.get("target_language", "ru"),
                model=params.get("translate_model"),
                backend=params.get("translate_backend", "ollama"),
                custom_api_url=params.get("translate_custom_api_url"),
                pdf_reflow=params.get("pdf_reflow", False),
                image_mode=params.get("image_translation_mode", "notes"),
                translation_mode=params.get("translation_mode", "block"),
            )
            update_job(session, job_id, stage="saving", progress=90)
            
            # Копируем переведенный файл в job_dir если он не там уже
            real_translated = os.path.realpath(translated_path)
            real_job_dir = os.path.realpath(job_dir)
            if not real_translated.startswith(real_job_dir):
                # Файл не в job_dir, копируем его туда
                translated_name = os.path.basename(translated_path)
                dest_translated = os.path.join(job_dir, translated_name)
                try:
                    shutil.copy2(translated_path, dest_translated)
                    translated_path = dest_translated
                    LOGGER.info("Job %s: Скопирован файл перевода в job_dir: %s", job_id, dest_translated)
                except Exception as e:
                    LOGGER.error("Job %s: Не удалось скопировать файл перевода: %s", job_id, e)
                    raise
            
            # Также копируем оригинальный файл в job_dir
            input_path = params["input_path"]
            if input_path and os.path.exists(input_path):
                real_input = os.path.realpath(input_path)
                if not real_input.startswith(real_job_dir):
                    input_name = os.path.basename(input_path)
                    input_ext = os.path.splitext(input_name)[1]
                    dest_input = os.path.join(job_dir, f"input_original{input_ext}")
                    try:
                        shutil.copy2(input_path, dest_input)
                        LOGGER.info("Job %s: Скопирован оригинальный файл в job_dir: %s", job_id, dest_input)
                    except Exception as e:
                        LOGGER.warning("Job %s: Не удалось скопировать оригинальный файл: %s", job_id, e)
            
            # Создаем manifest с правильной информацией
            manifest = []
            _add_manifest_item(manifest, translated_path, "translation", job_dir)
            
            update_job(
                session,
                job_id,
                status="success",
                stage="done",
                progress=100,
                finished_at=datetime.utcnow(),
                result_manifest=manifest,
            )
        except Exception as exc:  # pylint: disable=broad-except
            update_job(
                session,
                job_id,
                status="failed",
                stage="failed",
                progress=100,
                error_message=str(exc),
                error_traceback=traceback.format_exc(),
                finished_at=datetime.utcnow(),
            )
    return job_id


@celery_app.task(name="app.tasks.cleanup_expired_jobs")
def cleanup_expired_jobs():
    """Периодическая задача для очистки устаревших jobs и файлов"""
    from app import config_manager
    from app.transcriber import _get_transcribe_cache_dir, CACHE_TTL
    import time
    
    cfg = config_manager.get_config()
    ttl_days = cfg.get("JOB_TTL_DAYS", 14)
    
    LOGGER.info("Запуск очистки устаревших jobs (TTL: %d дней)", ttl_days)
    try:
        removed_count = prune_expired_jobs(ttl_days=ttl_days)
        LOGGER.info("Очистка завершена: удалено %d jobs", removed_count)
        
        # Также очищаем кэши транскрипций старше TTL
        cache_dir = _get_transcribe_cache_dir()
        cache_removed = 0
        if os.path.exists(cache_dir):
            cutoff_time = time.time() - CACHE_TTL.total_seconds()
            for filename in os.listdir(cache_dir):
                cache_path = os.path.join(cache_dir, filename)
                try:
                    if os.path.isfile(cache_path) and os.path.getmtime(cache_path) < cutoff_time:
                        os.remove(cache_path)
                        cache_removed += 1
                except Exception as e:
                    LOGGER.warning("Не удалось удалить кэш %s: %s", cache_path, e)
        
        if cache_removed > 0:
            LOGGER.info("Удалено %d устаревших кэшей транскрипций", cache_removed)
        
        return {"removed_jobs": removed_count, "removed_cache": cache_removed}
    except Exception as e:
        LOGGER.error("Ошибка при очистке устаревших jobs: %s", e, exc_info=True)
        raise
