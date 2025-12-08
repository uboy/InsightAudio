import json
import logging
import os
import traceback
from datetime import datetime
from typing import Dict, List, Optional

from celery import shared_task

LOGGER = logging.getLogger("insightaudio.tasks")

from app import summarizer, transcriber, translator
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

PRESETS = {
    "quality": {"beam_size": 5, "temperature": 0.0, "vad_filter": True, "loudnorm": True},
    "balanced": {"beam_size": 3, "temperature": 0.0, "vad_filter": True, "loudnorm": True},
    "fast": {"beam_size": 1, "temperature": 0.0, "vad_filter": True, "loudnorm": False},
}

STAGE_WEIGHTS = {
    "upload_saved": 5,
    "audio_preprocess": 10,
    "transcribing": 45,
    "summarizing": 20,
    "reviewing": 5,
    "translating": 10,
    "saving": 5,
}


def _calc_progress(stage: str, inner_fraction: float = 0.0) -> int:
    passed = 0
    total = 0
    stages = ["upload_saved", "audio_preprocess", "transcribing", "summarizing", "reviewing", "translating", "saving", "done"]
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


def _save_manifest(
    base_dir: str,
    transcript_text: str,
    summary_text: Optional[str],
    translation_path: Optional[str],
    segments: Optional[List[Dict]] = None,
    meta: Optional[Dict] = None,
) -> List[Dict]:
    manifest: List[Dict] = []
    transcript_path = os.path.join(base_dir, "transcript.txt")
    _write_text(transcript_path, transcript_text)
    manifest.append({"name": "transcript.txt", "kind": "transcript"})
    transcript_json_path = os.path.join(base_dir, "transcript.json")
    with open(transcript_json_path, "w", encoding="utf-8") as f:
        json.dump({"text": transcript_text, "segments": segments or []}, f, ensure_ascii=False, indent=2)
    manifest.append({"name": "transcript.json", "kind": "transcript_json"})

    if summary_text is not None:
        summary_path = os.path.join(base_dir, "summary.md")
        _write_text(summary_path, summary_text)
        manifest.append({"name": "summary.md", "kind": "summary"})

    if translation_path is not None and os.path.exists(translation_path):
        manifest.append({"name": os.path.basename(translation_path), "kind": "translation"})

    if meta is not None:
        meta_path = os.path.join(base_dir, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        manifest.append({"name": "meta.json", "kind": "meta"})
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
            wav_path, duration_sec = transcriber.convert_to_wav(
                audio_path,
                model=params.get("asr_model", "whisper-tiny"),
                loudnorm=preset.get("loudnorm", False),
            )
            params["duration_sec"] = duration_sec
            job.params_json = params

            update_job(session, job_id, stage="transcribing", progress=25)
            transcription = transcriber.transcribe(
                wav_path,
                model=params.get("asr_model", "whisper-tiny"),
                language=params.get("language"),
                beam_size=preset.get("beam_size"),
                temperature=preset.get("temperature"),
                vad_filter=preset.get("vad_filter"),
                no_speech_threshold=params.get("no_speech_threshold"),
                enable_diarization=params.get("enable_diarization", True),
                task=params.get("task"),
            )

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
                    from app import config_manager
                    cfg = config_manager.get_config()
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
            meta = {"params": params, "duration_sec": duration_sec, "model": params.get("asr_model")}
            manifest = _save_manifest(job_dir, transcript_text, summary_text, translation_path, segments=segments, meta=meta)
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
            manifest = [{"name": os.path.basename(translated_path), "kind": "translation"}]
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


@shared_task(name="app.tasks.cleanup_expired_jobs")
def cleanup_expired_jobs():
    try:
        removed = prune_expired_jobs()
        return {"removed": removed}
    except Exception:
        return {"removed": 0, "error": traceback.format_exc()}

