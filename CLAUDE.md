# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

InsightAudio is a fully local service for automatic transcription of audio/video files and generation of structured summaries using AI models. It also supports document translation (DOCX, PPTX, XLSX, PDF). The UI is in Russian. All processing happens locally for privacy.

## Architecture

```
Browser UI (Jinja2 + vanilla JS)
    ↓
FastAPI Web Server (port 55667)  ←  app/main.py
    ↓
Celery + Redis Task Queue       ←  app/celery_app.py, app/tasks.py
    ↓
Worker (transcription, summarization, translation)
    ↓
SQLite (SQLAlchemy ORM, WAL mode)  ←  app/db.py, app/db_models.py
```

**Three Docker services** (`docker-compose.yml`): `insightaudio` (web), `insightaudio-worker` (Celery solo pool), `insightaudio-redis` (Redis 7).

**Job pipeline**: Upload → Create Job in DB → Queue Celery task → Worker processes → Store results to `RESULTS_DIR/<user_id>/<job_id>/` → Client polls via SSE.

**Job types**: `audio` (transcription + optional summary), `doc_translate` (document translation).

**Job states**: `queued` → `running` → `completed` / `failed`.

### Key Modules

| Module | Purpose |
|---|---|
| `app/main.py` | FastAPI routes, middleware, startup/shutdown |
| `app/tasks.py` | Celery task definitions (audio_job, doc_job, cleanup_expired_jobs) |
| `app/transcriber.py` | Whisper/faster-whisper ASR integration with VAD and loudnorm |
| `app/summarizer.py` | Summary generation via Ollama, llama.cpp, or custom API |
| `app/translator.py` | Document translation for DOCX/PPTX/XLSX/PDF |
| `app/diarizer.py` | Speaker diarization via NVIDIA NeMo |
| `app/models.py` | Model management, downloading, availability checking |
| `app/config_manager.py` | Config loading with path normalization and env var overrides |
| `app/job_service.py` | Job CRUD operations, manifest validation |
| `app/db.py` | SQLAlchemy engine and session factory |
| `app/db_models.py` | SQLAlchemy models: User, Job |
| `app/ollama_utils.py` | Streaming Ollama API client with metrics |

### Configuration

- `config/default_settings.json` — master defaults (models list, MODEL_TUNING, limits). Never modified at runtime.
- `config/config.json` — runtime config, auto-generated from defaults on first run. Reloaded on each request via `config_manager.get_config()`.
- `config/prompt_templates.json` — summary prompt templates by context type (meeting, lecture, interview, etc.).
- Environment variables override config values (see docker-compose.yml).

### Session/Auth

UUID-based sessions via `insight_session` cookie (HttpOnly, SameSite=Lax, 90-day TTL). Users can only access their own jobs (ownership check + manifest validation on downloads).

## Build & Run Commands

```bash
# Docker (primary method)
docker-compose build
docker-compose up -d
# Access at http://localhost:55667

# View logs
docker-compose logs -f insightaudio
docker-compose logs -f insightaudio-worker

# Stop
docker-compose down

# Rebuild after code changes
docker-compose down && docker-compose build && docker-compose up -d
```

### Local Development (without Docker)

```bash
pip install -r requirements.txt

# Start Redis (required for Celery)
redis-server

# Start Celery worker (separate terminal)
celery -A app.celery_app.celery_app worker --loglevel=INFO -P solo

# Start web server
uvicorn app.main:app --reload --host 0.0.0.0 --port 55667
```

Requires `ffmpeg` on PATH for audio conversion. Requires Ollama running (`ollama serve`) for summarization features.

## Testing

```bash
pytest tests/
pytest tests/test_transcriber.py -v
pytest tests/test_security.py::test_path_traversal
```

Tests cover: Whisper model name parsing, model availability detection, path traversal prevention, manifest file validation. Uses pytest with monkeypatch for mocking.

## Key Environment Variables

| Variable | Purpose | Default |
|---|---|---|
| `CELERY_BROKER_URL` | Redis broker URL | `redis://localhost:6379/0` |
| `MODEL_DIR` | Path to model storage | `/models` |
| `CONFIG_DIR` | Path to config files | `/config` |
| `RESULTS_DIR` | Path to job results | `/app/results` |
| `LOG_DIR` | Path to log files | `/app/logs` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `INSIGHT_STALE_RUNNING_MINUTES` | Timeout for stale jobs | `60` |

## GPU Support

Uses `Dockerfile.gpu` (CUDA 12.4.1 + cuDNN9). GPU deploy section in `docker-compose.yml` is active by default. Set `USE_CUDA=true` in config. See `GPU_SETUP.md` for host setup.

## Important Patterns

- **Config is reloaded per-request**: `config_manager.get_config()` reads from disk each time; relative paths are normalized to absolute.
- **Celery worker uses solo pool**: One task at a time, no forking. Prefetch multiplier = 1, late acks.
- **File uploads are streamed**: 8MB chunks to avoid memory exhaustion.
- **ASR has a heartbeat system**: Progress updates emitted every 2-20 seconds depending on file size, with a safety timeout (default 1200s).
- **Summarization uses chunking**: Long transcripts split into overlapping chunks, summarized independently, then reduced to final summary.
- **All paths must be absolute**: The config manager normalizes relative paths. Path traversal protection strips `../` and validates against manifest.
