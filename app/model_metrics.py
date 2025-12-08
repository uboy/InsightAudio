import json
import os
import threading
from datetime import datetime
from typing import Any, Dict, Optional

from app import config_manager

CONFIG = config_manager.get_config()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.abspath(CONFIG.get("CONFIG_DIR", os.path.join(BASE_DIR, "..", "config")))
METRICS_FILE = os.path.join(CONFIG_DIR, "model_metrics.json")
_LOCK = threading.Lock()


def _ensure_metrics_file() -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    if not os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f)


def _load_metrics() -> Dict[str, Dict[str, Any]]:
    _ensure_metrics_file()
    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def _save_metrics(data: Dict[str, Dict[str, Any]]) -> None:
    with _LOCK:
        with open(METRICS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def _metrics_key(backend: str, server: str, scope: str, model: str) -> str:
    server_norm = server.rstrip("/")
    return f"{backend}:{server_norm}:{scope}:{model}"


def record_metrics(
    backend: str,
    server: str,
    scope: str,
    model: str,
    throughput: Optional[float],
    ttft: Optional[float],
    tpot_ms: Optional[float],
    eval_count: Optional[int] = None,
    eval_duration_ns: Optional[int] = None,
) -> None:
    """Сохраняет метрики для конкретной модели"""
    key = _metrics_key(backend, server, scope, model)
    data = _load_metrics()
    entry = {
        "backend": backend,
        "server": server.rstrip("/"),
        "scope": scope,
        "model": model,
        "throughput": throughput,
        "ttft": ttft,
        "tpot_ms": tpot_ms,
        "avg_eval_count": eval_count,
        "avg_eval_duration_ns": eval_duration_ns,
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    data[key] = entry
    _save_metrics(data)


def get_metrics(backend: str, server: str, scope: str, model: str) -> Optional[Dict[str, Any]]:
    key = _metrics_key(backend, server, scope, model)
    data = _load_metrics()
    return data.get(key)


def get_all_metrics() -> Dict[str, Dict[str, Any]]:
    return _load_metrics()

