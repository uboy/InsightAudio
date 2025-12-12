import json
import os
from typing import Any, Dict

from app import settings_loader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "config"))
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
PATH_KEYS = (
    "MODEL_DIR",
    "RESULTS_DIR",
    "LOG_DIR",
    "LLAMA_CPP_MODEL_DIR",
)


def _normalize_path(value: str) -> str:
    if not value:
        return value
    if os.path.isabs(value):
        return value
    return os.path.abspath(os.path.join(BASE_DIR, value))


def _normalize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg.setdefault("CONFIG_DIR", CONFIG_DIR)
    for key in PATH_KEYS:
        if key in cfg and isinstance(cfg[key], str):
            cfg[key] = _normalize_path(cfg[key])
    cfg.setdefault("LLAMA_CPP_MODELS", [])
    cfg.setdefault("ENABLE_SPEAKER_DIARIZATION", True)
    cfg.setdefault("DEFAULT_PDF_REFLOW", False)
    cfg.setdefault("DEFAULT_IMAGE_TRANSLATION_MODE", "notes")
    cfg.setdefault("DEFAULT_TRANSLATION_MODE", "block")
    cfg.setdefault("MODEL_TUNING", {})
    cfg.setdefault("CUSTOM_SUMMARY_API_HEADERS", {})
    cfg.setdefault("LOG_LEVEL", "INFO")
    cfg.setdefault("LOG_FILE_MAX_MB", 5)
    cfg.setdefault("LOG_BACKUP_COUNT", 5)
    return cfg


def _build_default_config() -> Dict[str, Any]:
    defaults = settings_loader.load_default_settings().get("default_config", {}).copy()
    return _normalize_config(defaults)


DEFAULT_CONFIG: Dict[str, Any] = _build_default_config()

def ensure_config():
    """Проверяет наличие config.json, если нет — пишет дефолт."""
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR, exist_ok=True)
    if not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=2)

def get_config():
    """Читает конфиг из файла."""
    ensure_config()
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return _normalize_config(cfg)

def set_config(new_cfg: dict):
    """Обновляет конфиг."""
    ensure_config()
    cfg = get_config()
    for k in new_cfg:
        cfg[k] = new_cfg[k]
    cfg = _normalize_config(cfg)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return cfg

def get_param(key, default=None):
    """Получает отдельный параметр конфига."""
    cfg = get_config()
    return cfg.get(key, default)

def set_param(key, value):
    """Устанавливает отдельный параметр конфига."""
    cfg = get_config()
    cfg[key] = value
    cfg = _normalize_config(cfg)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return True

# Тестирование
if __name__ == "__main__":
    print(get_config())
    set_param("AUDIO_LANGUAGE", "en")
    print(get_param("AUDIO_LANGUAGE"))
