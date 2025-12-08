import json
import os
from functools import lru_cache
from typing import Any, Dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "config"))
DEFAULT_SETTINGS_PATH = os.path.join(CONFIG_DIR, "default_settings.json")


def _read_default_settings() -> Dict[str, Any]:
    """
    Reads default settings. Falls back to config.json if default file is absent,
    to keep the service bootable in containers where only config.json exists.
    """
    if os.path.exists(DEFAULT_SETTINGS_PATH):
        with open(DEFAULT_SETTINGS_PATH, "r", encoding="utf-8") as fp:
            return json.load(fp)

    # Fallback: use config.json if present
    fallback_path = os.path.join(CONFIG_DIR, "config.json")
    if os.path.exists(fallback_path):
        with open(fallback_path, "r", encoding="utf-8") as fp:
            return json.load(fp)

    raise FileNotFoundError(
        f"Файл с настройками по умолчанию не найден: {DEFAULT_SETTINGS_PATH}. "
        f"Создайте его или скопируйте из примера (fallback также ищется в {fallback_path})."
    )


@lru_cache(maxsize=1)
def load_default_settings() -> Dict[str, Any]:
    """
    Возвращает словарь с настройками/константами по умолчанию из config/default_settings.json.
    Результат кешируется в рамках процесса.
    """
    return _read_default_settings()


