import json
import os
from functools import lru_cache
from typing import Any, Dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "config"))
DEFAULT_SETTINGS_PATH = os.path.join(CONFIG_DIR, "default_settings.json")


def _read_default_settings() -> Dict[str, Any]:
    if not os.path.exists(DEFAULT_SETTINGS_PATH):
        raise FileNotFoundError(
            f"Файл с настройками по умолчанию не найден: {DEFAULT_SETTINGS_PATH}. "
            "Создайте его или скопируйте из примера."
        )
    with open(DEFAULT_SETTINGS_PATH, "r", encoding="utf-8") as fp:
        return json.load(fp)


@lru_cache(maxsize=1)
def load_default_settings() -> Dict[str, Any]:
    """
    Возвращает словарь с настройками/константами по умолчанию из config/default_settings.json.
    Результат кешируется в рамках процесса.
    """
    return _read_default_settings()


