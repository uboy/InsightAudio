import os
import json

CONFIG_DIR = "../config"
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
DEFAULT_CONFIG = {
    "MODEL_DIR": "../models",
    "RESULTS_DIR": "/tmp",
    "DEFAULT_TRANSCRIBE_MODEL": "whisper-tiny",
    "DEFAULT_SUMMARIZE_MODEL": "llama3-8b",
    "SUMMARY_PROMPT": (
        "Подготовь структурированный пересказ по критериям:\n"
        "1. Тема/Название встречи/лекции\n"
        "2. Участники, роли\n"
        "3. Хронология вопросов и решений\n"
        "4. Ключевые задачи, обсуждения, решения\n"
        "5. Даты, документы, ссылки\n"
        "6. Итоги и открытые вопросы\n"
        "Оформи результат структурированно, разделы, список, в формате Markdown."
    ),
    "AUDIO_LANGUAGE": "ru"
}

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
    return cfg

def set_config(new_cfg: dict):
    """Обновляет конфиг."""
    ensure_config()
    cfg = get_config()
    for k in new_cfg:
        cfg[k] = new_cfg[k]
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
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return True

# Тестирование
if __name__ == "__main__":
    print(get_config())
    set_param("AUDIO_LANGUAGE", "en")
    print(get_param("AUDIO_LANGUAGE"))
