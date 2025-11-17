import os
import json
import subprocess

# Путь к файлу конфигурации моделей
CONFIG_DIR = "../config"
MODELS_JSON = os.path.join(CONFIG_DIR, "supported_models.json")
MODELS_STORAGE = "../models"

# Дефолтный перечень поддерживаемых моделей (можно расширять)
DEFAULT_MODELS = {
    "transcribe": [
        {"name": "whisper-tiny", "display_name": "Whisper Tiny (RU/EN)", "downloaded": False, "size": "1GB"},
        {"name": "vosk-ru", "display_name": "Vosk Russian", "downloaded": False, "size": "500MB"}
    ],
    "summarize": [
        {"name": "llama3-8b", "display_name": "Llama3 8B", "downloaded": False, "size": "4GB"},
        {"name": "gemma-7b", "display_name": "Gemma 7B", "downloaded": False, "size": "3GB"}
    ]
}

def ensure_models_json():
    if not os.path.exists(MODELS_JSON):
        with open(MODELS_JSON, "w") as f:
            json.dump(DEFAULT_MODELS, f, ensure_ascii=False, indent=2)

def list_all_models():
    ensure_models_json()
    with open(MODELS_JSON, "r") as f:
        models = json.load(f)
    for cat in models:
        for m in models[cat]:
            m["downloaded"] = is_model_downloaded(m["name"], cat)
    return models

def is_model_downloaded(model_name, model_type):
    """Проверяет наличие модели в каталоге хранения"""
    # Для Whisper — путь к весам/папке, для Ollama — `ollama list` (или файл-маркеры)
    if model_type == "transcribe":
        # Проверяем наличие папки/models/model_name
        model_path = os.path.join(MODELS_STORAGE, model_name)
        return os.path.exists(model_path)
    elif model_type == "summarize":
        # Для Ollama можно проверять списком моделей ollama list
        try:
            res = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            return model_name in res.stdout
        except Exception:
            return False
    return False

def download_model(model_name, model_type):
    """Скачивает модель, если не скачана; обновляет статус в клиенте"""
    # Для Whisper: python -m whisper или скачивание через huggingface
    # Для Vosk: скачивание архива модели и распаковка
    # Для Ollama: ollama pull <model>
    try:
        if model_type == "transcribe":
            if model_name.startswith("whisper"):
                # Whisper скачивает модели сам при первом запуске
                # Можно заранее прогреть: python -c 'import whisper; whisper.load_model("tiny")'
                import whisper
                whisper.load_model(model_name.replace("whisper-", ""))
                # Для production: сохраните вес в папку /models, если нужно
            elif model_name == "vosk-ru":
                # Example: скачивание vosk (URL замените реальным для prod!)
                model_url = "https://alphacephei.com/vosk/models/vosk-model-ru-0.22.zip"
                model_dest = os.path.join(MODELS_STORAGE, model_name)
                if not os.path.exists(model_dest):
                    os.makedirs(model_dest, exist_ok=True)
                    subprocess.run([
                        "wget", model_url, "-O", f"{model_dest}/vosk-model.zip"
                    ], check=True)
                    subprocess.run([
                        "unzip", f"{model_dest}/vosk-model.zip", "-d", model_dest
                    ], check=True)
        elif model_type == "summarize":
            # Ollama: ollama pull llama3:8b или ollama pull gemma:7b
            res = subprocess.run(["ollama", "pull", model_name], capture_output=True, text=True)
            if res.returncode != 0:
                raise Exception(f"Ollama error: {res.stderr}")
        # Обновляем файл конфигурации
        set_model_downloaded(model_name, model_type)
        return {"status": "OK", "message": f"Модель {model_name} скачана"}
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}

def set_model_downloaded(model_name, model_type):
    ensure_models_json()
    with open(MODELS_JSON, "r") as f:
        models = json.load(f)
    for m in models.get(model_type, []):
        if m["name"] == model_name:
            m["downloaded"] = True
    with open(MODELS_JSON, "w") as f:
        json.dump(models, f, ensure_ascii=False, indent=2)

# Для локального теста
if __name__ == "__main__":
    print(list_all_models())
    print(download_model("llama3-8b", "summarize"))
