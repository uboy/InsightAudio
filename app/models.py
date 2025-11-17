import copy
import json
import os
import subprocess
from typing import Dict, List, Optional

import requests

from app import config_manager, settings_loader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "config"))
MODELS_JSON = os.path.join(CONFIG_DIR, "supported_models.json")
MODELS_STORAGE = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))
DEFAULT_MODELS = copy.deepcopy(settings_loader.load_default_settings().get("models", {}))
DEFAULT_SUMMARIZE_MODELS = copy.deepcopy(DEFAULT_MODELS.get("summarize", []))


def ensure_models_json() -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    if not os.path.exists(MODELS_JSON):
        with open(MODELS_JSON, "w", encoding="utf-8") as f:
            json.dump(copy.deepcopy(DEFAULT_MODELS), f, ensure_ascii=False, indent=2)
        return

    with open(MODELS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    updated = False
    for section, defaults in DEFAULT_MODELS.items():
        names = {m["name"] for m in data.get(section, [])}
        for mdl in defaults:
            if mdl["name"] not in names:
                data.setdefault(section, []).append(copy.deepcopy(mdl))
                updated = True

    if updated:
        with open(MODELS_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def _load_models_file() -> Dict[str, List[Dict]]:
    ensure_models_json()
    with open(MODELS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def list_all_models(summary_backend: Optional[str] = None, custom_models: Optional[List[str]] = None):
    config = config_manager.get_config()
    backend = summary_backend or config.get("SUMMARY_BACKEND", "ollama")
    models = {
        "transcribe": _build_transcribe_models(),
        "summarize": _build_summarize_models(backend, config, custom_models),
    }
    return models


def _build_transcribe_models() -> List[Dict]:
    data = _load_models_file().get("transcribe", [])
    for item in data:
        item["downloaded"] = is_model_downloaded(item["name"], "transcribe")
    return sorted(data, key=lambda x: x["display_name"])


def _build_summarize_models(backend: str, config: dict, custom_models: Optional[List[str]]) -> List[Dict]:
    if backend == "ollama":
        return fetch_ollama_models()
    if backend == "llama_cpp":
        return fetch_llama_cpp_models(config)
    if backend == "custom_api" and custom_models:
        return [
            {"name": model.strip(), "display_name": f"{model.strip()} (API)", "downloaded": True, "size": "remote"}
            for model in custom_models
            if model.strip()
        ]
    fallback = _load_models_file().get("summarize", DEFAULT_SUMMARIZE_MODELS)
    for item in fallback:
        item.setdefault("downloaded", False)
    return fallback


def _ollama_base_url() -> str:
    cfg = config_manager.get_config()
    base = cfg.get("OLLAMA_API_BASE") or "http://localhost:11434"
    return base.rstrip("/")


def fetch_ollama_models() -> List[Dict]:
    base_url = _ollama_base_url()
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=15)
        response.raise_for_status()
    except requests.RequestException:
        return copy.deepcopy(DEFAULT_SUMMARIZE_MODELS)

    payload = response.json()
    candidates = payload.get("models") or payload.get("data") or []
    models: List[Dict] = []
    for item in candidates:
        name = item.get("name") or item.get("model")
        if not name:
            continue
        models.append(
            {
                "name": name,
                "display_name": f"{name} (Ollama)",
                "downloaded": True,
                "size": human_size(item.get("size")),
            }
        )
    return models or copy.deepcopy(DEFAULT_SUMMARIZE_MODELS)


def human_size(num_bytes: Optional[int]) -> str:
    if not isinstance(num_bytes, int):
        return "—"
    step = 1024.0
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < step:
            return f"{size:.1f}{unit}"
        size /= step
    return f"{size:.1f}PB"


def fetch_llama_cpp_models(config: dict) -> List[Dict]:
    dir_path = config.get("LLAMA_CPP_MODEL_DIR") or os.path.join(MODELS_STORAGE, "llama_cpp")
    dir_path = os.path.abspath(dir_path)
    models: List[Dict] = []
    if os.path.isdir(dir_path):
        for entry in os.listdir(dir_path):
            if entry.lower().endswith(".gguf"):
                models.append(
                    {
                        "name": entry,
                        "display_name": f"{entry} (llama.cpp)",
                        "downloaded": True,
                        "size": human_size(
                            os.path.getsize(os.path.join(dir_path, entry))
                            if os.path.exists(os.path.join(dir_path, entry))
                            else None
                        ),
                    }
                )
    if models:
        return models
    manual = config.get("LLAMA_CPP_MODELS", [])
    if isinstance(manual, list) and manual:
        return [
            {"name": m, "display_name": f"{m} (llama.cpp)", "downloaded": False, "size": "—"}
            for m in manual
        ]
    return copy.deepcopy(DEFAULT_SUMMARIZE_MODELS)


def is_model_downloaded(model_name, model_type):
    """Проверяет наличие модели в каталоге хранения"""
    if model_type == "transcribe":
        model_path = os.path.join(MODELS_STORAGE, model_name)
        return os.path.exists(model_path)
    if model_type == "summarize":
        # rely on backend-specific commands later (ollama list already used)
        return True
    return False


def download_model(model_name, model_type, backend: Optional[str] = None):
    """Скачивает модель, если не скачана; обновляет статус в клиенте"""
    try:
        if model_type == "transcribe":
            if model_name.startswith("whisper"):
                import whisper

                whisper.load_model(model_name.replace("whisper-", ""))
            elif model_name == "vosk-ru":
                model_url = "https://alphacephei.com/vosk/models/vosk-model-ru-0.22.zip"
                model_dest = os.path.join(MODELS_STORAGE, model_name)
                if not os.path.exists(model_dest):
                    os.makedirs(model_dest, exist_ok=True)
                    subprocess.run(
                        ["wget", model_url, "-O", f"{model_dest}/vosk-model.zip"],
                        check=True,
                    )
                    subprocess.run(
                        ["unzip", f"{model_dest}/vosk-model.zip", "-d", model_dest],
                        check=True,
                    )
        elif model_type == "summarize":
            target_backend = backend or config_manager.get_config().get("SUMMARY_BACKEND", "ollama")
            if target_backend == "ollama":
                base_url = _ollama_base_url()
                try:
                    response = requests.post(
                        f"{base_url}/api/pull",
                        json={"name": model_name},
                        timeout=600,
                    )
                    response.raise_for_status()
                except requests.RequestException as err:
                    raise Exception(f"Ollama API error: {err}") from err
            else:
                return {
                    "status": "WARNING",
                    "message": "Скачивание поддерживается только для backend=ollama. "
                    "Добавьте модели вручную для выбранного источника.",
                }
        set_model_downloaded(model_name, model_type)
        return {"status": "OK", "message": f"Модель {model_name} готова"}
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}


def set_model_downloaded(model_name, model_type):
    data = _load_models_file()
    for m in data.get(model_type, []):
        if m["name"] == model_name:
            m["downloaded"] = True
    with open(MODELS_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    print(list_all_models())
