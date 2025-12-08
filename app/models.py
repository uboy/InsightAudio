import copy
import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional

import requests

from app import config_manager, model_metrics, settings_loader

LOGGER = logging.getLogger("insightaudio.models")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "config"))
MODELS_JSON = os.path.join(CONFIG_DIR, "supported_models.json")
# Используем переменную окружения MODEL_DIR если она установлена (для Docker)
MODELS_STORAGE = os.environ.get("MODEL_DIR") or os.path.abspath(os.path.join(BASE_DIR, "..", "models"))
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


def list_all_models(
    summary_backend: Optional[str] = None,
    custom_models: Optional[List[str]] = None,
    translate_backend: Optional[str] = None,
    translate_custom_models: Optional[List[str]] = None,
):
    config = config_manager.get_config()
    backend = summary_backend or config.get("SUMMARY_BACKEND", "ollama")
    translate_bk = translate_backend or config.get("TRANSLATE_BACKEND", "ollama")
    models = {
        "transcribe": _build_transcribe_models(),
        "summarize": _build_summarize_models(backend, config, custom_models, scope="summarize"),
        "translate": _build_summarize_models(translate_bk, config, translate_custom_models, scope="translate"),
    }
    return models


def _build_transcribe_models() -> List[Dict]:
    data = _load_models_file().get("transcribe", [])
    # Создаем копию, чтобы не изменять исходные данные
    result = []
    for item in data:
        item_copy = item.copy()
        # Всегда проверяем реальное наличие файла, игнорируя статус в JSON
        item_copy["downloaded"] = is_model_downloaded(item["name"], "transcribe")
        result.append(item_copy)
    return sorted(result, key=lambda x: x["display_name"])


def _build_summarize_models(
    backend: str,
    config: dict,
    custom_models: Optional[List[str]],
    scope: str = "summarize",
) -> List[Dict]:
    if backend == "ollama":
        return fetch_ollama_models(scope=scope)
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


def fetch_ollama_models(scope: str = "summarize") -> List[Dict]:
    base_url = _ollama_base_url()
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=15)
        response.raise_for_status()
    except requests.RequestException:
        fallback = copy.deepcopy(DEFAULT_SUMMARIZE_MODELS)
        for entry in fallback:
            _attach_model_metrics(entry, base_url, scope)
        return fallback

    payload = response.json()
    candidates = payload.get("models") or payload.get("data") or []
    models: List[Dict] = []
    for item in candidates:
        name = item.get("name") or item.get("model")
        if not name:
            continue
        entry = {
            "name": name,
            "display_name": f"{name} (Ollama)",
            "downloaded": True,
            "size": human_size(item.get("size")),
        }
        _attach_model_metrics(entry, base_url, scope)
        models.append(entry)
    if not models:
        fallback = copy.deepcopy(DEFAULT_SUMMARIZE_MODELS)
        for entry in fallback:
            _attach_model_metrics(entry, base_url, scope)
        return fallback
    return models


def _attach_model_metrics(entry: Dict, base_url: str, scope: str) -> None:
    metrics = model_metrics.get_metrics("ollama", base_url, scope, entry["name"])
    if metrics:
        entry["metrics"] = metrics
        entry["metrics_summary"] = _format_metrics_summary(metrics)


def _format_metrics_summary(metrics: Dict[str, Any]) -> str:
    parts = []
    if metrics.get("ttft") is not None:
        parts.append(f"TTFT {metrics['ttft']:.2f}с")
    if metrics.get("tpot_ms") is not None:
        parts.append(f"TPOT {metrics['tpot_ms']:.0f}мс")
    if metrics.get("throughput") is not None:
        parts.append(f"{metrics['throughput']:.1f} ток/с")
    return ", ".join(parts)


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


def _get_whisper_cache_dir():
    """Получает путь к кешу Whisper
    Использует XDG_CACHE_HOME/whisper/ (который установлен в main.py как MODELS_STORAGE/.cache/)
    чтобы модели сохранялись в монтируемый volume models/.cache/whisper/
    """
    # XDG_CACHE_HOME уже установлен в main.py, используем его
    cache_home = os.environ.get("XDG_CACHE_HOME")
    if cache_home:
        cache_dir = os.path.join(cache_home, "whisper")
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir
    # Fallback: если XDG_CACHE_HOME не установлен, используем MODELS_STORAGE
    if MODELS_STORAGE:
        cache_dir = os.path.join(MODELS_STORAGE, ".cache", "whisper")
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir
    # Последний fallback: стандартный кеш
    cache_home = os.path.expanduser("~/.cache")
    cache_dir = os.path.join(cache_home, "whisper")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def is_model_downloaded(model_name, model_type):
    """Проверяет наличие модели в каталоге хранения"""
    if model_type == "transcribe":
        if model_name.startswith("whisper") or model_name.startswith("faster-whisper"):
            # Whisper сохраняет модели в кеш
            cache_dir = _get_whisper_cache_dir()
            LOGGER.debug("Проверка модели '%s' в каталоге: %s", model_name, cache_dir)
            
            if not os.path.exists(cache_dir):
                return False
            
            # Извлекаем базовое имя без префикса whisper- или faster-whisper-
            base_name = model_name.replace("whisper-", "").replace("faster-whisper-", "")
            
            # Определяем точное имя файла на основе модели
            # Для точной проверки: если модель называется whisper-large-v3, ищем только large-v3.pt
            # Если модель называется whisper-large (без версии), ищем только large.pt
            
            # Список файлов для проверки
            files_to_check = []
            
            # Обрабатываем .en модели
            if ".en" in base_name:
                base_clean = base_name.replace(".en", "")
                if "-int8" in base_clean:
                    # INT8 .en модели не поддерживаются напрямую, пропускаем
                    return False
                elif "-v2" in base_clean:
                    files_to_check = [f"{base_clean.replace('-v2', '')}.en-v2.pt", f"{base_clean.replace('-v2', '')}.en-v2.pt.bin"]
                elif "-v3" in base_clean:
                    files_to_check = [f"{base_clean.replace('-v3', '')}.en-v3.pt", f"{base_clean.replace('-v3', '')}.en-v3.pt.bin"]
                else:
                    files_to_check = [f"{base_clean}.en.pt", f"{base_clean}.en.pt.bin"]
            elif "-int8" in base_name:
                # INT8 модели используют те же файлы, что и обычные модели, но требуют специальной загрузки
                # Проверяем, что скачана именно INT8 версия, а не обычная
                base_clean = base_name.replace("-int8", "")
                
                # Определяем имя файла для проверки
                if "-v2" in base_clean:
                    file_to_check = f"{base_clean.replace('-v2', '')}-v2.pt"
                    file_to_check_bin = f"{base_clean.replace('-v2', '')}-v2.pt.bin"
                    regular_model_name = f"whisper-{base_clean}"
                elif "-v3" in base_clean:
                    file_to_check = f"{base_clean.replace('-v3', '')}-v3.pt"
                    file_to_check_bin = f"{base_clean.replace('-v3', '')}-v3.pt.bin"
                    regular_model_name = f"whisper-{base_clean}"
                else:
                    file_to_check = f"{base_clean}.pt"
                    file_to_check_bin = f"{base_clean}.pt.bin"
                    regular_model_name = f"whisper-{base_clean}"
                
                # Проверяем наличие файла
                model_path = os.path.join(cache_dir, file_to_check)
                model_path_bin = os.path.join(cache_dir, file_to_check_bin)
                file_exists = (os.path.exists(model_path) and os.path.getsize(model_path) > 0) or \
                             (os.path.exists(model_path_bin) and os.path.getsize(model_path_bin) > 0)
                
                if not file_exists:
                    LOGGER.debug("INT8 модель '%s': файл не найден", model_name)
                    return False
                
                # Проверяем, не скачана ли обычная версия модели
                # Если скачана обычная версия, INT8 не считается скачанной
                # (так как файлы одинаковые, но INT8 требует специальной загрузки)
                # Используем рекурсивный вызов, но избегаем бесконечной рекурсии через проверку имени
                if regular_model_name != model_name:
                    # Проверяем, скачана ли обычная версия
                    regular_downloaded = is_model_downloaded(regular_model_name, model_type)
                    if regular_downloaded:
                        # Если обычная версия скачана, INT8 не считается скачанной
                        # (так как файл может быть скачан как обычная модель, а не как INT8)
                        LOGGER.debug("INT8 модель '%s': обычная версия '%s' скачана, INT8 не считается скачанной", model_name, regular_model_name)
                        return False
                
                # Если файл существует и обычная версия не скачана, считаем INT8 скачанной
                # Однако, поскольку мы не можем различить, была ли модель скачана как INT8 или обычная,
                # лучше всегда возвращать False для INT8 моделей - они требуют явной загрузки
                LOGGER.debug("INT8 модель '%s': файл найден, но INT8 модели требуют явной загрузки с квантованием", model_name)
                return False
            else:
                # Обычная модель - проверяем версию точно
                if "-v2" in base_name:
                    # Для whisper-large-v2 ищем только large-v2.pt
                    base_clean = base_name.replace("-v2", "")
                    files_to_check = [f"{base_clean}-v2.pt", f"{base_clean}-v2.pt.bin"]
                elif "-v3" in base_name:
                    # Для whisper-large-v3 ищем только large-v3.pt
                    base_clean = base_name.replace("-v3", "")
                    files_to_check = [f"{base_clean}-v3.pt", f"{base_clean}-v3.pt.bin"]
                else:
                    # Базовая модель без версии - ищем только базовый файл
                    # Но сначала проверяем, что нет версионных файлов
                    if os.path.exists(cache_dir):
                        for file in os.listdir(cache_dir):
                            if file.startswith(base_name + "-v") and (file.endswith('.pt') or file.endswith('.pt.bin')):
                                # Если есть версионный файл (например, large-v3.pt), базовая модель не считается скачанной
                                LOGGER.debug("Найден версионный файл %s, базовая модель %s не считается скачанной", file, model_name)
                                return False
                    files_to_check = [f"{base_name}.pt", f"{base_name}.pt.bin"]
            
            # Проверяем файлы
            for model_file in files_to_check:
                model_path = os.path.join(cache_dir, model_file)
                if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
                    LOGGER.debug("Модель найдена: %s (размер: %d байт)", model_path, os.path.getsize(model_path))
                    return True
            
            # Также проверяем директорию с именем модели (для некоторых версий Whisper)
            if files_to_check:
                model_dir_base = files_to_check[0].replace(".pt", "").replace(".pt.bin", "")
                model_dir = os.path.join(cache_dir, model_dir_base)
                if os.path.exists(model_dir) and os.path.isdir(model_dir):
                    for file in os.listdir(model_dir):
                        if file.endswith('.pt') or file.endswith('.bin'):
                            file_path = os.path.join(model_dir, file)
                            if os.path.getsize(file_path) > 0:
                                LOGGER.debug("Модель найдена в директории: %s (размер: %d байт)", file_path, os.path.getsize(file_path))
                                return True
            
            LOGGER.debug("Модель '%s' не найдена в каталоге %s", model_name, cache_dir)
            return False
        elif model_name == "vosk-ru":
            model_path = os.path.join(MODELS_STORAGE, model_name)
            return os.path.exists(model_path)
    if model_type == "summarize":
        # rely on backend-specific commands later (ollama list already used)
        return True
    return False


def _extract_whisper_model_name(model_name: str) -> Optional[str]:
    """Извлекает базовое имя модели Whisper для whisper.load_model()"""
    if not model_name.startswith("whisper") and not model_name.startswith("faster-whisper"):
        return None
    # Убираем префиксы
    base = model_name.replace("whisper-", "").replace("faster-whisper-", "")
    # Убираем суффиксы типа .en, -int8, -v2, -v3
    base = base.split(".")[0]  # убираем .en
    base = base.replace("-int8", "")
    # Для whisper.load_model нужны имена: tiny, base, small, medium, large
    # large-v2 и large-v3 тоже работают как "large"
    if base.startswith("large"):
        return "large"
    if base in ("tiny", "base", "small", "medium", "large"):
        return base
    return None


def download_model(model_name, model_type, backend: Optional[str] = None):
    """Скачивает модель, если не скачана; обновляет статус в клиенте"""
    try:
        if model_type == "transcribe":
            if model_name.startswith("whisper"):
                try:
                    import whisper
                except ImportError:
                    raise ImportError("Пакет openai-whisper не установлен. Установите: pip install openai-whisper")
                whisper_model_name = _extract_whisper_model_name(model_name)
                if not whisper_model_name:
                    raise ValueError(f"Не удалось определить имя модели Whisper из '{model_name}'")
                # whisper.load_model() автоматически скачивает модель при первом использовании
                # XDG_CACHE_HOME уже установлен глобально в main.py, поэтому модели будут сохраняться в models/.cache/whisper/
                cache_dir = _get_whisper_cache_dir()
                
                # Создаем директорию с проверкой
                try:
                    os.makedirs(cache_dir, exist_ok=True)
                    # Проверяем, что директория действительно создана и доступна для записи
                    if not os.path.exists(cache_dir):
                        raise RuntimeError(f"Не удалось создать директорию {cache_dir}")
                    # Проверяем права на запись
                    test_file = os.path.join(cache_dir, ".test_write")
                    try:
                        with open(test_file, "w") as f:
                            f.write("test")
                        os.remove(test_file)
                    except Exception as e:
                        raise RuntimeError(f"Нет прав на запись в директорию {cache_dir}: {e}")
                except Exception as e:
                    LOGGER.error("Ошибка при создании/проверке директории %s: %s", cache_dir, e)
                    raise
                
                LOGGER.info("Скачивание модели Whisper '%s' в каталог: %s", whisper_model_name, cache_dir)
                LOGGER.info("MODELS_STORAGE=%s, XDG_CACHE_HOME=%s", MODELS_STORAGE, os.environ.get("XDG_CACHE_HOME"))
                
                # Whisper использует torch.hub для скачивания, который может игнорировать download_root
                # Устанавливаем TORCH_HOME для управления путем кеша PyTorch
                cache_home = os.path.dirname(cache_dir)
                old_torch_home = os.environ.get("TORCH_HOME")
                old_cache_home = os.environ.get("XDG_CACHE_HOME")
                
                try:
                    # Устанавливаем переменные окружения для управления путем кеша
                    os.environ["TORCH_HOME"] = cache_home
                    os.environ["XDG_CACHE_HOME"] = cache_home
                    LOGGER.info("Установлены TORCH_HOME=%s и XDG_CACHE_HOME=%s для скачивания", cache_home, cache_home)
                    
                    # Также устанавливаем через torch.hub.set_dir() если доступно
                    try:
                        import torch
                        torch.hub.set_dir(cache_home)
                        LOGGER.info("Установлен torch.hub каталог: %s", cache_home)
                    except Exception as e:
                        LOGGER.warning("Не удалось установить torch.hub каталог: %s", e)
                    
                    # Загружаем модель с явным указанием download_root
                    model = whisper.load_model(whisper_model_name, download_root=cache_dir)
                    LOGGER.info("whisper.load_model() завершен для '%s'", whisper_model_name)
                finally:
                    # Восстанавливаем переменные окружения
                    if old_torch_home:
                        os.environ["TORCH_HOME"] = old_torch_home
                    elif "TORCH_HOME" in os.environ:
                        del os.environ["TORCH_HOME"]
                    
                    if old_cache_home:
                        os.environ["XDG_CACHE_HOME"] = old_cache_home
                    elif "XDG_CACHE_HOME" in os.environ:
                        # Не удаляем, так как оно установлено глобально в main.py
                        LOGGER.info("XDG_CACHE_HOME оставлен установленным глобально")
                
                # Проверяем, что модель действительно скачалась в нужное место
                LOGGER.info("Проверка наличия модели '%s' в каталоге: %s", whisper_model_name, cache_dir)
                if os.path.exists(cache_dir):
                    files_in_cache = os.listdir(cache_dir)
                    LOGGER.info("Содержимое каталога %s: %s", cache_dir, files_in_cache)
                else:
                    LOGGER.warning("Каталог %s не существует", cache_dir)
                
                model_found = False
                # Проверяем файлы с разными вариантами имен (с версиями и без)
                model_files = [
                    f"{whisper_model_name}.pt",
                    f"{whisper_model_name}.pt.bin",
                    f"{whisper_model_name}-v2.pt",
                    f"{whisper_model_name}-v2.pt.bin",
                    f"{whisper_model_name}-v3.pt",
                    f"{whisper_model_name}-v3.pt.bin",
                ]
                for model_file in model_files:
                    model_path = os.path.join(cache_dir, model_file)
                    if os.path.exists(model_path):
                        size = os.path.getsize(model_path)
                        LOGGER.info("Найден файл: %s (размер: %d байт)", model_path, size)
                        if size > 0:
                            model_found = True
                            break
                        else:
                            LOGGER.warning("Файл %s пустой (0 байт)", model_path)
                
                # Если не нашли точное совпадение, ищем файлы, начинающиеся с имени модели
                # Но только если имя модели совпадает с началом имени файла (не просто префикс)
                if not model_found and os.path.exists(cache_dir):
                    for file in os.listdir(cache_dir):
                        if (file.endswith('.pt') or file.endswith('.pt.bin')):
                            # Проверяем, что имя файла начинается с имени модели и за ним идет либо конец, либо дефис с версией
                            if file.startswith(whisper_model_name):
                                # Проверяем, что после имени модели идет либо конец файла, либо дефис с версией
                                remaining = file[len(whisper_model_name):]
                                if remaining == '.pt' or remaining == '.pt.bin' or remaining.startswith('-v') or remaining.startswith('.en'):
                                    file_path = os.path.join(cache_dir, file)
                                    size = os.path.getsize(file_path)
                                    LOGGER.info("Найден файл модели (по префиксу): %s (размер: %d байт)", file_path, size)
                                    if size > 0:
                                        model_found = True
                                        break
                
                if not model_found:
                    # Проверяем наличие директории с именем модели
                    model_dir = os.path.join(cache_dir, whisper_model_name)
                    if os.path.exists(model_dir) and os.path.isdir(model_dir):
                        LOGGER.info("Проверка директории: %s", model_dir)
                        for file in os.listdir(model_dir):
                            if file.endswith('.pt') or file.endswith('.bin'):
                                file_path = os.path.join(model_dir, file)
                                size = os.path.getsize(file_path)
                                LOGGER.info("Найден файл в директории: %s (размер: %d байт)", file_path, size)
                                if size > 0:
                                    model_found = True
                                    break
                
                if not model_found:
                    # Проверяем стандартный кеш и другие возможные места
                    possible_caches = [
                        os.path.expanduser("~/.cache/whisper"),
                        "/root/.cache/whisper",
                        os.path.join(os.path.expanduser("~"), ".cache", "whisper"),
                    ]
                    for default_cache in possible_caches:
                        if os.path.exists(default_cache):
                            LOGGER.info("Проверка стандартного кеша: %s", default_cache)
                            for model_file in model_files:
                                src_path = os.path.join(default_cache, model_file)
                                if os.path.exists(src_path):
                                    size = os.path.getsize(src_path)
                                    LOGGER.info("Модель найдена в стандартном кеше: %s (размер: %d байт)", src_path, size)
                                    if size > 0:
                                        import shutil
                                        dst_path = os.path.join(cache_dir, model_file)
                                        LOGGER.info("Копирование модели: %s -> %s", src_path, dst_path)
                                        try:
                                            shutil.copy2(src_path, dst_path)
                                            if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
                                                LOGGER.info("Модель успешно скопирована")
                                                model_found = True
                                                break
                                        except Exception as e:
                                            LOGGER.error("Ошибка при копировании модели: %s", e)
                            if model_found:
                                break
                
                if not model_found:
                    # Выводим подробную информацию для отладки
                    LOGGER.error("Модель '%s' не найдена после скачивания", whisper_model_name)
                    LOGGER.error("Искали в каталоге: %s", cache_dir)
                    LOGGER.error("MODELS_STORAGE=%s", MODELS_STORAGE)
                    LOGGER.error("XDG_CACHE_HOME=%s", os.environ.get("XDG_CACHE_HOME"))
                    LOGGER.error("Содержимое каталога %s: %s", cache_dir, os.listdir(cache_dir) if os.path.exists(cache_dir) else "каталог не существует")
                    raise RuntimeError(f"Модель {whisper_model_name} не найдена после скачивания в {cache_dir}. Проверьте логи для подробностей.")
                
                # Проверяем еще раз, что модель найдена (для уверенности)
                if not is_model_downloaded(model_name, model_type):
                    LOGGER.warning("Модель '%s' не найдена после скачивания, но файлы были обнаружены", model_name)
                else:
                    LOGGER.info("Модель '%s' успешно проверена после скачивания", model_name)
                
                # Обновляем статус модели после успешного скачивания и проверки
                set_model_downloaded(model_name, model_type)
                LOGGER.info("Статус модели '%s' обновлен на 'скачано' в JSON файле", model_name)
            elif model_name.startswith("faster-whisper"):
                # faster-whisper использует другой API, модели скачиваются автоматически при первом использовании
                # Здесь просто проверяем, что пакет установлен
                try:
                    from faster_whisper import WhisperModel
                except ImportError:
                    raise ImportError("Пакет faster-whisper не установлен. Установите: pip install faster-whisper")
                # Для faster-whisper модели скачиваются автоматически при создании WhisperModel
                # Просто проверяем, что имя модели корректное
                whisper_model_name = _extract_whisper_model_name(model_name)
                if not whisper_model_name:
                    raise ValueError(f"Не удалось определить имя модели Faster-Whisper из '{model_name}'")
                # Модель будет скачана при первом использовании в transcriber
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
                    # Обновляем статус модели после успешного скачивания
                    set_model_downloaded(model_name, model_type)
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
                    # Обновляем статус модели после успешного скачивания
                    set_model_downloaded(model_name, model_type)
                except requests.RequestException as err:
                    raise Exception(f"Ollama API error: {err}") from err
            else:
                return {
                    "status": "WARNING",
                    "message": "Скачивание поддерживается только для backend=ollama. "
                    "Добавьте модели вручную для выбранного источника.",
                }
        return {"status": "OK", "message": f"Модель {model_name} готова"}
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        import traceback
        traceback_str = traceback.format_exc()
        return {"status": "ERROR", "message": error_msg, "traceback": traceback_str}


def set_model_downloaded(model_name, model_type):
    """Обновляет статус модели на 'скачано' в JSON файле"""
    data = _load_models_file()
    found = False
    for m in data.get(model_type, []):
        if m["name"] == model_name:
            m["downloaded"] = True
            found = True
            LOGGER.info("Обновление статуса модели '%s' (%s) на 'скачано' в %s", model_name, model_type, MODELS_JSON)
            break
    if not found:
        LOGGER.warning("Модель '%s' (%s) не найдена в списке моделей для обновления статуса", model_name, model_type)
    with open(MODELS_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    LOGGER.debug("Файл %s обновлен", MODELS_JSON)


if __name__ == "__main__":
    print(list_all_models())
