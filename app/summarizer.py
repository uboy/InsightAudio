import json
import logging
from typing import Dict, List, Optional

import requests

from app import config_manager
from app.ollama_utils import generate_with_metrics

DEFAULT_SUMMARY_PROMPT = (
    "Подготовь структурированный пересказ по критериям:\n"
    "1. Тема/Название встречи/лекции\n"
    "2. Участники, роли\n"
    "3. Хронология вопросов и решений\n"
    "4. Ключевые задачи, обсуждения, решения\n"
    "5. Даты, документы, ссылки\n"
    "6. Итоги и открытые вопросы\n"
    "Оформи результат структурированно, разделы, список, в формате Markdown."
)

LOGGER = logging.getLogger("insightaudio.summarizer")


def summarize(
    text: str,
    model: str = "llama3:8b",
    prompt: Optional[str] = None,
    backend: Optional[str] = None,
    custom_api_url: Optional[str] = None,
    custom_api_headers: Optional[Dict[str, str]] = None,
    llama_cpp_api_url: Optional[str] = None,
    segments: Optional[List[Dict[str, float]]] = None,
) -> str:
    """
    Генерирует пересказ (summary) текста.
    Args:
      text: исходная транскрипция
      model: имя модели
      prompt: промпт summary (если None — используем из конфига)
      backend: источник LLM (ollama | llama_cpp | custom_api)
    """
    cfg = config_manager.get_config()
    timeline_hint = build_timeline_hint(segments or [])
    extra = (
        "\n\nВажно: для каждого ключевого решения или момента указывай таймкод в формате [HH:MM:SS]. "
        "Используй данные таймлинии ниже, при необходимости уточняй время."
    )
    prompt_text = (prompt or cfg.get("SUMMARY_PROMPT", DEFAULT_SUMMARY_PROMPT)) + extra + timeline_hint
    backend_name = backend or cfg.get("SUMMARY_BACKEND", "ollama")

    LOGGER.info("Запуск summarizer: backend=%s, model=%s", backend_name, model)

    if backend_name == "ollama":
        return _summarize_via_ollama(cfg, text, model, prompt_text)
    if backend_name == "llama_cpp":
        api_url = llama_cpp_api_url or cfg.get("LLAMA_CPP_API_URL")
        return _summarize_via_llama_cpp(text, model, prompt_text, api_url)
    if backend_name == "custom_api":
        api_url = custom_api_url or cfg.get("CUSTOM_SUMMARY_API_URL")
        headers = custom_api_headers or cfg.get("CUSTOM_SUMMARY_API_HEADERS", {})
        return _summarize_via_custom_api(text, model, prompt_text, api_url, headers)

    raise ValueError(f"Неизвестный backend summary: {backend_name}")


def _compose_prompt(text: str, prompt: str) -> str:
    return f"{prompt.strip()}\n---\n{text.strip()}\n---\n"


def build_timeline_hint(segments: List[Dict[str, float]]) -> str:
    if not segments:
        return ""
    lines = []
    for segment in segments[:150]:
        start = segment.get("start")
        text = segment.get("text", "").strip()
        if not text:
            continue
        lines.append(f"[{format_timestamp(start)}] {text}")
    if not lines:
        return ""
    return "\n\nТаймлиния:\n" + "\n".join(lines)


def format_timestamp(seconds: float) -> str:
    if seconds is None:
        return "??:??:??"
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h:02}:{m:02}:{s:02}"
    return f"{m:02}:{s:02}"


def _summarize_via_ollama(cfg: dict, text: str, model: str, prompt: str) -> str:
    llm_prompt = _compose_prompt(text, prompt)
    base_url = (cfg.get("OLLAMA_API_BASE") or "http://localhost:11434").rstrip("/")
    tuning = (cfg.get("MODEL_TUNING") or {}).get(model, {}) or {}
    tuning_options = dict(tuning.get("options") or {})
    context_length = tuning.get("context_length")
    if context_length and "num_ctx" not in tuning_options:
        tuning_options["num_ctx"] = context_length
    options = {"temperature": 0.2}
    options.update(tuning_options)
    payload = {
        "model": model,
        "prompt": llm_prompt,
        "options": options,
    }
    return generate_with_metrics(base_url, payload, model, scope="summarize")


def _summarize_via_llama_cpp(text: str, model: str, prompt: str, api_url: Optional[str]) -> str:
    if not api_url:
        raise RuntimeError("Не указан LLAMA_CPP_API_URL для backend=llama_cpp")
    payload = {
        "model": model,
        "prompt": _compose_prompt(text, prompt),
        "n_predict": 512,
        "temperature": 0.2,
        "cache_prompt": True,
        "stream": False,
    }
    LOGGER.debug("llama.cpp POST %s", api_url)
    response = requests.post(api_url, json=payload, timeout=600)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict):
        for key in ("content", "summary", "result", "output", "response"):
            if key in data and isinstance(data[key], str):
                return data[key].strip()
        # llama.cpp HTTP server usually returns {"content":[{"text": "..."}]}
        if "choices" in data and data["choices"]:
            return data["choices"][0].get("text", "").strip()
        if "content" in data and isinstance(data["content"], list):
            texts = [chunk.get("text", "") for chunk in data["content"] if isinstance(chunk, dict)]
            return "".join(texts).strip()
    return json.dumps(data, ensure_ascii=False)


def _summarize_via_custom_api(
    text: str,
    model: str,
    prompt: str,
    api_url: Optional[str],
    headers: Optional[Dict[str, str]] = None,
) -> str:
    if not api_url:
        raise RuntimeError("Не указан URL пользовательского API summary")
    payload = {
        "model": model,
        "prompt": prompt,
        "text": text,
        "input": text,
    }
    LOGGER.debug("Custom summary API POST %s", api_url)
    response = requests.post(api_url, json=payload, headers=headers or {}, timeout=120)
    response.raise_for_status()
    try:
        data = response.json()
    except ValueError:
        return response.text.strip()

    for key in ("summary", "result", "output", "text"):
        value = data.get(key)
        if isinstance(value, str):
            return value.strip()
    return json.dumps(data, ensure_ascii=False)


if __name__ == "__main__":
    test_text = "Это пример транскрипта, на основе которого будет создан структурированный пересказ."
    result = summarize(test_text, model="llama3:8b")
    print(result)
