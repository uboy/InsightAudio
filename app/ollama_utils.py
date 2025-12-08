import json
import logging
import time
from typing import Any, Dict, Optional, Tuple

import requests

from app import model_metrics

LOGGER = logging.getLogger("insightaudio.ollama")


def _parse_streaming_response(response: requests.Response):
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            LOGGER.warning("Не удалось распарсить строку Ollama: %s", line)


def generate_with_metrics(
    base_url: str,
    payload: Dict[str, Any],
    model: str,
    scope: str,
    timeout: int = 600,
) -> str:
    """
    Выполняет запрос к Ollama с включенным streaming и сохраняет метрики.
    scope: 'summarize' или 'translate'
    """
    endpoint = f"{base_url}/api/generate"
    request_payload = dict(payload)
    request_payload["stream"] = True
    LOGGER.debug("Ollama POST %s", endpoint)
    start_time = time.perf_counter()
    ttft: Optional[float] = None
    eval_count: Optional[int] = None
    eval_duration = None
    text_chunks = []

    with requests.post(endpoint, json=request_payload, stream=True, timeout=timeout) as response:
        if response.status_code >= 400:
            try:
                error_payload = response.json()
                error_msg = error_payload.get("error") or error_payload.get("message") or response.text
            except ValueError:
                error_msg = response.text
            raise RuntimeError(f"Ollama error ({response.status_code}): {error_msg.strip()}")
        streamed = False
        for chunk in _parse_streaming_response(response):
            streamed = True
            if not isinstance(chunk, dict):
                continue
            chunk_text = chunk.get("response") or chunk.get("content")
            if chunk_text:
                if ttft is None:
                    ttft = time.perf_counter() - start_time
                text_chunks.append(chunk_text)
            if chunk.get("done"):
                if ttft is None:
                    ttft = time.perf_counter() - start_time
                eval_count = chunk.get("eval_count")
                eval_duration = chunk.get("eval_duration")
                break

        if not streamed:
            # fallback на обычный JSON ответ
            data = response.json()
            text_chunks.append(_extract_text_from_response(data))
            ttft = time.perf_counter() - start_time
            if isinstance(data, dict):
                eval_count = data.get("eval_count")
                eval_duration = data.get("eval_duration")

    text = "".join(filter(None, text_chunks)).strip()
    throughput = None
    tpot_ms = None
    if eval_count and eval_duration and eval_duration > 0:
        throughput = eval_count / (eval_duration / 1_000_000_000)
        tpot_ms = (eval_duration / eval_count) / 1_000_000

    if ttft is not None or throughput:
        model_metrics.record_metrics(
            backend="ollama",
            server=base_url,
            scope=scope,
            model=model,
            throughput=throughput,
            ttft=ttft,
            tpot_ms=tpot_ms,
            eval_count=eval_count,
            eval_duration_ns=eval_duration,
        )

    return text


def _extract_text_from_response(data: Any) -> str:
    if isinstance(data, dict):
        for key in ("response", "content", "summary"):
            if key in data and isinstance(data[key], str):
                return data[key]
        if "output" in data and isinstance(data["output"], list):
            return "".join(chunk.get("content", "") for chunk in data["output"]).strip()
    elif isinstance(data, list):
        text_chunks = []
        for chunk in data:
            if isinstance(chunk, dict):
                if "response" in chunk:
                    text_chunks.append(chunk["response"])
                elif "content" in chunk:
                    text_chunks.append(chunk["content"])
        if text_chunks:
            return "".join(text_chunks)
    return json.dumps(data, ensure_ascii=False)

