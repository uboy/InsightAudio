"""Тесты для парсинга имен моделей"""
import pytest
from app.models import _extract_whisper_model_name, _extract_faster_whisper_model_name


def test_extract_whisper_model_name():
    """Тест извлечения имени модели для openai-whisper"""
    assert _extract_whisper_model_name("whisper-tiny") == "tiny"
    assert _extract_whisper_model_name("whisper-base") == "base"
    assert _extract_whisper_model_name("whisper-small") == "small"
    assert _extract_whisper_model_name("whisper-medium") == "medium"
    assert _extract_whisper_model_name("whisper-large") == "large"
    assert _extract_whisper_model_name("whisper-large-v2") == "large"
    assert _extract_whisper_model_name("whisper-large-v3") == "large"
    assert _extract_whisper_model_name("whisper-base.en") == "base"
    assert _extract_whisper_model_name("whisper-large-int8") == "large"
    assert _extract_whisper_model_name("invalid-model") is None
    assert _extract_whisper_model_name("not-whisper") is None


def test_extract_faster_whisper_model_name():
    """Тест извлечения имени модели для faster-whisper (сохраняет версии)"""
    assert _extract_faster_whisper_model_name("faster-whisper-tiny") == "tiny"
    assert _extract_faster_whisper_model_name("faster-whisper-base") == "base"
    assert _extract_faster_whisper_model_name("faster-whisper-small") == "small"
    assert _extract_faster_whisper_model_name("faster-whisper-medium") == "medium"
    assert _extract_faster_whisper_model_name("faster-whisper-large") == "large"
    assert _extract_faster_whisper_model_name("faster-whisper-large-v2") == "large-v2"
    assert _extract_faster_whisper_model_name("faster-whisper-large-v3") == "large-v3"
    assert _extract_faster_whisper_model_name("faster-whisper-base.en") == "base"
    assert _extract_faster_whisper_model_name("faster-whisper-large-int8") == "large"
    # Также работает с префиксом whisper-
    assert _extract_faster_whisper_model_name("whisper-large-v3") == "large-v3"
    assert _extract_faster_whisper_model_name("invalid-model") is None
    assert _extract_faster_whisper_model_name("not-whisper") is None

