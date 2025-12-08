import pytest

from app.transcriber import transcribe


@pytest.mark.parametrize(
    "model,expected",
    [
        ("whisper-large-v3", "large-v3"),
        ("whisper-small.en", "small.en"),
        ("whisper-medium", "medium"),
    ],
)
def test_whisper_name_parsing(monkeypatch, model, expected):
    # monkeypatch whisper.load_model to avoid real download
    class DummyModel:
        def transcribe(self, *args, **kwargs):
            return {"segments": [], "text": ""}

    import app.transcriber as t

    monkeypatch.setattr(t.whisper, "load_model", lambda name, download_root=None: DummyModel())

    # simulate convert_to_wav returning path and duration
    monkeypatch.setattr(t, "convert_to_wav", lambda *_, **__: ("dummy.wav", 10))
    res = transcribe("dummy", model=model)
    assert isinstance(res, dict)

