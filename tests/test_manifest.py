"""Тесты для manifest и сохранения файлов"""
import os
import tempfile
import shutil
from app.tasks import _add_manifest_item, _save_manifest


def test_add_manifest_item():
    """Тест добавления элемента в manifest"""
    manifest = []
    with tempfile.TemporaryDirectory() as tmpdir:
        # Создаем тестовый файл
        test_file = os.path.join(tmpdir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test content")
        
        # Добавляем в manifest
        _add_manifest_item(manifest, test_file, "test", tmpdir)
        
        assert len(manifest) == 1
        assert manifest[0]["name"] == "test.txt"
        assert manifest[0]["kind"] == "test"
        assert manifest[0]["size_bytes"] > 0
        assert "created_at" in manifest[0]


def test_add_manifest_item_outside_dir():
    """Тест что файлы вне base_dir не добавляются в manifest"""
    manifest = []
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = os.path.join(tmpdir, "base")
        os.makedirs(base_dir, exist_ok=True)
        
        # Файл вне base_dir
        outside_file = os.path.join(tmpdir, "outside.txt")
        with open(outside_file, "w") as f:
            f.write("test")
        
        # Пытаемся добавить файл вне base_dir
        _add_manifest_item(manifest, outside_file, "test", base_dir)
        
        # Файл не должен быть добавлен
        assert len(manifest) == 0


def test_save_manifest():
    """Тест сохранения manifest с файлами"""
    with tempfile.TemporaryDirectory() as tmpdir:
        transcript_text = "Test transcript"
        summary_text = "Test summary"
        
        # Создаем тестовые файлы
        input_file = os.path.join(tmpdir, "input.mp3")
        with open(input_file, "w") as f:
            f.write("fake audio")
        
        wav_file = os.path.join(tmpdir, "audio.wav")
        with open(wav_file, "w") as f:
            f.write("fake wav")
        
        # Сохраняем manifest
        manifest = _save_manifest(
            tmpdir,
            transcript_text,
            summary_text,
            None,
            segments=[],
            meta={"test": "meta"},
            input_original_path=input_file,
            audio_wav_path=wav_file,
        )
        
        # Проверяем что manifest создан
        assert len(manifest) > 0
        
        # Проверяем что файлы созданы
        assert os.path.exists(os.path.join(tmpdir, "transcript.txt"))
        assert os.path.exists(os.path.join(tmpdir, "transcript.json"))
        assert os.path.exists(os.path.join(tmpdir, "summary.md"))
        assert os.path.exists(os.path.join(tmpdir, "meta.json"))
        
        # Проверяем что все элементы имеют нужные поля
        for item in manifest:
            assert "name" in item
            assert "kind" in item
            assert "size_bytes" in item
            assert "created_at" in item

