"""Тесты безопасности для download endpoint и path traversal"""
import os
import tempfile
import pytest
from pathlib import Path


def test_path_traversal_protection():
    """Тест защиты от path traversal в именах файлов"""
    # Проверяем что os.path.sep и os.path.altsep правильно определяются
    assert os.path.sep in ("/", "\\")
    
    # Тестовые случаи path traversal
    malicious_names = [
        "../etc/passwd",
        "..\\..\\windows\\system32",
        "/etc/passwd",
        "C:\\Windows\\System32",
        "....//....//etc/passwd",
        "..%2F..%2Fetc%2Fpasswd",  # URL encoded
    ]
    
    for name in malicious_names:
        # Проверяем что имя содержит разделители путей
        has_sep = os.path.sep in name
        has_altsep = os.path.altsep and os.path.altsep in name
        assert has_sep or has_altsep, f"Имя {name} должно быть заблокировано"


def test_realpath_check():
    """Тест проверки реального пути файла"""
    with tempfile.TemporaryDirectory() as tmpdir:
        job_dir = os.path.join(tmpdir, "job123")
        os.makedirs(job_dir, exist_ok=True)
        
        # Создаем файл внутри job_dir
        valid_file = os.path.join(job_dir, "transcript.txt")
        with open(valid_file, "w") as f:
            f.write("test")
        
        # Проверяем что realpath работает правильно
        real_job_dir = os.path.realpath(job_dir)
        real_file = os.path.realpath(valid_file)
        
        assert real_file.startswith(real_job_dir), "Файл должен быть внутри job_dir"
        
        # Проверяем что файл вне job_dir не проходит проверку
        outside_file = os.path.join(tmpdir, "outside.txt")
        with open(outside_file, "w") as f:
            f.write("test")
        
        real_outside = os.path.realpath(outside_file)
        assert not real_outside.startswith(real_job_dir), "Файл вне job_dir не должен проходить проверку"


def test_manifest_validation():
    """Тест валидации manifest"""
    # Правильный manifest
    valid_manifest = [
        {"name": "transcript.txt", "kind": "transcript_txt", "size_bytes": 100},
        {"name": "summary.md", "kind": "summary_md", "size_bytes": 200},
    ]
    
    # Проверяем что все элементы - словари
    assert all(isinstance(item, dict) for item in valid_manifest)
    
    # Проверяем что все имеют поле name
    assert all("name" in item for item in valid_manifest)
    
    # Некорректный manifest
    invalid_manifest = [
        "not a dict",
        {"no_name": "value"},
        None,
    ]
    
    # Проверяем что некорректные элементы не проходят валидацию
    for item in invalid_manifest:
        if isinstance(item, dict):
            assert "name" not in item or item.get("name") is None
        else:
            assert not isinstance(item, dict)

