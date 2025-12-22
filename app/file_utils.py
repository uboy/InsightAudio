import os
import re
import shutil
import subprocess
from datetime import datetime

RESULTS_DIR = "/tmp"  # Можно сменить на config/results/ для логики с volume

def handle_upload(upload_file, save_dir=RESULTS_DIR, chunk_size: int = 8 * 1024 * 1024):
    """
    Обрабатывает загрузку файла потоково (chunks 1-8MB) без чтения целиком в память.
    Returns: абсолютный путь к файлу
    """
    filename = clean_filename(upload_file.filename)
    save_path = os.path.join(save_dir, filename)
    upload_file.file.seek(0)
    
    # Потоковая запись чанками
    with open(save_path, "wb") as f:
        while True:
            chunk = upload_file.file.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
    
    return save_path

def clean_filename(filename):
    """
    Очищает имя файла от лишних символов, добавляет временную метку
    """
    name = os.path.basename(filename or "upload")
    base, ext = os.path.splitext(name)
    # Оставляем только безопасные символы, чтобы исключить path traversal и странные юникод-символы
    safe_base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    if not safe_base:
        safe_base = "upload"
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{safe_base}_{timestamp}{ext}"

def convert_to_wav(input_path):
    """
    Конвертирует любой поддерживаемый аудио/видео-файл в wav (моно, 16кГц).
    Returns: путь к wav-файлу
    """
    base, ext = os.path.splitext(input_path)
    if ext.lower() == ".wav":
        return input_path
    wav_path = base + ".wav"
    cmd = [
        "ffmpeg", "-y", "-i", input_path, "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1", wav_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        raise RuntimeError(f"Ошибка конвертации FFmpeg: {e}")
    return wav_path

def save_results(results, save_dir=RESULTS_DIR):
    """
    Сохраняет результаты транскрипции и summary.
    Args:
      results (dict): {"transcript": "...", "summary": "..."}
      save_dir (str): директория для сохранения файлов
    Returns:
      List[str]: список файлов (имена)
    """
    files = []
    if "transcript" in results:
        transcript_file = os.path.join(save_dir, "result_transcript.txt")
        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write(results["transcript"])
        files.append("result_transcript.txt")
    if "summary" in results:
        summary_file = os.path.join(save_dir, "result_summary.txt")
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(results["summary"])
        files.append("result_summary.txt")
    return files

def cleanup_tmp_files(save_dir=RESULTS_DIR, pattern="*"):
    """
    Очищает временные файлы по шаблону (например, перед запуском нового кейса)
    """
    print(f"Очистка временной папки: {save_dir}")
    for f in os.listdir(save_dir):
        if pattern in f or pattern == "*":
            try:
                path = os.path.join(save_dir, f)
                if os.path.isfile(path):
                    os.remove(path)
            except Exception as e:
                print(f"Ошибка удаления файла {f}: {e}")

# Тестирование
if __name__ == "__main__":
    test_file = "input.mp4"
    wav_path = convert_to_wav(test_file)
    print("Converted file:", wav_path)
    res = save_results({"transcript": "Пример транскрипта", "summary": "Пример пересказа"})
    print("Saved files:", res)
    cleanup_tmp_files()
