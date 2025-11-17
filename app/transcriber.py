import os
import subprocess
import whisper

def transcribe(input_path, model="whisper-tiny"):
    """
    Транскрипция аудио/видео-файла.
    Args:
      input_path (str): абсолютный путь к файлу (аудио или видео)
      model (str): название используемой модели, например 'whisper-tiny', 'whisper-base'
    Returns:
      transcript (str): текстовая транскрипция
    """
    audio_path = convert_to_wav(input_path)
    print(f"Transcribing: {audio_path} with {model}")

    # Whisper: загрузка модели
    if model.startswith("whisper"):
        # whisper-tiny, whisper-base и др.
        model_name = model.split("-")[1] if '-' in model else model
        whisper_model = whisper.load_model(model_name)
        result = whisper_model.transcribe(audio_path, language="ru")  # Язык можно сделать параметром!
        return result.get("text", "")
    # Здесь можно добавить другие модели (Vosk, Whisper.cpp, etc.)
    else:
        # Заглушка
        return "Транскрипция для данной модели не реализована"

def convert_to_wav(input_path):
    """
    Конвертация любого входного файла (видео/аудио) в wav.
    Сохраняет файл рядом с исходником, если уже wav — просто возвращает путь.
    """
    base, ext = os.path.splitext(input_path)
    if ext.lower() == ".wav":
        return input_path
    wav_path = base + ".wav"
    # ffmpeg конвертация аудио/видео
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        wav_path
    ], check=True)
    return wav_path

# Для тестов из консоли:
if __name__ == "__main__":
    test_path = "test.mp3"
    result = transcribe(test_path, model="whisper-tiny")
    print(result)
