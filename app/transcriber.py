import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Callable

import whisper

from app import config_manager, diarizer
from app.models import _get_whisper_cache_dir


CACHE_TTL = timedelta(days=7)
SPEAKER_PREFIX = "Speaker"


def _get_device() -> str:
    """
    Определяет устройство для Whisper (CUDA или CPU).
    Проверяет доступность CUDA и настройки конфигурации.
    """
    import logging
    logger = logging.getLogger("insightaudio.transcriber")
    
    config = config_manager.get_config()
    use_cuda = config.get("USE_CUDA", True)
    logger.info("USE_CUDA из конфига: %s", use_cuda)
    
    if not use_cuda:
        logger.info("USE_CUDA отключен в конфиге, используется CPU")
        return "cpu"
    
    try:
        import torch
        logger.info("PyTorch версия: %s", torch.__version__)
        logger.info("PyTorch собран с CUDA: %s", torch.version.cuda if hasattr(torch.version, 'cuda') else "неизвестно")
        
        # Проверяем доступность CUDA
        cuda_available = torch.cuda.is_available()
        logger.info("torch.cuda.is_available(): %s", cuda_available)
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            logger.info("Количество CUDA устройств: %d", device_count)
            if device_count > 0:
                try:
                    device_name = torch.cuda.get_device_name(0)
                    device_capability = torch.cuda.get_device_capability(0)
                    logger.info("CUDA устройство #0: %s (Compute Capability: %s.%s)", 
                               device_name, device_capability[0], device_capability[1])
                    
                    # Проверяем доступность cuDNN (пробуем создать простой тензор и выполнить операцию)
                    try:
                        test_tensor = torch.zeros(1, device="cuda")
                        # Пробуем операцию, которая требует cuDNN
                        test_result = torch.nn.functional.conv1d(test_tensor.unsqueeze(0), torch.ones(1, 1, 1, device="cuda"))
                        del test_tensor, test_result
                        torch.cuda.empty_cache()
                        logger.info("CUDA доступна, cuDNN работает, используется GPU")
                        return "cuda"
                    except Exception as cudnn_error:
                        logger.warning("CUDA доступна, но cuDNN не работает: %s. Используется CPU", str(cudnn_error))
                        logger.warning("Ошибка cuDNN может быть из-за несовместимости версий. Рекомендуется использовать Dockerfile.gpu с правильной версией cuDNN")
                        return "cpu"
                except Exception as cuda_error:
                    logger.warning("Ошибка при проверке CUDA устройства: %s. Используется CPU", str(cuda_error))
                    return "cpu"
            else:
                logger.warning("CUDA доступна, но устройств не найдено, используется CPU")
                return "cpu"
        else:
            # Дополнительная диагностика
            logger.info("CUDA недоступна, проверяем причины...")
            try:
                # Проверяем переменные окружения
                import os
                cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
                if cuda_visible_devices is not None:
                    logger.info("CUDA_VISIBLE_DEVICES: %s", cuda_visible_devices)
                else:
                    logger.info("CUDA_VISIBLE_DEVICES не установлена")
                
                # Проверяем наличие CUDA библиотек
                try:
                    import subprocess
                    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        logger.info("nvidia-smi доступен, но PyTorch не видит CUDA")
                        logger.info("Возможные причины: PyTorch собран без CUDA поддержки или версия CUDA не совместима")
                    else:
                        logger.info("nvidia-smi недоступен или вернул ошибку")
                except FileNotFoundError:
                    logger.info("nvidia-smi не найден в PATH")
                except Exception as e:
                    logger.debug("Ошибка при проверке nvidia-smi: %s", e)
            except Exception as diag_e:
                logger.debug("Ошибка при диагностике CUDA: %s", diag_e)
            
            logger.info("Используется CPU")
            return "cpu"
    except ImportError:
        logger.warning("PyTorch не установлен, используется CPU")
        return "cpu"
    except Exception as e:
        logger.error("Ошибка при проверке CUDA: %s", e, exc_info=True)
        logger.warning("Используется CPU из-за ошибки")
        return "cpu"


def transcribe(
    input_path: str,
    model: str = "whisper-tiny",
    *,
    asr_engine: str = "auto",  # auto | faster_whisper | openai_whisper
    language: Optional[str] = None,
    beam_size: Optional[int] = None,
    temperature: Optional[float] = None,
    vad_filter: Optional[bool] = None,
    no_speech_threshold: Optional[float] = None,
    enable_diarization: bool = True,
    task: Optional[str] = None,
    progress_callback: Optional[Callable[[int, float], None]] = None,  # Для faster-whisper прогресса: (progress_percent, last_segment_end)
) -> Dict[str, Union[str, List[Dict]]]:
    """
    Транскрипция аудио/видео-файла.
    Returns:
        {"text": "...", "segments": [{"start": 0.0, "end": 1.5, "text": "..."}]}
    """
    import logging
    logger = logging.getLogger("insightaudio.transcriber")
    file_hash = _get_file_hash(input_path)
    
    # Определяем движок для использования
    if asr_engine == "auto":
        # Автоматический выбор: предпочитаем faster-whisper для поддерживаемых моделей
        if model.startswith("faster-whisper"):
            asr_engine = "faster_whisper"
        elif model.startswith("whisper"):
            # Проверяем доступность faster-whisper
            try:
                from faster_whisper import WhisperModel
                # Если faster-whisper доступен, используем его для whisper моделей
                asr_engine = "faster_whisper"
            except ImportError:
                asr_engine = "openai_whisper"
        else:
            asr_engine = "openai_whisper"
    logger.info("Transcribe: engine=%s model=%s language=%s enable_diarization=%s", asr_engine, model, language, enable_diarization)
    
    # Кэш ключ должен включать движок
    cache_key_parts = [
        file_hash,
        asr_engine,
        model,
        str(language or ""),
        str(task or ""),
        str(beam_size or ""),
        str(temperature or ""),
        str(vad_filter) if asr_engine == "faster_whisper" else "",
        str(no_speech_threshold or ""),
    ]
    cache_key = "_".join(cache_key_parts)
    cached_transcript = _load_transcript_cache(cache_key)
    if cached_transcript:
        return cached_transcript

    audio_path, duration_sec = convert_to_wav(input_path, model, file_hash=file_hash)
    config = config_manager.get_config()
    
    # Обрабатываем значение language: "auto" -> None для автоопределения языка
    # Whisper не поддерживает language="auto", нужно передавать None или не передавать параметр вообще
    original_language = language
    if language:
        language_str = str(language).strip().lower()
        if language_str in ("auto", ""):
            language = None  # None означает автоопределение языка в Whisper
        else:
            language = language_str
    elif language is None:
        # Если language не передан, используем значение по умолчанию из конфига
        default_lang = config.get("AUDIO_LANGUAGE", "ru")
        if default_lang and str(default_lang).strip().lower() not in ("auto", ""):
            language = str(default_lang).strip().lower()
        else:
            language = None  # Автоопределение
    
    # Логируем для отладки
    logger.debug("Language processing: original=%s, final=%s", original_language, language)

    diarization_segments = []
    # Diarization поддерживается только для Whisper моделей
    if enable_diarization and config.get("ENABLE_SPEAKER_DIARIZATION", True) and model.startswith("whisper"):
        diarization_segments = diarizer.diarize(audio_path)

    # Поддержка Vosk Russian
    if model == "vosk-ru":
        try:
            import json as _json
            from vosk import Model, KaldiRecognizer, SetLogLevel
            SetLogLevel(-1)  # Отключаем логи Vosk
            
            model_path = os.path.join(config_manager.get_config().get("MODEL_DIR", "../models"), "vosk-ru")
            if not os.path.exists(model_path):
                return {"text": f"Модель Vosk Russian не найдена в {model_path}. Скачайте её через интерфейс.", "segments": [], "duration_sec": duration_sec}
            
            vosk_model = Model(model_path)
            recognizer = KaldiRecognizer(vosk_model, 16000)
            recognizer.SetWords(True)
            
            import soundfile as sf
            data, sample_rate = sf.read(audio_path)
            if sample_rate != 16000:
                import subprocess
                temp_wav = audio_path + ".resampled.wav"
                subprocess.run(
                    ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", temp_wav],
                    check=True,
                    capture_output=True
                )
                data, _ = sf.read(temp_wav)
                os.remove(temp_wav)
            
            segments = []
            text_parts = []
            current_start = None
            current_text = []
            
            chunk_size = 4000
            for i in range(0, len(data), chunk_size):
                chunk = data[i:i+chunk_size].tobytes()
                if recognizer.AcceptWaveform(chunk):
                    result_json = _json.loads(recognizer.Result())
                    if "result" in result_json:
                        for word_info in result_json["result"]:
                            word_start = word_info.get("start", 0)
                            word_end = word_info.get("end", 0)
                            word_text = word_info.get("word", "")
                            if current_start is None:
                                current_start = word_start
                            current_text.append(word_text)
                            if word_end - current_start > 5.0:  # Сегмент длиной ~5 секунд
                                segments.append({
                                    "start": current_start,
                                    "end": word_end,
                                    "text": " ".join(current_text)
                                })
                                text_parts.append(" ".join(current_text))
                                current_start = None
                                current_text = []
                else:
                    partial = _json.loads(recognizer.PartialResult())
                    if "partial" in partial:
                        pass  # Игнорируем частичные результаты
            
            # Финальный результат
            final_result = _json.loads(recognizer.FinalResult())
            if "result" in final_result:
                for word_info in final_result["result"]:
                    word_start = word_info.get("start", 0)
                    word_end = word_info.get("end", 0)
                    word_text = word_info.get("word", "")
                    if current_start is None:
                        current_start = word_start
                    current_text.append(word_text)
            
            if current_text:
                segments.append({
                    "start": current_start or 0,
                    "end": duration_sec or 0,
                    "text": " ".join(current_text)
                })
                text_parts.append(" ".join(current_text))
            
            transcript_text = " ".join(text_parts)
            # Vosk не поддерживает diarization, поэтому передаем пустой список
            segments = _assign_speakers_to_segments(segments, [], enable_diarization=False)
            transcript_data = {"text": transcript_text, "segments": segments, "duration_sec": duration_sec}
            _save_transcript_cache(cache_key, transcript_data)
            return transcript_data
        except ImportError:
            return {"text": "Библиотека vosk не установлена. Установите: pip install vosk", "segments": [], "duration_sec": duration_sec}
        except Exception as e:
            import logging
            logging.getLogger("insightaudio.transcriber").error("Vosk error: %s", e, exc_info=True)
            return {"text": f"Ошибка при транскрипции Vosk: {str(e)}", "segments": [], "duration_sec": duration_sec}
    
    # Вызываем соответствующую функцию в зависимости от движка
    if asr_engine == "faster_whisper":
        return transcribe_with_faster_whisper(
            audio_path=audio_path,
            model=model,
            language=language,
            beam_size=beam_size,
            temperature=temperature,
            vad_filter=vad_filter,
            no_speech_threshold=no_speech_threshold,
            enable_diarization=enable_diarization,
            diarization_segments=diarization_segments,
            duration_sec=duration_sec,
            cache_key=cache_key,
            progress_callback=progress_callback,
        )
    elif asr_engine == "openai_whisper":
        return transcribe_with_openai_whisper(
            audio_path=audio_path,
            model=model,
            language=language,
            beam_size=beam_size,
            temperature=temperature,
            no_speech_threshold=no_speech_threshold,
            enable_diarization=enable_diarization,
            diarization_segments=diarization_segments,
            duration_sec=duration_sec,
            cache_key=cache_key,
            task=task,
        )
    else:
        return {"text": f"Неизвестный движок ASR: {asr_engine}", "segments": [], "duration_sec": duration_sec}


def transcribe_with_openai_whisper(
    audio_path: str,
    model: str,
    language: Optional[str],
    beam_size: Optional[int],
    temperature: Optional[float],
    no_speech_threshold: Optional[float],
    enable_diarization: bool,
    diarization_segments: List,
    duration_sec: float,
    cache_key: str,
    task: Optional[str] = None,
) -> Dict[str, Union[str, List[Dict]]]:
    """
    Транскрипция через openai-whisper.
    vad_filter не поддерживается в openai-whisper.
    """
    import logging
    logger = logging.getLogger("insightaudio.transcriber")
    
    if not model.startswith("whisper"):
        return {"text": "Транскрипция для данной модели не реализована", "segments": [], "duration_sec": duration_sec}

    if model.startswith("whisper-"):
        whisper_name = model[len("whisper-") :]
    elif model.startswith("whisper"):
        whisper_name = model.replace("whisper", "", 1).lstrip("-") or model
    else:
        whisper_name = model
    
    cache_dir = _get_whisper_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    
    device = _get_device()
    logger.info("Загрузка модели OpenAI Whisper '%s' на устройство: %s", whisper_name, device)
    
    whisper_model = whisper.load_model(whisper_name, download_root=cache_dir, device=device)
    
    # Формируем параметры для transcribe
    transcribe_kwargs = {
        "verbose": False,
    }
    if language is not None:
        transcribe_kwargs["language"] = language
    if temperature is not None:
        transcribe_kwargs["temperature"] = temperature
    if beam_size is not None:
        transcribe_kwargs["beam_size"] = beam_size
    if task is not None:
        transcribe_kwargs["task"] = task
    if no_speech_threshold is not None:
        transcribe_kwargs["no_speech_threshold"] = no_speech_threshold
    # vad_filter не поддерживается в openai-whisper
    
    result = whisper_model.transcribe(audio_path, **transcribe_kwargs)
    segments = [
        {"start": seg.get("start"), "end": seg.get("end"), "text": seg.get("text", "").strip()}
        for seg in result.get("segments", [])
        if seg.get("text")
    ]
    segments = _assign_speakers_to_segments(segments, diarization_segments, enable_diarization)
    transcript_text = _format_segments(segments)
    transcript_data = {"text": transcript_text, "segments": segments, "duration_sec": duration_sec}
    _save_transcript_cache(cache_key, transcript_data)
    return transcript_data


def transcribe_with_faster_whisper(
    audio_path: str,
    model: str,
    language: Optional[str],
    beam_size: Optional[int],
    temperature: Optional[float],
    vad_filter: Optional[bool],
    no_speech_threshold: Optional[float],
    enable_diarization: bool,
    diarization_segments: List,
    duration_sec: float,
    cache_key: str,
    progress_callback: Optional[Callable[[int, float], None]] = None,
) -> Dict[str, Union[str, List[Dict]]]:
    """
    Транскрипция через faster-whisper.
    Поддерживает vad_filter и прогресс через callback.
    """
    import logging
    logger = logging.getLogger("insightaudio.transcriber")
    
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return {"text": "Библиотека faster-whisper не установлена. Установите: pip install faster-whisper", "segments": [], "duration_sec": duration_sec}
    
    # Извлекаем имя модели для faster-whisper
    if model.startswith("faster-whisper-"):
        whisper_name = model[len("faster-whisper-") :]
    elif model.startswith("whisper-"):
        whisper_name = model[len("whisper-") :]
    elif model.startswith("whisper"):
        whisper_name = model.replace("whisper", "", 1).lstrip("-") or model
    else:
        whisper_name = model
    
    # Определяем устройство
    device = _get_device()
    compute_type = "float16" if device == "cuda" else "int8"
    logger.info("Загрузка модели Faster-Whisper '%s' на устройство: %s (compute_type: %s)", whisper_name, device, compute_type)
    
    cache_dir = _get_whisper_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    
    # Создаем модель faster-whisper с обработкой ошибок cuDNN
    try:
        whisper_model = WhisperModel(
            whisper_name,
            device=device,
            compute_type=compute_type,
            download_root=cache_dir,
        )
    except Exception as cudnn_error:
        error_msg = str(cudnn_error).lower()
        if "cudnn" in error_msg or "libcudnn" in error_msg or "invalid handle" in error_msg:
            logger.warning("Ошибка cuDNN при загрузке модели на GPU: %s. Переключаемся на CPU", str(cudnn_error))
            device = "cpu"
            compute_type = "int8"
            logger.info("Повторная загрузка модели Faster-Whisper '%s' на устройство: %s (compute_type: %s)", whisper_name, device, compute_type)
            whisper_model = WhisperModel(
                whisper_name,
                device=device,
                compute_type=compute_type,
                download_root=cache_dir,
            )
        else:
            raise
    
    # Формируем параметры для transcribe
    transcribe_kwargs = {}
    if language is not None:
        transcribe_kwargs["language"] = language
    if beam_size is not None:
        transcribe_kwargs["beam_size"] = beam_size
    if temperature is not None:
        transcribe_kwargs["temperature"] = temperature
    if vad_filter is not None:
        transcribe_kwargs["vad_filter"] = vad_filter
    if no_speech_threshold is not None:
        transcribe_kwargs["no_speech_threshold"] = no_speech_threshold
    
    # Выполняем транскрипцию с поддержкой прогресса
    segments = []
    last_progress_time = 0
    import time
    
    try:
        segments_generator, info = whisper_model.transcribe(audio_path, **transcribe_kwargs)
    except Exception as cudnn_runtime_err:
        err_text = str(cudnn_runtime_err).lower()
        if "cudnn" in err_text or "sublibrary_version_mismatch" in err_text or "cudnn_status" in err_text:
            logger.warning("Ошибка cuDNN во время инференса: %s. Переключаемся на CPU и повторяем.", cudnn_runtime_err)
            # Пересоздаем модель на CPU и пробуем снова
            device = "cpu"
            compute_type = "int8"
            whisper_model = WhisperModel(
                whisper_name,
                device=device,
                compute_type=compute_type,
                download_root=cache_dir,
            )
            segments_generator, info = whisper_model.transcribe(audio_path, **transcribe_kwargs)
        else:
            raise
    
    # Обрабатываем сегменты с обновлением прогресса
    for segment in segments_generator:
        segments.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
        })
        
        # Обновляем прогресс каждые ≤2 секунды
        if progress_callback and duration_sec and duration_sec > 0:
            current_time = time.time()
            if current_time - last_progress_time >= 2.0:
                inner_progress = min(1.0, segment.end / duration_sec)
                progress_percent = int(inner_progress * 100)
                try:
                    progress_callback(progress_percent, segment.end)
                except Exception as e:
                    logger.warning("Ошибка в progress_callback: %s", e)
                last_progress_time = current_time
    
    # Форматируем сегменты
    formatted_segments = [
        {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
        for seg in segments
        if seg["text"]
    ]
    
    formatted_segments = _assign_speakers_to_segments(formatted_segments, diarization_segments, enable_diarization)
    transcript_text = _format_segments(formatted_segments)
    transcript_data = {"text": transcript_text, "segments": formatted_segments, "duration_sec": duration_sec}
    _save_transcript_cache(cache_key, transcript_data)
    return transcript_data


def _get_file_hash(file_path: str) -> str:
    """Вычисляет MD5 хеш файла"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def convert_to_wav(
    input_path: str,
    model: str = "whisper-tiny",
    file_hash: Optional[str] = None,
    loudnorm: bool = False,
) -> tuple[str, float]:
    """
    Конвертация любого входного файла (видео/аудио) в wav.
    Использует кеширование на основе хеша исходного файла и модели транскрипции.
    Returns:
        (wav_path, duration_sec)
    """
    base, ext = os.path.splitext(input_path)
    file_hash = file_hash or _get_file_hash(input_path)
    cache_key = f"{file_hash}_{model}_{'loudnorm' if loudnorm else 'noloudnorm'}"
    
    cache_dir = _get_wav_cache_dir()
    cached_wav_path = os.path.join(cache_dir, f"{cache_key}.wav")
    cached_duration_path = os.path.join(cache_dir, f"{cache_key}.duration.json")
    
    # Проверяем, есть ли кешированный файл и не старше ли он 1 суток
    if os.path.exists(cached_wav_path) and os.path.exists(cached_duration_path):
        file_time = datetime.fromtimestamp(os.path.getmtime(cached_wav_path))
        if datetime.now() - file_time < CACHE_TTL:
            try:
                with open(cached_duration_path, "r", encoding="utf-8") as f:
                    duration_data = json.load(f)
                    duration_sec = duration_data.get("duration_sec", 0.0)
                    return cached_wav_path, duration_sec
            except Exception:
                pass
    
    if ext.lower() == ".wav" and not loudnorm:
        try:
            shutil.copy2(input_path, cached_wav_path)
            duration_sec = _probe_duration_sec(cached_wav_path)
            with open(cached_duration_path, "w", encoding="utf-8") as f:
                json.dump({"duration_sec": duration_sec}, f)
            return cached_wav_path, duration_sec or 0.0
        except Exception:
            duration_sec = _probe_duration_sec(input_path)
            return input_path, duration_sec or 0.0
    
    # Конвертируем файл
    wav_path = base + ".wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
    ]
    if loudnorm:
        cmd.extend(["-af", "loudnorm=I=-16:LRA=11:TP=-1.5"])
    cmd.append(wav_path)
    subprocess.run(cmd, check=True)
    
    duration_sec = _probe_duration_sec(wav_path) or 0.0
    
    # Копируем в кеш для переиспользования
    try:
        shutil.copy2(wav_path, cached_wav_path)
        with open(cached_duration_path, "w", encoding="utf-8") as f:
            json.dump({"duration_sec": duration_sec}, f)
        wav_path = cached_wav_path
    except Exception:
        pass
    
    return wav_path, duration_sec


def _probe_duration_sec(path: str) -> Optional[float]:
    try:
        import json as _json
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        data = _json.loads(result.stdout)
        fmt = data.get("format") or {}
        dur = fmt.get("duration")
        if dur:
            return float(dur)
    except Exception:
        return None
    return None


def _get_wav_cache_dir() -> str:
    config = config_manager.get_config()
    results_dir = os.path.abspath(config.get("RESULTS_DIR", "/tmp"))
    cache_dir = os.path.join(results_dir, ".wav_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _get_transcribe_cache_dir() -> str:
    config = config_manager.get_config()
    results_dir = os.path.abspath(config.get("RESULTS_DIR", "/tmp"))
    cache_dir = os.path.join(results_dir, ".transcribe_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _load_transcript_cache(cache_key: str) -> Optional[Dict[str, Union[str, List[Dict]]]]:
    cache_dir = _get_transcribe_cache_dir()
    cache_path = os.path.join(cache_dir, f"{cache_key}.json")
    if not os.path.exists(cache_path):
        return None
    file_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
    if datetime.now() - file_time >= CACHE_TTL:
        try:
            os.remove(cache_path)
        except OSError:
            pass
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except (json.JSONDecodeError, OSError):
        return None
    return None


def _save_transcript_cache(cache_key: str, data: Dict[str, Union[str, List[Dict]]]) -> None:
    cache_dir = _get_transcribe_cache_dir()
    cache_path = os.path.join(cache_dir, f"{cache_key}.json")
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except OSError:
        pass


def _assign_speakers_to_segments(segments, speaker_segments, enable_diarization: bool = True):
    """
    Назначает спикеров сегментам транскрипции.
    Если диаризация выключена или недоступна, все сегменты получают "Speaker 1".
    """
    if not enable_diarization or not speaker_segments:
        # Если диаризация выключена или нет данных о спикерах - все сегменты от одного спикера
        for seg in segments:
            seg["speaker"] = f"{SPEAKER_PREFIX} 1"
        return segments
    
    # Если диаризация включена и есть данные о спикерах - назначаем по времени
    for seg in segments:
        midpoint = (seg.get("start", 0) + seg.get("end", 0)) / 2
        speaker = _find_speaker_for_time(midpoint, speaker_segments)
        seg["speaker"] = speaker
    return segments


def _find_speaker_for_time(time_point, speaker_segments):
    for seg in speaker_segments:
        if seg.start <= time_point <= seg.end:
            return seg.speaker
    return speaker_segments[0].speaker if speaker_segments else f"{SPEAKER_PREFIX} 1"


def _format_segments(segments):
    lines = []
    current_speaker = None
    buffer = []
    start_time = None

    def flush():
        if not buffer:
            return
        text = " ".join(buffer).strip()
        if not text:  # Пропускаем пустые буферы
            return
        timestamp = _format_timestamp(start_time if start_time is not None else 0)
        lines.append(f"[{timestamp}] {current_speaker}: {text}")

    for seg in segments:
        speaker = seg.get("speaker") or f"{SPEAKER_PREFIX} 1"
        text = seg.get("text", "").strip()
        if not text:  # Пропускаем пустые сегменты
            continue
        if speaker != current_speaker:
            flush()  # Сохраняем предыдущий буфер перед сменой спикера
            buffer = []
            current_speaker = speaker
            start_time = seg.get("start", 0)
        buffer.append(text)
    
    # Финальный flush только если есть что сохранить
    if buffer:
        flush()
    return "\n".join(lines).strip()


def _format_timestamp(seconds):
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h:02}:{m:02}:{s:02}"
    return f"{m:02}:{s:02}"


if __name__ == "__main__":
    test_path = "test.mp3"
    result = transcribe(test_path, model="whisper-tiny")
    print(result)
