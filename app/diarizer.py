import json
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import List, Optional

import soundfile as sf

from app import config_manager

try:
    from nemo.collections.asr.models.msdd_models import NeuralDiarizer
    from nemo.collections.asr.parts.utils.diarization_utils import rttm_to_labels

    _NEMO_AVAILABLE = True
except ImportError:
    NeuralDiarizer = None  # type: ignore
    rttm_to_labels = None  # type: ignore
    _NEMO_AVAILABLE = False

LOGGER = logging.getLogger("insightaudio.diarizer")

DIARIZATION_MODEL = "diar_msdd_telephony"
CONFIG = config_manager.get_config()


@dataclass
class SpeakerSegment:
    speaker: str
    start: float
    end: float


class NemoSpeakerDiarizer:
    _model: Optional[NeuralDiarizer] = None

    @classmethod
    def _load_model(cls) -> NeuralDiarizer:
        if not _NEMO_AVAILABLE:
            raise RuntimeError("NVIDIA NeMo недоступен")
        if cls._model is None:
            cls._model = NeuralDiarizer.from_pretrained(DIARIZATION_MODEL)
        return cls._model

    @staticmethod
    def diarize(audio_path: str) -> List[SpeakerSegment]:
        if not os.path.exists(audio_path):
            return []
        temp_dir = tempfile.mkdtemp(prefix="diar_", dir=CONFIG.get("RESULTS_DIR", "/tmp"))
        manifest_path = os.path.join(temp_dir, "manifest.json")
        rttm_path = os.path.join(temp_dir, "predicted.rttm")

        info = sf.info(audio_path)
        duration = info.duration
        manifest_entry = {
            "audio_filepath": audio_path,
            "offset": 0,
            "duration": duration,
            "label": "infer",
            "text": "-",
            "rttm_filepath": rttm_path,
            "uem_filepath": "",
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(manifest_entry) + "\n")

        model = NemoSpeakerDiarizer._load_model()
        diarizer_params = {
            "manifest_filepath": manifest_path,
            "out_dir": temp_dir,
            "oracle_vad": False,
            "vad_model": "vad_multilingual_marblenet",
            "oracle_num_speakers": False,
            "speaker_embeddings": "titanet_large",
        }
        model.diarize(diarizer_params)

        if not os.path.exists(rttm_path):
            return []
        diarization = rttm_to_labels(rttm_path)
        segments: List[SpeakerSegment] = []
        for entry in diarization:
            segments.append(
                SpeakerSegment(
                    speaker=entry["speaker"],
                    start=float(entry["start_time"]),
                    end=float(entry["end_time"]),
                )
            )
        return segments


def diarize(audio_path: str) -> List[SpeakerSegment]:
    """Public helper to run Nemo diarization."""
    try:
        if not _NEMO_AVAILABLE:
            LOGGER.warning("NeMo недоступен, диаризация пропущена")
            return []
        return NemoSpeakerDiarizer.diarize(audio_path)
    except Exception:
        return []

