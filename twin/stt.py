"""Offline speech-to-text using faster-whisper."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


class Transcriber:
    def __init__(self, model_size: str = "small", language: str = ""):
        self.model_size = model_size
        self.language = language or None
        self._model = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            # int8 on CPU works everywhere; faster-whisper switches to GPU automatically
            # when CUDA libraries are present ("auto").
            self._model = WhisperModel(self.model_size, device="auto", compute_type="auto")
        return self._model

    def file(self, path: Path) -> list[str]:
        """Transcribe an audio or video file. Returns one string per spoken segment."""
        model = self._load()
        segments, _info = model.transcribe(
            str(path), language=self.language, vad_filter=True, beam_size=1
        )
        return [s.text.strip() for s in segments if s.text.strip()]

    def audio(self, samples: np.ndarray) -> str:
        """Transcribe a float32 mono 16 kHz array (microphone capture)."""
        model = self._load()
        segments, _info = model.transcribe(
            samples, language=self.language, vad_filter=True, beam_size=1
        )
        return " ".join(s.text.strip() for s in segments).strip()


def record_until_silence(
    max_seconds: float = 20.0,
    silence_seconds: float = 1.2,
    threshold: float = 0.012,
    wait_for_speech_seconds: float = 30.0,
) -> np.ndarray | None:
    """Record from the default microphone. Starts when speech begins, stops after silence.

    Returns float32 mono audio at 16 kHz, or None if nobody spoke.
    """
    import sounddevice as sd

    block = int(0.1 * SAMPLE_RATE)
    frames: list[np.ndarray] = []
    started = False
    quiet_blocks = 0
    quiet_limit = int(silence_seconds / 0.1)
    begin = time.time()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
        while True:
            data, _ = stream.read(block)
            data = data[:, 0]
            level = float(np.sqrt(np.mean(data**2)))
            if level >= threshold:
                started = True
                quiet_blocks = 0
            elif started:
                quiet_blocks += 1

            if started:
                frames.append(data.copy())
                if quiet_blocks >= quiet_limit:
                    break
                if len(frames) * 0.1 >= max_seconds:
                    break
            elif time.time() - begin > wait_for_speech_seconds:
                return None

    return np.concatenate(frames) if frames else None
