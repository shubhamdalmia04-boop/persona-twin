"""State shared between the conversation thread (writer) and the display loop (reader)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

ENVELOPE_FPS = 30


@dataclass
class DisplayState:
    status: str = "idle"          # idle | listening | thinking | speaking
    subtitle: str = ""
    quit: bool = False
    envelope: np.ndarray | None = field(default=None, repr=False)
    visemes: np.ndarray | None = field(default=None, repr=False)
    speech_started: float = 0.0

    def start_speech(self, envelope: np.ndarray, visemes: np.ndarray | None = None) -> None:
        self.speech_started = time.monotonic()
        self.envelope = envelope
        self.visemes = visemes
        self.status = "speaking"

    def end_speech(self) -> None:
        self.envelope = None
        self.visemes = None
        self.status = "idle"

    def mouth(self) -> tuple[float, float]:
        """(openness 0..1, width -1..1) right now. Falls back to loudness if no visemes."""
        track = self.visemes
        if track is not None:
            i = int((time.monotonic() - self.speech_started) * ENVELOPE_FPS)
            if 0 <= i < len(track):
                return float(track[i, 0]), float(track[i, 1])
            return 0.0, 0.0
        return self.amplitude(), 0.0

    def amplitude(self) -> float:
        env = self.envelope
        if env is None:
            return 0.0
        i = int((time.monotonic() - self.speech_started) * ENVELOPE_FPS)
        return float(env[i]) if 0 <= i < len(env) else 0.0
