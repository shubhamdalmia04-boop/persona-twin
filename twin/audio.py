"""Small audio helpers shared by the voice and display code."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Read a WAV/AIFF/FLAC file as float32 mono. Returns (samples, sample_rate)."""
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data.mean(axis=1), int(sr)


def envelope(samples: np.ndarray, sample_rate: int, fps: int = 30) -> np.ndarray:
    """Loudness per display frame, scaled to 0..1. Drives the avatar animation."""
    hop = max(1, sample_rate // fps)
    n = len(samples) // hop
    if n == 0:
        return np.zeros(1, dtype=np.float32)
    frames = samples[: n * hop].reshape(n, hop)
    rms = np.sqrt((frames**2).mean(axis=1))
    peak = np.percentile(rms, 95)
    if peak <= 1e-6:
        return np.zeros(n, dtype=np.float32)
    env = np.clip(rms / peak, 0.0, 1.0)
    # light smoothing so the motion does not flicker
    kernel = np.array([0.25, 0.5, 0.25])
    return np.convolve(env, kernel, mode="same").astype(np.float32)


def play_blocking(samples: np.ndarray, sample_rate: int) -> None:
    import sounddevice as sd

    sd.play(samples, sample_rate)
    sd.wait()


def viseme_track(samples: np.ndarray, sample_rate: int, fps: int = 30) -> np.ndarray:
    """Mouth shape per display frame, shape (frames, 2), float32.

    Column 0 = openness 0..1 (loudness). Column 1 = width -1..1, from the spectral centroid:
    low, dark sounds (oo, oh) purse the lips (-1); bright sounds (ee) widen them (+1);
    hissy consonants (s, f) keep the mouth nearly shut. A cheap stand-in for real phoneme
    alignment that looks convincing at conversation speed.
    """
    hop = max(16, sample_rate // fps)
    n = len(samples) // hop
    if n == 0:
        return np.zeros((1, 2), dtype=np.float32)
    frames = samples[: n * hop].reshape(n, hop).astype(np.float32)
    rms = np.sqrt((frames**2).mean(axis=1))
    peak = np.percentile(rms, 95)
    if peak <= 1e-6:
        return np.zeros((n, 2), dtype=np.float32)
    env = np.clip(rms / peak, 0.0, 1.0)

    spectrum = np.abs(np.fft.rfft(frames * np.hanning(hop), axis=1))
    freqs = np.fft.rfftfreq(hop, 1.0 / sample_rate)
    centroid = (spectrum * freqs).sum(axis=1) / (spectrum.sum(axis=1) + 1e-9)
    width = np.clip(np.log2(np.maximum(centroid, 1.0) / 1200.0) * 0.8, -1.0, 1.0)

    opening = env**0.8
    opening = opening * (1.0 - 0.3 * np.abs(width))          # ee and oo open less than aa
    opening = np.where(centroid > 3500, opening * 0.4, opening)  # fricatives
    silent = env < 0.08
    opening = np.where(silent, 0.0, opening)
    width = np.where(silent, 0.0, width)

    track = np.stack([opening, width], axis=1)
    kernel = np.array([0.25, 0.5, 0.25])
    for col in range(2):
        track[:, col] = np.convolve(track[:, col], kernel, mode="same")
    return track.astype(np.float32)
