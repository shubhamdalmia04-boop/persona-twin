"""Choosing and storing the reference clips that a voice-cloning model imitates.

Zero-shot cloning (XTTS) needs a few seconds to a minute of clean, single-speaker speech.
`VoiceBank` scans the person's recordings, picks the cleanest ~12 second windows (high
signal-to-noise ratio, mostly speech, no clipping) and keeps the best three in
`<twin folder>/voice/`. Pure numpy + stdlib, so it is easy to test.
"""
from __future__ import annotations

import json
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REF_SAMPLE_RATE = 22050


# ------------------------------------------------------------------ audio io
def decode_audio_file(path: Path, sample_rate: int = REF_SAMPLE_RATE) -> np.ndarray:
    """Decode any audio/video file to mono float32. Uses PyAV, installed with faster-whisper."""
    from faster_whisper.audio import decode_audio

    return decode_audio(str(path), sampling_rate=sample_rate)


def media_duration_seconds(path: Path) -> float | None:
    """Length of an audio/video file from its header (no decoding). None if unknown."""
    try:
        import av

        with av.open(str(path)) as container:
            if container.duration:
                return container.duration / 1_000_000
    except Exception:
        pass
    return None


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())


# ------------------------------------------------------------------ clip selection
@dataclass
class Clip:
    start: float
    end: float
    score: float


def _frame_rms(x: np.ndarray, hop: int) -> np.ndarray:
    n = len(x) // hop
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    return np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(axis=1))


def best_windows(
    samples: np.ndarray, sample_rate: int, clip_seconds: float = 12.0, n: int = 2
) -> list[Clip]:
    """Return up to n non-overlapping windows of clean speech, best first."""
    hop = int(sample_rate * 0.025)
    rms = _frame_rms(samples, hop)
    if len(rms) * 0.025 < 6.0:  # need at least six seconds of audio
        return []

    if float(np.percentile(rms, 90)) < 0.005:  # essentially silent
        return []

    # Background level per 2-second block (10th percentile = the quiet moments between words).
    # Measuring it locally matters: a window that mixes a noisy stretch with a silent pause
    # must not be credited with the pause's low noise level.
    block = int(2.0 / 0.025)
    n_blocks = max(1, len(rms) // block)
    block_floor = np.array(
        [np.percentile(rms[i * block : (i + 1) * block] if i < n_blocks - 1 else rms[i * block :], 10)
         for i in range(n_blocks)]
    )
    local_floor = np.repeat(block_floor, block)[: len(rms)]
    if len(local_floor) < len(rms):
        local_floor = np.pad(local_floor, (0, len(rms) - len(local_floor)), mode="edge")

    win = min(int(clip_seconds / 0.025), len(rms))
    step = int(1.0 / 0.025)
    candidates: list[Clip] = []
    for start in range(0, len(rms) - win + 1, step):
        seg = rms[start : start + win]
        w_floor = max(float(np.median(local_floor[start : start + win])), 1e-5)
        w_level = float(np.percentile(seg, 90))
        if w_level < 0.01:
            continue
        voiced_frac = float((seg > max(3 * w_floor, 0.2 * w_level)).mean())
        if voiced_frac < 0.5:
            continue
        snr = min(20 * np.log10(w_level / w_floor), 35.0)
        raw = samples[start * hop : (start + win) * hop]
        clipped = float((np.abs(raw) > 0.98).mean())
        score = 0.5 * snr / 35.0 + 0.5 * min(voiced_frac / 0.75, 1.0) - 10 * clipped
        candidates.append(Clip(start * 0.025, (start + win) * 0.025, score))

    chosen: list[Clip] = []
    for clip in sorted(candidates, key=lambda c: -c.score):
        if all(clip.end <= c.start or clip.start >= c.end for c in chosen):
            chosen.append(clip)
        if len(chosen) == n:
            break
    return chosen


# ------------------------------------------------------------------ storage
class VoiceBank:
    """The stored reference clips for one twin."""

    def __init__(self, twin_folder: Path, max_clips: int = 3):
        self.dir = Path(twin_folder) / "voice"
        self.max_clips = max_clips
        self.index_path = self.dir / "refs.json"
        self.entries: list[dict] = []  # {"file", "score", "source"}
        self._pending: list[tuple[float, np.ndarray, int, str]] = []
        if self.index_path.exists():
            self.entries = json.loads(self.index_path.read_text(encoding="utf-8"))

    def paths(self) -> list[Path]:
        return [self.dir / e["file"] for e in self.entries if (self.dir / e["file"]).exists()]

    def consider(self, samples: np.ndarray, sample_rate: int, source: str) -> int:
        """Look for good clips in this recording. Returns how many candidates were found."""
        clips = best_windows(samples, sample_rate)
        for c in clips:
            seg = samples[int(c.start * sample_rate) : int(c.end * sample_rate)].copy()
            self._pending.append((c.score, seg, sample_rate, source))
        return len(clips)

    def commit(self) -> list[Path]:
        """Merge pending candidates with stored clips and keep the best few."""
        self.dir.mkdir(parents=True, exist_ok=True)
        pool = [dict(e, samples=None) for e in self.entries]
        pool += [
            {"file": None, "score": s, "source": src, "samples": seg, "sr": sr}
            for s, seg, sr, src in self._pending
        ]
        pool.sort(key=lambda e: -e["score"])
        keep, drop = pool[: self.max_clips], pool[self.max_clips :]

        for e in drop:
            if e["file"]:
                (self.dir / e["file"]).unlink(missing_ok=True)
        self.entries = []
        for e in keep:
            if e["file"] is None:
                e["file"] = f"ref_{uuid.uuid4().hex[:8]}.wav"
                write_wav(self.dir / e["file"], e["samples"], e["sr"])
            self.entries.append({"file": e["file"], "score": round(e["score"], 3), "source": e["source"]})
        self.index_path.write_text(json.dumps(self.entries, indent=1), encoding="utf-8")
        self._pending = []
        return self.paths()

    def add_file(self, path: Path) -> int:
        """Add clips from a file the user picked by hand (higher priority than automatic ones)."""
        samples = decode_audio_file(path)
        found = self.consider(samples, REF_SAMPLE_RATE, path.name)
        # boost hand-picked material so it wins over automatically found clips
        self._pending = [(s + 1.0, seg, sr, src) for s, seg, sr, src in self._pending]
        return found

    def clear(self) -> None:
        for e in self.entries:
            (self.dir / e["file"]).unlink(missing_ok=True)
        self.entries, self._pending = [], []
        if self.index_path.exists():
            self.index_path.unlink()
