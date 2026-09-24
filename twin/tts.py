"""Text-to-speech.

Every backend implements `synthesize(text) -> Path` returning an audio file.

* ClonedTTS  - speaks in the person's voice with Coqui XTTS-v2, imitating the reference clips
               that `ingest` (or `run.py voice --add`) stored. Fully offline once installed.
               Fast on an NVIDIA GPU; slow on CPU (several seconds per sentence).
               XTTS-v2 weights are licensed for NON-COMMERCIAL use (Coqui CPML). This project is
               meant for personal use; check the licence before going beyond that.
* SystemTTS  - the operating system's offline voices via pyttsx3. Not the person's voice, but
               works everywhere and instantly. Used as the fallback.
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Protocol

from .config import DATA_DIR, Settings, twin_dir
from .voice import VoiceBank, write_wav

XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
XTTS_SAMPLE_RATE = 24000
TERMS_MARKER = DATA_DIR / ".xtts_terms_accepted"
_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_CLAUSE_BREAK = re.compile(r"(?<=[,;:\u0964])\s+")


class TTSBackend(Protocol):
    def synthesize(self, text: str) -> Path: ...


def pick_language(text: str, default: str = "en") -> str:
    """XTTS needs a language code per call. Devanagari text switches to Hindi."""
    return "hi" if _DEVANAGARI.search(text) else default


def chunk_text(text: str, limit: int = 230) -> list[str]:
    """XTTS degrades on long inputs (~250 characters). Split at clause breaks, then spaces."""
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []
    pieces: list[str] = []
    current = ""
    for clause in _CLAUSE_BREAK.split(text):
        for part in _hard_wrap(clause, limit):
            if current and len(current) + len(part) + 1 > limit:
                pieces.append(current)
                current = part
            else:
                current = f"{current} {part}".strip()
    if current:
        pieces.append(current)
    return pieces


def _hard_wrap(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    out, current = [], ""
    for word in text.split():
        if current and len(current) + len(word) + 1 > limit:
            out.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        out.append(current)
    return out


def _com_init():
    """Windows speech (SAPI) is a COM service: every thread that uses it must initialise COM first.

    Without this, speaking from a worker thread fails with 'CoInitialize has not been called'.
    Returns a function to undo it, or None where COM does not apply.
    """
    if os.name != "nt":
        return None
    try:
        import pythoncom

        pythoncom.CoInitialize()
        return pythoncom.CoUninitialize
    except Exception:
        try:
            import comtypes

            comtypes.CoInitialize()
            return comtypes.CoUninitialize
        except Exception:
            return None


class SystemTTS:
    def __init__(self, rate: int = 170):
        self.rate = rate
        self._tmp = Path(tempfile.mkdtemp(prefix="twin_tts_"))
        self._n = 0

    def synthesize(self, text: str) -> Path:
        import pyttsx3

        self._n += 1
        out = self._tmp / f"utterance_{self._n}.wav"
        release = _com_init()
        try:
            # A fresh engine per call avoids hangs seen when reusing one across runAndWait().
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.save_to_file(text, str(out))
            engine.runAndWait()
            engine.stop()
        finally:
            if release:
                release()
        return out


class ClonedTTS:
    def __init__(self, reference_wavs: list[Path], language: str = "en"):
        if not reference_wavs:
            raise RuntimeError("No voice reference clips. Run: python run.py voice --name <twin> --add <recording>")
        self.refs = [str(p) for p in reference_wavs]
        self.language = language
        self._model = None
        self._latents = None
        self._tmp = Path(tempfile.mkdtemp(prefix="twin_xtts_"))
        self._n = 0

    def _load(self):
        if self._model is not None:
            return self._model
        if not TERMS_MARKER.exists():
            raise RuntimeError(
                "Voice cloning terms not accepted yet. Run: python install.py --voice-clone"
            )
        os.environ["COQUI_TOS_AGREED"] = "1"  # the user accepted the licence in install.py
        import torch
        from TTS.api import TTS

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("[Voice cloning is running on the CPU, so each sentence takes a few seconds. "
                  "Use --tts system for instant replies with a generic voice.]", flush=True)
        model = TTS(XTTS_MODEL).to(device).synthesizer.tts_model
        self._latents = self._conditioning(model)
        self._model = model
        return model

    def _conditioning(self, model):
        """Voice latents from the reference clips.

        Newer torchaudio versions need extra codec libraries just to read a WAV file, so read the
        clips with soundfile and hand XTTS the audio directly. If this XTTS version does not offer
        those methods, fall back to its own path-based loader.
        """
        try:
            import soundfile as sf
            import torch

            audios, embeddings = [], []
            for path in self.refs:
                data, rate = sf.read(path, dtype="float32", always_2d=True)
                audio = torch.from_numpy(data.mean(axis=1)).unsqueeze(0)[:, : rate * 30].to(model.device)
                embeddings.append(model.get_speaker_embedding(audio, rate))
                audios.append(audio)
            latents = model.get_gpt_cond_latents(torch.cat(audios, dim=-1), rate, length=30, chunk_length=4)
            return latents, torch.stack(embeddings).mean(dim=0)
        except Exception as exc:
            print(f"[voice: using XTTS's own audio loader ({type(exc).__name__}: {exc})]", flush=True)
            return model.get_conditioning_latents(audio_path=self.refs)

    def synthesize(self, text: str) -> Path:
        import numpy as np

        model = self._load()
        gpt_cond_latent, speaker_embedding = self._latents
        lang = pick_language(text, self.language)
        waves = []
        for piece in chunk_text(text):
            out = model.inference(piece, lang, gpt_cond_latent, speaker_embedding)
            waves.append(np.asarray(out["wav"], dtype=np.float32))
            waves.append(np.zeros(int(0.12 * XTTS_SAMPLE_RATE), dtype=np.float32))  # brief pause
        self._n += 1
        path = self._tmp / f"utterance_{self._n}.wav"
        write_wav(path, np.concatenate(waves) if waves else np.zeros(1, dtype=np.float32), XTTS_SAMPLE_RATE)
        return path


def make_tts(kind: str, twin_name: str, settings: Settings, log=print) -> TTSBackend:
    """kind: 'auto' (clone if possible, else system), 'clone', or 'system'."""
    if kind == "system":
        return SystemTTS()

    refs = VoiceBank(twin_dir(twin_name)).paths()
    problem = None
    if not refs:
        problem = "no voice reference clips yet (add audio/video to the ingest folder or use `run.py voice --add`)"
    elif not TERMS_MARKER.exists():
        problem = "voice cloning is not installed (run `python install.py --voice-clone`)"
    else:
        try:
            import TTS  # noqa: F401
        except Exception as exc:  # missing, or installed but broken (e.g. a missing dependency)
            problem = f"the voice package does not load ({type(exc).__name__}: {exc})"

    if problem is None:
        return ClonedTTS(refs, settings.tts_language)
    if kind == "clone":
        raise SystemExit(f"Cannot use the cloned voice: {problem}.")
    log(f"Using the generic system voice: {problem}.")
    return SystemTTS()
