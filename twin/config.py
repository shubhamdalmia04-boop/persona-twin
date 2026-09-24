"""Central settings. Every value can be overridden with an environment variable.

Precedence for models: environment variable > data/profile.json (written by install.py from a
hardware check) > built-in default.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("TWIN_DATA_DIR", ROOT / "data"))

# MediaPipe face-landmark model used to build the animated face (downloaded once by install.py).
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/"
    "face_landmarker.task"
)
FACE_MODEL_PATH = ROOT / "models" / "face_landmarker.task"


def load_profile() -> dict:
    path = DATA_DIR / "profile.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError):
        return {}


def _setting(env: str, profile_key: str | None, default):
    """Resolve at call time so changes to the environment or profile are picked up."""
    def factory():
        value = os.environ.get(env)
        if value:
            return type(default)(value)
        if profile_key and load_profile().get(profile_key):
            return type(default)(load_profile()[profile_key])
        return default
    return field(default_factory=factory)


@dataclass
class Settings:
    # Local LLM served by Ollama (https://ollama.com). Any chat model you have pulled works.
    llm_model: str = _setting("TWIN_LLM_MODEL", "llm_model", "qwen2.5:3b")
    ollama_url: str = _setting("OLLAMA_URL", None, "http://localhost:11434")
    temperature: float = _setting("TWIN_TEMPERATURE", None, 0.7)

    # Multilingual embedding model used for memory search.
    embed_model: str = _setting("TWIN_EMBED_MODEL", None, "paraphrase-multilingual-MiniLM-L12-v2")

    # faster-whisper model size: tiny | base | small | medium | large-v3
    whisper_model: str = _setting("TWIN_WHISPER_MODEL", "whisper_model", "base")
    # Optional language hint for Whisper (e.g. "en", "hi"). Empty = auto-detect.
    whisper_language: str = _setting("TWIN_WHISPER_LANGUAGE", None, "")

    top_k: int = 5            # memories retrieved per turn
    history_turns: int = 6    # conversation turns kept in the prompt
    style_examples: int = 12  # sample messages shown to the model as style guide

    # Default language for the cloned voice (XTTS code, e.g. "en", "hi", "es").
    # Devanagari text automatically switches to Hindi.
    tts_language: str = _setting("TWIN_TTS_LANGUAGE", None, "en")

    # Total minutes of audio/video to transcribe when building a twin (the rest is skipped).
    media_minutes: int = _setting("TWIN_MEDIA_MINUTES", None, 90)

    # A single language code (e.g. "hi") is a shortcut for both whisper_language and
    # tts_language; set via --language or the app's Language choice. TWIN_WHISPER_LANGUAGE /
    # TWIN_TTS_LANGUAGE, if set, always win over this for their own purpose.
    language: str = _setting("TWIN_LANGUAGE", None, "")

    # Resolution of the animated face in pixels. Lower = faster on weak CPUs.
    face_size: int = _setting("TWIN_FACE_SIZE", None, 384)


def slugify(name: str) -> str:
    slug = re.sub(r"[^\w\-]+", "-", name.strip().lower(), flags=re.UNICODE).strip("-")
    return slug or "twin"


def twin_dir(name: str) -> Path:
    return DATA_DIR / slugify(name)
