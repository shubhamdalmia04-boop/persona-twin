"""Language choices offered in the app.

Codes are XTTS-v2's set (17 languages) since that is the tighter constraint - faster-whisper
recognises far more languages than XTTS can speak. If cloned-voice output is ever needed in a
language outside this list, leave the voice as "system" (generic) and set TWIN_WHISPER_LANGUAGE
directly to any Whisper-supported code.
"""
from __future__ import annotations

# Display name -> language code (used for both speech recognition and the cloned voice).
LANGUAGES: dict[str, str] = {
    "Detect automatically": "",
    "English": "en",
    "Hindi": "hi",
    "Hindi + English mixed (Hinglish)": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Polish": "pl",
    "Turkish": "tr",
    "Russian": "ru",
    "Dutch": "nl",
    "Czech": "cs",
    "Arabic": "ar",
    "Chinese": "zh-cn",
    "Japanese": "ja",
    "Hungarian": "hu",
    "Korean": "ko",
}

DEFAULT_LANGUAGE = "Detect automatically"


def code_for(display_name: str) -> str:
    """The language code for a display name; unknown names fall back to auto-detect."""
    return LANGUAGES.get(display_name, "")


def name_for(code: str) -> str:
    """A display name for a stored code (first match); unknown/empty codes fall back to auto-detect."""
    if not code:
        return DEFAULT_LANGUAGE
    return next((name for name, c in LANGUAGES.items() if c == code), code)
