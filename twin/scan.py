"""Look inside a folder and summarise what it holds, before anything is processed.

Standard library only: the launcher window uses this to show counts and to suggest which
speaker in the chats is the person.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .chatparse import looks_like_whatsapp, speakers_in
from .filetypes import classify, walk

_HEAD_BYTES = 64 * 1024
_MAX_CHAT_BYTES = 8 * 1024 * 1024


@dataclass
class ScanResult:
    counts: Counter = field(default_factory=Counter)      # chats, texts, docs, audio, video, photos, other
    sizes: Counter = field(default_factory=Counter)       # bytes per category
    speakers: list[tuple[str, int]] = field(default_factory=list)
    truncated: bool = False

    @property
    def total_files(self) -> int:
        return sum(self.counts.values())

    def summary(self) -> str:
        labels = [("chats", "chat exports"), ("texts", "text files"), ("docs", "Word/PDF documents"),
                  ("audio", "audio files"), ("video", "videos"), ("photos", "photos"),
                  ("other", "other files (ignored)")]
        parts = [f"{self.counts[k]} {label}" for k, label in labels if self.counts[k]]
        text = ", ".join(parts) if parts else "nothing usable found"
        return text + (" (very large folder: scan stopped early)" if self.truncated else "")


def _read_text(path: Path, limit: int) -> str:
    try:
        with open(path, "rb") as f:
            return f.read(limit).decode("utf-8", errors="ignore")
    except OSError:
        return ""


def scan_folder(folder: Path, progress=None, speaker_files: int = 25) -> ScanResult:
    """Classify every file and collect chat speaker names. `progress(n_files)` is optional."""
    result = ScanResult()
    speakers: Counter = Counter()
    chats_read = 0
    limit = 300_000
    for i, path in enumerate(walk(folder, limit), start=1):
        kind = classify(path)
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        category = {"doc": "docs", "audio": "audio", "video": "video", "photo": "photos", "other": "other"}.get(kind)

        if kind in ("chat_or_text", "json"):
            is_json = kind == "json"
            head = _read_text(path, _HEAD_BYTES)
            is_chat = False
            if is_json and size <= _MAX_CHAT_BYTES:
                found = speakers_in(_read_text(path, _MAX_CHAT_BYTES), is_json=True)
                is_chat = bool(found)
                if is_chat and chats_read < speaker_files:
                    speakers.update(found)
                    chats_read += 1
            elif not is_json and looks_like_whatsapp(head):
                is_chat = True
                if chats_read < speaker_files:
                    speakers.update(speakers_in(_read_text(path, _MAX_CHAT_BYTES)))
                    chats_read += 1
            category = "chats" if is_chat else ("texts" if not is_json else "other")

        result.counts[category] += 1
        result.sizes[category] += size
        if progress and i % 500 == 0:
            progress(i)
    result.truncated = result.total_files >= limit
    result.speakers = speakers.most_common(12)
    return result
