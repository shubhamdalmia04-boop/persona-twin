"""File-type tables shared by the scanner (GUI) and the ingest pipeline. Standard library only."""
from __future__ import annotations

from pathlib import Path

AUDIO_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".opus", ".amr", ".wma"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".3gp", ".wmv"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
HEIC_EXT = {".heic", ".heif"}          # iPhone photos, converted on ingest when pillow-heif is present
TEXT_EXT = {".txt", ".md"}
JSON_EXT = {".json"}
DOC_EXT = {".docx", ".pdf"}

# Folders never worth descending into (system, cache, dev and cloud-sync internals).
SKIP_DIRS = {
    "node_modules", "__pycache__", "$recycle.bin", "system volume information", "appdata",
    "windows", "program files", "program files (x86)", "programdata", ".git", ".venv",
}
SKIP_FILES = {".ds_store", "thumbs.db", "desktop.ini"}


def classify(path: Path) -> str:
    """One of: chat_or_text, json, doc, audio, video, photo, other."""
    ext = path.suffix.lower()
    if ext in TEXT_EXT:
        return "chat_or_text"
    if ext in JSON_EXT:
        return "json"
    if ext in DOC_EXT:
        return "doc"
    if ext in AUDIO_EXT:
        return "audio"
    if ext in VIDEO_EXT:
        return "video"
    if ext in IMAGE_EXT | HEIC_EXT:
        return "photo"
    return "other"


def walk(folder: Path, limit: int = 300_000):
    """Yield files under folder, skipping hidden/system directories. Stops after `limit` files."""
    count = 0
    stack = [Path(folder)]
    while stack:
        current = stack.pop()
        try:
            entries = sorted(current.iterdir(), key=lambda p: p.name.lower())
        except OSError:
            continue
        for entry in entries:
            name = entry.name.lower()
            try:
                if entry.is_dir():
                    if name in SKIP_DIRS or name.startswith("."):
                        continue
                    stack.append(entry)
                elif entry.is_file() and name not in SKIP_FILES and not name.startswith("."):
                    yield entry
                    count += 1
                    if count >= limit:
                        return
            except OSError:
                continue
