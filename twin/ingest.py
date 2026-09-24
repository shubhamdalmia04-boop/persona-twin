"""Turn a folder of someone's material into a searchable memory + style profile.

Point it at any folder (sub-folders are searched too). It understands:
  * chat exports: WhatsApp .txt, and Telegram/Facebook-style .json  (only the person's messages)
  * other .txt / .md, Word .docx and PDF files: treated as written by the person
  * audio (.mp3 .m4a .opus ...) and video (.mp4 .mov ...): transcribed offline with Whisper
  * photos (.jpg .png .heic ...): a few are kept as avatar candidates
  * about.txt: facts you want the twin to know (relationship, birthplace, habits, ...)

Audio/video are assumed to contain only the person speaking. Recordings with several speakers
will mix other people's words into the memory (speaker separation is a later step), so leave
them out (include_media=False) or point at a folder with just this person's recordings.
"""
from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .chatparse import (  # noqa: F401  (re-exported for callers and tests)
    Message, is_person, looks_like_whatsapp, parse_json_messages, parse_whatsapp,
)
from .config import Settings, twin_dir
from .docs import read_document
from .filetypes import (  # noqa: F401
    AUDIO_EXT, DOC_EXT, HEIC_EXT, IMAGE_EXT, JSON_EXT, TEXT_EXT, VIDEO_EXT, classify, walk,
)
from .memory import MemoryStore
from .voice import REF_SAMPLE_RATE, VoiceBank, decode_audio_file, media_duration_seconds

MAX_PHOTOS = 30            # avatar candidates kept per twin (the rest are only counted)
UNKNOWN_MEDIA_SECONDS = 300


# --------------------------------------------------------------------------- chunking
def chat_to_chunks(
    messages: list[Message], person: str, ref: str
) -> tuple[list[dict], list[str]]:
    """Return (memory chunks, style-sample messages) for one conversation."""
    chunks: list[dict] = []
    style: list[str] = []
    for i, (speaker, text) in enumerate(messages):
        if not is_person(speaker, person):
            continue
        if 3 <= len(text) <= 240 and "\n" not in text:
            style.append(text)
        if len(text) < 12:
            continue
        prev = messages[i - 1] if i > 0 else None
        if prev and not is_person(prev[0], person):
            body = f"{prev[0]}: {prev[1]}\n{person}: {text}"
        else:
            body = f"{person}: {text}"
        chunks.append({"text": body, "source": "chat", "ref": ref})
    return chunks, style


def merge_segments(segments: Iterable[str], max_chars: int = 400) -> list[str]:
    out: list[str] = []
    buf = ""
    for seg in segments:
        if buf and len(buf) + len(seg) + 1 > max_chars:
            out.append(buf)
            buf = seg
        else:
            buf = f"{buf} {seg}".strip()
    if buf:
        out.append(buf)
    return out


def paragraphs_to_chunks(text: str, max_chars: int = 600) -> list[str]:
    import re

    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return merge_segments(paras, max_chars=max_chars)


# --------------------------------------------------------------------------- consent
CONSENT_TEXT = (
    "Before continuing: you are about to build an AI simulation of a real person from "
    "their private material.\n"
    "Only continue if you are that person, or you have their permission, or (if they have "
    "died) you are legally and ethically entitled to do so, for example as their family.\n"
    "Type YES to confirm: "
)


def confirm_consent(assume_yes: bool = False) -> bool:
    if assume_yes:
        return True
    return input(CONSENT_TEXT).strip().lower() in {"yes", "y"}


# --------------------------------------------------------------------------- photos
def save_as_jpeg_or_copy(src: Path, dest_dir: Path, stem: str) -> Path | None:
    """Copy a photo into dest_dir. HEIC/HEIF (iPhone) is converted to JPEG when possible."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in HEIC_EXT:
        dest = dest_dir / f"{stem}.jpg"
        try:
            import pillow_heif
            from PIL import Image

            pillow_heif.register_heif_opener()
            Image.open(src).convert("RGB").save(dest, "JPEG", quality=92)
            return dest
        except Exception:
            return None
    dest = dest_dir / f"{stem}{src.suffix.lower()}"
    shutil.copy(src, dest)
    return dest


# --------------------------------------------------------------------------- pipeline
def ingest_folder(
    name: str,
    folder: Path,
    speaker: str,
    settings: Settings,
    transcriber=None,
    store: MemoryStore | None = None,
    log=print,
    voice_bank: VoiceBank | None = None,
    decode=decode_audio_file,
    duration=media_duration_seconds,
    include_media: bool = True,
    media_minutes: int | None = None,
    face_photo: Path | None = None,
    about_text: str = "",
    listener: str = "",
    language: str = "",
    llm_model: str = "",
) -> dict:
    folder = Path(folder)
    if not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")

    tdir = twin_dir(name)
    (tdir / "photos").mkdir(parents=True, exist_ok=True)
    if store is None:
        store = MemoryStore(tdir, embed_model=settings.embed_model)
    if voice_bank is None:
        voice_bank = VoiceBank(tdir)

    if face_photo:
        saved = save_as_jpeg_or_copy(Path(face_photo), tdir, "avatar")
        for old in tdir.glob("avatar.*"):  # only one avatar photo at a time
            if saved is None or old != saved:
                old.unlink(missing_ok=True)
        log("  avatar  face photo saved" if saved else "  avatar  could not read the face photo")

    if listener.strip() or language.strip() or llm_model.strip():
        persona_path = tdir / "persona.json"
        saved = json.loads(persona_path.read_text(encoding="utf-8")) if persona_path.exists() else {}
        if listener.strip():
            saved["listener"] = listener.strip()
        if language.strip():
            saved["language"] = language.strip()
        if llm_model.strip():
            saved["llm_model"] = llm_model.strip()
        persona_path.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")

    chunks: list[dict] = []
    style: list[str] = []
    media: list[Path] = []
    seen_speakers: Counter = Counter()
    matched = 0
    counts = {"chats": 0, "texts": 0, "docs": 0, "audio": 0, "video": 0,
              "photos": 0, "photos_kept": 0, "skipped": 0, "other": 0}

    # ---- pass 1: text, documents, photos (fast) -------------------------------
    for path in walk(folder):
        kind = classify(path)
        rel = str(path.relative_to(folder))

        if path.name.lower() == "about.txt":
            shutil.copy(path, tdir / "about.txt")
            chunks += [
                {"text": t, "source": "about", "ref": rel}
                for t in paragraphs_to_chunks(path.read_text(encoding="utf-8", errors="ignore"))
            ]
            log(f"  about   {rel}")

        elif kind in ("chat_or_text", "json"):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                counts["skipped"] += 1
                continue
            messages: list[Message] = []
            try:
                if kind == "json":
                    messages = parse_json_messages(text)
                elif looks_like_whatsapp(text):
                    messages = parse_whatsapp(text)
            except (ValueError, TypeError, AttributeError):
                log(f"  skipped {rel} (could not parse)")
                counts["skipped"] += 1
                continue
            if messages:
                c, s = chat_to_chunks(messages, speaker, rel)
                chunks += c
                style += s
                counts["chats"] += 1
                names = Counter(sp for sp, _ in messages)
                seen_speakers.update(names)
                mine = sum(n for sp, n in names.items() if is_person(sp, speaker))
                matched += mine
                log(f"  chat    {rel}: {len(c)} memories")
                if mine == 0:
                    log(f"          WARNING: no messages from '{speaker}' in this chat. "
                        f"Names in it: {', '.join(f'{n} ({c})' for n, c in names.most_common(5))}")
            elif kind == "chat_or_text":
                for t in paragraphs_to_chunks(text):
                    chunks.append({"text": t, "source": "text", "ref": rel})
                counts["texts"] += 1
                log(f"  text    {rel}")
            else:
                counts["other"] += 1

        elif kind == "doc":
            text = read_document(path)
            if text.strip():
                chunks += [{"text": t, "source": "text", "ref": rel} for t in paragraphs_to_chunks(text)]
                counts["docs"] += 1
                log(f"  doc     {rel}")
            else:
                counts["skipped"] += 1
                log(f"  skipped {rel} (no readable text; scanned PDFs need OCR)")

        elif kind in ("audio", "video"):
            if include_media:
                media.append(path)
            else:
                counts["skipped"] += 1

        elif kind == "photo":
            counts["photos"] += 1
            if counts["photos_kept"] < MAX_PHOTOS:
                if save_as_jpeg_or_copy(path, tdir / "photos", f"{counts['photos_kept']:03d}_{path.stem}"):
                    counts["photos_kept"] += 1

        else:
            counts["other"] += 1

    # ---- pass 2: recordings, short voice notes first, within a time budget -----
    budget = (media_minutes if media_minutes is not None else settings.media_minutes) * 60
    used = 0.0
    media.sort(key=lambda p: (classify(p) != "audio", p.stat().st_size))
    for path in media:
        rel = str(path.relative_to(folder))
        kind = classify(path)
        seconds = duration(path) or UNKNOWN_MEDIA_SECONDS
        if used + seconds > budget:
            counts["skipped"] += 1
            log(f"  skipped {rel} (over the {budget // 60:.0f}-minute recording limit)")
            continue
        if transcriber is None:
            from .stt import Transcriber

            transcriber = Transcriber(settings.whisper_model, settings.whisper_language or language)
        log(f"  {kind:<7} {rel}: transcribing ({seconds / 60:.1f} min)...")
        try:
            pieces = merge_segments(transcriber.file(path))
        except Exception as exc:
            counts["skipped"] += 1
            log(f"          could not transcribe: {exc}")
            continue
        used += seconds
        chunks += [{"text": t, "source": kind, "ref": rel} for t in pieces]
        style += [t for t in pieces if 3 <= len(t) <= 240]
        counts[kind] += 1
        try:  # also look for clean speech to use as the cloned-voice reference
            found = voice_bank.consider(decode(path), REF_SAMPLE_RATE, rel)
            log(f"          {found} clean voice clip(s) found")
        except Exception as exc:  # never let voice scanning break ingestion
            log(f"          voice scan skipped: {exc}")

    if about_text.strip():
        about_path = tdir / "about.txt"
        existing = about_path.read_text(encoding="utf-8", errors="ignore").strip() if about_path.exists() else ""
        about_path.write_text((existing + "\n\n" + about_text.strip()).strip(), encoding="utf-8")
        chunks += [{"text": t, "source": "about", "ref": "typed notes"} for t in paragraphs_to_chunks(about_text)]
        log("  about   your notes added")
    if counts["chats"] and matched == 0:
        log(f"WARNING: none of the chat messages are from '{speaker}', so the twin learned nothing from the chats. "
            "Pick the person's name exactly as it appears in the chats and build again.")

    store.add(chunks)
    store.save()
    counts["voice_clips"] = len(voice_bank.commit())

    style_path = tdir / "style.json"
    old = json.loads(style_path.read_text(encoding="utf-8")) if style_path.exists() else []
    merged = list(dict.fromkeys(old + style))  # de-duplicate, keep order
    style_path.write_text(json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")

    meta = {
        "name": name,
        "speaker": speaker,
        "consent_confirmed_at": datetime.now(timezone.utc).isoformat(),
        "memories": len(store),
        "style_samples": len(merged),
        "matched_messages": matched,
        "chat_speakers": seen_speakers.most_common(8),
        **counts,
    }
    (tdir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def pick_style_examples(twin_folder: Path, n: int, seed: int = 7) -> list[str]:
    path = Path(twin_folder) / "style.json"
    if not path.exists():
        return []
    samples = json.loads(path.read_text(encoding="utf-8"))
    random.Random(seed).shuffle(samples)
    return samples[:n]
