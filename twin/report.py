"""A plain-language health report for a twin: what it learned and what is missing."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def describe_twin(folder: Path) -> list[str]:
    folder = Path(folder)
    meta_path = folder / "meta.json"
    if not meta_path.exists():
        return ["No twin found here yet."]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    lines = [f"Twin: {meta.get('name')}  (name used in the chats: '{meta.get('speaker')}')"]

    chunks = []
    chunks_path = folder / "chunks.json"
    if chunks_path.exists():
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    by_source = Counter(c.get("source", "?") for c in chunks)
    lines.append(f"Memories: {len(chunks)}  (" + ", ".join(f"{k} {v}" for k, v in sorted(by_source.items())) + ")")
    lines.append(f"Real messages kept as tone examples: {meta.get('style_samples', 0)}")
    lines.append(f"Voice reference clips: {meta.get('voice_clips', 0)}")
    lines.append("Face photo chosen: " + ("yes" if list(folder.glob("avatar.*")) else "no (picked automatically)"))
    about = folder / "about.txt"
    lines.append("Notes about them (about.txt): " + (f"{len(about.read_text(encoding='utf-8', errors='ignore').split())} words" if about.exists() else "none"))
    persona = folder / "persona.json"
    listener = language = llm_model = ""
    if persona.exists():
        saved = json.loads(persona.read_text(encoding="utf-8"))
        listener = saved.get("listener", "")
        language = saved.get("language", "")
        llm_model = saved.get("llm_model", "")
    lines.append(f"Who is talking to them: {listener or 'not set'}")
    from .languages import name_for

    lines.append(f"Language: {name_for(language) if language else 'auto-detect'}")
    if llm_model:
        lines.append(f"Language model for this twin: {llm_model} (must be pulled in Ollama, or chat will fail)")

    speakers = meta.get("chat_speakers") or []
    if speakers:
        lines.append("Names seen in the chats: " + ", ".join(f"{n} ({c})" for n, c in speakers))

    tips = []
    if meta.get("chats") and not meta.get("matched_messages"):
        tips.append("NONE of the chat messages matched the person's name, so it learned nothing from the chats. "
                    "Build again and pick the name exactly as listed above.")
    elif meta.get("matched_messages", 0) < 200:
        tips.append("Few chat messages from them were found. More chat history makes it sound much more like them.")
    if not about.exists():
        tips.append("Add notes about them (relationship, nicknames, sayings, stories): the twin can only state facts it has.")
    if not listener:
        tips.append("Tell it who is talking (for example 'his daughter Meera') so it answers like it would to you.")
    if tips:
        lines.append("")
        lines += ["- " + t for t in tips]
    return lines
