"""Chat-export parsing. Standard library only, so the launcher can use it without extra packages."""
from __future__ import annotations

import json
import re
from collections import Counter

_WA_LINE = re.compile(
    r"^\[?(\d{1,4}[/.\-]\d{1,2}[/.\-]\d{1,4}),?\s+"
    r"(\d{1,2}[:.]\d{2}(?:[:.]\d{2})?(?:\s?[APap]\.?[Mm]\.?)?)\]?"
    r"\s*(?:-\s*)?"
    r"([^:]{1,60}?):\s(.*)$"
)
_WA_START = re.compile(r"^\[?\d{1,4}[/.\-]\d{1,2}[/.\-]\d{1,4},?\s+\d{1,2}[:.]\d{2}")
_SKIP_MESSAGES = {
    "<media omitted>",
    "image omitted",
    "video omitted",
    "audio omitted",
    "sticker omitted",
    "this message was deleted",
    "you deleted this message",
    "null",
}

Message = tuple[str, str]  # (speaker, text)


# --------------------------------------------------------------------------- parsing
def _clean(text: str) -> str:
    return text.replace("\u200e", "").replace("\u200f", "").replace("\ufeff", "").strip()


def looks_like_whatsapp(text: str) -> bool:
    lines = text.splitlines()[:60]
    return sum(1 for ln in lines if _WA_LINE.match(_clean(ln))) >= 3


def parse_whatsapp(text: str) -> list[Message]:
    messages: list[list[str]] = []
    current: list[str] | None = None
    for raw in text.splitlines():
        line = _clean(raw)
        m = _WA_LINE.match(line)
        if m:
            current = [m.group(3).strip(), m.group(4).strip()]
            messages.append(current)
        elif _WA_START.match(line):
            current = None  # system notice such as "Messages are end-to-end encrypted"
        elif current is not None and line:
            current[1] += "\n" + line
    return [
        (s, t)
        for s, t in messages
        if t.strip() and t.strip().lower() not in _SKIP_MESSAGES
    ]


def parse_json_messages(text: str) -> list[Message]:
    data = json.loads(text)
    if isinstance(data, dict):
        data = data.get("messages") or data.get("data") or []
    out: list[Message] = []
    for m in data:
        if not isinstance(m, dict):
            continue
        speaker = next(
            (str(m[k]) for k in ("from", "sender", "sender_name", "speaker", "author") if m.get(k)),
            "",
        )
        body = next((m[k] for k in ("text", "message", "content", "body") if m.get(k)), "")
        if isinstance(body, list):  # Telegram style: mix of strings and {"text": ...}
            body = "".join(p if isinstance(p, str) else str(p.get("text", "")) for p in body)
        body = str(body).strip()
        if speaker and body:
            out.append((speaker, body))
    return out


def is_person(speaker: str, person: str) -> bool:
    a, b = speaker.strip().casefold(), person.strip().casefold()
    return bool(b) and (a == b or b in a)


def speakers_in(text: str, is_json: bool = False) -> Counter:
    """How many messages each person wrote in a chat export (empty if it is not a chat)."""
    try:
        if is_json:
            msgs = parse_json_messages(text)
        elif looks_like_whatsapp(text):
            msgs = parse_whatsapp(text)
        else:
            return Counter()
    except (ValueError, TypeError, AttributeError):
        return Counter()
    return Counter(speaker for speaker, _ in msgs)
