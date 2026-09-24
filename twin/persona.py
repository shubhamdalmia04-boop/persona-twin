"""Builds the prompt that makes the LLM speak as the person.

Small local models follow *examples* far better than descriptions, so the prompt is built from
three layers:
  1. a short system prompt (who they are, who they are talking to, rules, facts about them),
  2. real exchanges from their chats, shown as earlier turns of the conversation: a few fixed
     ones for tone, plus the ones most similar to the current message,
  3. the actual conversation so far.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from .ingest import pick_style_examples

LANGUAGE_NAMES = {"hi": "Hindi", "es": "Spanish", "fr": "French", "de": "German", "it": "Italian",
                  "pt": "Portuguese", "pl": "Polish", "tr": "Turkish", "ru": "Russian", "nl": "Dutch",
                  "cs": "Czech", "ar": "Arabic", "zh-cn": "Chinese", "ja": "Japanese", "hu": "Hungarian",
                  "ko": "Korean", "en": "English"}

Pair = tuple[str, str, str]  # (other person's name, what they said, the person's real reply)


def parse_pair(text: str, speaker: str) -> Pair | None:
    """Split a stored chat chunk 'Other: hello\\nSpeaker: hi' into its two halves."""
    m = re.match(rf"^(?P<other>[^\n:]{{1,60}}): (?P<msg>.+?)\n{re.escape(speaker)}: (?P<reply>.+)$", text, re.DOTALL)
    if not m or not m["msg"].strip() or not m["reply"].strip():
        return None
    return m["other"].strip(), m["msg"].strip(), m["reply"].strip()


class Persona:
    def __init__(self, name: str, twin_folder: Path, style_examples: int = 12, listener: str | None = None,
                 language: str | None = None):
        self.name = name
        self.folder = Path(twin_folder)
        about = self.folder / "about.txt"
        self.about = about.read_text(encoding="utf-8", errors="ignore").strip() if about.exists() else ""
        self.style = pick_style_examples(self.folder, style_examples)
        self.speaker = name
        meta = self.folder / "meta.json"
        if meta.exists():
            try:
                self.speaker = json.loads(meta.read_text(encoding="utf-8")).get("speaker") or name
            except ValueError:
                pass
        saved_data: dict = {}
        saved = self.folder / "persona.json"
        if saved.exists():
            try:
                saved_data = json.loads(saved.read_text(encoding="utf-8"))
            except ValueError:
                pass
        self.listener = (listener or saved_data.get("listener") or "").strip()
        self.language = (language or saved_data.get("language") or "").strip()
        self._pairs: list[Pair] = []

    # ------------------------------------------------------------------ examples
    def index_pairs(self, chunks: list[dict]) -> None:
        """Remember which stored chunks are real chat exchanges (call once after loading memory)."""
        self._pairs = [p for c in chunks if c.get("source") == "chat" and (p := parse_pair(c["text"], self.speaker))]

    def split_memories(self, memories: list[dict], max_pairs: int = 3, max_facts: int = 4) -> tuple[list[Pair], list[dict]]:
        """Separate retrieved memories into chat exchanges (used as examples) and facts (used as notes)."""
        pairs: list[Pair] = []
        facts: list[dict] = []
        for m in memories:
            pair = parse_pair(m["text"], self.speaker) if m.get("source") == "chat" else None
            if pair is not None:
                if len(pairs) < max_pairs:
                    pairs.append(pair)
            elif len(facts) < max_facts:
                facts.append(m)
        return pairs, facts

    def style_pairs(self, n: int = 3, exclude: list[Pair] | None = None) -> list[Pair]:
        """A few fixed, typical exchanges that keep the voice consistent whatever is asked."""
        skip = set(exclude or [])
        usable = [p for p in self._pairs if 3 <= len(p[2]) <= 200 and len(p[1]) <= 200 and p not in skip]
        random.Random(7).shuffle(usable)
        return usable[:n]

    @staticmethod
    def as_turns(pairs: list[Pair]) -> list[dict]:
        turns: list[dict] = []
        for _, said, reply in pairs:
            turns += [{"role": "user", "content": said}, {"role": "assistant", "content": reply}]
        return turns

    # ------------------------------------------------------------------ prompt
    def system_prompt(self, memories: list[dict]) -> str:
        who = f"You are chatting with {self.listener}." if self.listener else \
            f"You are chatting with someone who knows or wants to know {self.name}."
        parts = [
            f"You are {self.name}. {who}",
            "",
            "Rules:",
            f"1. Always answer as {self.name}, in the first person, as if this were a real chat. Never say "
            "you are an assistant or language model, and never explain or lecture.",
            f"2. Copy how {self.name} really talks: the same words, language and spelling, emojis, pet names "
            "and habits. Real replies are short. The earlier turns of this conversation are real exchanges; "
            "sound like those.",
            "3. Only state personal facts, memories, stories or opinions supported by the MEMORIES or ABOUT "
            "notes below or by the earlier turns. If something is not covered, say so the way "
            f"{self.name} naturally would (for example \"I don't remember\") and ask something back. "
            "Never invent specific events, dates, names or promises.",
            f"4. If someone sincerely asks whether you are the real {self.name}, say honestly that you are "
            f"an AI simulation of {self.name}, built from {self.name}'s own messages and recordings.",
            f"5. Do not make commitments, give medical, legal or financial instructions, or claim to "
            f"authorise anything on behalf of {self.name}.",
            "6. Keep it to one to three short sentences. Your reply is spoken aloud.",
        ]
        if self.language:
            name = LANGUAGE_NAMES.get(self.language, self.language)
            parts.append(
                f"7. Reply in {name}, the language {self.name} spoke. If the message you receive is in a "
                f"different language, still answer in {name} unless {self.name}'s own real messages above "
                "mix languages, in which case mix them the same natural way."
            )
        if self.about:
            parts += ["", "ABOUT:", self.about]
        if self.style:
            parts += ["", f"MORE OF {self.name.upper()}'S REAL MESSAGES (for tone):"]
            parts += [f"- {s}" for s in self.style]
        if memories:
            parts += ["", "MEMORIES (most relevant first):"]
            parts += [f"- {m['text']}" for m in memories]
        else:
            parts += ["", "MEMORIES: nothing relevant found for this message."]
        return "\n".join(parts)
