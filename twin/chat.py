"""One conversation with a twin: retrieve memories -> local LLM -> speech."""
from __future__ import annotations

import re
from pathlib import Path

from .config import Settings, twin_dir
from .llm import chat_stream
from .memory import MemoryStore
from .persona import Persona
from .speech import SpeechPipeline
from .state import DisplayState

_SENTENCE_BREAK = re.compile(r"(?<=[.!?\u0964\u2026])\s+|\n+")


def split_sentences(buffer: str) -> tuple[list[str], str]:
    """Split streamed text into finished sentences and an unfinished remainder."""
    parts = _SENTENCE_BREAK.split(buffer)
    if len(parts) == 1:
        return [], buffer
    return [p.strip() for p in parts[:-1] if p.strip()], parts[-1]


class Conversation:
    def __init__(
        self,
        name: str,
        settings: Settings,
        tts=None,
        state: DisplayState | None = None,
        speech: SpeechPipeline | None = None,
        fallback_tts=None,
        listener: str | None = None,
        language: str | None = None,
        show_memories: bool = False,
    ):
        self.name = name
        self.settings = settings
        self.folder: Path = twin_dir(name)
        if not (self.folder / "chunks.json").exists():
            raise SystemExit(
                f"No twin named '{name}' yet. Build one first:\n"
                f"  python run.py ingest --name \"{name}\" --path <folder> --speaker \"<name in chats>\""
            )
        self.store = MemoryStore(self.folder, embed_model=settings.embed_model)
        self.persona = Persona(name, self.folder, settings.style_examples, listener=listener, language=language)
        self.persona.index_pairs(self.store.chunks)
        self.show_memories = show_memories
        self.tts = tts
        self.state = state or DisplayState()
        self.speech = speech or SpeechPipeline(tts, self.state, speaker=name, fallback=fallback_tts)
        self.history: list[dict] = []

    def _print_memories(self, memories: list[dict], pairs, style) -> None:
        print(f"[remembered {len(memories)} thing(s); using {len(pairs)} similar exchange(s), "
              f"{len(style)} tone example(s)]", flush=True)
        for m in memories[:6]:
            snippet = m["text"].replace("\n", "  ->  ")
            print(f"   ({m['score']:.2f}) {snippet[:150]}", flush=True)
        if not memories:
            print("   (nothing in memory matched this message)", flush=True)

    # ------------------------------------------------------------------ turn
    def respond(self, user_text: str) -> str:
        """Answer one message. Sentences are queued for speech as they stream in; returns when done."""
        s = self.settings
        memories = self.store.search(user_text, max(s.top_k, 8), min_score=0.15)
        pairs, facts = self.persona.split_memories(memories)
        style = self.persona.style_pairs(3, exclude=pairs)
        if self.show_memories:
            self._print_memories(memories, pairs, style)
        messages = (
            [{"role": "system", "content": self.persona.system_prompt(facts)}]
            + Persona.as_turns(style + pairs)   # real exchanges: fixed tone examples, then the most relevant
            + self.history[-s.history_turns * 2 :]
            + [{"role": "user", "content": user_text}]
        )

        self.state.subtitle = ""
        self.state.status = "thinking"
        buffer, spoken = "", []
        try:
            for piece in chat_stream(messages, s.llm_model, s.ollama_url, s.temperature):
                if self.state.quit:
                    break
                buffer += piece
                sentences, buffer = split_sentences(buffer)
                for sentence in sentences:
                    spoken.append(sentence)
                    self.speech.submit(sentence)
            if buffer.strip() and not self.state.quit:
                spoken.append(buffer.strip())
                self.speech.submit(buffer.strip())
            self.speech.wait()
        finally:
            if self.state.status == "thinking":
                self.state.status = "idle"

        reply = " ".join(spoken)
        self.history += [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": reply},
        ]
        return reply
