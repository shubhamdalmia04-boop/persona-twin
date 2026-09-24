"""Tests for the parts that need no models or hardware. Run: python -m unittest -v"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from twin.audio import envelope
from twin.chat import split_sentences
from twin.config import Settings, slugify
from twin.ingest import (
    chat_to_chunks, ingest_folder, is_person, looks_like_whatsapp,
    merge_segments, parse_json_messages, parse_whatsapp,
)
from twin.memory import MemoryStore
from twin.persona import Persona
from twin.state import DisplayState

ANDROID = """12/03/2021, 10:15 - Messages and calls are end-to-end encrypted. No one outside of this chat can read them.
12/03/2021, 10:15 - Ravi Kumar: I went to the old market today
12/03/2021, 10:16 - Meera: How was it?
12/03/2021, 10:17 - Ravi Kumar: Crowded but the jalebi was worth it
and the tea afterwards too
12/03/2021, 10:18 - Ravi Kumar: <Media omitted>
12/03/2021, 10:19 - Meera: Save me some next time
"""

IOS = """[12/03/21, 10:15:30 AM] Ravi Kumar: I went to the old market today
[12/03/21, 10:16:02 AM] Meera: How was it?
[12/03/21, 10:17:11\u202fAM] Ravi Kumar: Crowded but the jalebi was worth it
"""


def fake_embed(texts):
    """Deterministic bag-of-letters embedding, good enough to test ranking."""
    out = np.zeros((len(texts), 26), dtype=np.float32)
    for i, t in enumerate(texts):
        for ch in t.lower():
            if "a" <= ch <= "z":
                out[i, ord(ch) - 97] += 1
    return out


class Parsing(unittest.TestCase):
    def test_android_whatsapp(self):
        self.assertTrue(looks_like_whatsapp(ANDROID))
        msgs = parse_whatsapp(ANDROID)
        self.assertEqual(len(msgs), 4)  # system line and media line dropped
        self.assertEqual(msgs[0], ("Ravi Kumar", "I went to the old market today"))
        self.assertIn("tea afterwards", msgs[2][1])  # continuation line joined

    def test_ios_whatsapp(self):
        msgs = parse_whatsapp(IOS)
        self.assertEqual([m[0] for m in msgs], ["Ravi Kumar", "Meera", "Ravi Kumar"])

    def test_json(self):
        data = [{"from": "A", "text": "hello there"}, {"sender": "B", "message": ["hi ", {"text": "you"}]}]
        self.assertEqual(parse_json_messages(json.dumps(data)), [("A", "hello there"), ("B", "hi you")])

    def test_is_person(self):
        self.assertTrue(is_person("Ravi Kumar", "ravi"))
        self.assertFalse(is_person("Meera", "Ravi"))

    def test_chunks_and_style(self):
        chunks, style = chat_to_chunks(parse_whatsapp(ANDROID), "Ravi Kumar", "chat.txt")
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[1]["text"].startswith("Meera: How was it?"))
        self.assertIn("I went to the old market today", style)

    def test_merge_segments(self):
        self.assertEqual(merge_segments(["a" * 300, "b" * 300], max_chars=400), ["a" * 300, "b" * 300])
        self.assertEqual(merge_segments(["a", "b"], max_chars=400), ["a b"])


class Memory(unittest.TestCase):
    def test_search_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            store = MemoryStore(Path(d), embed_fn=fake_embed)
            store.add([{"text": "zebra zoo zigzag", "source": "t", "ref": "a"},
                       {"text": "apple banana cherry", "source": "t", "ref": "b"}])
            store.save()
            again = MemoryStore(Path(d), embed_fn=fake_embed)
            self.assertEqual(len(again), 2)
            hit = again.search("zebra zoo", k=1)[0]
            self.assertEqual(hit["ref"], "a")

    def test_empty(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(MemoryStore(Path(d), embed_fn=fake_embed).search("x"), [])


class Pipeline(unittest.TestCase):
    def test_ingest_and_persona(self):
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
            (Path(src) / "chat.txt").write_text(ANDROID, encoding="utf-8")
            (Path(src) / "about.txt").write_text("Ravi is Meera's brother.\n\nHe loves old markets.", encoding="utf-8")
            (Path(src) / "letter.md").write_text("Dear Meera,\n\nRemember the monsoon trip.", encoding="utf-8")
            import twin.config as cfg, twin.ingest as ing
            cfg.DATA_DIR = Path(dst)
            store = MemoryStore(Path(dst) / "ravi", embed_fn=fake_embed)
            meta = ingest_folder("ravi", Path(src), "Ravi Kumar", Settings(), store=store, log=lambda *_: None)
            self.assertGreaterEqual(meta["memories"], 4)
            self.assertEqual(meta["chats"], 1)
            persona = Persona("Ravi", Path(dst) / "ravi")
            prompt = persona.system_prompt(store.search("market jalebi", k=2))
            self.assertIn("AI simulation of Ravi", prompt)
            self.assertIn("loves old markets", prompt)
            self.assertIn("jalebi", prompt)


class Misc(unittest.TestCase):
    def test_sentences(self):
        done, rest = split_sentences("Hello there. How are you? I am")
        self.assertEqual(done, ["Hello there.", "How are you?"])
        self.assertEqual(rest, "I am")
        self.assertEqual(split_sentences("no end yet"), ([], "no end yet"))
        done, rest = split_sentences("नमस्ते। आप कैसे हैं? ठीक")
        self.assertEqual(len(done), 2)

    def test_envelope(self):
        sr = 16000
        t = np.linspace(0, 1, sr, endpoint=False)
        loud_then_quiet = np.concatenate([np.sin(2 * np.pi * 220 * t), 0.02 * np.sin(2 * np.pi * 220 * t)])
        env = envelope(loud_then_quiet.astype(np.float32), sr, fps=30)
        self.assertEqual(len(env), 60)
        self.assertGreater(env[10], 0.8)
        self.assertLess(env[50], 0.2)

    def test_state_amplitude(self):
        s = DisplayState()
        self.assertEqual(s.amplitude(), 0.0)
        s.start_speech(np.ones(300, dtype=np.float32))
        self.assertEqual(s.status, "speaking")
        self.assertEqual(s.amplitude(), 1.0)
        s.end_speech()
        self.assertEqual(s.amplitude(), 0.0)

    def test_slug(self):
        self.assertEqual(slugify(" Asha Verma! "), "asha-verma")


if __name__ == "__main__":
    unittest.main()
