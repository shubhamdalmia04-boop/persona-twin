"""Voice reference selection, TTS helpers, and ingest integration (no models needed)."""
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

import twin.config as cfg
from twin.config import Settings
from twin.ingest import ingest_folder
from twin.memory import MemoryStore
from twin.tts import SystemTTS, chunk_text, make_tts, pick_language
from twin.voice import VoiceBank, best_windows
from tests.test_core import fake_embed

SR = 22050


def speechlike(seconds, noise, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    env = 0.3 * (np.sin(2 * np.pi * 2 * t) > -0.5)
    return (env * np.sin(2 * np.pi * 180 * t) + noise * rng.standard_normal(len(t))).astype(np.float32)


def three_part_recording():
    noisy = speechlike(20, 0.08, 1)                                   # 0-20 s speech buried in noise
    silence = 0.001 * np.random.default_rng(2).standard_normal(20 * SR).astype(np.float32)  # 20-40 s
    clean = speechlike(20, 0.002, 3)                                  # 40-60 s clean speech
    return np.concatenate([noisy, silence, clean])


class ClipSelection(unittest.TestCase):
    def test_prefers_clean_speech(self):
        clips = best_windows(three_part_recording(), SR, clip_seconds=12, n=1)
        self.assertEqual(len(clips), 1)
        self.assertGreaterEqual(clips[0].start, 39.9)   # inside the clean part
        self.assertLessEqual(clips[0].end, 60.1)

    def test_non_overlapping(self):
        clips = best_windows(three_part_recording(), SR, clip_seconds=8, n=2)
        self.assertEqual(len(clips), 2)
        a, b = sorted(clips, key=lambda c: c.start)
        self.assertLessEqual(a.end, b.start)

    def test_too_short_or_silent(self):
        self.assertEqual(best_windows(speechlike(4, 0.001), SR), [])
        self.assertEqual(best_windows(np.zeros(20 * SR, dtype=np.float32), SR), [])


class Bank(unittest.TestCase):
    def test_keeps_best_three_and_deletes_others(self):
        with tempfile.TemporaryDirectory() as d:
            bank = VoiceBank(Path(d), max_clips=3)
            seg = speechlike(8, 0.002)
            for score in (0.2, 0.9, 0.5, 0.7):
                bank._pending.append((score, seg, SR, f"s{score}"))
            paths = bank.commit()
            self.assertEqual(len(paths), 3)
            self.assertEqual(sorted(e["score"] for e in bank.entries), [0.5, 0.7, 0.9])
            with wave.open(str(paths[0])) as w:
                self.assertEqual(w.getframerate(), SR)
                self.assertEqual(w.getnframes(), len(seg))
            # a new, better candidate evicts the weakest stored clip and its file
            bank2 = VoiceBank(Path(d), max_clips=3)
            weakest = min(bank2.entries, key=lambda e: e["score"])["file"]
            bank2._pending.append((0.95, seg, SR, "new"))
            bank2.commit()
            self.assertFalse((Path(d) / "voice" / weakest).exists())
            self.assertEqual(len(VoiceBank(Path(d)).paths()), 3)

    def test_clear(self):
        with tempfile.TemporaryDirectory() as d:
            bank = VoiceBank(Path(d))
            bank._pending.append((0.5, speechlike(8, 0.002), SR, "x"))
            bank.commit()
            bank.clear()
            self.assertEqual(VoiceBank(Path(d)).paths(), [])


class TTSHelpers(unittest.TestCase):
    def test_language(self):
        self.assertEqual(pick_language("hello"), "en")
        self.assertEqual(pick_language("नमस्ते दोस्त"), "hi")
        self.assertEqual(pick_language("hola", default="es"), "es")

    def test_chunking(self):
        self.assertEqual(chunk_text("  short  "), ["short"])
        self.assertEqual(chunk_text(""), [])
        long = ", ".join(["this is a fairly long clause of words"] * 12)
        pieces = chunk_text(long, limit=100)
        self.assertTrue(all(len(p) <= 100 for p in pieces))
        self.assertEqual(" ".join(pieces).replace(",", "").split(), long.replace(",", "").split())
        blob = "word " * 200
        self.assertTrue(all(len(p) <= 100 for p in chunk_text(blob, limit=100)))

    def test_make_tts_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            cfg.DATA_DIR = Path(d)
            logs = []
            self.assertIsInstance(make_tts("auto", "nobody", Settings(), log=logs.append), SystemTTS)
            self.assertIn("system voice", logs[0])
            with self.assertRaises(SystemExit):
                make_tts("clone", "nobody", Settings())


class IngestVoice(unittest.TestCase):
    class FakeTranscriber:
        def file(self, path):
            return ["this is a spoken sentence about the old market"]

    def _run(self, decode):
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
            (Path(src) / "memo.mp3").write_bytes(b"not real audio")
            cfg.DATA_DIR = Path(dst)
            store = MemoryStore(Path(dst) / "t", embed_fn=fake_embed)
            meta = ingest_folder("t", Path(src), "T", Settings(), transcriber=self.FakeTranscriber(),
                                 store=store, log=lambda *_: None, decode=decode)
            return meta, VoiceBank(Path(dst) / "t").paths()

    def test_reference_clips_created(self):
        meta, paths = self._run(lambda p: three_part_recording())
        self.assertEqual(meta["audio"], 1)
        # only the clean 20 s part qualifies, so exactly one 12 s reference clip fits
        self.assertEqual(meta["voice_clips"], 1)
        self.assertEqual(len(paths), 1)

    def test_voice_scan_failure_does_not_break_ingest(self):
        def boom(_):
            raise RuntimeError("cannot decode")
        meta, paths = self._run(boom)
        self.assertEqual(meta["voice_clips"], 0)
        self.assertEqual(meta["memories"], 1)


if __name__ == "__main__":
    unittest.main()
