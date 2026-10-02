"""The bundled fictional sample twin must keep working: it is how new users first try the app."""
import tempfile
import unittest
from pathlib import Path

import twin.config as cfg
from twin.chatparse import speakers_in
from twin.config import Settings
from twin.ingest import ingest_folder
from twin.memory import MemoryStore
from twin.persona import Persona
from tests.test_core import fake_embed

SAMPLE = Path(__file__).resolve().parent.parent / "examples" / "sample-twin"


class SampleTwin(unittest.TestCase):
    def test_maggie_writes_most_messages(self):
        chat = (SAMPLE / "WhatsApp Chat with Nana Maggie.txt").read_text(encoding="utf-8")
        self.assertEqual(speakers_in(chat).most_common(1)[0][0], "Maggie Thorne")

    def test_sample_builds_a_twin(self):
        old = cfg.DATA_DIR
        with tempfile.TemporaryDirectory() as dst:
            cfg.DATA_DIR = Path(dst)
            try:
                store = MemoryStore(Path(dst) / "maggie", embed_fn=fake_embed)
                ingest_folder("maggie", SAMPLE, "Maggie Thorne", Settings(), store=store,
                              log=lambda *_: None, listener="her granddaughter Lily")
                self.assertGreater(len(store), 25)
                persona = Persona("Maggie", Path(dst) / "maggie")
                persona.index_pairs(store.chunks)
                self.assertGreater(len(persona.style_pairs(3)), 0)  # real exchanges become tone examples
                self.assertIn("fish pie", persona.about)
                self.assertIn("granddaughter Lily", persona.system_prompt([]))
            finally:
                cfg.DATA_DIR = old


if __name__ == "__main__":
    unittest.main()
