"""End-to-end conversation test against a fake Ollama server (no model needed)."""
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import twin.config as cfg
from twin.chat import Conversation
from twin.config import Settings
from twin.ingest import ingest_folder
from twin.llm import OllamaNotReady, check_ollama
from twin.memory import MemoryStore
from tests.test_core import ANDROID, fake_embed

RECEIVED = []


class FakeOllama(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        body = json.dumps({"models": [{"name": "qwen2.5:7b"}]}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        RECEIVED.append(json.loads(self.rfile.read(length)))
        self.send_response(200); self.end_headers()
        for piece in ["The jalebi ", "was great. ", "Come along ", "next time!"]:
            self.wfile.write((json.dumps({"message": {"content": piece}, "done": False}) + "\n").encode())
        self.wfile.write((json.dumps({"done": True}) + "\n").encode())


class ChatFlow(unittest.TestCase):
    def test_full_turn(self):
        server = HTTPServer(("127.0.0.1", 0), FakeOllama)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{server.server_port}"
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
            (Path(src) / "chat.txt").write_text(ANDROID, encoding="utf-8")
            cfg.DATA_DIR = Path(dst)
            store = MemoryStore(Path(dst) / "ravi", embed_fn=fake_embed)
            ingest_folder("ravi", Path(src), "Ravi Kumar", Settings(), store=store, log=lambda *_: None)

            settings = Settings(ollama_url=url, llm_model="qwen2.5:7b")
            check_ollama(url, "qwen2.5:7b")
            with self.assertRaises(OllamaNotReady):
                check_ollama(url, "missing-model")

            conv = Conversation("ravi", settings)
            conv.store._embed_fn = fake_embed
            reply = conv.respond("tell me about the market jalebi")
            self.assertEqual(reply, "The jalebi was great. Come along next time!")
            msgs = RECEIVED[-1]["messages"]
            self.assertEqual(msgs[0]["role"], "system")
            self.assertEqual(msgs[-1], {"role": "user", "content": "tell me about the market jalebi"})
            # the real chat exchange is shown as earlier turns: Meera said it, Ravi answered
            turns = [(m["role"], m["content"]) for m in msgs[1:-1]]
            self.assertIn(("user", "How was it?"), turns)
            self.assertIn(("assistant", "Crowded but the jalebi was worth it\nand the tea afterwards too"), turns)
            conv.respond("and then?")
            msgs = RECEIVED[-1]["messages"]
            self.assertEqual(msgs[-1]["content"], "and then?")
            self.assertEqual(msgs[-2], {"role": "assistant", "content": "The jalebi was great. Come along next time!"})
            self.assertEqual(msgs[-3], {"role": "user", "content": "tell me about the market jalebi"})
        server.shutdown()


if __name__ == "__main__":
    unittest.main()
