"""Language selection: the languages table, persona prompt, ingest persistence, and commands."""
import json
import tempfile
import unittest
from pathlib import Path

import twin.config as cfg
from twin.config import Settings
from twin.ingest import ingest_folder
from twin.languages import DEFAULT_LANGUAGE, LANGUAGES, code_for, name_for
from twin.memory import MemoryStore
from twin.persona import Persona
from tests.test_core import ANDROID, fake_embed
from tests.test_folder import FakeTranscriber, make_library


class LanguageTable(unittest.TestCase):
    def test_code_for_and_name_for(self):
        self.assertEqual(code_for("Hindi"), "hi")
        self.assertEqual(code_for("Hindi + English mixed (Hinglish)"), "hi")
        self.assertEqual(code_for(DEFAULT_LANGUAGE), "")
        self.assertEqual(code_for("not a real language"), "")
        self.assertEqual(name_for("hi"), "Hindi")
        self.assertEqual(name_for(""), DEFAULT_LANGUAGE)
        self.assertEqual(name_for("xx"), "xx")   # unknown code: fall back to showing the code itself

    def test_every_entry_has_a_code_or_is_the_default(self):
        for name, code in LANGUAGES.items():
            if name != DEFAULT_LANGUAGE:
                self.assertTrue(code, name)


class PersonaLanguage(unittest.TestCase):
    def test_language_steers_the_prompt(self):
        with tempfile.TemporaryDirectory() as d:
            hindi = Persona("Papa", Path(d), language="hi")
            prompt = hindi.system_prompt([])
            self.assertIn("Reply in Hindi", prompt)
            self.assertIn("the language Papa spoke", prompt)
            none = Persona("Papa", Path(d))
            self.assertNotIn("Reply in", none.system_prompt([]))

    def test_saved_language_persists_and_can_be_overridden(self):
        with tempfile.TemporaryDirectory() as d:
            tdir = Path(d)
            tdir.mkdir(exist_ok=True)
            (tdir / "persona.json").write_text(json.dumps({"listener": "his son", "language": "hi"}))
            saved = Persona("Papa", tdir)
            self.assertEqual(saved.language, "hi")
            self.assertEqual(saved.listener, "his son")   # unrelated field is untouched
            overridden = Persona("Papa", tdir, language="es")
            self.assertEqual(overridden.language, "es")

    def test_unknown_code_falls_back_to_the_code_itself(self):
        with tempfile.TemporaryDirectory() as d:
            p = Persona("Papa", Path(d), language="mr")   # Marathi: not in LANGUAGE_NAMES
            self.assertIn("Reply in mr", p.system_prompt([]))


class IngestLanguage(unittest.TestCase):
    def test_language_is_saved(self):
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
            make_library(Path(src))
            cfg.DATA_DIR = Path(dst)
            store = MemoryStore(Path(dst) / "papa", embed_fn=fake_embed)
            ingest_folder("papa", Path(src), "Ravi Kumar", Settings(), transcriber=FakeTranscriber(), store=store,
                          log=lambda *_: None, include_media=False, language="hi", listener="his daughter Meera")
            saved = json.loads((Path(dst) / "papa" / "persona.json").read_text())
            self.assertEqual(saved, {"listener": "his daughter Meera", "language": "hi"})

    def test_language_used_to_construct_the_transcriber_when_none_given(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
            make_library(Path(src))
            cfg.DATA_DIR = Path(dst)
            store = MemoryStore(Path(dst) / "papa", embed_fn=fake_embed)
            captured = {}

            class Recording(FakeTranscriber):
                def __init__(self, model_size, language):
                    captured["language"] = language

            with patch("twin.stt.Transcriber", Recording):
                ingest_folder("papa", Path(src), "Ravi Kumar", Settings(), store=store, log=lambda *_: None,
                              media_minutes=10, language="hi",
                              duration=lambda p: 30.0)
            self.assertEqual(captured["language"], "hi")

    def test_language_not_overwritten_when_only_listener_changes(self):
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
            make_library(Path(src))
            cfg.DATA_DIR = Path(dst)
            tdir = Path(dst) / "papa"
            store = MemoryStore(tdir, embed_fn=fake_embed)
            ingest_folder("papa", Path(src), "Ravi Kumar", Settings(), include_media=False, store=store,
                          log=lambda *_: None, language="hi")
            store2 = MemoryStore(tdir, embed_fn=fake_embed)
            ingest_folder("papa", Path(src), "Ravi Kumar", Settings(), include_media=False, store=store2,
                          log=lambda *_: None, listener="his son Arjun")
            saved = json.loads((tdir / "persona.json").read_text())
            self.assertEqual(saved, {"listener": "his son Arjun", "language": "hi"})


class RunPyLanguageResolution(unittest.TestCase):
    def test_saved_language_helper(self):
        import run

        with tempfile.TemporaryDirectory() as d:
            cfg.DATA_DIR = Path(d)
            self.assertEqual(run._saved_language("nobody"), "")
            tdir = Path(d) / "papa"
            tdir.mkdir()
            (tdir / "persona.json").write_text(json.dumps({"language": "hi"}))
            self.assertEqual(run._saved_language("papa"), "hi")
            (tdir / "persona.json").write_text("not json")
            self.assertEqual(run._saved_language("papa"), "")


class LauncherLanguage(unittest.TestCase):
    def test_ingest_and_chat_commands_carry_language(self):
        from launcher import core

        ing = core.ingest_command("Papa", "D:/x", "Papa", language="hi")
        self.assertEqual(ing[ing.index("--language") + 1], "hi")
        self.assertNotIn("--language", core.ingest_command("Papa", "D:/x", "Papa"))

        chat = core.chat_command("Papa", "Window (for testing)", language="hi")
        self.assertEqual(chat[chat.index("--language") + 1], "hi")
        self.assertNotIn("--language", core.chat_command("Papa", "Window (for testing)"))

    def test_llm_model_is_opt_in_via_ingest_command(self):
        from launcher import core

        with_model = core.ingest_command("Papa", "D:/x", "Papa", language="hi", llm_model="llama3.1:8b")
        self.assertEqual(with_model[with_model.index("--llm-model") + 1], "llama3.1:8b")
        # picking the language alone must NOT imply the model override
        self.assertNotIn("--llm-model", core.ingest_command("Papa", "D:/x", "Papa", language="hi"))


class DownloadLanguageModel(unittest.TestCase):
    def _serve(self, lines, expect_ok=True):
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(200); self.send_header("Content-Length", "2"); self.end_headers()
                self.wfile.write(b"{}")

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200); self.end_headers()
                for line in lines:
                    self.wfile.write((json.dumps(line) + "\n").encode())

        server = HTTPServer(("127.0.0.1", 0), Handler)
        import threading
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_port}"

    def test_pull_model_starts_server_and_downloads(self):
        from twin.ollama import pull_model

        api = self._serve([{"status": "pulling manifest"}, {"status": "success"}])
        out = []
        # server already "up" at this url, so ensure_ollama_server should not need to spawn anything
        self.assertTrue(pull_model("unused-exe-path", "llama3.1:8b", api=api, log=out.append))
        self.assertIn("  pulling manifest", out)

    def test_download_language_model_without_ollama_installed(self):
        from unittest.mock import patch
        from launcher import core

        with patch("twin.hardware.find_ollama", return_value=None):
            out = []
            self.assertFalse(core.download_language_model("llama3.1:8b", out.append))
            self.assertTrue(any("not installed" in l for l in out))

    def test_download_language_model_success(self):
        from unittest.mock import patch
        from launcher import core

        api = self._serve([{"status": "pulling manifest"}, {"status": "success"}])
        with patch("twin.hardware.find_ollama", return_value="fake-ollama"), \
             patch("twin.ollama.OLLAMA_URL", api):
            out = []
            self.assertTrue(core.download_language_model("llama3.1:8b", out.append))


class LLMModelForLanguage(unittest.TestCase):
    def test_recommend_llm_for_language(self):
        from twin.hardware import recommend_llm_for_language

        self.assertIsNone(recommend_llm_for_language("en", 16, False, False))
        self.assertIsNone(recommend_llm_for_language("", 16, False, False))
        hi = recommend_llm_for_language("hi", 7.3, False, False)
        self.assertEqual(hi["model"], "llama3.1:8b")
        self.assertFalse(hi["fits"])
        self.assertTrue(recommend_llm_for_language("hi", 16, False, False)["fits"])
        self.assertTrue(recommend_llm_for_language("hi", 4, True, False)["fits"])   # GPU compensates for low RAM

    def test_saved_llm_model_overrides_settings(self):
        import run

        with tempfile.TemporaryDirectory() as d:
            cfg.DATA_DIR = Path(d)
            tdir = Path(d) / "papa"
            tdir.mkdir()
            (tdir / "persona.json").write_text(json.dumps({"llm_model": "llama3.1:8b"}))
            self.assertEqual(run._saved_persona_value("papa", "llm_model"), "llama3.1:8b")
            self.assertEqual(run._saved_persona_value("papa", "language"), "")
            self.assertEqual(run._saved_persona_value("nobody", "llm_model"), "")

    def test_llm_model_saved_by_ingest(self):
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
            make_library(Path(src))
            cfg.DATA_DIR = Path(dst)
            store = MemoryStore(Path(dst) / "papa", embed_fn=fake_embed)
            ingest_folder("papa", Path(src), "Ravi Kumar", Settings(), include_media=False, store=store,
                          log=lambda *_: None, language="hi", llm_model="llama3.1:8b")
            saved = json.loads((Path(dst) / "papa" / "persona.json").read_text())
            self.assertEqual(saved, {"language": "hi", "llm_model": "llama3.1:8b"})


if __name__ == "__main__":
    unittest.main()
