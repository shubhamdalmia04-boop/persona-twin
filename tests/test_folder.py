"""Any-folder input: scanning, documents, photos, recording budget, launcher and build helpers."""
import json
import sys
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path

import twin.config as cfg
from twin.chatparse import speakers_in
from twin.config import Settings
from twin.docs import read_docx, read_pdf
from twin.filetypes import classify, walk
from twin.hardware import find_ollama
from twin.ingest import ingest_folder
from twin.memory import MemoryStore
from twin.scan import scan_folder
from twin.voice import VoiceBank
from tests.test_core import ANDROID, fake_embed
from tests.test_voice import speechlike, SR


def make_docx(path: Path, paragraphs):
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    xml = ('<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/'
           f'wordprocessingml/2006/main"><w:body>{body}</w:body></w:document>')
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("word/document.xml", xml)


def tiny_pdf(text):
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] /Contents 4 0 R "
            "/Resources << /Font << /F1 5 0 R >> >> >>"]
    stream = f"BT /F1 18 Tf 20 100 Td ({text}) Tj ET"
    objs += [f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
             "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offsets = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{off:010d} 00000 n \n".encode() for off in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return out


def make_library(root: Path):
    (root / "chats").mkdir()
    (root / "chats" / "family.txt").write_text(ANDROID, encoding="utf-8")
    (root / "notes.txt").write_text("Just a shopping list\n\nmilk, eggs", encoding="utf-8")
    make_docx(root / "letter.docx", ["Dear Meera,", "Remember the monsoon trip we took to the hills."])
    (root / "scan.pdf").write_bytes(tiny_pdf("Recipe for the family jalebi"))
    (root / "about.txt").write_text("Ravi is Meera's brother.", encoding="utf-8")
    (root / "memo.mp3").write_bytes(b"x" * 100)
    (root / "clip.mp4").write_bytes(b"x" * 1000)
    (root / "photos").mkdir()
    for i in range(40):
        (root / "photos" / f"img{i:02d}.jpg").write_bytes(b"jpegbytes")
    (root / "photos" / "iphone.heic").write_bytes(b"heic")
    (root / "junk.xyz").write_bytes(b"?")
    (root / ".hidden").mkdir()
    (root / ".hidden" / "secret.txt").write_text("skip me")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "x.txt").write_text("skip me too")


class Basics(unittest.TestCase):
    def test_classify(self):
        cases = {"a.TXT": "chat_or_text", "a.json": "json", "a.docx": "doc", "a.pdf": "doc", "a.OPUS": "audio",
                 "a.MOV": "video", "a.HEIC": "photo", "a.png": "photo", "a.zip": "other"}
        for name, kind in cases.items():
            self.assertEqual(classify(Path(name)), kind, name)

    def test_walk_skips_hidden_and_system_dirs(self):
        with tempfile.TemporaryDirectory() as d:
            make_library(Path(d))
            names = {p.name for p in walk(Path(d))}
            self.assertNotIn("secret.txt", names)
            self.assertNotIn("x.txt", names)
            self.assertIn("family.txt", names)

    def test_docx(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.docx"
            make_docx(p, ["First paragraph.", "", "Second one."])
            self.assertEqual(read_docx(p), "First paragraph.\n\nSecond one.")
            self.assertEqual(read_docx(Path(d) / "missing.docx"), "")
            (Path(d) / "bad.docx").write_bytes(b"not a zip")
            self.assertEqual(read_docx(Path(d) / "bad.docx"), "")

    def test_pdf(self):
        try:
            import pypdf  # noqa: F401
        except ImportError:
            self.skipTest("pypdf not installed")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.pdf"
            p.write_bytes(tiny_pdf("Hello from the PDF"))
            self.assertIn("Hello from the PDF", read_pdf(p))
            (Path(d) / "bad.pdf").write_bytes(b"garbage")
            self.assertEqual(read_pdf(Path(d) / "bad.pdf"), "")

    def test_speakers(self):
        counts = speakers_in(ANDROID)
        self.assertEqual(counts["Ravi Kumar"], 2)
        self.assertEqual(counts["Meera"], 2)
        self.assertEqual(len(speakers_in("plain text, not a chat")), 0)
        data = json.dumps([{"from": "A", "text": "hello"}, {"from": "A", "text": "again"}, {"from": "B", "text": "hi"}])
        self.assertEqual(speakers_in(data, is_json=True).most_common(1), [("A", 2)])


class Scanning(unittest.TestCase):
    def test_scan_summary_and_speakers(self):
        with tempfile.TemporaryDirectory() as d:
            make_library(Path(d))
            r = scan_folder(Path(d))
            self.assertEqual(r.counts["chats"], 1)
            self.assertEqual(r.counts["texts"], 2)        # notes.txt + about.txt
            self.assertEqual(r.counts["docs"], 2)
            self.assertEqual(r.counts["audio"], 1)
            self.assertEqual(r.counts["video"], 1)
            self.assertEqual(r.counts["photos"], 41)
            self.assertEqual(r.counts["other"], 1)
            self.assertEqual(r.speakers[0][0], "Ravi Kumar")
            self.assertIn("41 photos", r.summary())
            self.assertFalse(r.truncated)

    def test_empty_folder(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(scan_folder(Path(d)).summary(), "nothing usable found")


class FakeTranscriber:
    def __init__(self):
        self.seen = []

    def file(self, path):
        self.seen.append(path.name)
        return ["spoken words about the market and the monsoon"]


class IngestFolder(unittest.TestCase):
    def _ingest(self, **kw):
        self.src, self.dst = tempfile.TemporaryDirectory(), tempfile.TemporaryDirectory()
        self.addCleanup(self.src.cleanup)
        self.addCleanup(self.dst.cleanup)
        make_library(Path(self.src.name))
        cfg.DATA_DIR = Path(self.dst.name)
        self.tr = FakeTranscriber()
        store = MemoryStore(Path(self.dst.name) / "ravi", embed_fn=fake_embed)
        durations = {"memo.mp3": 60.0, "clip.mp4": 3600.0}
        return ingest_folder(
            "ravi", Path(self.src.name), "Ravi Kumar", Settings(), transcriber=self.tr, store=store,
            log=lambda *_: None, decode=lambda p: speechlike(20, 0.002),
            duration=lambda p: durations[p.name], **kw)

    def test_everything_supported_is_used(self):
        meta = self._ingest(media_minutes=10)
        self.assertEqual((meta["chats"], meta["docs"], meta["texts"]), (1, 2, 1))
        self.assertEqual(meta["photos"], 41)
        self.assertEqual(meta["photos_kept"], 30)                 # capped
        self.assertTrue((Path(self.dst.name) / "ravi" / "about.txt").exists())
        store = MemoryStore(Path(self.dst.name) / "ravi", embed_fn=fake_embed)
        texts = " ".join(c["text"] for c in store.chunks)
        self.assertIn("monsoon trip", texts)                       # from the .docx

    def test_recording_budget_skips_the_long_video(self):
        meta = self._ingest(media_minutes=10)
        self.assertEqual(self.tr.seen, ["memo.mp3"])              # 60 s fits, 3600 s does not
        self.assertEqual((meta["audio"], meta["video"]), (1, 0))
        self.assertGreaterEqual(meta["skipped"], 1)
        self.assertEqual(meta["voice_clips"], 1)

    def test_media_can_be_excluded(self):
        meta = self._ingest(include_media=False)
        self.assertEqual(self.tr.seen, [])
        self.assertEqual(meta["voice_clips"], 0)

    def test_face_photo_is_saved_and_replaced(self):
        with tempfile.TemporaryDirectory() as extra:
            a, b = Path(extra) / "a.jpg", Path(extra) / "b.png"
            a.write_bytes(b"one")
            b.write_bytes(b"two")
            self._ingest(face_photo=a)
            tdir = Path(self.dst.name) / "ravi"
            self.assertEqual((tdir / "avatar.jpg").read_bytes(), b"one")
            ingest_folder("ravi", Path(self.src.name), "Ravi Kumar", Settings(), include_media=False,
                          store=MemoryStore(tdir, embed_fn=fake_embed), log=lambda *_: None, face_photo=b)
            self.assertEqual([p.name for p in tdir.glob("avatar.*")], ["avatar.png"])

    def test_listener_is_saved(self):
        self._ingest(include_media=False, listener="his daughter Meera")
        tdir = Path(self.dst.name) / "ravi"
        self.assertEqual(json.loads((tdir / "persona.json").read_text())["listener"], "his daughter Meera")

    def test_about_text_is_appended_and_indexed(self):
        self._ingest(include_media=False, about_text="Ravi always called Meera 'chhoti'.")
        tdir = Path(self.dst.name) / "ravi"
        self.assertIn("chhoti", (tdir / "about.txt").read_text())
        store = MemoryStore(tdir, embed_fn=fake_embed)
        self.assertIn("chhoti", " ".join(c["text"] for c in store.chunks))

    def test_warns_when_speaker_name_does_not_match_the_chats(self):
        logs = []
        self.src, self.dst = tempfile.TemporaryDirectory(), tempfile.TemporaryDirectory()
        self.addCleanup(self.src.cleanup)
        self.addCleanup(self.dst.cleanup)
        make_library(Path(self.src.name))
        cfg.DATA_DIR = Path(self.dst.name)
        meta = ingest_folder(
            "wrong", Path(self.src.name), "Someone Else", Settings(), include_media=False,
            store=MemoryStore(Path(self.dst.name) / "wrong", embed_fn=fake_embed), log=logs.append)
        self.assertEqual(meta["matched_messages"], 0)
        self.assertTrue(any("no messages from" in l for l in logs))
        self.assertTrue(any("learned nothing from the chats" in l for l in logs))
        self.assertEqual(dict(meta["chat_speakers"]), {"Ravi Kumar": 2, "Meera": 2})

    def test_correct_speaker_name_produces_no_warning(self):
        logs = []
        meta = self._ingest(include_media=False)
        self.tr = FakeTranscriber()
        cfg.DATA_DIR = Path(self.dst.name)
        meta = ingest_folder("ravi2", Path(self.src.name), "Ravi Kumar", Settings(), include_media=False,
                             store=MemoryStore(Path(self.dst.name) / "ravi2", embed_fn=fake_embed), log=logs.append)
        self.assertGreater(meta["matched_messages"], 0)
        self.assertFalse(any("no messages from" in l for l in logs))


class PersonaPrompt(unittest.TestCase):
    def test_pairs_used_as_examples_and_facts_kept_separate(self):
        from twin.persona import Persona

        with tempfile.TemporaryDirectory() as d:
            tdir = Path(d)
            (tdir / "meta.json").write_text(json.dumps({"speaker": "Ravi Kumar"}))
            (tdir / "about.txt").write_text("Ravi is Meera's brother.")
            chunks = [
                {"text": "Meera: How was it?\nRavi Kumar: Crowded but great", "source": "chat", "ref": "a"},
                {"text": "Ravi is Meera's brother.", "source": "about", "ref": "about.txt"},
            ]
            p = Persona("ravi", tdir)
            p.index_pairs(chunks)
            memories = [{"text": "Meera: How was it?\nRavi Kumar: Crowded but great", "source": "chat", "score": 0.9},
                       {"text": "Ravi is Meera's brother.", "source": "about", "score": 0.5}]
            pairs, facts = p.split_memories(memories)
            self.assertEqual(pairs, [("Meera", "How was it?", "Crowded but great")])
            self.assertEqual(facts, [memories[1]])
            prompt = p.system_prompt(facts)
            self.assertIn("Ravi is Meera's brother.", prompt)
            self.assertNotIn("Crowded but great", prompt)   # pairs go in as chat turns, not the prompt text

    def test_listener_changes_the_prompt(self):
        from twin.persona import Persona

        with tempfile.TemporaryDirectory() as d:
            p = Persona("Ravi", Path(d), listener="his daughter Meera")
            self.assertIn("You are chatting with his daughter Meera.", p.system_prompt([]))
            default = Persona("Ravi", Path(d))
            self.assertIn("someone who knows or wants to know Ravi", default.system_prompt([]))

    def test_listener_persists_across_sessions(self):
        from twin.persona import Persona

        with tempfile.TemporaryDirectory() as d:
            Persona("Ravi", Path(d), listener="her son Rahul")
            (Path(d) / "persona.json").write_text(json.dumps({"listener": "her son Rahul"}))
            again = Persona("Ravi", Path(d))
            self.assertEqual(again.listener, "her son Rahul")


class TwinReport(unittest.TestCase):
    def test_describe_twin(self):
        from twin.report import describe_twin

        with tempfile.TemporaryDirectory() as d:
            tdir = Path(d)
            self.assertIn("No twin found", describe_twin(tdir)[0])
            (tdir / "meta.json").write_text(json.dumps({
                "name": "Ravi", "speaker": "Ravi Kumar", "style_samples": 4, "voice_clips": 0,
                "matched_messages": 0, "chats": 1, "chat_speakers": [["Ravi Kumar", 3], ["Meera", 2]]}))
            (tdir / "chunks.json").write_text(json.dumps([{"source": "chat"}, {"source": "about"}]))
            lines = "\n".join(describe_twin(tdir))
            self.assertIn("Memories: 2", lines)
            self.assertIn("Face photo chosen: no", lines)
            self.assertIn("Who is talking to them: not set", lines)
            self.assertIn("Language: auto-detect", lines)
            self.assertIn("NONE of the chat messages matched", lines)
            self.assertIn("Ravi Kumar (3), Meera (2)", lines)

    def test_describe_twin_with_language_and_model(self):
        from twin.report import describe_twin

        with tempfile.TemporaryDirectory() as d:
            tdir = Path(d)
            (tdir / "meta.json").write_text(json.dumps({
                "name": "Papa", "speaker": "Papa", "style_samples": 4, "voice_clips": 1,
                "matched_messages": 300, "chats": 1, "chat_speakers": [["Papa", 300]]}))
            (tdir / "chunks.json").write_text("[]")
            (tdir / "persona.json").write_text(json.dumps(
                {"listener": "his daughter Meera", "language": "hi", "llm_model": "llama3.1:8b"}))
            lines = "\n".join(describe_twin(tdir))
            self.assertIn("Language: Hindi", lines)
            self.assertIn("llama3.1:8b", lines)
            self.assertNotIn("NONE of the chat messages matched", lines)


class Launcher(unittest.TestCase):
    def setUp(self):
        from launcher import core
        self.core = core

    def test_pth_content(self):
        text = self.core.pth_content("python312.zip", r"..\..\app")
        self.assertEqual(text.splitlines(), ["python312.zip", ".", "Lib/site-packages", "../../app", "import site"])

    def test_chat_command_variants(self):
        c = self.core.chat_command
        window = c("Asha", "Window (for testing)")
        self.assertIn("--windowed", window)
        self.assertEqual(window[window.index("--display") + 1], "single")
        full = c("Asha", "Fullscreen / projector", mirror=True, screen=1, microphone=True)
        self.assertNotIn("--windowed", full)
        self.assertEqual(full[full.index("--screen") + 1], "1")
        self.assertTrue({"--mirror", "--voice"} <= set(full))
        text = c("Asha", "Text only (no picture)", speak=False)
        self.assertEqual(text[text.index("--display") + 1], "none")
        self.assertNotIn("--screen", text)
        self.assertIn("--no-speech", text)
        self.assertEqual(c("Asha", "Hologram pyramid")[c("Asha", "Hologram pyramid").index("--display") + 1], "pyramid")
        with_photo = c("Asha", "Window (for testing)", photo="face.jpg")
        self.assertEqual(with_photo[with_photo.index("--photo") + 1], "face.jpg")
        self.assertNotIn("--photo", window)
        self.assertEqual(self.core.soundtest_command()[-1], "soundtest")
        personal = c("Asha", "Window (for testing)", listener="his daughter Meera", show_memories=True)
        self.assertEqual(personal[personal.index("--listener") + 1], "his daughter Meera")
        self.assertIn("--show-memories", personal)
        self.assertEqual(self.core.inspect_command("Asha")[-3:], ["inspect", "--name", "Asha"])

    def test_ingest_command(self):
        cmd = self.core.ingest_command("Asha", "C:/stuff", "Asha V", "face.jpg", include_media=False, media_minutes=45)
        for part in ("--i-have-permission", "--no-media", "--face-photo", "face.jpg"):
            self.assertIn(part, cmd)
        self.assertEqual(cmd[cmd.index("--media-minutes") + 1], "45")
        self.assertEqual(cmd[cmd.index("--speaker") + 1], "Asha V")
        rich = self.core.ingest_command("Asha", "C:/x", "Asha V", about_file="notes.txt", listener="her son Rahul")
        self.assertEqual(rich[rich.index("--about-file") + 1], "notes.txt")
        self.assertEqual(rich[rich.index("--listener") + 1], "her son Rahul")

    def test_list_twins(self):
        with tempfile.TemporaryDirectory() as d:
            original = self.core.home_dir
            self.core.home_dir = lambda: Path(d)
            try:
                self.assertEqual(self.core.list_twins(), [])
                (Path(d) / "data" / "asha").mkdir(parents=True)
                (Path(d) / "data" / "asha" / "meta.json").write_text(json.dumps({"name": "Asha"}))
                (Path(d) / "data" / "bad").mkdir()
                (Path(d) / "data" / "bad" / "meta.json").write_text("{not json")
                self.assertEqual(self.core.list_twins(), ["Asha", "bad"])
                self.assertFalse(self.core.is_setup_done())
            finally:
                self.core.home_dir = original

    def test_unpack_embedded_python(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            archive = root / "embed.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("python.exe", b"MZ")
                z.writestr("python312.zip", b"stdlib")
                z.writestr("python312._pth", "python312.zip\n.\n\n#import site\n")
            py_dir, app = root / "runtime" / "python", root / "app"
            self.core.unpack_embedded_python(archive, py_dir, app)
            pth = (py_dir / "python312._pth").read_text().splitlines()
            self.assertEqual(pth[0], "python312.zip")
            self.assertIn("import site", pth)
            self.assertNotIn("#import site", pth)
            self.assertTrue(any(line.endswith("app") for line in pth))
            self.assertTrue((py_dir / "Lib" / "site-packages").is_dir())

    def test_stream_and_chat_session(self):
        lines = []
        code = self.core.stream([sys.executable, "-c", "print('héllo'); print('two')"], lines.append)
        self.assertEqual((code, lines), (0, ["héllo", "two"]))

        got, done = [], threading.Event()
        session = self.core.ChatSession()
        script = "import sys\nfor l in sys.stdin: print('echo:' + l.strip(), flush=True)"
        session.start([sys.executable, "-u", "-c", script], got.append, lambda code: done.set())
        self.assertTrue(session.send("hi there"))
        for _ in range(100):
            if got:
                break
            threading.Event().wait(0.05)
        self.assertEqual(got, ["echo:hi there"])
        session.stop()
        self.assertTrue(done.wait(5))
        self.assertFalse(session.send("late"))

    def test_find_ollama_is_safe(self):
        result = find_ollama()
        self.assertTrue(result is None or isinstance(result, str))


class OllamaOutput(unittest.TestCase):
    def test_clean_output_keeps_only_the_last_redraw(self):
        from launcher.core import clean_output, is_progress_line
        raw = ("\x1b[?2026h\x1b[?25l\x1b[1Gpulling manifest \u280b \x1b[K\x1b[?25h\x1b[?2026l"
               "\x1b[?2026h\x1b[?25l\x1b[1Gpulling manifest \u2819 \x1b[K\x1b[?25h\x1b[?2026l\n")
        self.assertEqual(clean_output(raw), "pulling manifest \u2819")
        bar = "pulling 5ee4f07cdb9b:   1% |  | 20 MB/1.9 GB  3.4 MB/s  9m26s\x1b[K\x1b[?25h\x1b[A\x1b[1Gpulling manifest \x1b[K\n"
        self.assertEqual(clean_output(bar), "pulling manifest")
        self.assertEqual(clean_output("\x1b[K\r\n"), "")
        self.assertEqual(clean_output("plain text\r\n"), "plain text")
        self.assertTrue(is_progress_line("pulling 5ee4f07cdb9b:  50% | 1 GB"))
        self.assertFalse(is_progress_line("Successfully installed foo"))


class FakeOllamaPull(unittest.TestCase):
    def _serve(self, lines):
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
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_port}"

    def test_progress_is_summarised(self):
        import install
        from twin.ollama import ollama_up
        lines = [{"status": "pulling manifest"}]
        lines += [{"status": "pulling abc", "digest": "abc", "total": 1000, "completed": n} for n in range(0, 1001, 10)]
        lines += [{"status": "verifying sha256 digest"}, {"status": "success"}]
        api, out = self._serve(lines), []
        self.assertTrue(ollama_up(api))
        self.assertTrue(install.pull_model_api("qwen2.5:3b", api, out.append))
        self.assertEqual(out[0], "  pulling manifest")
        self.assertLessEqual(len([l for l in out if "downloading" in l]), 22)   # ~5% steps, not 100 lines
        self.assertIn("100%", [l for l in out if "downloading" in l][-1])
        self.assertEqual(out[-2:], ["  verifying sha256 digest", "  success"])

    def test_error_and_unreachable(self):
        import install
        from twin.ollama import ollama_up
        out = []
        api = self._serve([{"error": "pull model manifest: file does not exist"}])
        self.assertFalse(install.pull_model_api("nope", api, out.append))
        self.assertIn("file does not exist", out[-1])
        self.assertFalse(install.pull_model_api("x", "http://127.0.0.1:9", out.append))
        self.assertFalse(ollama_up("http://127.0.0.1:9"))


class InstallerHelpers(unittest.TestCase):
    def test_constraints_from_freeze(self):
        import install
        freeze = "torch==2.14.0\nnumpy==2.5.3\npip==25.0\nsetuptools==84.0.0\n-e git+x#egg=y\nfoo @ file:///x\n# c\nPillow==12.3.0\n"
        self.assertEqual(install.constraints_from_freeze(freeze).splitlines(),
                         ["torch==2.14.0", "numpy==2.5.3", "Pillow==12.3.0"])


class Packaging(unittest.TestCase):
    def test_payload_has_program_files_only(self):
        import build_exe
        with tempfile.TemporaryDirectory() as d:
            dest = build_exe.make_payload(Path(d) / "payload")
            names = {p.name for p in dest.iterdir()}
            self.assertEqual(names, {"run.py", "install.py", "requirements.txt", "requirements-face.txt",
                                     "requirements-voice.txt", "twin"})
            self.assertTrue((dest / "twin" / "face.py").exists())
            self.assertFalse(list(dest.rglob("__pycache__")))
            self.assertFalse((dest / "tests").exists())


if __name__ == "__main__":
    unittest.main()
