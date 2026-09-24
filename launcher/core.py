"""Everything the launcher does that is not drawing windows. Standard library only.

Layout on disk (next to PersonaTwin.exe, or the project folder when run from source):
    app/       the program files (extracted from the exe on every start)
    runtime/   a private Python with all packages (created by "Set up")
    data/      your twins: memories, voice clips, photos. Personal data: keep it safe.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import webbrowser
import zipfile
from pathlib import Path
from typing import Callable

EMBED_PY_VERSION = "3.12.8"
EMBED_URL = "https://www.python.org/ftp/python/{v}/python-{v}-embed-amd64.zip"
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"
OLLAMA_INSTALLER_URL = "https://ollama.com/download/OllamaSetup.exe"
OLLAMA_PAGE = "https://ollama.com/download"
OLLAMA_API = "http://localhost:11434/api/tags"

FROZEN = bool(getattr(sys, "frozen", False))
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

Log = Callable[[str], None]

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def clean_output(line: str) -> str:
    """Make a command's output readable in a text box.

    Progress bars redraw one line many times using cursor codes; keep only the final redraw and
    strip the escape sequences (which otherwise show up as empty boxes).
    """
    parts = re.split(r"\r|\x1b\[1G", line)
    for part in reversed(parts):
        text = _CONTROL.sub("", _ANSI.sub("", part)).rstrip()
        if text.strip():
            return text
    return ""


def is_progress_line(text: str) -> bool:
    return text.startswith("pulling ") and ("%" in text or "manifest" in text)


# ------------------------------------------------------------------ locations
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def home_dir() -> Path:
    return Path(sys.executable).resolve().parent if FROZEN else project_root()


def payload_dir() -> Path:
    """Where the program files are bundled inside the exe."""
    return Path(getattr(sys, "_MEIPASS", project_root())) / "app_payload"


def app_dir() -> Path:
    return home_dir() / "app" if FROZEN else project_root()


def data_dir() -> Path:
    return home_dir() / "data"


def runtime_dir() -> Path:
    return home_dir() / "runtime"


def runtime_python() -> Path:
    if os.name == "nt":
        return runtime_dir() / "python" / "python.exe"
    return runtime_dir() / "venv" / "bin" / "python"


def setup_marker() -> Path:
    return data_dir() / "setup_done.json"


def is_setup_done() -> bool:
    return setup_marker().exists() and runtime_python().exists()


def sync_app() -> None:
    """When running from the exe, unpack the bundled program files next to it."""
    if not FROZEN:
        return
    src, dst = payload_dir(), app_dir()
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    data_dir().mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ processes
def base_env() -> dict:
    env = dict(os.environ)
    env.update(
        TWIN_DATA_DIR=str(data_dir()),
        TWIN_NO_VENV="1",
        PYTHONUNBUFFERED="1",
        PYTHONIOENCODING="utf-8",
        PYTHONUTF8="1",
        HF_HUB_DISABLE_SYMLINKS_WARNING="1",
    )
    try:  # make sure a freshly installed Ollama is found even if PATH was not refreshed
        from twin.hardware import find_ollama

        exe = find_ollama()
        if exe:
            env["PATH"] = str(Path(exe).parent) + os.pathsep + env.get("PATH", "")
    except Exception:
        pass
    return env


def stream(cmd: list[str], on_line: Log, cwd: Path | None = None) -> int:
    """Run a command, sending each output line to on_line. Returns the exit code."""
    with subprocess.Popen(
        cmd, cwd=str(cwd) if cwd else None, env=base_env(),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        text=True, encoding="utf-8", errors="replace", bufsize=1, creationflags=NO_WINDOW,
    ) as proc:
        assert proc.stdout is not None
        last_progress = 0.0
        for raw in proc.stdout:
            text = clean_output(raw)
            if not text:
                continue
            if is_progress_line(text):  # download bars: show a fresh line every few seconds
                now = time.monotonic()
                if now - last_progress < 3 and "100%" not in text:
                    continue
                last_progress = now
            on_line(text)
        return proc.wait()


class ChatSession:
    """A running `run.py chat` process: send text in, receive the twin's lines out."""

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, cmd: list[str], on_line: Log, on_exit: Callable[[int], None]) -> None:
        self.proc = subprocess.Popen(
            cmd, cwd=str(app_dir()), env=base_env(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1, creationflags=NO_WINDOW,
        )

        def pump() -> None:
            assert self.proc is not None and self.proc.stdout is not None
            for raw in self.proc.stdout:
                text = clean_output(raw)
                if text:
                    on_line(text)
            code = self.proc.wait()
            self.proc.stdout.close()
            if self.proc.stdin:
                try:
                    self.proc.stdin.close()
                except OSError:
                    pass
            on_exit(code)

        threading.Thread(target=pump, daemon=True).start()

    def send(self, text: str) -> bool:
        if not self.running or self.proc is None or self.proc.stdin is None:
            return False
        try:
            self.proc.stdin.write(text.strip() + "\n")
            self.proc.stdin.flush()
            return True
        except OSError:
            return False

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()


# ------------------------------------------------------------------ downloads and Python runtime
def download(url: str, dest: Path, log: Log) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as response, open(dest, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done, next_mark = 0, 10
        while True:
            block = response.read(1024 * 256)
            if not block:
                break
            out.write(block)
            done += len(block)
            if total and done * 100 // total >= next_mark:
                log(f"  {done * 100 // total}% ({done // 1_048_576} MB)")
                next_mark += 10


def pth_content(zip_name: str, app_relative: str) -> str:
    """The `pythonXY._pth` file for the embedded Python: enables pip packages and finds our code."""
    return "\n".join([zip_name, ".", "Lib/site-packages", app_relative.replace("\\", "/"), "import site", ""])


def unpack_embedded_python(archive: Path, py_dir: Path, app: Path) -> None:
    """Extract the official embeddable Python and configure it for pip packages + our code."""
    with zipfile.ZipFile(archive) as z:
        z.extractall(py_dir)
    zip_name = next(py_dir.glob("python*.zip")).name
    pth = next(py_dir.glob("python*._pth"))
    pth.write_text(pth_content(zip_name, os.path.relpath(app, py_dir)), encoding="utf-8")
    (py_dir / "Lib" / "site-packages").mkdir(parents=True, exist_ok=True)


def bootstrap_runtime(log: Log) -> None:
    """Create the private Python. On Windows this downloads the official embeddable Python."""
    rt = runtime_dir()
    rt.mkdir(parents=True, exist_ok=True)
    py = runtime_python()

    if os.name == "nt":
        py_dir = rt / "python"
        if not py.exists():
            if platform.machine().lower() not in ("amd64", "x86_64"):
                raise RuntimeError("This needs 64-bit Windows on an Intel/AMD processor (ARM is not supported yet).")
            archive = rt / "python-embed.zip"
            download(EMBED_URL.format(v=EMBED_PY_VERSION), archive, log)
            unpack_embedded_python(archive, py_dir, app_dir())
            archive.unlink(missing_ok=True)
            log("Python runtime unpacked.")
    else:
        if FROZEN:
            raise RuntimeError("The packaged app is Windows only. On macOS/Linux run: python launcher/main.py")
        if not py.exists():
            log("Creating a private Python environment...")
            subprocess.check_call([sys.executable, "-m", "venv", str(rt / "venv")])

    if subprocess.call([str(py), "-m", "pip", "--version"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, creationflags=NO_WINDOW) != 0:
        get_pip = rt / "get-pip.py"
        download(GET_PIP_URL, get_pip, log)
        if stream([str(py), str(get_pip), "--no-warn-script-location"], log) != 0:
            raise RuntimeError("Could not install pip.")
        get_pip.unlink(missing_ok=True)


def run_install(log: Log, voice_clone: bool, only_llm: bool = False) -> int:
    cmd = [str(runtime_python()), "-u", str(app_dir() / "install.py"), "--here"]
    cmd += ["--only-llm"] if only_llm else []
    cmd += ["--voice-clone", "--accept-xtts-terms"] if voice_clone else ["--no-voice-clone"]
    return stream(cmd, log, cwd=app_dir())


def full_setup(log: Log, voice_clone: bool) -> bool:
    """Runtime + packages + models. Returns True when the language model is also ready."""
    bootstrap_runtime(log)
    code = run_install(log, voice_clone)
    if code != 0:
        raise RuntimeError(f"Setup step failed (exit code {code}). Scroll up for details, then try again.")
    data_dir().mkdir(parents=True, exist_ok=True)
    setup_marker().write_text(json.dumps({"voice_clone": voice_clone}), encoding="utf-8")
    from twin.hardware import find_ollama

    return find_ollama() is not None


# ------------------------------------------------------------------ Ollama
def ollama_ready() -> bool:
    try:
        with urllib.request.urlopen(OLLAMA_API, timeout=2):
            return True
    except Exception:
        return False


def ensure_ollama_running(log: Log, wait_seconds: int = 20) -> bool:
    from twin.hardware import find_ollama

    if ollama_ready():
        return True
    exe = find_ollama()
    if not exe:
        return False
    log("Starting Ollama...")
    subprocess.Popen([exe, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)
    for _ in range(wait_seconds):
        if ollama_ready():
            return True
        time.sleep(1)
    return False


def download_language_model(model: str, log: Log) -> bool:
    """Pull an additional Ollama model (e.g. a better one for a specific language).

    Runs in the launcher process itself - no subprocess needed, since it is just HTTP calls to
    the local Ollama server. Returns False (with a reason logged) if Ollama is not installed.
    """
    from twin.hardware import find_ollama
    from twin.ollama import pull_model

    exe = find_ollama()
    if not exe:
        log("Ollama is not installed yet. Finish setup on the '1. Set up' tab first.")
        return False
    return pull_model(exe, model, log=log)


def install_ollama(log: Log) -> bool:
    """Download and start Ollama's own installer (you click through it). Returns True if found afterwards."""
    from twin.hardware import find_ollama

    if os.name != "nt":
        webbrowser.open(OLLAMA_PAGE)
        log(f"Please install Ollama from {OLLAMA_PAGE}, then press the button again.")
        return False
    installer = Path(tempfile.gettempdir()) / "OllamaSetup.exe"
    try:
        download(OLLAMA_INSTALLER_URL, installer, log)
        log("Running the Ollama installer. Follow its window, then come back here.")
        subprocess.call([str(installer)])
    except Exception as exc:
        log(f"Could not run the installer ({exc}). Opening the download page instead.")
        webbrowser.open(OLLAMA_PAGE)
    return find_ollama() is not None


# ------------------------------------------------------------------ twins and commands
def list_twins() -> list[str]:
    root = data_dir()
    names = []
    if root.exists():
        for meta in sorted(root.glob("*/meta.json")):
            try:
                names.append(json.loads(meta.read_text(encoding="utf-8")).get("name") or meta.parent.name)
            except (OSError, ValueError):
                names.append(meta.parent.name)
    return names


def _run_py() -> list[str]:
    return [str(runtime_python()), "-u", str(app_dir() / "run.py")]


def inspect_command(name: str) -> list[str]:
    return _run_py() + ["inspect", "--name", name]


def ingest_command(name: str, folder: str, speaker: str, face_photo: str = "",
                   include_media: bool = True, media_minutes: int = 90,
                   about_file: str = "", listener: str = "", language: str = "", llm_model: str = "") -> list[str]:
    cmd = _run_py() + ["ingest", "--name", name, "--path", folder, "--speaker", speaker,
                       "--i-have-permission", "--media-minutes", str(media_minutes)]
    if face_photo:
        cmd += ["--face-photo", face_photo]
    if not include_media:
        cmd += ["--no-media"]
    if about_file:
        cmd += ["--about-file", about_file]
    if listener:
        cmd += ["--listener", listener]
    if language:
        cmd += ["--language", language]
    if llm_model:
        cmd += ["--llm-model", llm_model]
    return cmd


LAYOUTS = {
    "Window (for testing)": ("single", True),
    "Fullscreen / projector": ("single", False),
    "Hologram pyramid": ("pyramid", False),
    "Text only (no picture)": ("none", False),
}


def soundtest_command() -> list[str]:
    return _run_py() + ["soundtest"]


def chat_command(name: str, layout_label: str, mirror: bool = False, screen: int = 0,
                 microphone: bool = False, voice: str = "auto", speak: bool = True, photo: str = "",
                 listener: str = "", language: str = "", show_memories: bool = False) -> list[str]:
    display, windowed = LAYOUTS.get(layout_label, ("single", True))
    cmd = _run_py() + ["chat", "--name", name, "--display", display, "--tts", voice]
    if display != "none":
        cmd += ["--screen", str(screen)]
        if windowed:
            cmd += ["--windowed"]
        if mirror:
            cmd += ["--mirror"]
    if photo:
        cmd += ["--photo", photo]
    if listener:
        cmd += ["--listener", listener]
    if language:
        cmd += ["--language", language]
    if show_memories:
        cmd += ["--show-memories"]
    if microphone:
        cmd += ["--voice"]
    if not speak:
        cmd += ["--no-speech"]
    return cmd


def open_path(path: Path) -> None:
    """Show a folder in the file manager."""
    path.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])
