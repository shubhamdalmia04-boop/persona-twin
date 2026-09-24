#!/usr/bin/env python3
"""One-time setup. Needs internet once; afterwards everything runs offline.

    python install.py

What it does:
  1. creates a private virtual environment (.venv)
  2. installs the Python dependencies
  3. downloads the speech-to-text and memory-search models
  4. installs the animated-face tools (MediaPipe) and downloads its small face model
  5. optionally installs voice cloning (Coqui XTTS-v2, non-commercial licence)
  6. checks your hardware and picks model sizes that stay usable on it (data/profile.json)
  7. pulls the local language model through Ollama (if Ollama is installed)
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
sys.path.insert(0, str(ROOT))

from twin.config import DATA_DIR, FACE_MODEL_PATH, FACE_MODEL_URL, Settings  # noqa: E402  (stdlib only)
from twin.hardware import detect, find_ollama  # noqa: E402
from twin.ollama import ensure_ollama_server, pull_model_api  # noqa: E402


os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # Windows cache warning is harmless noise
QUIET = "--no-warn-script-location"  # pip's "script X is not on PATH" warnings are irrelevant here


def pip_install(py: str, *args: str) -> int:
    return subprocess.call([py, "-m", "pip", "install", QUIET, *args])


def constraints_from_freeze(freeze_text: str) -> str:
    """Turn `pip freeze` output into a constraints file that keeps every installed package as is.

    Used when adding voice cloning: if its packages cannot work with the versions already
    installed, pip fails cleanly instead of silently changing (and breaking) the base install.
    """
    skip = {"pip", "setuptools", "wheel"}
    keep = []
    for line in freeze_text.splitlines():
        line = line.strip()
        if "==" not in line or line.startswith(("-", "#")) or " @ " in line:
            continue
        if line.split("==")[0].lower() not in skip:
            keep.append(line)
    return "\n".join(keep) + "\n"


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def step(msg: str) -> None:
    print(f"\n==> {msg}", flush=True)


XTTS_TERMS = (
    "Voice cloning uses Coqui XTTS-v2 (about 2 GB download).\n"
    "Its model licence (Coqui Public Model License) allows NON-COMMERCIAL use only.\n"
    "Personal use is fine; selling or offering it as a paid product is not.\n"
)


def install_voice_cloning(py: str, accepted_flag: bool) -> None:
    step("Voice cloning (XTTS-v2)")
    if sys.version_info >= (3, 13):
        print("Warning: coqui-tts supports Python 3.10-3.12. On newer Python the install may fail.")
    print(XTTS_TERMS)
    accepted = accepted_flag or input("Type YES to accept these terms and install: ").strip().lower() in {"yes", "y"}
    if not accepted:
        print("Skipped voice cloning. Replies will use a generic voice. Re-run with --voice-clone to add it.")
        return
    frozen = subprocess.run([py, "-m", "pip", "freeze", "--all"], capture_output=True, text=True).stdout
    constraints = DATA_DIR / "constraints.txt"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    constraints.write_text(constraints_from_freeze(frozen), encoding="utf-8")
    if pip_install(py, "-r", str(ROOT / "requirements-voice.txt"), "-c", str(constraints)) != 0:
        print(
            "\nVOICE CLONING COULD NOT BE INSTALLED. Its packages may not match the versions already installed,\n"
            "may need the Microsoft C++ Build Tools, or a download failed. Nothing else was changed and\n"
            "everything else works: replies will use a generic voice."
        )
        return
    check = subprocess.run([py, "-c", "import TTS"], capture_output=True, text=True)
    if check.returncode != 0:
        tail = "\n".join(check.stderr.strip().splitlines()[-6:])
        print(
            "\nVOICE CLONING INSTALL INCOMPLETE: the voice package installed but would not load:\n"
            f"{tail}\n"
            "Everything else works: replies will use a generic voice."
        )
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / ".xtts_terms_accepted").write_text("accepted non-commercial terms\n", encoding="utf-8")
    env = dict(os.environ, COQUI_TOS_AGREED="1")
    result = subprocess.call(
        [py, "-c", "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')"], env=env
    )
    if result != 0:
        print("Could not download the voice model now. It will be retried on first use (needs internet).")


def write_profile(reprofile: bool) -> None:
    path = DATA_DIR / "profile.json"
    if path.exists() and not reprofile:
        return
    info = detect()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=1), encoding="utf-8")
    ram = f"{info['ram_gb']} GB RAM" if info["ram_gb"] else "unknown RAM"
    gpu = "NVIDIA GPU" if info["nvidia_gpu"] else ("Apple Silicon" if info["apple_silicon"] else "no dedicated GPU")
    print(f"Detected {ram}, {gpu}: {info['tier']}. Using language model {info['llm_model']} "
          f"and speech model {info['whisper_model']}.\n"
          f"(Change them by editing {path} or setting TWIN_LLM_MODEL / TWIN_WHISPER_MODEL.)")


def install_face(py: str) -> None:
    step("Animated face (MediaPipe)")
    if pip_install(py, "-r", str(ROOT / "requirements-face.txt")) != 0:
        print("Could not install MediaPipe. The display will use a simpler photo avatar instead.")
        return
    if FACE_MODEL_PATH.exists():
        return
    try:
        FACE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL_PATH)
        print(f"Downloaded face model to {FACE_MODEL_PATH}")
    except Exception as exc:
        FACE_MODEL_PATH.unlink(missing_ok=True)
        print(f"Could not download the face model ({exc}). Download it manually from\n  {FACE_MODEL_URL}\n"
              f"and save it as {FACE_MODEL_PATH}. Until then the display uses a simpler photo avatar.")


def install_everything(args, settings) -> None:
    if args.here:
        py = sys.executable
    else:
        step("Creating virtual environment (.venv)")
        if not venv_python().exists():
            subprocess.check_call([sys.executable, "-m", "venv", str(VENV)])
        py = str(venv_python())


    step("Installing Python packages (this can take several minutes)")
    if pip_install(py, "--upgrade", "pip") != 0:
        print("Could not upgrade pip; continuing with the current version.")
    if pip_install(py, "-r", str(ROOT / "requirements.txt")) != 0:
        raise SystemExit("Installing the required packages failed. Check the internet connection and try again.")

    if platform.system() == "Linux":
        print(
            "\nLinux note: if audio or voice fails, install system libraries with\n"
            "  sudo apt install espeak-ng libportaudio2 libsndfile1"
        )

    if not args.no_face:
        install_face(py)

    if not args.skip_models:
        step(f"Downloading memory-search model ({settings.embed_model})")
        subprocess.check_call(
            [py, "-c", f"from sentence_transformers import SentenceTransformer as S; S({settings.embed_model!r})"]
        )
        step(f"Downloading speech-to-text model (whisper {settings.whisper_model})")
        subprocess.check_call(
            [py, "-c", f"from faster_whisper import WhisperModel as W; W({settings.whisper_model!r})"]
        )

    if not args.no_voice_clone:
        wants = args.voice_clone or (
            sys.stdin.isatty()
            and input("\nInstall voice cloning so replies sound like the person? [y/N] ").strip().lower() in {"y", "yes"}
        )
        if wants:
            install_voice_cloning(py, args.accept_xtts_terms)



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-models", action="store_true", help="skip model downloads")
    parser.add_argument("--voice-clone", action="store_true", help="install voice cloning without asking")
    parser.add_argument("--no-voice-clone", action="store_true", help="do not install voice cloning")
    parser.add_argument("--accept-xtts-terms", action="store_true",
                        help="accept the non-commercial XTTS-v2 licence non-interactively")
    parser.add_argument("--here", action="store_true",
                        help="install into the Python running this script instead of creating .venv")
    parser.add_argument("--only-llm", action="store_true", help="only download the language model")
    parser.add_argument("--no-face", action="store_true", help="skip the animated-face tools")
    parser.add_argument("--reprofile", action="store_true", help="re-detect hardware and reset model sizes")
    args = parser.parse_args()
    step("Checking your hardware")
    write_profile(args.reprofile)
    settings = Settings()  # picks up the profile just written

    if sys.version_info < (3, 10):
        raise SystemExit("Python 3.10 or newer is required.")

    if not args.only_llm:
        install_everything(args, settings)

    step(f"Language model ({settings.llm_model}) via Ollama")
    ollama = find_ollama()
    if ollama is None and args.here:
        print("Ollama is not installed yet (the app will offer to install it).")
    elif ollama is None:
        print(
            "Ollama is not installed. Get it from https://ollama.com (free), start it, then run\n"
            f"  ollama pull {settings.llm_model}\n"
            "or simply run this installer again."
        )
    else:
        ok = ensure_ollama_server(ollama) and pull_model_api(settings.llm_model)
        result = 0 if ok else subprocess.call([ollama, "pull", settings.llm_model])
        if result != 0:
            print(
                "Could not pull the model. Make sure the Ollama app (or `ollama serve`) is running, then run\n"
                f"  ollama pull {settings.llm_model}"
            )

    step("Setup finished")
    if args.here:
        print("From now on no internet is needed.")
        return
    print(
        "From now on no internet is needed.\n\n"
        "Next steps:\n"
        "  python run.py ingest --name Asha --path ./asha_material --speaker \"Asha Verma\"\n"
        "  python run.py chat --name Asha --display single\n"
    )


if __name__ == "__main__":
    main()
