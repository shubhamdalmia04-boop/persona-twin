#!/usr/bin/env python3
"""persona-twin command line.

  python run.py ingest --name Asha --path ./asha_material --speaker "Asha Verma"
  python run.py chat   --name Asha --display single
  python run.py list
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _use_project_venv() -> None:
    """If install.py created .venv, transparently run inside it."""
    if os.environ.get("TWIN_NO_VENV"):
        return
    venv = ROOT / ".venv"
    py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if py.exists() and Path(sys.prefix).resolve() != venv.resolve():
        if os.name == "nt":
            sys.exit(subprocess.call([str(py), *sys.argv]))
        os.execv(str(py), [str(py), *sys.argv])


_use_project_venv()

import json
import os

from twin.config import DATA_DIR, Settings, twin_dir  # noqa: E402
from twin.ingest import IMAGE_EXT, confirm_consent, ingest_folder  # noqa: E402


def cmd_ingest(args: argparse.Namespace) -> None:
    if not confirm_consent(args.i_have_permission):
        raise SystemExit("Cancelled. Nothing was processed.")
    print(f"Building '{args.name}' from {args.path}")
    meta = ingest_folder(
        args.name,
        Path(args.path),
        args.speaker,
        Settings(),
        include_media=not args.no_media,
        media_minutes=args.media_minutes,
        face_photo=Path(args.face_photo) if args.face_photo else None,
        about_text=Path(args.about_file).read_text(encoding="utf-8", errors="ignore") if args.about_file else "",
        listener=args.listener or "",
        language=args.language or "",
        llm_model=args.llm_model or "",
    )
    print(
        f"\nDone: {meta['memories']} memories, {meta['style_samples']} style samples, "
        f"{meta['photos']} photos seen, {meta['voice_clips']} voice reference clip(s)."
    )
    if not meta["voice_clips"]:
        print("No clean voice clips found: replies will use a generic voice unless you add some with\n"
              f"  python run.py voice --name \"{args.name}\" --add <recording>")
    print(f"Next: python run.py chat --name \"{args.name}\"")


def cmd_voice(args: argparse.Namespace) -> None:
    from twin.voice import VoiceBank

    bank = VoiceBank(twin_dir(args.name))
    if args.clear:
        bank.clear()
        print("Voice references removed.")
        return
    if args.add:
        for f in args.add:
            path = Path(f)
            if not path.is_file():
                raise SystemExit(f"File not found: {path}")
            print(f"{path.name}: {bank.add_file(path)} clean clip(s) found")
        bank.commit()
    refs = bank.paths()
    print(f"{len(refs)} reference clip(s) for '{args.name}':")
    for e in bank.entries:
        print(f"  {e['file']}  score {e['score']}  from {e['source']}")
    if args.say:
        from twin.audio import load_audio, play_blocking
        from twin.tts import make_tts

        tts = make_tts(args.tts, args.name, Settings())
        print("Synthesizing (the first run loads the model and can take a minute)...")
        samples, sr = load_audio(tts.synthesize(args.say))
        play_blocking(samples, sr)


def cmd_soundtest(_: argparse.Namespace) -> None:
    """Find out why nothing is audible: speaker, the system voice, or the voice in a worker thread."""
    import threading

    import numpy as np

    try:
        import sounddevice as sd
    except Exception as exc:
        raise SystemExit(f"Sound output is not available: {exc}")
    outputs = [(i, d["name"]) for i, d in enumerate(sd.query_devices()) if d["max_output_channels"] > 0]
    print("Output devices: " + "; ".join(f"{i}: {n}" for i, n in outputs))
    print(f"Default output: {sd.default.device[1]}")

    print("1) Playing a short beep. You should hear it now...")
    t = np.arange(int(44100 * 0.6)) / 44100
    sd.play((0.3 * np.sin(2 * np.pi * 440 * t)).astype("float32"), 44100)
    sd.wait()

    print("2) Making speech with the system voice (in a worker thread, as the app does)...")
    from twin.audio import load_audio, play_blocking
    from twin.tts import SystemTTS

    result: dict = {}

    def work() -> None:
        try:
            result["wav"] = SystemTTS().synthesize("Testing. One, two, three.")
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"

    worker = threading.Thread(target=work)
    worker.start()
    worker.join()
    if "error" in result:
        raise SystemExit(f"   The system voice FAILED: {result['error']}")
    samples, rate = load_audio(result["wav"])
    seconds = len(samples) / rate
    print(f"   created {seconds:.1f} seconds of speech")
    if seconds < 0.3:
        raise SystemExit("   The voice produced almost no audio: the Windows speech engine is not working.")
    play_blocking(samples, rate)
    print("Done. Heard the beep but not the voice: the Windows voice is the problem. "
          "Heard neither: check the output device and volume.")


def cmd_languages(_: argparse.Namespace) -> None:
    from twin.languages import LANGUAGES

    print("Language codes (use with --language):")
    seen = set()
    for name, code in LANGUAGES.items():
        if code and code not in seen:
            seen.add(code)
            print(f"  {code:<6} {name}")


def cmd_inspect(args: argparse.Namespace) -> None:
    from twin.report import describe_twin

    print("\n".join(describe_twin(twin_dir(args.name))))


def cmd_list(_: argparse.Namespace) -> None:
    if not DATA_DIR.exists() or not any(DATA_DIR.iterdir()):
        print("No twins yet. Create one with: python run.py ingest ...")
        return
    for folder in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        print(folder.name)


def _saved_persona_value(name: str, key: str) -> str:
    persona_path = twin_dir(name) / "persona.json"
    if persona_path.exists():
        try:
            return json.loads(persona_path.read_text(encoding="utf-8")).get(key, "")
        except (OSError, ValueError):
            pass
    return ""


def _saved_language(name: str) -> str:
    return _saved_persona_value(name, "language")


def _photos(name: str, override: str | None = None) -> list[Path]:
    """Photos to build the avatar from. --photo wins; otherwise everything ingested."""
    if override:
        path = Path(override)
        if not path.is_file():
            raise SystemExit(f"Photo not found: {path}")
        return [path]
    tdir = twin_dir(name)
    avatar = sorted(tdir.glob("avatar.*"))
    if avatar:  # the photo chosen when the twin was built wins
        return avatar[:1]
    folder = tdir / "photos"
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT)


def cmd_chat(args: argparse.Namespace) -> None:
    from twin.chat import Conversation
    from twin.llm import OllamaNotReady, check_ollama
    from twin.state import DisplayState

    settings = Settings()
    language = args.language or _saved_language(args.name)
    if language:
        if not os.environ.get("TWIN_TTS_LANGUAGE"):
            settings.tts_language = language
        if not os.environ.get("TWIN_WHISPER_LANGUAGE"):
            settings.whisper_language = language
    llm_model = args.llm_model or _saved_persona_value(args.name, "llm_model")
    if llm_model and not os.environ.get("TWIN_LLM_MODEL"):
        settings.llm_model = llm_model
    elif language and not os.environ.get("TWIN_LLM_MODEL") and not llm_model:
        from twin.hardware import recommend_llm_for_language, total_ram_gb, has_nvidia_gpu, is_apple_silicon

        tip = recommend_llm_for_language(language, total_ram_gb(), has_nvidia_gpu(), is_apple_silicon())
        if tip:
            print(f"[tip: {tip['reason']} If replies in this language feel off, try:\n"
                  f"  ollama pull {tip['model']}\n"
                  f"  python run.py chat --name \"{args.name}\" --llm-model {tip['model']}]", flush=True)
    try:
        check_ollama(settings.ollama_url, settings.llm_model)
    except OllamaNotReady as exc:
        raise SystemExit(str(exc))

    state = DisplayState()
    tts = None
    if not args.no_speech:
        from twin.tts import make_tts

        tts = make_tts(args.tts, args.name, settings)
    fallback = None
    if tts is not None:
        from twin.tts import ClonedTTS, SystemTTS

        fallback = SystemTTS() if isinstance(tts, ClonedTTS) else None
    conversation = Conversation(args.name, settings, tts, state, fallback_tts=fallback,
                                listener=args.listener, language=language, show_memories=args.show_memories)

    transcriber = None
    if args.voice:
        from twin.stt import Transcriber

        transcriber = Transcriber(settings.whisper_model, settings.whisper_language)

    def loop() -> None:
        print("Type a message and press Enter. /quit to leave." if not args.voice else "Listening. Ctrl+C to leave.")
        try:
            while not state.quit:
                if transcriber is not None:
                    from twin.stt import record_until_silence

                    state.status = "listening"
                    audio = record_until_silence()
                    if audio is None:
                        continue
                    state.status = "thinking"
                    text = transcriber.audio(audio)
                    if text:
                        print(f"You: {text}")
                else:
                    try:
                        text = input("You: ").strip()
                    except EOFError:
                        break
                if text in ("/quit", "/exit"):
                    break
                if text:
                    print("[thinking... the first reply can take a minute on a laptop]", flush=True)
                    conversation.respond(text)
        except KeyboardInterrupt:
            pass
        finally:
            state.quit = True

    if args.display == "none":
        loop()
        return

    from twin.display import run_display

    threading.Thread(target=loop, daemon=True).start()
    run_display(
        state,
        args.name,
        _photos(args.name, args.photo),
        layout=args.display,
        mirror=args.mirror,
        windowed=args.windowed,
        screen_index=args.screen,
        avatar_mode=args.avatar,
        face_size=settings.face_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="persona-twin")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="build a twin from a folder of material")
    p.add_argument("--name", required=True, help="name of the twin (also its folder name)")
    p.add_argument("--path", required=True, help="folder with chats, recordings, photos, about.txt")
    p.add_argument("--speaker", required=True, help="the person's name exactly as it appears in the chats")
    p.add_argument("--i-have-permission", action="store_true", help="skip the interactive consent prompt")
    p.add_argument("--no-media", action="store_true", help="ignore audio and video files")
    p.add_argument("--media-minutes", type=int, default=None,
                   help="max minutes of audio/video to transcribe (default 90, short voice notes first)")
    p.add_argument("--face-photo", help="a clear front-facing photo of the person, used for the animated face")
    p.add_argument("--about-file", help="a text file of notes about them: relationship, nicknames, sayings, stories")
    p.add_argument("--listener", help="who will talk to the twin, e.g. \"his daughter Meera\"")
    p.add_argument("--language", help="the person's language, e.g. \"hi\" for Hindi (used for transcription, "
                   "the cloned voice, and replies; run 'python run.py languages' to list codes)")
    p.add_argument("--llm-model", help="use a specific Ollama language model for this twin instead of the default "
                   "(must already be pulled, e.g. with 'ollama pull llama3.1:8b')")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("chat", help="talk to a twin")
    p.add_argument("--name", required=True)
    p.add_argument("--display", choices=["none", "single", "pyramid"], default="none",
                   help="none = terminal only; single = projector/screen; pyramid = 4-view hologram pyramid")
    p.add_argument("--mirror", action="store_true", help="flip horizontally (rear projection / pyramid calibration)")
    p.add_argument("--windowed", action="store_true", help="show in a window instead of fullscreen (for testing)")
    p.add_argument("--screen", type=int, default=0, help="which monitor to use, 0 = primary, 1 = projector, ...")
    p.add_argument("--avatar", choices=["auto", "puppet", "photo"], default="auto",
                   help="auto = animated talking face if possible, else the photo with a glow")
    p.add_argument("--photo", help="use this photo for the face instead of picking from the ingested ones")
    p.add_argument("--voice", action="store_true", help="speak to the twin with the microphone")
    p.add_argument("--no-speech", action="store_true", help="text replies only, no spoken audio")
    p.add_argument("--listener", help="who is talking to the twin, e.g. \"his daughter Meera\" (overrides the saved one)")
    p.add_argument("--language", help="override the twin's saved language for this session, e.g. \"hi\"")
    p.add_argument("--llm-model", help="use this Ollama language model instead of the default or saved one "
                   "(must already be pulled)")
    p.add_argument("--show-memories", action="store_true", help="print what it remembered for each message (debugging)")
    p.add_argument("--tts", choices=["auto", "clone", "system"], default="auto",
                   help="auto = the person's cloned voice if available, else a generic voice")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("voice", help="manage and test the cloned voice")
    p.add_argument("--name", required=True)
    p.add_argument("--add", nargs="+", metavar="FILE", help="add reference clips from audio/video files")
    p.add_argument("--clear", action="store_true", help="remove all stored reference clips")
    p.add_argument("--say", metavar="TEXT", help="speak this text to test the voice")
    p.add_argument("--tts", choices=["auto", "clone", "system"], default="clone")
    p.set_defaults(func=cmd_voice)

    p = sub.add_parser("languages", help="list supported language codes")
    p.set_defaults(func=cmd_languages)

    p = sub.add_parser("inspect", help="show what a twin has learned and what is missing")
    p.add_argument("--name", required=True)
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("soundtest", help="check that speakers and the voice work")
    p.set_defaults(func=cmd_soundtest)

    p = sub.add_parser("list", help="list twins on this machine")
    p.set_defaults(func=cmd_list)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
