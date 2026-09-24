#!/usr/bin/env python3
"""Build the single-file Windows app.

Run this ON WINDOWS (PyInstaller builds for the system it runs on):

    py -3.12 build_exe.py

Result: dist/PersonaTwin-windows.zip containing PersonaTwin.exe and README-FIRST.txt.
Only this build machine needs Python. The people who receive the zip do not.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
PAYLOAD = BUILD / "app_payload"
DIST = ROOT / "dist"
FILES = ["run.py", "install.py", "requirements.txt", "requirements-face.txt", "requirements-voice.txt"]


def make_payload(dest: Path = PAYLOAD, root: Path = ROOT) -> Path:
    """The program files that get embedded in the exe (no tests, data or caches)."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for name in FILES:
        shutil.copy(root / name, dest / name)
    shutil.copytree(root / "twin", dest / "twin", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return dest


def main() -> None:
    if os.name != "nt":
        print("Warning: this builds for the system it runs on. For a Windows .exe, run it on Windows.")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pyinstaller"])
    make_payload()
    subprocess.check_call([
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onefile", "--noconsole",
        "--name", "PersonaTwin", "--paths", str(ROOT),
        "--add-data", f"{PAYLOAD}{os.pathsep}app_payload",
        "--distpath", str(DIST / "exe"), "--workpath", str(BUILD / "pyi"), "--specpath", str(BUILD),
        str(ROOT / "launcher" / "main.py"),
    ])
    exe = DIST / "exe" / ("PersonaTwin.exe" if os.name == "nt" else "PersonaTwin")
    archive = DIST / "PersonaTwin-windows.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(exe, exe.name)
        z.write(ROOT / "README-FIRST.txt", "README-FIRST.txt")
    print(f"\nDone: {archive}")


if __name__ == "__main__":
    main()
