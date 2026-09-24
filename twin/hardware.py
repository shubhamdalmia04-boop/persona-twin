"""Detects the machine and recommends model sizes that stay usable on it.

Most laptops have 8-16 GB of RAM and no dedicated GPU, so the default is a small local
language model and a small speech-recognition model. A gaming/workstation GPU or a large-RAM
Apple Silicon Mac gets the bigger models. install.py stores the result in data/profile.json;
edit that file (or set TWIN_LLM_MODEL / TWIN_WHISPER_MODEL) to override.
"""
from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess


def parse_meminfo(text: str) -> float | None:
    """Total RAM in GB from the contents of /proc/meminfo."""
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            return round(int(line.split()[1]) / 1024 / 1024, 1)
    return None


def total_ram_gb() -> float | None:
    try:
        system = platform.system()
        if system == "Linux":
            with open("/proc/meminfo", encoding="utf-8") as f:
                return parse_meminfo(f.read())
        if system == "Darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return round(int(out.strip()) / 1024**3, 1)
        if system == "Windows":
            class MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            status = MemStatus()
            status.dwLength = ctypes.sizeof(MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))  # type: ignore[attr-defined]
            return round(status.ullTotalPhys / 1024**3, 1)
    except Exception:
        pass
    return None


def find_ollama() -> str | None:
    """Path to the ollama executable, including the default install spots that are often not on PATH."""
    found = shutil.which("ollama")
    if found:
        return found
    candidates = []
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA", "")
        candidates += [os.path.join(local, "Programs", "Ollama", "ollama.exe"),
                       r"C:\Program Files\Ollama\ollama.exe"]
    else:
        candidates += ["/usr/local/bin/ollama", "/opt/homebrew/bin/ollama", "/usr/bin/ollama"]
    return next((c for c in candidates if c and os.path.exists(c)), None)


def has_nvidia_gpu() -> bool:
    return shutil.which("nvidia-smi") is not None


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def recommend(ram_gb: float | None, nvidia: bool, apple_silicon: bool) -> dict:
    ram = ram_gb or 0
    strong = nvidia or (apple_silicon and ram >= 16) or ram >= 32
    if ram_gb is not None and ram < 6:
        llm, whisper, tier = "qwen2.5:1.5b", "tiny", "low-memory laptop"
    elif strong:
        llm, whisper, tier = "qwen2.5:7b", "small", "strong machine"
    else:
        llm, whisper, tier = "qwen2.5:3b", "base", "typical laptop"
    return {"llm_model": llm, "whisper_model": whisper, "tier": tier}


# Models known to officially support a given language well, in case the general-purpose default
# (Qwen 2.5) is weak in it. Qwen2.5's documented strong languages are mostly European/East Asian;
# Llama 3.1 explicitly lists Hindi among its 8 supported languages, for example.
_LANGUAGE_MODELS: dict[str, tuple[str, str, float]] = {
    # code: (ollama model tag, why it's suggested, approximate download size in GB)
    "hi": ("llama3.1:8b", "Llama 3.1 8B officially supports Hindi; the default model's Hindi is hit-or-miss.", 4.7),
}


def recommend_llm_for_language(language: str, ram_gb: float | None, nvidia: bool, apple_silicon: bool) -> dict | None:
    """A better language-model suggestion for the given language code, or None if the default is fine.

    Returns {"model": ..., "reason": ..., "size_gb": ..., "fits": bool} - "fits" is a rough RAM
    check so the caller can decide whether to mention it at all on very small machines.
    """
    choice = _LANGUAGE_MODELS.get(language)
    if not choice:
        return None
    model, reason, size_gb = choice
    strong = nvidia or (apple_silicon and (ram_gb or 0) >= 16) or (ram_gb or 0) >= 16
    fits = strong or (ram_gb or 0) >= 8
    return {"model": model, "reason": reason, "size_gb": size_gb, "fits": fits}


def detect() -> dict:
    ram, nvidia, apple = total_ram_gb(), has_nvidia_gpu(), is_apple_silicon()
    return {"ram_gb": ram, "nvidia_gpu": nvidia, "apple_silicon": apple, **recommend(ram, nvidia, apple)}
