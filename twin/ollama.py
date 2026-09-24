"""Talking to a local Ollama server over its HTTP API. Standard library only, so both the
launcher (a separate, lightweight process) and install.py (run inside the runtime Python) can
use it without extra dependencies.
"""
from __future__ import annotations

import json
import subprocess
import time
import urllib.request
from typing import Callable

OLLAMA_URL = "http://localhost:11434"
Log = Callable[[str], None]


def ollama_up(api: str | None = None) -> bool:
    api = api or OLLAMA_URL
    try:
        with urllib.request.urlopen(f"{api}/api/tags", timeout=3):
            return True
    except Exception:
        return False


def ensure_ollama_server(exe: str, api: str | None = None, wait: int = 25) -> bool:
    """Make sure the Ollama server answers, starting `ollama serve` if it is not running."""
    api = api or OLLAMA_URL
    if ollama_up(api):
        return True
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.Popen([exe, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     stdin=subprocess.DEVNULL, creationflags=flags)
    for _ in range(wait):
        if ollama_up(api):
            return True
        time.sleep(1)
    return False


def pull_model_api(model: str, api: str | None = None, log: Log = print) -> bool:
    """Download a model through Ollama's HTTP API and print a tidy progress line every 5%.

    (The `ollama pull` command draws an animated bar with terminal codes that look like garbage
    when its output is captured by a window.)
    """
    api = api or OLLAMA_URL
    body = json.dumps({"model": model, "name": model, "stream": True}).encode()
    request = urllib.request.Request(f"{api}/api/pull", data=body, headers={"Content-Type": "application/json"})
    try:
        response = urllib.request.urlopen(request, timeout=600)
    except Exception as exc:
        log(f"Could not reach Ollama ({exc}).")
        return False
    last: dict[str, int] = {}
    status_seen = ""
    with response:
        for raw in response:
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            if msg.get("error"):
                log(f"Ollama reported an error: {msg['error']}")
                return False
            total, done, digest = msg.get("total"), msg.get("completed"), msg.get("digest", "")
            if total and done is not None:
                pct = int(done * 100 / total)
                if pct >= last.get(digest, -5) + 5 or pct == 100:
                    last[digest] = pct
                    log(f"  downloading {model}: {pct}% ({done / 1e9:.2f} of {total / 1e9:.2f} GB)")
            elif msg.get("status") and msg["status"] != status_seen:
                status_seen = msg["status"]
                log(f"  {status_seen}")
            if msg.get("status") == "success":
                return True
    return False


def pull_model(exe: str, model: str, api: str | None = None, log: Log = print) -> bool:
    """Start the server if needed, then download a model. What both install.py and the
    launcher's "Download now" button call."""
    api = api or OLLAMA_URL
    if not ensure_ollama_server(exe, api):
        log("Could not start Ollama.")
        return False
    return pull_model_api(model, api, log)
