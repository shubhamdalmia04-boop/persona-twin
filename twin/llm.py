"""Talks to a local Ollama server. Nothing leaves the machine."""
from __future__ import annotations

import json
from typing import Iterator

import requests


class OllamaNotReady(RuntimeError):
    pass


def check_ollama(url: str, model: str) -> None:
    try:
        r = requests.get(f"{url}/api/tags", timeout=3)
        r.raise_for_status()
    except requests.RequestException as exc:
        raise OllamaNotReady(
            f"Cannot reach Ollama at {url}. Install it from https://ollama.com and make sure "
            "it is running (the desktop app, or `ollama serve`)."
        ) from exc
    installed = {m.get("name", "") for m in r.json().get("models", [])}
    wanted = model if ":" in model else f"{model}:latest"
    if wanted not in installed:
        raise OllamaNotReady(
            f"Model '{model}' is not downloaded yet. Run: ollama pull {model}"
        )


def chat_stream(
    messages: list[dict], model: str, url: str, temperature: float = 0.7
) -> Iterator[str]:
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "keep_alive": "30m",  # avoid reloading the model (slow on a laptop) between messages
        "options": {"temperature": temperature, "num_predict": 200},
    }
    with requests.post(f"{url}/api/chat", json=payload, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            piece = data.get("message", {}).get("content", "")
            if piece:
                yield piece
            if data.get("done"):
                break
