"""Turns sentences into speech without gaps where possible.

Two worker threads: one synthesizes audio for sentence N+1 while the other plays sentence N.
On a laptop without a GPU, synthesis (especially voice cloning) is the slowest step, so this
overlap hides a good part of it. It also drives the on-screen subtitle and mouth animation,
which are updated at the moment each sentence starts playing.
"""
from __future__ import annotations

import queue
import threading
from typing import Callable

from .audio import envelope, load_audio, play_blocking, viseme_track
from .state import ENVELOPE_FPS, DisplayState


class SpeechPipeline:
    def __init__(
        self,
        tts,
        state: DisplayState,
        speaker: str = "",
        fallback=None,
        loader: Callable = load_audio,
        player: Callable = play_blocking,
        out: Callable[[str], None] = print,
    ):
        self.tts, self.state, self.speaker = tts, state, speaker
        self.fallback = fallback  # used if the main voice fails (e.g. cloned voice -> generic voice)
        self._loader, self._player, self._out = loader, player, out
        self._synth_q: queue.Queue = queue.Queue()
        self._play_q: queue.Queue = queue.Queue(maxsize=2)
        for target in (self._synth_loop, self._play_loop):
            threading.Thread(target=target, daemon=True).start()

    # ------------------------------------------------------------------ public
    def submit(self, text: str) -> None:
        self._synth_q.put(text)

    def wait(self) -> None:
        """Block until everything submitted has been spoken (or shown, in text-only mode)."""
        self._synth_q.join()
        self._play_q.join()

    def close(self) -> None:
        self._synth_q.put(None)
        self._play_q.put(None)

    # ------------------------------------------------------------------ workers
    def _synth_loop(self) -> None:
        while True:
            text = self._synth_q.get()
            try:
                if text is None:
                    return
                item = (text, None, 0, None, None)  # text-only unless synthesis succeeds
                while self.tts is not None and not self.state.quit:
                    try:
                        samples, sr = self._loader(self.tts.synthesize(text))
                        item = (
                            text, samples, sr,
                            envelope(samples, sr, ENVELOPE_FPS),
                            viseme_track(samples, sr, ENVELOPE_FPS),
                        )
                        break
                    except Exception as exc:
                        if self.fallback is not None:
                            self._out(f"[main voice failed ({exc}); switching to the generic voice]")
                            self.tts, self.fallback = self.fallback, None
                            continue
                        self._out(f"[voice unavailable: {exc}. Continuing with text only.]")
                        self.tts = None
                self._play_q.put(item)
            finally:
                self._synth_q.task_done()

    def _play_loop(self) -> None:
        while True:
            item = self._play_q.get()
            try:
                if item is None:
                    return
                text, samples, sr, env, visemes = item
                self._out(f"{self.speaker}: {text}" if self.speaker else text)
                self.state.subtitle = text
                if samples is not None and not self.state.quit:
                    self.state.start_speech(env, visemes)
                    try:
                        self._player(samples, sr)
                    finally:
                        self.state.end_speech()
            finally:
                self._play_q.task_done()
