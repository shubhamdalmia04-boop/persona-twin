"""Speech pipeline, mouth-shape analysis, hardware profile and settings precedence."""
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

import twin.config as cfg
from twin.audio import viseme_track
from twin.config import Settings
from twin.hardware import parse_meminfo, recommend
from twin.speech import SpeechPipeline
from twin.state import DisplayState

SR = 22050


def tone(freq, seconds=1.0, amp=0.5):
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class Visemes(unittest.TestCase):
    def test_shapes_follow_spectrum(self):
        dark = viseme_track(tone(300), SR)      # "oo"-like
        bright = viseme_track(tone(2800), SR)   # "ee"-like
        mid = viseme_track(tone(1200), SR)
        self.assertEqual(dark.shape[1], 2)
        self.assertLess(dark[10, 1], -0.5)
        self.assertGreater(bright[10, 1], 0.5)
        self.assertLess(abs(mid[10, 1]), 0.3)
        self.assertGreater(mid[10, 0], dark[10, 0])   # open vowel opens wider than a closed one

    def test_silence_and_pauses_close_the_mouth(self):
        self.assertEqual(float(np.abs(viseme_track(np.zeros(SR, np.float32), SR)).max()), 0.0)
        speech_then_pause = np.concatenate([tone(1000), np.zeros(SR, np.float32)])
        track = viseme_track(speech_then_pause, SR)
        self.assertGreater(track[10, 0], 0.5)
        self.assertLess(track[-5, 0], 0.05)
        self.assertEqual(len(track), 60)

    def test_hissy_sounds_stay_nearly_shut(self):
        loud = np.concatenate([tone(1200, 0.5), tone(6000, 0.5)])
        track = viseme_track(loud, SR)
        self.assertLess(track[-8, 0], 0.5 * track[7, 0])


class StateMouth(unittest.TestCase):
    def test_mouth_reads_the_track(self):
        s = DisplayState()
        self.assertEqual(s.mouth(), (0.0, 0.0))
        track = np.tile(np.array([[0.7, -0.4]], dtype=np.float32), (300, 1))
        s.start_speech(np.ones(300, np.float32), track)
        self.assertEqual(s.mouth(), (np.float32(0.7), np.float32(-0.4)))
        s.end_speech()
        self.assertEqual(s.mouth(), (0.0, 0.0))


class FakeTTS:
    def __init__(self, fail_on=None, delay=0.05):
        self.calls, self.fail_on, self.delay = [], fail_on, delay

    def synthesize(self, text):
        self.calls.append((text, time.monotonic()))
        time.sleep(self.delay)
        if self.fail_on and self.fail_on in text:
            raise RuntimeError("model missing")
        return text  # the "path" is just the text


class Pipeline(unittest.TestCase):
    def _make(self, tts, play_time=0.15):
        self.out, self.played = [], []
        self.state = DisplayState()

        def player(samples, sr):
            self.played.append((time.monotonic(), self.state.status))
            time.sleep(play_time)

        return SpeechPipeline(tts, self.state, speaker="Asha", out=self.out.append,
                              loader=lambda text: (tone(800, 0.2), SR), player=player)

    def test_plays_in_order_and_overlaps_synthesis_with_playback(self):
        tts = FakeTTS()
        sp = self._make(tts)
        for s in ["One.", "Two.", "Three."]:
            sp.submit(s)
        sp.wait()
        self.assertEqual(self.out, ["Asha: One.", "Asha: Two.", "Asha: Three."])
        self.assertTrue(all(status == "speaking" for _, status in self.played))
        # sentence 2 was synthesized while sentence 1 was still playing
        first_play_start = self.played[0][0]
        second_synth_start = tts.calls[1][1]
        self.assertLess(second_synth_start, first_play_start + 0.15)
        self.assertEqual(self.state.status, "idle")

    def test_main_voice_failure_switches_to_fallback_voice(self):
        fallback = FakeTTS()
        self.out, self.played, self.state = [], [], DisplayState()
        sp = SpeechPipeline(FakeTTS(fail_on="One"), self.state, speaker="Asha", fallback=fallback, out=self.out.append,
                            loader=lambda text: (tone(800, 0.2), SR),
                            player=lambda samples, sr: self.played.append(1))
        for s in ["One.", "Two."]:
            sp.submit(s)
        sp.wait()
        self.assertEqual(len(self.played), 2)                      # both sentences were spoken
        self.assertEqual(len(fallback.calls), 2)                   # ... by the fallback voice
        self.assertEqual(sum("switching to the generic voice" in l for l in self.out), 1)

    def test_text_only_mode(self):
        sp = self._make(None)
        sp.submit("Hello there.")
        sp.wait()
        self.assertEqual(self.out, ["Asha: Hello there."])
        self.assertEqual(self.played, [])
        self.assertEqual(self.state.subtitle, "Hello there.")

    def test_voice_failure_falls_back_to_text_once(self):
        sp = self._make(FakeTTS(fail_on="Two"))
        for s in ["One.", "Two.", "Three."]:
            sp.submit(s)
        sp.wait()
        spoken = [l for l in self.out if l.startswith("Asha:")]
        self.assertEqual(spoken, ["Asha: One.", "Asha: Two.", "Asha: Three."])
        self.assertEqual(sum("voice unavailable" in l for l in self.out), 1)
        self.assertEqual(len(self.played), 1)      # only the first sentence was actually spoken


class WindowsVoice(unittest.TestCase):
    def test_com_init_is_a_noop_off_windows(self):
        from twin.tts import _com_init
        if os.name != "nt":
            self.assertIsNone(_com_init())


class Hardware(unittest.TestCase):
    def test_meminfo(self):
        self.assertEqual(parse_meminfo("MemTotal:       16384000 kB\nMemFree: 1 kB"), 15.6)
        self.assertIsNone(parse_meminfo("nothing here"))

    def test_recommendations(self):
        self.assertEqual(recommend(8, False, False)["llm_model"], "qwen2.5:3b")
        self.assertEqual(recommend(16, False, False)["llm_model"], "qwen2.5:3b")
        self.assertEqual(recommend(4, False, False)["llm_model"], "qwen2.5:1.5b")
        self.assertEqual(recommend(16, True, False)["llm_model"], "qwen2.5:7b")
        self.assertEqual(recommend(16, False, True)["llm_model"], "qwen2.5:7b")
        self.assertEqual(recommend(64, False, False)["whisper_model"], "small")
        self.assertEqual(recommend(None, False, False)["llm_model"], "qwen2.5:3b")


class SettingsPrecedence(unittest.TestCase):
    def test_env_beats_profile_beats_default(self):
        old = cfg.DATA_DIR
        with tempfile.TemporaryDirectory() as d:
            cfg.DATA_DIR = Path(d)
            try:
                os.environ.pop("TWIN_LLM_MODEL", None)
                self.assertEqual(Settings().llm_model, "qwen2.5:3b")          # default
                (Path(d) / "profile.json").write_text(json.dumps({"llm_model": "llama3.2:3b", "whisper_model": "tiny"}))
                s = Settings()
                self.assertEqual((s.llm_model, s.whisper_model), ("llama3.2:3b", "tiny"))   # profile
                os.environ["TWIN_LLM_MODEL"] = "custom:1b"
                self.assertEqual(Settings().llm_model, "custom:1b")           # environment
                os.environ["TWIN_FACE_SIZE"] = "256"
                self.assertEqual(Settings().face_size, 256)
            finally:
                os.environ.pop("TWIN_LLM_MODEL", None)
                os.environ.pop("TWIN_FACE_SIZE", None)
                cfg.DATA_DIR = old


if __name__ == "__main__":
    unittest.main()
