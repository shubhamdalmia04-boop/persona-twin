"""Animated-face tests (numpy + OpenCV, no MediaPipe model or display needed)."""
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from twin.face import (
    Anchors, BlinkClock, Eye, PuppetFace, anchors_from_landmarks, build_bases, crop_head,
    head_motion, prepare_puppet, read_rgb, score_face,
)


def landmarks(mouth_gap=0.0, nose_dx=0.0, eye_h=12.0):
    """A synthetic 478-point mesh with the indices the puppet reads, on a 600x800 image."""
    p = np.zeros((478, 2), dtype=np.float32)
    p[61], p[291] = (260, 560), (340, 560)                     # mouth corners
    p[13], p[14] = (300, 560 - mouth_gap / 2), (300, 560 + mouth_gap / 2)
    p[152], p[10] = (300, 700), (300, 300)                     # chin, forehead
    p[234], p[454] = (200, 500), (400, 500)                    # cheeks
    p[1] = (300 + nose_dx, 500)                                # nose tip
    p[33], p[133], p[159], p[145] = (232, 450), (272, 450), (252, 450 - eye_h / 2), (252, 450 + eye_h / 2)
    p[263], p[362], p[386], p[374] = (368, 450), (328, 450), (348, 450 - eye_h / 2), (348, 450 + eye_h / 2)
    return p


def anchors(size=384):
    a = anchors_from_landmarks(landmarks())
    return a


class AnchorTests(unittest.TestCase):
    def test_from_landmarks(self):
        a = anchors_from_landmarks(landmarks(mouth_gap=4))
        self.assertAlmostEqual(a.mouth_cx, 300)
        self.assertAlmostEqual(a.mouth_cy, 560)
        self.assertAlmostEqual(a.mouth_w, 80)
        self.assertAlmostEqual(a.mouth_gap, 4)
        self.assertAlmostEqual(a.face_h, 400)
        self.assertAlmostEqual(a.eyes[0].w, 40)
        self.assertAlmostEqual(a.eyes[0].h, 12)
        self.assertLess(a.eyes[0].cx, a.eyes[1].cx)

    def test_transform(self):
        a = anchors_from_landmarks(landmarks()).transform(0.5, -10, -20)
        self.assertAlmostEqual(a.mouth_cx, 140)
        self.assertAlmostEqual(a.mouth_w, 40)
        self.assertAlmostEqual(a.face_h, 200)

    def test_score_prefers_closed_mouth_frontal_open_eyes(self):
        good = score_face(anchors_from_landmarks(landmarks()), 800)
        smiling = score_face(anchors_from_landmarks(landmarks(mouth_gap=12)), 800)
        turned = score_face(anchors_from_landmarks(landmarks(nose_dx=30)), 800)
        eyes_shut = score_face(anchors_from_landmarks(landmarks(eye_h=1)), 800)
        self.assertGreater(good, smiling)
        self.assertGreater(good, eyes_shut)
        self.assertEqual(turned, 0.0)


class CropAndBases(unittest.TestCase):
    def setUp(self):
        self.img = np.random.default_rng(0).integers(0, 255, (800, 600, 3), dtype=np.uint8)
        self.crop, self.a = crop_head(self.img, anchors_from_landmarks(landmarks()), 256)

    def test_crop_shape_and_anchor_mapping(self):
        self.assertEqual(self.crop.shape, (256, 256, 3))
        self.assertAlmostEqual(self.a.face_cx, 128, delta=1)
        self.assertGreater(self.a.mouth_cy, self.a.eyes[0].cy)
        self.assertLess(self.a.chin_y, 256)

    def test_crop_handles_face_near_border(self):
        p = landmarks()
        p[:, 1] -= 250  # face now runs off the top of the image
        crop, a = crop_head(self.img, anchors_from_landmarks(p), 256)
        self.assertEqual(crop.shape, (256, 256, 3))

    def test_bases_geometry(self):
        b = build_bases(256, self.a)
        self.assertEqual(b.shape, (4, 256, 256, 2))
        jaw = b[0]
        mouth_y, mouth_x = int(self.a.mouth_cy), int(self.a.mouth_cx)
        self.assertEqual(float(np.abs(jaw[: mouth_y - 2]).max()), 0.0)    # nothing moves above the lips
        self.assertGreater(jaw[mouth_y + 20, mouth_x, 1], 0)               # jaw drops downwards
        self.assertEqual(float(jaw[..., 0].max()), 0.0)                    # purely vertical
        widen, purse = b[1], b[2]
        right = mouth_x + int(self.a.mouth_w * 0.4)
        self.assertGreater(widen[mouth_y, right, 0], 0)                    # right corner moves right
        self.assertLess(purse[mouth_y, right, 0], 0)                       # ... or in when pursing
        eye = self.a.eyes[0]
        self.assertGreater(b[3][int(eye.cy - eye.h / 2), int(eye.cx), 1], 0)  # eyelid moves down
        self.assertEqual(float(np.abs(b[3][int(self.a.chin_y):]).max()), 0.0)


class PuppetTests(unittest.TestCase):
    def setUp(self):
        img = np.full((256, 256, 3), 180, dtype=np.uint8)
        cv2.rectangle(img, (90, 180), (170, 190), (150, 60, 70), -1)   # a "mouth" to move
        img[100:120, 60:200] = (60, 80, 110)                            # an "eye band"
        _, self.a = crop_head(np.zeros((800, 600, 3), np.uint8), anchors_from_landmarks(landmarks()), 256)
        self.p = PuppetFace(img, self.a)

    def test_rest_pose_stays_close_to_source(self):
        out = self.p.render(0, 0, 0, t=0.0)
        self.assertEqual(out.shape, self.p.image.shape)
        self.assertEqual(out.dtype, np.uint8)
        self.assertLess(np.abs(out.astype(int) - self.p.image.astype(int)).mean(), 6)

    def test_mouth_open_changes_lower_face_only(self):
        rest = self.p.render(0, 0, 0, t=0).astype(int)
        talk = self.p.render(1.0, 0, 0, t=0).astype(int)
        diff = np.abs(talk - rest).sum(axis=2)
        my = int(self.a.mouth_cy)
        self.assertGreater(diff[my : my + 40].sum(), 20 * diff[: my - 30].sum() + 1)
        # dark mouth interior painted where the lips parted
        interior = talk[my + 2 : my + 6, int(self.a.mouth_cx) - 3 : int(self.a.mouth_cx) + 3]
        self.assertLess(interior.mean(), 120)

    def test_blink_changes_eye_region(self):
        rest = self.p.render(0, 0, 0, t=0).astype(int)
        blink = self.p.render(0, 0, 1.0, t=0).astype(int)
        e = self.a.eyes[0]
        region = np.abs(blink - rest)[int(e.cy) - 12 : int(e.cy) + 12, int(e.cx) - 20 : int(e.cx) + 20]
        self.assertGreater(region.sum(), 0)

    def test_render_is_fast_enough_for_a_laptop(self):
        self.p.render(0.5, 0.2, 0, t=0)
        start = time.perf_counter()
        for i in range(30):
            self.p.render(0.5, np.sin(i), 0, t=i / 30)
        per_frame_ms = (time.perf_counter() - start) / 30 * 1000
        self.assertLess(per_frame_ms, 25)   # 256px frames; budget for 30 fps on a slow core


class MotionTests(unittest.TestCase):
    def test_blink_clock(self):
        clock = BlinkClock()
        values = [clock.value(t / 100) for t in range(0, 3000)]
        self.assertTrue(all(0.0 <= v <= 1.0 for v in values))
        self.assertGreaterEqual(sum(1 for v in values if v > 0.99), 3)     # several blinks in 30 s
        self.assertLess(sum(1 for v in values if v > 0), 0.2 * len(values))  # eyes open most of the time

    def test_head_motion_is_subtle(self):
        for t in np.linspace(0, 60, 500):
            rot, dx, dy = head_motion(float(t), 1.0)
            self.assertLess(abs(rot), 0.03)
            self.assertLess(abs(dx), 0.01)
            self.assertLess(abs(dy), 0.01)


class ExplodingDetector:
    def unavailable_reason(self):
        return None

    def detect(self, rgb):
        raise RuntimeError("incompatible mediapipe")


class FakeDetector:
    def __init__(self, table, reason=None):
        self.table, self.reason = table, reason

    def unavailable_reason(self):
        return self.reason

    def detect(self, rgb):
        return self.table.get(int(rgb[0, 0, 0]))


class PreparePuppet(unittest.TestCase):
    def _photo(self, folder, name, marker):
        img = np.full((800, 600, 3), marker, dtype=np.uint8)   # marker lets FakeDetector tell photos apart
        path = Path(folder) / name
        cv2.imwrite(str(path), img)
        return path

    def test_picks_best_photo_and_builds(self):
        with tempfile.TemporaryDirectory() as d:
            smile = self._photo(d, "smile.jpg", 10)
            calm = self._photo(d, "calm.jpg", 20)
            noface = self._photo(d, "none.jpg", 30)
            det = FakeDetector({10: landmarks(mouth_gap=14), 20: landmarks(), 30: None})
            logs = []
            puppet = prepare_puppet([smile, noface, calm], 128, detector=det, log=logs.append)
            self.assertIsInstance(puppet, PuppetFace)
            self.assertEqual(puppet.image.shape, (128, 128, 3))
            self.assertAlmostEqual(puppet.a.mouth_gap, 0, places=3)   # chose the closed-mouth photo

    def test_failure_message_says_why(self):
        with tempfile.TemporaryDirectory() as d:
            blank = self._photo(d, "none.jpg", 30)
            turned = self._photo(d, "turned.jpg", 40)
            det = FakeDetector({30: None, 40: landmarks(nose_dx=40)})
            logs = []
            self.assertIsNone(prepare_puppet([blank, turned], 128, detector=det, log=logs.append))
            self.assertIn("checked 2 photo(s): 1 with no face found, 1 not clearly front-facing", logs[-1])

    def test_detector_crash_falls_back_instead_of_raising(self):
        with tempfile.TemporaryDirectory() as d:
            photo = self._photo(d, "a.jpg", 10)
            logs = []
            self.assertIsNone(prepare_puppet([photo], 128, detector=ExplodingDetector(), log=logs.append))
            self.assertIn("incompatible mediapipe", logs[-1])

    def test_graceful_fallbacks(self):
        logs = []
        self.assertIsNone(prepare_puppet([], 128, detector=FakeDetector({}, reason="no model"), log=logs.append))
        self.assertIn("no model", logs[0])
        with tempfile.TemporaryDirectory() as d:
            photo = self._photo(d, "a.jpg", 10)
            self.assertIsNone(prepare_puppet([photo], 128, detector=FakeDetector({10: None}), log=logs.append))
            self.assertIn("1 with no face found", logs[-1])
            self.assertIsNone(read_rgb(Path(d) / "missing.jpg"))


if __name__ == "__main__":
    unittest.main()
