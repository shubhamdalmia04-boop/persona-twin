"""A lightweight talking face that runs in real time on an ordinary CPU-only laptop.

Neural talking-head models (LivePortrait, MuseTalk, Wav2Lip...) need a GPU. This module takes a
different route, the classic 2D "puppet": find the face in ONE photo, precompute a handful of
smooth warp fields (jaw opening, mouth widening/pursing, eye blink), and each frame blend those
fields with a few numbers (how open, how wide, how closed the eyes) and remap the photo with
OpenCV. It costs a few milliseconds per frame.

What you get: a photo whose jaw drops and lips widen or round in time with the voice, natural
blinking, gentle head sway, and a dark mouth interior with teeth painted in when the mouth opens.
What you do not get: photoreal lip-sync, side views, or emotion. Use a front-facing photo with a
CLOSED mouth and open eyes for the best result (a big smile with teeth will look odd).

MediaPipe supplies the landmarks (one-time detection per photo). Everything downstream needs only
numpy + OpenCV and works from a small set of `Anchors`, so it is easy to test.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .config import FACE_MODEL_PATH

# MediaPipe face-mesh landmark indices used here.
_MOUTH_L, _MOUTH_R, _LIP_TOP, _LIP_BOT = 61, 291, 13, 14
_CHIN, _FOREHEAD, _CHEEK_L, _CHEEK_R, _NOSE = 152, 10, 234, 454, 1
_EYES = (
    dict(outer=33, inner=133, top=159, bottom=145),    # image-left eye
    dict(outer=263, inner=362, top=386, bottom=374),   # image-right eye
)


# ---------------------------------------------------------------------------- anchors
@dataclass(frozen=True)
class Eye:
    cx: float
    cy: float
    w: float
    h: float


@dataclass(frozen=True)
class Anchors:
    """The few facial measurements the puppet needs, in pixel coordinates."""

    mouth_cx: float
    mouth_cy: float       # height of the (closed) lip line
    mouth_w: float
    mouth_gap: float      # inner-lip opening in the source photo (0 = closed)
    chin_y: float
    forehead_y: float
    face_cx: float
    face_w: float
    nose_x: float
    eyes: tuple[Eye, Eye]

    @property
    def face_h(self) -> float:
        return self.chin_y - self.forehead_y

    def transform(self, scale: float, dx: float, dy: float) -> "Anchors":
        """Map coordinates x' = x*scale + dx, y' = y*scale + dy."""
        fx = lambda v: v * scale + dx
        fy = lambda v: v * scale + dy
        return replace(
            self,
            mouth_cx=fx(self.mouth_cx), mouth_cy=fy(self.mouth_cy),
            mouth_w=self.mouth_w * scale, mouth_gap=self.mouth_gap * scale,
            chin_y=fy(self.chin_y), forehead_y=fy(self.forehead_y),
            face_cx=fx(self.face_cx), face_w=self.face_w * scale, nose_x=fx(self.nose_x),
            eyes=tuple(Eye(fx(e.cx), fy(e.cy), e.w * scale, e.h * scale) for e in self.eyes),
        )


def anchors_from_landmarks(pts: np.ndarray) -> Anchors:
    """Reduce MediaPipe's 468+ landmarks (N, 2) to the anchors the puppet uses."""
    p = np.asarray(pts, dtype=np.float64)
    eyes = []
    for e in _EYES:
        corners = (p[e["outer"]] + p[e["inner"]]) / 2
        lids = (p[e["top"]] + p[e["bottom"]]) / 2
        eyes.append(
            Eye(
                cx=float((corners[0] + lids[0]) / 2),
                cy=float((corners[1] + lids[1]) / 2),
                w=float(np.linalg.norm(p[e["outer"]] - p[e["inner"]])),
                h=float(abs(p[e["top"]][1] - p[e["bottom"]][1])),
            )
        )
    return Anchors(
        mouth_cx=float((p[_MOUTH_L][0] + p[_MOUTH_R][0]) / 2),
        mouth_cy=float((p[_LIP_TOP][1] + p[_LIP_BOT][1]) / 2),
        mouth_w=float(np.linalg.norm(p[_MOUTH_R] - p[_MOUTH_L])),
        mouth_gap=float(abs(p[_LIP_BOT][1] - p[_LIP_TOP][1])),
        chin_y=float(p[_CHIN][1]),
        forehead_y=float(p[_FOREHEAD][1]),
        face_cx=float((p[_CHEEK_L][0] + p[_CHEEK_R][0]) / 2),
        face_w=float(abs(p[_CHEEK_R][0] - p[_CHEEK_L][0])),
        nose_x=float(p[_NOSE][0]),
        eyes=tuple(eyes),
    )


def score_face(a: Anchors, image_height: int) -> float:
    """How good a source photo this is for the puppet (0 = unusable, 1 = ideal)."""
    inter = abs(a.eyes[1].cx - a.eyes[0].cx)
    if inter < 1 or a.face_h <= 0:
        return 0.0
    mid = (a.eyes[0].cx + a.eyes[1].cx) / 2
    frontal = 1 - min(1.0, abs(a.nose_x - mid) / (0.45 * inter))
    if frontal < 0.4:
        return 0.0
    size = min(a.face_h / image_height, 0.5) / 0.5
    closed = float(np.clip(1 - a.mouth_gap / (0.10 * a.mouth_w), 0, 1))
    ratio = np.mean([e.h / max(e.w, 1e-6) for e in a.eyes])
    eyes_open = float(np.clip((ratio - 0.08) / 0.14, 0, 1))
    return 0.35 * size + 0.25 * frontal + 0.20 * closed + 0.20 * eyes_open


def crop_head(rgb: np.ndarray, a: Anchors, size: int) -> tuple[np.ndarray, Anchors]:
    """Cut a square around the head (hair, ears, neck included) and resize to size x size."""
    import cv2

    h = a.face_h
    side = int(round(2.1 * h))
    cx = a.face_cx
    cy = (a.forehead_y + a.chin_y) / 2 - 0.08 * h
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    H, W = rgb.shape[:2]
    pad = max(0, -x0, -y0, x0 + side - W, y0 + side - H)
    src = cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_REPLICATE) if pad else rgb
    crop = src[y0 + pad : y0 + pad + side, x0 + pad : x0 + pad + side]
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    s = size / side
    return np.ascontiguousarray(crop), a.transform(s, -x0 * s, -y0 * s)


# ---------------------------------------------------------------------------- warp fields
def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def build_bases(size: int, a: Anchors) -> np.ndarray:
    """Displacement fields for unit parameters, shape (4, size, size, 2), float32.

    Order: jaw open, mouth widen, mouth purse, blink. A field says how far the picture content
    at each pixel moves (dx, dy) when that parameter is 1.
    """
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    h, mw = a.face_h, a.mouth_w
    bases = np.zeros((4, size, size, 2), dtype=np.float32)

    # Jaw: everything below the lip line drops, fading out under the chin and towards the sides.
    up = _smoothstep((ys - a.mouth_cy) / (0.06 * h))
    down = 1 - _smoothstep((ys - (a.chin_y - 0.03 * h)) / (0.10 * h))
    side = np.exp(-0.5 * ((xs - a.mouth_cx) / (0.30 * a.face_w)) ** 2)
    bases[0, ..., 1] = up * down * side * (0.07 * h)

    # Mouth widening / pursing: stretch or squeeze the lips horizontally about the mouth centre.
    g = np.exp(-0.5 * (((xs - a.mouth_cx) / (0.75 * mw)) ** 2 + ((ys - a.mouth_cy) / (0.5 * mw)) ** 2))
    bases[1, ..., 0] = (xs - a.mouth_cx) * 0.12 * g
    bases[2, ..., 0] = -(xs - a.mouth_cx) * 0.22 * g

    # Blink: pull the upper eyelid skin down over the eye.
    for e in a.eyes:
        top = e.cy - e.h / 2
        gauss = np.exp(-0.5 * (((xs - e.cx) / (0.62 * e.w)) ** 2 + ((ys - top) / (0.8 * e.h + 2.0)) ** 2))
        bases[3, ..., 1] += gauss * (e.h * 1.05 + 0.5)
    return bases


class BlinkClock:
    """Deterministic random blinking: one blink every 2-5.5 seconds, 0.18 s long."""

    def __init__(self, seed: int = 3, count: int = 4000, duration: float = 0.18):
        gaps = np.random.default_rng(seed).uniform(2.0, 5.5, count)
        self.starts = np.cumsum(gaps)
        self.duration = duration

    def value(self, t: float) -> float:
        i = int(np.searchsorted(self.starts, t, side="right")) - 1
        if i < 0:
            return 0.0
        dt = t - self.starts[i]
        return float(math.sin(math.pi * dt / self.duration)) if dt < self.duration else 0.0


def head_motion(t: float, talk: float) -> tuple[float, float, float]:
    """Gentle idle sway. Returns (rotation radians, dx, dy) with dx, dy as fractions of the image size."""
    rot = 0.010 * math.sin(2 * math.pi * t / 6.3) + 0.006 * math.sin(2 * math.pi * t / 3.1 + 1.0)
    rot += 0.007 * talk * math.sin(2 * math.pi * t * 1.3)
    dx = 0.003 * math.sin(2 * math.pi * t / 7.7)
    dy = 0.002 * math.sin(2 * math.pi * t / 5.1 + 2.0) + 0.002 * talk * math.sin(2 * math.pi * t * 1.9)
    return rot, dx, dy


# ---------------------------------------------------------------------------- the puppet
class PuppetFace:
    """Animates a head photo. `render()` returns an RGB uint8 array (size x size x 3)."""

    def __init__(self, image_rgb: np.ndarray, anchors: Anchors):
        self.image = np.ascontiguousarray(image_rgb)
        self.size = self.image.shape[0]
        self.a = anchors
        self.bases = build_bases(self.size, anchors)
        ys, xs = np.mgrid[0:self.size, 0:self.size].astype(np.float32)
        self.base_map = np.stack([xs, ys], axis=-1)
        self._pivot = (anchors.face_cx, anchors.chin_y)
        self._gx = xs - self._pivot[0]
        self._gy = ys - self._pivot[1]
        self.blinks = BlinkClock()

    def _mouth_interior(self, out: np.ndarray, open_: float, width: float, shift: tuple[float, float]) -> None:
        a = self.a
        gap = open_ * 0.07 * a.face_h * 0.92           # lower inner lip drops by about this much
        if gap < 1.2:
            return
        half_w = 0.5 * a.mouth_w * (0.80 + 0.10 * max(width, 0) - 0.28 * max(-width, 0))
        cx, top = a.mouth_cx + shift[0], a.mouth_cy + shift[1]
        cy, half_h = top + gap / 2, gap / 2
        x0, x1 = int(cx - half_w - 3), int(cx + half_w + 4)
        y0, y1 = int(top - 3), int(top + gap + 4)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, self.size), min(y1, self.size)
        if x1 <= x0 or y1 <= y0:
            return
        ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        ex, ey = (xs - cx) / half_w, (ys - cy) / (half_h + 1e-3)
        dist = np.sqrt(ex * ex + ey * ey)
        alpha = np.clip((1.0 - dist) * max(2.0, gap * 0.35), 0, 1)[..., None]
        cavity = np.array([48, 20, 26], dtype=np.float32)
        teeth = np.array([226, 220, 208], dtype=np.float32)
        strength = float(np.clip((open_ - 0.12) * 3.0, 0, 1)) * 0.9
        t = (_smoothstep((-ey - 0.05) / 0.5) * np.clip(1 - ex * ex, 0, 1))[..., None] * strength
        colour = cavity * (1 - t) + teeth * t
        roi = out[y0:y1, x0:x1].astype(np.float32)
        out[y0:y1, x0:x1] = (roi * (1 - alpha) + colour * alpha).astype(np.uint8)

    def render(self, open_: float = 0.0, width: float = 0.0, blink: float = 0.0, t: float = 0.0) -> np.ndarray:
        import cv2

        open_ = float(np.clip(open_, 0, 1))
        width = float(np.clip(width, -1, 1))
        coefs = np.array([open_, max(width, 0), max(-width, 0), np.clip(blink, 0, 1)], dtype=np.float32)
        flow = np.tensordot(coefs, self.bases, axes=(0, 0))  # (size, size, 2)

        rot, dx, dy = head_motion(t, open_)
        s = self.size
        flow[..., 0] += -rot * self._gy + dx * s
        flow[..., 1] += rot * self._gx + dy * s

        out = cv2.remap(self.image, self.base_map - flow, None, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT)
        # keep the painted mouth interior aligned with the head sway
        gx, gy = self.a.mouth_cx - self._pivot[0], self.a.mouth_cy - self._pivot[1]
        shift = (-rot * gy + dx * s, rot * gx + dy * s)
        self._mouth_interior(out, open_, width, shift)
        return out


# ---------------------------------------------------------------------------- detection
class LandmarkDetector:
    """MediaPipe FaceLandmarker (Tasks API). Needs the model file downloaded by install.py."""

    def __init__(self, model_path: Path = FACE_MODEL_PATH):
        self.model_path = Path(model_path)
        self._landmarker = None

    def unavailable_reason(self) -> str | None:
        if not self.model_path.exists():
            return "face model not downloaded (run: python install.py)"
        try:
            import mediapipe  # noqa: F401
            import cv2  # noqa: F401
        except ImportError:
            return "mediapipe is not installed (run: python install.py)"
        return None

    def detect(self, rgb: np.ndarray) -> np.ndarray | None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        if self._landmarker is None:
            options = vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(self.model_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
            )
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
        result = self._landmarker.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        )
        if not result.face_landmarks:
            return None
        h, w = rgb.shape[:2]
        return np.array([[p.x * w, p.y * h] for p in result.face_landmarks[0]], dtype=np.float32)


def read_rgb(path: Path) -> np.ndarray | None:
    """Read an image (unicode paths and EXIF rotation handled) as RGB."""
    import cv2

    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def prepare_puppet(
    photos: list[Path], size: int = 384, detector: LandmarkDetector | None = None,
    log=print, max_photos: int = 40,
) -> PuppetFace | None:
    """Pick the most suitable photo (frontal, closed mouth, open eyes) and build the puppet.

    Returns None (after logging why) if the puppet cannot be built, so callers can fall back.
    """
    detector = detector or LandmarkDetector()
    reason = detector.unavailable_reason()
    if reason:
        log(f"Animated face unavailable: {reason}.")
        return None
    best = None  # (score, rgb, anchors)
    checked = no_face = unusable = 0
    try:
        for path in photos[:max_photos]:
            rgb = read_rgb(path)
            if rgb is None:
                continue
            checked += 1
            pts = detector.detect(rgb)
            if pts is None:
                no_face += 1
                continue
            anchors = anchors_from_landmarks(pts)
            score = score_face(anchors, rgb.shape[0])
            if score <= 0:
                unusable += 1
            elif best is None or score > best[0]:
                best = (score, rgb, anchors)
        if best is None:
            log(
                f"Animated face unavailable: checked {checked} photo(s): {no_face} with no face found, "
                f"{unusable} not clearly front-facing. Choose a clear, front-facing photo of the person "
                "(mouth closed, eyes open)."
            )
            return None
        crop, crop_anchors = crop_head(best[1], best[2], size)
        return PuppetFace(crop, crop_anchors)
    except Exception as exc:  # e.g. an incompatible MediaPipe/OpenCV version: fall back, never crash
        log(f"Animated face unavailable: {type(exc).__name__}: {exc}")
        return None
