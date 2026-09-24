"""Projection display.

A black stage (black is invisible on a projector and on a Pepper's-ghost pyramid) showing the
person's photo, a glow that follows their voice, and subtitles.

Layouts
  single   one full-screen picture: projector on a wall, screen, or projection film/scrim
  pyramid  four copies arranged in a cross for a 4-sided "hologram" pyramid placed on a
           flat screen or tablet (heads point outward; the pyramid's small end sits at the centre)

`--mirror` flips the picture horizontally, needed for rear projection and for some pyramid
setups. Calibrate with a piece of text first.

Avatars
  PuppetAvatar  animated talking face built from the photo (twin/face.py), CPU-only
  GlowAvatar    fallback: the photo with a glow that follows the voice
Both draw the same way, so a GPU neural renderer can be added later as another class.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import pygame

from .state import DisplayState

BG = (0, 0, 0)
TEXT = (238, 236, 228)
DIM = (128, 128, 124)
GLOW = (150, 185, 255)
FONT_STACK = "notosans,notosansdevanagari,nirmalaui,mangal,arialunicodems,dejavusans,arial"


def wrap_text(font: pygame.font.Font, text: str, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if not current or font.size(trial)[0] <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


class _DiscAvatar:
    """Shared drawing: a soft glow ring and a circular crop of the face surface."""

    def __init__(self) -> None:
        self._masks: dict[int, pygame.Surface] = {}

    def _mask(self, size: int) -> pygame.Surface:
        if size not in self._masks:
            mask = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.circle(mask, (255, 255, 255, 255), (size // 2, size // 2), size // 2)
            self._masks[size] = mask
        return self._masks[size]

    def _glow(self, surf: pygame.Surface, cx: int, cy: int, box: int, amp: float) -> None:
        glow = pygame.Surface((box * 2, box * 2), pygame.SRCALPHA)
        base = box // 2
        for k in range(6):
            radius = int(base * (1.03 + 0.055 * k * (0.35 + amp)))
            alpha = int(90 * (1 - k / 6) * (0.3 + 0.7 * amp))
            pygame.draw.circle(glow, (*GLOW, alpha), (box, box), radius, width=max(2, box // 90))
        surf.blit(glow, (cx - box, cy - box))

    def _blit_disc(self, surf: pygame.Surface, face: pygame.Surface, cx: int, cy: int, size: int) -> None:
        size = max(8, size)
        face = pygame.transform.smoothscale(face, (size, size))
        face.blit(self._mask(size), (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        surf.blit(face, face.get_rect(center=(cx, cy)))


class GlowAvatar(_DiscAvatar):
    """The photo (or an initial) with a glow that swells with the voice."""

    def __init__(self, photo: Path | None, name: str):
        super().__init__()
        self.name = name
        self.square: pygame.Surface | None = None
        if photo:
            img = pygame.image.load(str(photo)).convert_alpha()
            side = min(img.get_size())
            x = (img.get_width() - side) // 2
            y = max(0, int((img.get_height() - side) * 0.25))  # bias towards the face
            self.square = img.subsurface((x, y, side, side)).copy()

    def draw(self, surf: pygame.Surface, t: float, state: DisplayState, cx: int, cy: int, box: int) -> None:
        amp = state.amplitude()
        self._glow(surf, cx, cy, box, amp)
        size = int(box * (1.0 + 0.03 * amp + 0.004 * math.sin(t * 1.4)))
        if self.square is not None:
            self._blit_disc(surf, self.square, cx, cy, size)
            return
        size = max(8, size)
        face = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.circle(face, (36, 40, 52), (size // 2, size // 2), size // 2)
        font = pygame.font.SysFont(FONT_STACK, size // 2)
        letter = font.render((self.name[:1] or "?").upper(), True, TEXT)
        face.blit(letter, letter.get_rect(center=(size // 2, size // 2)))
        surf.blit(face, face.get_rect(center=(cx, cy)))


class PuppetAvatar(_DiscAvatar):
    """The animated talking face. Rendering costs a few milliseconds on a plain CPU."""

    def __init__(self, puppet) -> None:
        super().__init__()
        self.puppet = puppet
        self._open = 0.0
        self._width = 0.0
        self._last_t = 0.0

    @staticmethod
    def _approach(current: float, target: float, dt: float, up: float = 0.03, down: float = 0.07) -> float:
        tau = up if target > current else down
        return current + (target - current) * (1 - math.exp(-dt / tau))

    def draw(self, surf: pygame.Surface, t: float, state: DisplayState, cx: int, cy: int, box: int) -> None:
        dt = max(1e-3, min(0.1, t - self._last_t))
        self._last_t = t
        target_open, target_width = state.mouth()
        self._open = self._approach(self._open, target_open, dt)
        self._width = self._approach(self._width, target_width, dt)

        frame = self.puppet.render(self._open, self._width, self.puppet.blinks.value(t), t)
        side = frame.shape[0]
        face = pygame.image.frombuffer(frame.tobytes(), (side, side), "RGB").convert_alpha()
        self._glow(surf, cx, cy, box, 0.35 * state.amplitude())  # subtler than the photo avatar
        self._blit_disc(surf, face, cx, cy, box)


class Stage:
    def __init__(self, name: str, photos: list[Path], mode: str = "auto", face_size: int = 384, log=print):
        self.name = name
        self.avatar: GlowAvatar | PuppetAvatar | None = None
        if mode in ("auto", "puppet") and photos:
            from .face import prepare_puppet

            log("Preparing the animated face...")
            puppet = prepare_puppet(photos, face_size, log=log)
            if puppet is not None:
                self.avatar = PuppetAvatar(puppet)
            elif mode == "puppet":
                log("Falling back to the simple photo avatar.")
        if self.avatar is None:
            self.avatar = GlowAvatar(photos[0] if photos else None, name)
        self._photo = photos[0] if photos else None
        self._log = log
        self._fonts: dict[int, pygame.font.Font] = {}

    def font(self, size: int) -> pygame.font.Font:
        size = max(10, size)
        if size not in self._fonts:
            self._fonts[size] = pygame.font.SysFont(FONT_STACK, size)
        return self._fonts[size]

    def render(self, size: tuple[int, int], state: DisplayState, t: float, subtitles: bool = True) -> pygame.Surface:
        w, h = size
        surf = pygame.Surface(size)
        surf.fill(BG)

        box = int(min(w, h) * 0.52)
        try:
            self.avatar.draw(surf, t, state, w // 2, int(h * 0.44), box)
        except Exception as exc:  # a broken animation must not take the whole display down
            if isinstance(self.avatar, PuppetAvatar):
                self._log(f"Animated face failed ({type(exc).__name__}: {exc}); using the simple avatar.")
                self.avatar = GlowAvatar(self._photo, self.name)
                self.avatar.draw(surf, t, state, w // 2, int(h * 0.44), box)
            else:
                raise

        label = self.font(int(h * 0.032)).render(f"AI simulation of {self.name}", True, DIM)
        surf.blit(label, label.get_rect(midtop=(w // 2, int(h * 0.03))))

        if subtitles:
            font = self.font(int(h * 0.05))
            if state.status in ("listening", "thinking") and not state.subtitle:
                text, colour = state.status.capitalize(), DIM
            else:
                text, colour = state.subtitle, TEXT
            lines = wrap_text(font, text, int(w * 0.86))[-4:]
            y = int(h * 0.78)
            for line in lines:
                img = font.render(line, True, colour)
                surf.blit(img, img.get_rect(midtop=(w // 2, y)))
                y += int(font.get_linesize() * 1.1)
        return surf


def compose(stage: Stage, size: tuple[int, int], state: DisplayState, t: float, layout: str, mirror: bool) -> pygame.Surface:
    w, h = size
    if layout == "pyramid":
        m = min(w, h)
        s, gap = int(m * 0.34), int(m * 0.08)
        scene = stage.render((s, s), state, t, subtitles=False)
        if mirror:
            scene = pygame.transform.flip(scene, True, False)
        out = pygame.Surface(size)
        out.fill(BG)
        cx, cy = w // 2, h // 2
        out.blit(scene, (cx - s // 2, cy - gap - s))                                  # top
        out.blit(pygame.transform.rotate(scene, 180), (cx - s // 2, cy + gap))        # bottom
        out.blit(pygame.transform.rotate(scene, 90), (cx - gap - s, cy - s // 2))     # left
        out.blit(pygame.transform.rotate(scene, -90), (cx + gap, cy - s // 2))        # right
        return out

    scene = stage.render(size, state, t)
    return pygame.transform.flip(scene, True, False) if mirror else scene


def run_display(
    state: DisplayState,
    name: str,
    photos: list[Path],
    layout: str = "single",
    mirror: bool = False,
    windowed: bool = False,
    screen_index: int = 0,
    avatar_mode: str = "auto",
    face_size: int = 384,
) -> None:
    """Blocks until the window is closed (Esc or Q) or state.quit is set. Run on the main thread."""
    pygame.init()
    pygame.display.set_caption(f"{name} - AI simulation")
    if windowed:
        screen = pygame.display.set_mode((1100, 760), pygame.RESIZABLE, display=screen_index)
    else:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN, display=screen_index)
    pygame.mouse.set_visible(windowed)

    stage = Stage(name, photos, avatar_mode, face_size)
    clock = pygame.time.Clock()
    start = time.monotonic()
    try:
        while not state.quit:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    state.quit = True
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                    state.quit = True
            frame = compose(stage, screen.get_size(), state, time.monotonic() - start, layout, mirror)
            screen.blit(frame, (0, 0))
            pygame.display.flip()
            clock.tick(30)
    finally:
        state.quit = True
        pygame.quit()
