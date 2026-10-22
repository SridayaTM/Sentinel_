"""Camera self-check: is the camera still giving usable, correctly-aimed pictures?

Feed every frame as a grayscale image (2-D array, 0..255, any size) with its capture time.
Reported states (anything but "ok" means E1 must be switched off):
  warming  - learning the normal view (first seconds, or after rebaseline())
  ok       - fine
  frozen   - frames are identical for several seconds (stuck stream)
  covered  - picture is flat (lens covered, black or white)
  dark     - picture is too dark to see vehicles
  blurred  - picture has lost its detail (fog on lens, defocus)
  moved    - the view has shifted from the learned reference (camera knocked or turned)

A fault needs `bad_s` seconds in a row before it is reported, and recovery needs `good_s` clear seconds,
so one odd frame never flips the state.
All thresholds are PLACEHOLDERS: check them on real footage (day, dusk, rain) and freeze them.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

STATES = ("warming", "ok", "frozen", "covered", "dark", "blurred", "moved")


@dataclass(frozen=True)
class CamConfig:
    work_w: int = 160               # analysis size (pictures are reduced to about this width)
    work_h: int = 90
    warmup_s: float = 3.0
    ref_interval_s: float = 0.5     # reference frames are sampled this often while warming
    ref_frames: int = 5
    bad_s: float = 2.0              # fault must persist this long
    good_s: float = 2.0             # and be absent this long to recover
    frozen_s: float = 3.0
    frozen_diff: float = 0.02       # mean abs difference (0..255) between consecutive frames
    cover_std: float = 6.0          # flat picture
    dark_mean: float = 25.0
    blur_ratio: float = 0.1         # blurred when detail falls below this share of the learned detail
    blur_abs: float = 2.0           # ...or below this absolute detail (variance of the Laplacian on the reduced picture)
    moved_px: float = 2.5           # shift (in analysis pixels) from the reference
    moved_check_s: float = 0.5


def _reduce(gray: np.ndarray, w: int, h: int) -> np.ndarray:
    g = np.asarray(gray, dtype=np.float32)
    if g.ndim == 3:
        g = g.mean(axis=2)
    gh, gw = g.shape
    bh, bw = max(1, gh // h), max(1, gw // w)
    g = g[: (gh // bh) * bh, : (gw // bw) * bw]
    return g.reshape(g.shape[0] // bh, bh, g.shape[1] // bw, bw).mean(axis=(1, 3))


def _lap_var(a: np.ndarray) -> float:
    lap = (-4 * a[1:-1, 1:-1] + a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:])
    return float(lap.var())


def _shift(ref: np.ndarray, cur: np.ndarray) -> float:
    """Size of the picture shift between two same-shaped pictures, in pixels (phase correlation)."""
    def prep(x):
        x = x - x.mean()
        s = x.std()
        x = x / (s if s > 1e-6 else 1.0)
        return x * np.outer(np.hanning(x.shape[0]), np.hanning(x.shape[1]))
    a, b = np.fft.fft2(prep(ref)), np.fft.fft2(prep(cur))
    r = a * np.conj(b)
    r /= np.abs(r) + 1e-9
    corr = np.fft.ifft2(r).real
    iy, ix = np.unravel_index(int(np.argmax(corr)), corr.shape)
    h, w = corr.shape
    dy = iy - h if iy > h // 2 else iy
    dx = ix - w if ix > w // 2 else ix
    return float(np.hypot(dx, dy))


class CameraHealth:
    def __init__(self, cfg: CamConfig = CamConfig()):
        self.cfg = cfg
        self.rebaseline()
        self.transitions: list[tuple[float, str]] = []

    def rebaseline(self) -> None:
        """Forget the reference and learn the view again (e.g. after the camera was deliberately re-aimed)."""
        self.state = "warming"
        self._t0: Optional[float] = None
        self._prev: Optional[np.ndarray] = None
        self._frozen_since: Optional[float] = None
        self._ref: Optional[np.ndarray] = None
        self._ref_buf: list[np.ndarray] = []
        self._ref_lap: Optional[float] = None
        self._last_ref_t = -1e9
        self._last_move_t = -1e9
        self._moved_now = False
        self._raw: Optional[str] = None
        self._raw_since = 0.0
        self._clear_since: Optional[float] = None

    @property
    def ok(self) -> bool:
        return self.state == "ok"

    # ---------------------------------------------------------------- per frame
    def update(self, t: float, gray) -> list[tuple[float, str]]:
        """Returns the state changes caused by this frame: [(time, new_state)]."""
        cfg = self.cfg
        a = _reduce(gray, cfg.work_w, cfg.work_h)
        if self._t0 is None:
            self._t0 = t
        mean, std = float(a.mean()), float(a.std())

        if self._prev is not None and self._prev.shape == a.shape:
            if float(np.abs(a - self._prev).mean()) < cfg.frozen_diff:
                if self._frozen_since is None:
                    self._frozen_since = t
            else:
                self._frozen_since = None
        else:
            self._frozen_since = None
        self._prev = a

        raw: Optional[str] = None
        if self._frozen_since is not None and t - self._frozen_since >= cfg.frozen_s:
            raw = "frozen"
        elif std < cfg.cover_std:
            raw = "covered"
        elif mean < cfg.dark_mean:
            raw = "dark"
        else:
            lap = _lap_var(a)
            floor = cfg.blur_abs if self._ref_lap is None else max(cfg.blur_abs, cfg.blur_ratio * self._ref_lap)
            if lap < floor:
                raw = "blurred"

        # build the reference from clean frames, then watch for a shifted view
        if raw is None and self._ref is None and t - self._last_ref_t >= cfg.ref_interval_s:
            self._ref_buf.append(a)
            self._last_ref_t = t
            if len(self._ref_buf) >= cfg.ref_frames and t - self._t0 >= cfg.warmup_s:
                self._ref = np.median(np.stack(self._ref_buf), axis=0)
                self._ref_lap = float(np.median([_lap_var(x) for x in self._ref_buf]))
                self._ref_buf = []
        if raw is None and self._ref is not None and self._ref.shape == a.shape \
                and t - self._last_move_t >= cfg.moved_check_s:
            self._last_move_t = t
            self._moved_now = _shift(self._ref, a) >= cfg.moved_px
        if raw is None and self._ref is not None and self._moved_now:
            raw = "moved"

        return self._settle(t, raw)

    def _settle(self, t: float, raw: Optional[str]) -> list[tuple[float, str]]:
        cfg = self.cfg
        before = self.state
        if raw != self._raw:
            self._raw, self._raw_since = raw, t
        if raw is not None:
            self._clear_since = None
            if t - self._raw_since >= (0.0 if raw == "frozen" else cfg.bad_s):
                self.state = raw
        else:
            if self._clear_since is None:
                self._clear_since = t
            if self.state in ("warming",):
                if self._ref is not None:
                    self.state = "ok"
            elif self.state != "ok" and t - self._clear_since >= cfg.good_s:
                self.state = "ok"
        if self.state != before:
            self.transitions.append((t, self.state))
            return [(t, self.state)]
        return []
