"""E2: impact-sound evidence from the microphone, by rules on loudness and shape (no model inside).

Input: mono audio, float32 in -1..1, fed in any chunk size. Output: impact events.

An impact is accepted only when ALL of these hold:
  - loudness jumps up quickly: a step of at least `onset_step_db` between two frames AND the level is
    at least `rise_db` over the recent background (a truck slowly passing does not do this);
  - it is loud in absolute terms (`min_db`), and the unit is not listening to its own siren/speaker;
  - it is broadband (spectral flatness), not a tone (a horn or a beep is tonal);
  - it dies away: the level falls `decay_db` below its peak within `max_width_s`
    (a horn or a steady revving engine stays at its peak);
  - the microphone is healthy (not dead, not clipping flat) and has had `warmup_s` of background.

Thresholds are PLACEHOLDERS: tune them on recorded roadside audio and real impact sounds, then freeze.
A pretrained sound classifier can be added later as a second opinion; it is not needed for this rule.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

SCORE_VERSION = 1


@dataclass(frozen=True)
class E2Config:
    sample_rate: int = 16000
    frame_s: float = 0.02
    history_s: float = 5.0          # background window
    warmup_s: float = 1.0           # frames needed before any event is allowed
    rise_db: float = 15.0           # level over background
    onset_step_db: float = 10.0     # jump between two neighbouring frames
    min_db: float = -45.0           # dBFS, absolute floor for a candidate
    decay_db: float = 15.0          # must fall this far below the peak (or to within end_margin_db of the background)...
    end_margin_db: float = 6.0
    max_width_s: float = 0.5        # ...within this long
    min_flatness: float = 0.12      # spectral flatness (0 = pure tone, ~0.5 = white noise)
    refractory_s: float = 1.0
    mute_pad_s: float = 0.25        # own-speaker windows are widened by this much each side
    dead_db: float = -80.0
    dead_s: float = 3.0
    clip_level: float = 0.99
    clip_frac: float = 0.5          # share of recent frames hitting the rail
    clip_window_s: float = 2.0


@dataclass(frozen=True)
class E2Event:
    t_onset: float
    t_fire: float
    peak_db: float
    rise_db: float
    width_s: float
    flatness: float
    score: float
    score255: int
    score_version: int = SCORE_VERSION


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


class E2Detector:
    def __init__(self, cfg: E2Config = E2Config()):
        self.cfg = cfg
        self.n = int(round(cfg.sample_rate * cfg.frame_s))
        self._win = np.hanning(self.n).astype(np.float64)
        self._buf = np.zeros(0, dtype=np.float32)
        self._t_next: Optional[float] = None
        self._hist: deque[float] = deque(maxlen=int(cfg.history_s / cfg.frame_s))
        self._clip: deque[bool] = deque(maxlen=int(cfg.clip_window_s / cfg.frame_s))
        self._prev_db: Optional[float] = None
        self._cand: Optional[dict] = None
        self._quiet_until = -math.inf
        self._dead_since: Optional[float] = None
        self._frames_seen = 0
        self.mutes: list[tuple[float, float]] = []
        self.rejections: list[tuple[float, str]] = []
        self.health = "ok"

    # ------------------------------------------------------------------ public
    def mute(self, t0: float, t1: float) -> None:
        """Tell the detector the unit's own speaker was (or will be) playing in [t0, t1]."""
        self.mutes.append((t0, t1))

    def feed(self, t0: float, samples) -> list[E2Event]:
        """`t0` = capture time of samples[0]. Chunks must be contiguous in time."""
        x = np.asarray(samples, dtype=np.float32).ravel()
        if self._t_next is None:
            self._t_next = t0
        self._buf = np.concatenate([self._buf, x])
        out: list[E2Event] = []
        while len(self._buf) >= self.n:
            frame, self._buf = self._buf[: self.n], self._buf[self.n:]
            ev = self._frame(self._t_next, frame)
            self._t_next += self.cfg.frame_s
            if ev is not None:
                out.append(ev)
        return out

    # ----------------------------------------------------------------- internals
    def _own_speaker(self, t: float) -> bool:
        p = self.cfg.mute_pad_s
        return any(a - p <= t <= b + p for a, b in self.mutes)

    def _flatness(self, frame: np.ndarray) -> float:
        spec = np.abs(np.fft.rfft(frame.astype(np.float64) * self._win)) ** 2
        spec = spec[1:] + 1e-18
        return float(np.exp(np.mean(np.log(spec))) / np.mean(spec))

    def _update_health(self, t: float, db: float, peak_abs: float) -> None:
        cfg = self.cfg
        self._clip.append(peak_abs >= cfg.clip_level)
        if db < cfg.dead_db:
            if self._dead_since is None:
                self._dead_since = t
        else:
            self._dead_since = None
        if self._dead_since is not None and t - self._dead_since >= cfg.dead_s:
            self.health = "mic_dead"
        elif len(self._clip) == self._clip.maxlen and sum(self._clip) / len(self._clip) >= cfg.clip_frac:
            self.health = "clipping"
        else:
            self.health = "ok"

    def _frame(self, t: float, frame: np.ndarray) -> Optional[E2Event]:
        cfg = self.cfg
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        db = 20.0 * math.log10(rms + 1e-9)
        self._update_health(t, db, float(np.max(np.abs(frame))))
        self._frames_seen += 1
        warm = self._frames_seen * cfg.frame_s >= cfg.warmup_s and len(self._hist) >= int(cfg.warmup_s / cfg.frame_s)
        base = float(np.median(self._hist)) if self._hist else db
        prev = self._prev_db
        self._prev_db = db

        if self.health != "ok":
            self._cand = None
            self._hist.clear()           # relearn the background after a fault
            self._frames_seen = 0
            return None

        c = self._cand
        if c is None:
            self._hist.append(db)        # background only learns while no candidate is active
            if t < self._quiet_until or not warm or prev is None:
                return None
            if db - base >= cfg.rise_db and db - prev >= cfg.onset_step_db and db >= cfg.min_db:
                if self._own_speaker(t):
                    self.rejections.append((t, "own_speaker"))
                    self._quiet_until = t + cfg.refractory_s
                    return None
                self._cand = {"t0": t, "peak": db, "base": base, "flat": [self._flatness(frame)]}
            return None

        # candidate in progress
        c["peak"] = max(c["peak"], db)
        c["flat"].append(self._flatness(frame))
        width = t - c["t0"]
        if db <= max(c["peak"] - cfg.decay_db, c["base"] + cfg.end_margin_db):
            self._cand = None
            self._quiet_until = t + cfg.refractory_s
            flat = float(np.median(c["flat"]))
            if flat < cfg.min_flatness:
                self.rejections.append((c["t0"], "tonal"))
                return None
            if self._own_speaker(t) or self._own_speaker(c["t0"]):
                self.rejections.append((c["t0"], "own_speaker"))
                return None
            rise = c["peak"] - c["base"]
            score = (0.4 * _clamp01((rise - cfg.rise_db) / 25.0)
                     + 0.3 * _clamp01(flat / 0.4)
                     + 0.3 * _clamp01((c["peak"] - cfg.min_db) / 30.0))
            score = _clamp01(score)
            return E2Event(t_onset=c["t0"], t_fire=t, peak_db=round(c["peak"], 2), rise_db=round(rise, 2),
                           width_s=round(width, 3), flatness=round(flat, 3),
                           score=round(score, 4), score255=int(round(score * 255)))
        if width > cfg.max_width_s:
            self.rejections.append((c["t0"], "too_long"))
            self._cand = None
            self._quiet_until = t + cfg.refractory_s
        return None
