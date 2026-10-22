"""E1: vision crash evidence, built from tracked boxes (no CNN inside).

Input per frame: tracked detections (track id, class name, box x1 y1 x2 y2) and a
capture timestamp in seconds. Output: crash evidence events.

The rule follows the evidence policy in the Veganodi v2 documents:
  two tracks, each steady for several frames before contact, with no ID swap
  -> their ground footprints touch
  -> sharp loss of speed in the vehicle(s) that were moving
  -> both stationary inside one zone for a set time.
  Rejected if either track was lost or new at the moment of contact, or if the
  camera is out of calibration.  Two-wheeler fall rule: a sudden, sustained change in
  box shape, a sharp stop, then no motion.

Speeds are in object-sizes per second (size = sqrt(box width x height)), so the
rule does not depend on image resolution or on a homography.

ALL THRESHOLDS BELOW ARE PLACEHOLDERS. They must be tuned on a tuning set of clips and
then frozen before any test set is opened.  The score is a heuristic (SCORE_VERSION 1);
change the version whenever the thresholds or the formula change.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Callable, Optional

SCORE_VERSION = 1
Box = tuple[float, float, float, float]
Det = tuple[int, str, Box]  # (track id, class name, box)


@dataclass(frozen=True)
class E1Config:
    min_track_frames: int = 6       # steady frames needed before contact
    max_gap_s: float = 0.35         # a track unseen for longer than this is lost, and restarts
    speed_window: int = 3           # frames used to smooth speed
    v_move: float = 0.6             # sizes/s: at or above this the vehicle is "moving"
    v_stop: float = 0.15            # sizes/s: below this it is "stationary"
    contact_gap: float = 0.05       # footprint gap, as a fraction of the smaller object, that counts as contact
    footprint_frac: float = 0.4     # bottom share of the box used as the ground footprint
    drop_ratio: float = 0.6         # share of pre-contact speed that must be lost
    decel_window_s: float = 1.5     # time after contact in which the drop must happen
    stationary_s: float = 2.0       # both must stay still this long
    max_wait_s: float = 8.0         # give up on a contact that never settles
    jump_ratio: float = 1.5         # one-frame centre jump (in sizes) that signals an ID swap
    v_max: float = 6.0              # sizes/s: fastest believable motion; a jump is only suspicious beyond v_max * frame interval
    pre_window_s: float = 1.0       # look-back for pre-contact speed and for jump checks
    ar_change: float = 1.6          # two-wheeler: box aspect ratio must change by this factor
    ar_hold_s: float = 0.4          # ...and stay changed this long
    refractory_s: float = 10.0      # one crash, one event: suppress events near a recent one
    merge_dist: float = 4.0         # sizes: "near" for suppression
    same_object_iou: float = 0.6    # two tracks whose boxes overlap this much are one vehicle seen twice
    two_wheeler_classes: tuple[str, ...] = ("two_wheeler",)


@dataclass(frozen=True)
class E1Event:
    kind: str                       # "collision" or "two_wheeler_fall"
    t_contact: float
    t_fire: float
    track_ids: tuple[int, ...]
    classes: tuple[str, ...]
    at: tuple[float, float]         # contact point in pixels (midpoint of the two tracks)
    score: float                    # 0..1
    score255: int                   # packet field
    score_version: int = SCORE_VERSION
    headings_deg: tuple[float, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass
class _Sample:
    t: float
    cx: float
    cy: float
    w: float
    h: float
    speed: float


class _Track:
    def __init__(self, tid: int, cls: str):
        self.tid, self.cls = tid, cls
        self.samples: list[_Sample] = []
        self.cont = 0
        self.last_jump_t: Optional[float] = None
        self.last_seen_t = -math.inf
        self.box: Box = (0, 0, 0, 0)

    @property
    def last(self) -> _Sample:
        return self.samples[-1]

    def size(self) -> float:
        recent = self.samples[-10:]
        return statistics.median(math.sqrt(max(s.w * s.h, 1e-9)) for s in recent)

    def add(self, t: float, box: Box, cfg: E1Config, gap_s: Optional[float] = None) -> None:
        gap_s = cfg.max_gap_s if gap_s is None else gap_s
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        size = max(math.sqrt(max(w * h, 1e-9)), 1e-6)
        prev = self.samples[-1] if self.samples else None
        if prev is None or t - prev.t > gap_s:
            self.cont = 1
        else:
            self.cont += 1
            if math.hypot(cx - prev.cx, cy - prev.cy) > max(cfg.jump_ratio, cfg.v_max * (t - prev.t)) * size:
                self.last_jump_t = t
        speed = 0.0
        k = min(cfg.speed_window, self.cont - 1)
        if k >= 1:
            ref = self.samples[-k]
            dt = t - ref.t
            if dt > 1e-9:
                speed = math.hypot(cx - ref.cx, cy - ref.cy) / dt / size
        self.samples.append(_Sample(t, cx, cy, w, h, speed))
        if len(self.samples) > 240:
            del self.samples[:-240]
        self.last_seen_t = t
        self.box = box


@dataclass
class _Pair:
    state: str                      # "pending" | "rejected" | "done"
    t_contact: float = 0.0
    pre: tuple[float, float] = (0.0, 0.0)
    at: tuple[float, float] = (0.0, 0.0)
    headings: tuple[float, float] = (0.0, 0.0)
    contact_strength: float = 1.0
    last_contact_t: float = 0.0
    still_since: Optional[float] = None
    quality: float = 1.0


def _footprint(box: Box, frac: float) -> Box:
    x1, y1, x2, y2 = box
    return (x1, y2 - frac * (y2 - y1), x2, y2)


def _gap(a: Box, b: Box) -> float:
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return math.hypot(dx, dy)


def _overlap_area(a: Box, b: Box) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _iou(a: Box, b: Box) -> float:
    inter = _overlap_area(a, b)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


class E1Detector:
    def __init__(self, cfg: E1Config = E1Config()):
        self.cfg = cfg
        self.tracks: dict[int, _Track] = {}
        self.pairs: dict[tuple[int, int], _Pair] = {}
        self.rejections: list[tuple[float, tuple[int, ...], str]] = []   # for tuning and tests
        self.suppressed: list[E1Event] = []
        self.disabled = False
        self._recent: list[E1Event] = []
        self._fall: dict[int, dict] = {}
        self._dts: list[float] = []
        self._last_t: Optional[float] = None
        self._gap_s = cfg.max_gap_s

    # ------------------------------------------------------------------ public
    def update(
        self,
        t: float,
        dets: list[Det],
        calibration_ok: bool = True,
        zone_of: Optional[Callable[[float, float], object]] = None,
    ) -> list[E1Event]:
        cfg = self.cfg
        zone_of = zone_of or (lambda x, y: 0)
        if not calibration_ok:
            # Pixel geometry is not trustworthy: E1 is off and forgets everything it saw.
            self.disabled = True
            self.tracks.clear()
            self.pairs.clear()
            self._fall.clear()
            return []
        self.disabled = False

        # the 'track lost' limit follows the real frame interval, so slow frame rates still work
        if self._last_t is not None and t > self._last_t:
            self._dts.append(t - self._last_t)
            del self._dts[:-20]
            self._gap_s = max(cfg.max_gap_s, 2.5 * statistics.median(self._dts))
        self._last_t = t

        present: list[_Track] = []
        for tid, cls, box in dets:
            tr = self.tracks.get(tid)
            if tr is None:
                tr = self.tracks[tid] = _Track(tid, cls)
            tr.add(t, box, cfg, self._gap_s)
            present.append(tr)
        for tid in [k for k, tr in self.tracks.items() if t - tr.last_seen_t > 5.0]:
            del self.tracks[tid]
            self._fall.pop(tid, None)
        for key in [k for k in self.pairs if k[0] not in self.tracks or k[1] not in self.tracks]:
            del self.pairs[key]

        events: list[E1Event] = []
        events += self._collisions(t, present, zone_of)
        events += self._falls(t, present)
        out = []
        for ev in events:
            if self._is_duplicate(ev):
                self.suppressed.append(ev)
            else:
                self._recent.append(ev)
                out.append(ev)
        self._recent = [e for e in self._recent if t - e.t_fire <= cfg.refractory_s]
        return out

    # ------------------------------------------------------------- collisions
    def _in_contact(self, a: _Track, b: _Track) -> bool:
        cfg = self.cfg
        fa, fb = _footprint(a.box, cfg.footprint_frac), _footprint(b.box, cfg.footprint_frac)
        small = min(a.size(), b.size())
        return _gap(fa, fb) <= cfg.contact_gap * small

    def _pre_speed(self, tr: _Track, t: float) -> float:
        lo = t - self.cfg.pre_window_s
        run = tr.samples[-tr.cont:]
        vals = [s.speed for s in run if s.t >= lo]
        return statistics.median(vals) if vals else 0.0

    def _heading(self, tr: _Track, t: float) -> float:
        run = [s for s in tr.samples[-tr.cont:] if s.t >= t - self.cfg.pre_window_s]
        if len(run) < 2:
            return 0.0
        return math.degrees(math.atan2(run[-1].cy - run[0].cy, run[-1].cx - run[0].cx))

    def _collisions(self, t: float, present: list[_Track], zone_of) -> list[E1Event]:
        cfg = self.cfg
        events: list[E1Event] = []
        seen_now = {tr.tid for tr in present}
        contact_now: set[tuple[int, int]] = set()

        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                a, b = (present[i], present[j]) if present[i].tid < present[j].tid else (present[j], present[i])
                key = (a.tid, b.tid)
                if key not in self.pairs and _iou(a.box, b.box) >= cfg.same_object_iou:
                    if not self.rejections or self.rejections[-1][1:] != (key, "duplicate_box"):
                        self.rejections.append((t, key, "duplicate_box"))
                    continue                     # one vehicle detected twice (e.g. as bus and truck)
                if not self._in_contact(a, b):
                    continue
                contact_now.add(key)
                p = self.pairs.get(key)
                if p is not None:
                    p.last_contact_t = t
                    continue
                self.pairs[key] = self._evaluate_contact(t, a, b)

        # forget finished contacts so a later, separate contact is judged afresh
        for key, p in list(self.pairs.items()):
            if p.state in ("rejected", "done") and key not in contact_now and t - p.last_contact_t > 1.0:
                del self.pairs[key]

        for key, p in list(self.pairs.items()):
            if p.state != "pending":
                continue
            a, b = self.tracks[key[0]], self.tracks[key[1]]
            age = t - p.t_contact
            if age > cfg.max_wait_s:
                self._reject(t, key, "no_stop", drop=True)
                continue
            if t - a.last_seen_t > self._gap_s or t - b.last_seen_t > self._gap_s:
                self._reject(t, key, "lost_after_contact", drop=True)
                continue
            if a.tid not in seen_now or b.tid not in seen_now:
                continue  # not updated this frame; wait, the gap limit above decides

            decel_ok, drop_obs = self._decel(a, b, p)
            if age >= cfg.decel_window_s and not decel_ok:
                self._reject(t, key, "no_sharp_decel", drop=True)
                continue

            still_now = (a.last.speed < cfg.v_stop and b.last.speed < cfg.v_stop
                         and zone_of(a.last.cx, a.last.cy) == zone_of(b.last.cx, b.last.cy))
            if still_now:
                if p.still_since is None:
                    p.still_since = t
            else:
                p.still_since = None

            if p.still_since is not None and t - p.still_since >= cfg.stationary_s and decel_ok:
                score = 0.30 * p.contact_strength + 0.40 * _clamp01(drop_obs) + 0.30 * p.quality
                score = _clamp01(score)
                p.state = "done"
                p.last_contact_t = t
                events.append(E1Event(
                    kind="collision", t_contact=p.t_contact, t_fire=t,
                    track_ids=key, classes=(a.cls, b.cls), at=p.at,
                    score=round(score, 4), score255=int(round(score * 255)),
                    headings_deg=p.headings,
                ))
        return events

    def _evaluate_contact(self, t: float, a: _Track, b: _Track) -> _Pair:
        cfg = self.cfg
        key = (a.tid, b.tid)
        p = _Pair(state="rejected", last_contact_t=t)
        if a.cont < cfg.min_track_frames or b.cont < cfg.min_track_frames:
            self.rejections.append((t, key, "track_not_steady"))
            return p
        for tr in (a, b):
            if tr.last_jump_t is not None and t - tr.last_jump_t <= cfg.pre_window_s:
                self.rejections.append((t, key, "id_swap_suspected"))
                return p
        pre = (self._pre_speed(a, t), self._pre_speed(b, t))
        if max(pre) < cfg.v_move:
            self.rejections.append((t, key, "no_moving_vehicle"))
            return p
        quality = 1.0
        if min(a.cont, b.cont) < 2 * cfg.min_track_frames:
            quality -= 0.3
        fa, fb = _footprint(a.box, cfg.footprint_frac), _footprint(b.box, cfg.footprint_frac)
        p.state = "pending"
        p.t_contact = t
        p.pre = pre
        p.at = ((a.last.cx + b.last.cx) / 2, (a.last.cy + b.last.cy) / 2)
        p.headings = (self._heading(a, t), self._heading(b, t))
        p.contact_strength = 1.0 if _overlap_area(fa, fb) > 0 else 0.6
        p.quality = quality
        return p

    def _decel(self, a: _Track, b: _Track, p: _Pair) -> tuple[bool, float]:
        cfg = self.cfg
        drops = []
        for tr, pre in ((a, p.pre[0]), (b, p.pre[1])):
            if pre < cfg.v_move:
                continue  # was not moving: nothing to lose
            window = [s.speed for s in tr.samples
                      if p.t_contact <= s.t <= p.t_contact + cfg.decel_window_s]
            if not window:
                return False, 0.0
            drops.append(1.0 - min(window) / pre)
        if not drops:
            return False, 0.0
        return all(d >= cfg.drop_ratio for d in drops), sum(drops) / len(drops)

    def _reject(self, t: float, key: tuple[int, ...], reason: str, drop: bool) -> None:
        self.rejections.append((t, key, reason))
        if drop and key in self.pairs:
            p = self.pairs[key]
            p.state = "rejected"
            p.last_contact_t = t

    # ----------------------------------------------------------- two-wheelers
    def _falls(self, t: float, present: list[_Track]) -> list[E1Event]:
        cfg = self.cfg
        events: list[E1Event] = []
        for tr in present:
            if tr.cls not in cfg.two_wheeler_classes:
                continue
            st = self._fall.get(tr.tid)
            run = tr.samples[-tr.cont:]
            if st is None:
                recent = [s for s in run if s.t >= t - cfg.ar_hold_s]
                base = [s for s in run if t - 3.0 <= s.t <= t - cfg.ar_hold_s - 0.2]
                if len(base) < 4 or len(recent) < 2 or tr.cont < cfg.min_track_frames:
                    continue
                ar_r = statistics.median(s.w / max(s.h, 1e-6) for s in recent)
                ar_b = statistics.median(s.w / max(s.h, 1e-6) for s in base)
                factor = max(ar_r / ar_b, ar_b / ar_r)
                pre = statistics.median(s.speed for s in base)
                if factor >= cfg.ar_change and pre >= cfg.v_move:
                    self._fall[tr.tid] = {"state": "pending", "t_change": t - cfg.ar_hold_s,
                                          "pre": pre, "still_since": None, "factor": factor}
                continue
            if st["state"] != "pending":
                continue
            age = t - st["t_change"]
            if age > cfg.max_wait_s:
                st["state"] = "done"
                self.rejections.append((t, (tr.tid,), "fall_no_stop"))
                continue
            window = [s.speed for s in run
                      if st["t_change"] <= s.t <= st["t_change"] + cfg.decel_window_s + cfg.ar_hold_s]
            drop = 1.0 - min(window) / st["pre"] if window else 0.0
            if age >= cfg.decel_window_s + cfg.ar_hold_s and drop < cfg.drop_ratio:
                st["state"] = "done"
                self.rejections.append((t, (tr.tid,), "fall_no_sharp_decel"))
                continue
            if tr.last.speed < cfg.v_stop:
                if st["still_since"] is None:
                    st["still_since"] = t
            else:
                st["still_since"] = None
            if st["still_since"] is not None and t - st["still_since"] >= cfg.stationary_s and drop >= cfg.drop_ratio:
                st["state"] = "done"
                quality = 1.0 if tr.cont >= 2 * cfg.min_track_frames else 0.7
                score = min(0.6, 0.35 + 0.25 * _clamp01(drop)) * quality
                events.append(E1Event(
                    kind="two_wheeler_fall", t_contact=st["t_change"], t_fire=t,
                    track_ids=(tr.tid,), classes=(tr.cls,), at=(tr.last.cx, tr.last.cy),
                    score=round(score, 4), score255=int(round(score * 255)),
                    headings_deg=(self._heading(tr, t),),
                    notes=("single-track rule: weaker evidence, capped score",),
                ))
        return events

    # ------------------------------------------------------------ suppression
    def _is_duplicate(self, ev: E1Event) -> bool:
        cfg = self.cfg
        for old in self._recent:
            if ev.t_fire - old.t_fire > cfg.refractory_s:
                continue
            size = max(self._size_of(ev.track_ids), self._size_of(old.track_ids), 1.0)
            if math.hypot(ev.at[0] - old.at[0], ev.at[1] - old.at[1]) <= cfg.merge_dist * size:
                return True
        return False

    def _size_of(self, ids: tuple[int, ...]) -> float:
        sizes = [self.tracks[i].size() for i in ids if i in self.tracks and self.tracks[i].samples]
        return max(sizes) if sizes else 0.0
