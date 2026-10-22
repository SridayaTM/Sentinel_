"""Tests for the E1 crash rule, on synthetic tracks. Every scene is a small physical story:
positions are integrated from speed profiles, boxes are cut from positions.  No GPU, no camera."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import pytest

from e1 import E1Config, E1Detector, E1Event


# --------------------------------------------------------------------- harness
def ramp(v0: float, t_start: float, dur: float, v1: float = 0.0) -> Callable[[float], float]:
    """Constant v0 until t_start, linear to v1 over dur, then v1."""
    def v(t: float) -> float:
        if t <= t_start:
            return v0
        if t >= t_start + dur:
            return v1
        return v0 + (v1 - v0) * (t - t_start) / dur
    return v


def path(x0: float, speed: Callable[[float], float], dt: float = 0.001, tmax: float = 40.0):
    n = int(tmax / dt) + 1
    xs = [x0]
    for k in range(1, n):
        xs.append(xs[-1] + speed((k - 1) * dt) * dt)

    def pos(t: float) -> float:
        k = min(max(int(round(t / dt)), 0), n - 1)
        return xs[k]
    return pos


@dataclass
class Veh:
    ident: int
    cls: str
    x: Callable[[float], float]
    cy: float = 300.0
    size: Callable[[float], tuple[float, float]] = lambda t: (120.0, 70.0)
    visible: Callable[[float], bool] = lambda t: True
    id_at: Optional[Callable[[float], int]] = None
    x_offset: Callable[[float], float] = lambda t: 0.0


def run(vehicles, seconds, fps=10, jitter=0.0, zone_of=None, calib=lambda t: True, cfg=None):
    det = E1Detector(cfg or E1Config())
    events: list[E1Event] = []
    for k in range(int(seconds * fps)):
        t = k / fps + (jitter * (1 if k % 2 else -1))
        dets = []
        for v in vehicles:
            if not v.visible(t):
                continue
            cx, cy = v.x(t) + v.x_offset(t), v.cy
            w, h = v.size(t)
            dets.append((v.id_at(t) if v.id_at else v.ident, v.cls,
                         (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
        events += det.update(t, dets, calibration_ok=calib(t), zone_of=zone_of)
    return det, events


def head_on(x0a=100.0, x0b=900.0, t_hit=1.1, **kw):
    """Two cars drive at each other at 300 px/s (3.3 car-sizes/s) and stop within 0.3 s."""
    a = Veh(1, "car", path(x0a, ramp(300, t_hit, 0.3)), **kw)
    b = Veh(2, "car", path(x0b, ramp(-300, t_hit, 0.3)), **kw)
    return a, b


def reasons(det):
    return [r for _, _, r in det.rejections]


# ------------------------------------------------------------------ real crashes
def test_head_on_crash_fires_exactly_once_with_sane_event():
    a, b = head_on()
    det, ev = run([a, b], 8)
    assert len(ev) == 1
    e = ev[0]
    assert e.kind == "collision" and set(e.track_ids) == {1, 2}
    assert 1.0 < e.t_contact < 1.5            # contact when the boxes meet
    assert e.t_fire - e.t_contact > 2.0       # waited for the stationary period
    assert 440 < e.at[0] < 560 and e.at[1] == pytest.approx(300)
    assert 0.0 <= e.score <= 1.0 and e.score255 == round(e.score * 255)
    assert e.score >= 0.7                     # a clean, textbook crash scores high
    assert {round(h) for h in e.headings_deg} == {0, 180}


def test_rear_end_into_a_parked_car_fires():
    a = Veh(1, "car", path(100, ramp(300, 1.55, 0.3)))
    b = Veh(2, "car", path(700, lambda t: 0.0))
    det, ev = run([a, b], 8)
    assert len(ev) == 1 and set(ev[0].track_ids) == {1, 2}


def test_rear_end_where_the_struck_car_is_shoved_then_stops_fires():
    shove = lambda t: 100.0 if 1.6 <= t < 1.9 else 0.0
    a = Veh(1, "car", path(100, ramp(300, 1.55, 0.3)))
    b = Veh(2, "car", path(700, shove))
    det, ev = run([a, b], 9)
    assert len(ev) == 1


def test_works_at_5_fps_with_timestamp_jitter():
    a, b = head_on(x0a=-400, x0b=1300, t_hit=2.6)
    _, ev5 = run([a, b], 12, fps=5)
    assert len(ev5) == 1
    a, b = head_on()
    _, evj = run([a, b], 8, fps=10, jitter=0.012)
    assert len(evj) == 1


def test_one_crash_one_event_even_when_a_third_car_piles_in():
    a, b = head_on()
    c = Veh(3, "car", path(-300, ramp(300, 2.15, 0.3)))
    det, ev = run([a, b, c], 12)
    assert len(ev) == 1
    assert len(det.suppressed) >= 1           # the pile-up pair was seen and merged, not lost


def test_event_not_repeated_over_a_long_run():
    a, b = head_on()
    _, ev = run([a, b], 25)
    assert len(ev) == 1


# ----------------------------------------------------------- hard negatives (T13)
def test_hard_braking_that_stops_short_is_not_a_crash():
    a = Veh(1, "car", path(100, ramp(300, 1.0, 0.5)))
    b = Veh(2, "car", path(615, lambda t: 0.0))      # A ends 20 px short of B
    det, ev = run([a, b], 8)
    assert ev == []


def test_overlap_in_the_image_from_depth_is_not_contact():
    # A is near the camera (lower in the image), B is far (higher, smaller). Both brake to a stop
    # with their image boxes overlapping, but their ground footprints are far apart in depth.
    a = Veh(1, "car", path(100, ramp(300, 0.9, 0.3)), cy=320, size=lambda t: (120, 70))
    b = Veh(2, "car", path(300, ramp(100, 0.9, 0.3)), cy=270, size=lambda t: (70, 40))
    det, ev = run([a, b], 10)
    assert ev == []
    # premise: at rest the full boxes overlap, so a box-based rule would have fired
    ax, bx = a.x(9.0), b.x(9.0)
    box_a = (ax - 60, 285, ax + 60, 355)
    box_b = (bx - 35, 250, bx + 35, 290)
    ox = min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])
    oy = min(box_a[3], box_b[3]) - max(box_a[1], box_b[1])
    assert ox > 0 and oy > 0


def test_stopped_queue_with_touching_boxes_is_not_a_crash():
    a = Veh(1, "car", path(300, lambda t: 0.0))
    b = Veh(2, "car", path(400, lambda t: 0.0))      # footprints overlap, nobody ever moved
    det, ev = run([a, b], 8)
    assert ev == []


def test_gentle_creep_into_a_stopped_car_is_below_the_moving_threshold():
    a = Veh(1, "car", path(100, ramp(27, 99, 1)))     # 0.3 sizes/s, under v_move
    b = Veh(2, "car", path(700, lambda t: 0.0))
    det, ev = run([a, b], 30)
    assert ev == [] and "no_moving_vehicle" in reasons(det)


def test_sideswipe_that_keeps_driving_is_not_a_crash():
    a = Veh(1, "car", path(100, ramp(300, 99, 1)))
    b = Veh(2, "car", path(300, ramp(250, 99, 1)))
    det, ev = run([a, b], 14)
    assert ev == [] and "no_sharp_decel" in reasons(det)


def test_tap_and_go_where_one_car_drives_off_before_the_stationary_time():
    a_speed = lambda t: (300 if t < 1.1 else (300 * (1 - (t - 1.1) / 0.3) if t < 1.4 else (0.0 if t < 2.4 else 300.0)))
    a = Veh(1, "car", path(100, a_speed))
    b = Veh(2, "car", path(700, lambda t: 0.0))
    det, ev = run([a, b], 10)
    assert ev == []


# ----------------------------------------------------------------- track quality
def test_track_hidden_at_contact_is_rejected():
    a, b = head_on()
    b.visible = lambda t: not (1.0 <= t < 1.7)
    det, ev = run([a, b], 8)
    assert ev == [] and "track_not_steady" in reasons(det)


def test_track_id_swap_at_contact_is_rejected():
    a, b = head_on()
    b.id_at = lambda t: 2 if t < 1.15 else 5
    det, ev = run([a, b], 8)
    assert ev == []


def test_position_jump_before_contact_is_rejected_as_id_swap():
    a, b = head_on(x0b=1100.0)
    b.x_offset = lambda t: -200.0 if t >= 0.9 else 0.0
    det, ev = run([a, b], 9)
    assert ev == [] and "id_swap_suspected" in reasons(det)


def test_track_lost_after_contact_gives_no_event():
    a, b = head_on()
    a.visible = lambda t: t < 1.6
    det, ev = run([a, b], 9)
    assert ev == [] and "lost_after_contact" in reasons(det)


def test_stationary_in_different_zones_gives_no_event():
    a, b = head_on()
    det, ev = run([a, b], 9, zone_of=lambda x, y: 0 if x < 500 else 1)
    assert ev == []
    det, ev = run([a, b], 9, zone_of=lambda x, y: 0)           # same zone: fires
    assert len(ev) == 1


# ------------------------------------------------------------------ calibration
def test_invalid_calibration_turns_e1_off_for_the_whole_scene():
    a, b = head_on()
    det, ev = run([a, b], 8, calib=lambda t: False)
    assert ev == [] and det.disabled


def test_calibration_lost_mid_event_cancels_it_and_it_does_not_come_back():
    a, b = head_on()
    det, ev = run([a, b], 10, calib=lambda t: not (2.0 <= t < 4.0))
    assert ev == []


# ---------------------------------------------------------------- two-wheelers
def bike(turn_speed_after: Callable[[float], float], flat_after: float = 3.0):
    size = lambda t: (80.0, 70.0) if t < flat_after else (100.0, 40.0)   # upright then flat: aspect 1.1 -> 2.5
    return Veh(7, "two_wheeler", path(100, turn_speed_after), size=size)


def test_two_wheeler_fall_fires_with_a_capped_score():
    det, ev = run([bike(ramp(250, 3.0, 0.4))], 10)
    assert len(ev) == 1
    e = ev[0]
    assert e.kind == "two_wheeler_fall" and e.track_ids == (7,)
    assert e.score <= 0.6
    a, b = head_on()
    _, crash = run([a, b], 8)
    assert e.score < crash[0].score                      # single-track evidence is weaker


def test_two_wheeler_that_changes_shape_but_keeps_riding_is_not_a_fall():
    det, ev = run([bike(lambda t: 250.0)], 12)
    assert ev == [] and "fall_no_sharp_decel" in reasons(det)


def test_two_wheeler_that_turns_and_slows_gradually_is_not_a_fall():
    det, ev = run([bike(ramp(250, 3.0, 6.0))], 16)
    assert ev == []


def test_a_car_never_triggers_the_two_wheeler_rule():
    car = Veh(1, "car", path(100, ramp(250, 3.0, 0.4)),
              size=lambda t: (80.0, 70.0) if t < 3.0 else (100.0, 40.0))
    det, ev = run([car], 10)
    assert ev == []


# ----------------------------------------------------------------- determinism
def test_same_input_gives_identical_events():
    a, b = head_on()
    _, e1 = run([a, b], 8)
    a, b = head_on()
    _, e2 = run([a, b], 8)
    assert e1 == e2


def test_empty_and_single_vehicle_frames_are_harmless():
    det = E1Detector()
    assert det.update(0.0, []) == []
    assert det.update(0.1, [(1, "car", (0, 0, 100, 60))]) == []
    assert det.update(0.2, [(1, "car", (0, 0, 100, 60)), (1, "car", (0, 0, 100, 60))]) == []


def test_one_vehicle_detected_twice_is_not_a_crash():
    # the same car braking to a stop, reported as two tracks with slightly different boxes (bus + truck)
    sp = ramp(300, 1.1, 0.3)
    a = Veh(1, "bus", path(100, sp), size=lambda t: (120.0, 70.0))
    b = Veh(2, "truck", path(104, sp), size=lambda t: (114.0, 66.0))
    det, ev = run([a, b], 9)
    assert ev == [] and "duplicate_box" in reasons(det)
