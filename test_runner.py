"""Tests for the runner core (no model, no camera)."""

from __future__ import annotations

import json

import numpy as np

from e2 import E2Detector
from runner import Pipeline
from test_e1 import head_on, ramp, path, Veh
from test_e2 import SR, add, background, impact


def frames(vehicles, seconds, fps=10):
    for k in range(int(seconds * fps)):
        t = k / fps
        tracked = []
        for v in vehicles:
            cx, cy = v.x(t), v.cy
            w, h = v.size(t)
            tracked.append((v.ident, v.cls, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
        yield t, tracked


def test_crash_becomes_one_json_line_with_all_fields():
    a, b = head_on()
    p = Pipeline()
    out = []
    for t, tr in frames([a, b], 8):
        out += p.on_frame(t, tr)
    crashes = [e for e in out if e["src"] == "E1"]
    assert len(crashes) == 1
    e = json.loads(json.dumps(crashes[0]))                       # must survive JSON
    assert e["v"] == 1 and e["kind"] == "collision" and 0 <= e["score255"] <= 255
    assert set(e["classes"]) == {"car"} and len(e["at"]) == 2 and "zone" in e and e["t"] > e["t_contact"]


def test_calibration_callback_switches_e1_off():
    a, b = head_on()
    p = Pipeline(calibration_ok=lambda t: False)
    out = []
    for t, tr in frames([a, b], 8):
        out += p.on_frame(t, tr)
    assert out == []


def test_camera_stall_is_reported_with_the_gap():
    p = Pipeline()
    p.on_frame(0.0, [])
    p.on_frame(0.1, [])
    out = p.on_frame(5.1, [])
    assert [e["kind"] for e in out] == ["camera_stalled", "camera_back"]
    assert out[0]["gap_s"] == 5.0


def test_out_of_order_frame_is_ignored():
    p = Pipeline()
    p.on_frame(1.0, [])
    assert p.on_frame(0.5, []) == []
    assert p.frames == 1


def test_audio_impact_and_mic_health_events():
    p = Pipeline(e2=E2Detector())
    sig = add(background(8), 4.0, impact())
    out = []
    for i in range(0, len(sig), 1600):
        out += p.on_audio(i / SR, sig[i:i + 1600])
    imp = [e for e in out if e["src"] == "E2"]
    assert len(imp) == 1 and imp[0]["kind"] == "impact" and 3.9 < imp[0]["t"] < 4.2
    out = p.on_audio(8.0, np.zeros(5 * SR, dtype=np.float32))
    assert any(e["kind"] == "mic_mic_dead" or e["kind"] == "mic_dead" for e in out)


def test_audio_without_e2_is_a_noop():
    assert Pipeline().on_audio(0.0, np.zeros(100, dtype=np.float32)) == []


def test_same_input_same_output():
    def go():
        a, b = head_on()
        p = Pipeline()
        o = []
        for t, tr in frames([a, b], 8):
            o += p.on_frame(t, tr)
        return o
    assert go() == go()
