"""Tests for the E2 impact-sound rule on synthetic audio."""

from __future__ import annotations

import numpy as np
import pytest

from e2 import E2Config, E2Detector

SR = 16000


def rng(seed=0):
    return np.random.default_rng(seed)


def background(seconds, level_db=-50.0, seed=0):
    """Road hum: low-pass-ish noise at a steady level."""
    r = rng(seed)
    x = r.standard_normal(int(seconds * SR))
    x = np.convolve(x, np.ones(8) / 8, mode="same")
    x /= np.sqrt(np.mean(x ** 2))
    return (x * 10 ** (level_db / 20)).astype(np.float32)


def add(sig, start_s, burst):
    i = int(start_s * SR)
    sig = sig.copy()
    sig[i:i + len(burst)] += burst[: len(sig) - i]
    return sig


def impact(level_db=-15.0, decay_s=0.08, dur=0.6, seed=1):
    r = rng(seed)
    n = int(dur * SR)
    t = np.arange(n) / SR
    noise = r.standard_normal(n)
    env = np.exp(-t / decay_s)
    x = noise * env + 0.8 * np.sin(2 * np.pi * 70 * t) * env
    x /= np.max(np.abs(x))
    return (x * 10 ** (level_db / 20)).astype(np.float32)


def tone(freq=420.0, level_db=-15.0, dur=1.2):
    t = np.arange(int(dur * SR)) / SR
    x = np.sin(2 * np.pi * freq * t) + 0.5 * np.sin(2 * np.pi * 2 * freq * t)
    x /= np.max(np.abs(x))
    return (x * 10 ** (level_db / 20)).astype(np.float32)


def run(sig, chunk=1000, mutes=(), cfg=None):
    det = E2Detector(cfg or E2Config())
    for a, b in mutes:
        det.mute(a, b)
    ev = []
    for i in range(0, len(sig), chunk):
        ev += det.feed(i / SR, sig[i:i + chunk])
    return det, ev


def test_clear_impact_fires_once_with_a_sane_event():
    sig = add(background(8), 4.0, impact())
    det, ev = run(sig)
    assert len(ev) == 1
    e = ev[0]
    assert 3.95 < e.t_onset < 4.1
    assert e.t_fire > e.t_onset and e.width_s <= 0.5
    assert e.rise_db >= 15 and 0.0 <= e.score <= 1.0 and e.score255 == round(e.score * 255)
    assert e.score > 0.6


def test_chunk_size_does_not_change_the_result():
    sig = add(background(8), 4.0, impact())
    _, a = run(sig, chunk=137)
    _, b = run(sig, chunk=4000)
    _, c = run(sig, chunk=320)
    assert a == b == c and len(a) == 1


def test_car_horn_is_rejected():
    sig = add(background(8), 4.0, tone())
    det, ev = run(sig)
    assert ev == [] and "too_long" in [r for _, r in det.rejections]


def test_short_tonal_beep_is_rejected_as_tonal():
    beep = tone(dur=0.12)
    beep *= np.exp(-np.arange(len(beep)) / SR / 0.02).astype(np.float32)   # quick decay so it passes the width rule
    sig = add(background(8), 4.0, beep)
    det, ev = run(sig)
    assert ev == [] and "tonal" in [r for _, r in det.rejections]


def test_slow_loud_passing_vehicle_is_rejected_by_the_step_rule():
    n = int(8 * SR)
    ramp = np.linspace(0, 1, n) ** 2
    sig = background(8) + (background(8, level_db=-12, seed=5) * ramp).astype(np.float32)
    det, ev = run(sig)
    assert ev == []


def test_steady_loud_engine_does_not_fire():
    sig = background(8, level_db=-25)
    det, ev = run(sig)
    assert ev == []


def test_quiet_click_below_the_absolute_floor_is_rejected():
    sig = add(background(8, level_db=-75), 4.0, impact(level_db=-40))   # 25 dB over a silent background, but too quiet
    det, ev = run(sig)
    assert ev == []


def test_no_event_during_warmup():
    sig = add(background(3), 0.3, impact())
    det, ev = run(sig)
    assert ev == []


def test_own_speaker_window_is_ignored():
    sig = add(background(8), 4.0, impact())
    det, ev = run(sig, mutes=[(3.9, 4.8)])
    assert ev == [] and "own_speaker" in [r for _, r in det.rejections]
    det, ev = run(sig)                   # same audio, no mute: fires
    assert len(ev) == 1


def test_dead_microphone_reports_fault():
    det, ev = run(np.zeros(int(8 * SR), dtype=np.float32))
    assert det.health == "mic_dead" and ev == []


def test_impact_right_after_a_dead_period_is_not_trusted_until_relearned():
    sig = add(np.zeros(int(8 * SR), dtype=np.float32), 6.0, impact())
    det, ev = run(sig)
    assert ev == []


def test_clipping_microphone_reports_fault():
    r = rng(3)
    sig = np.sign(r.standard_normal(int(6 * SR))).astype(np.float32)      # rails all the time
    det, ev = run(sig)
    assert det.health == "clipping" and ev == []


def test_recovers_after_a_fault_and_relearns_the_background():
    dead = np.zeros(int(5 * SR), dtype=np.float32)
    live = add(background(8), 5.0, impact())
    det, ev = run(np.concatenate([dead, live]))
    assert det.health == "ok" and len(ev) == 1
    assert ev[0].t_onset > 5.0 + 4.9          # onset at 10.0 s, not before


def test_two_impacts_far_apart_give_two_events_and_close_ones_one():
    sig = add(add(background(14), 4.0, impact()), 9.0, impact(seed=2))
    _, ev = run(sig)
    assert len(ev) == 2
    sig = add(add(background(10), 4.0, impact()), 4.4, impact(seed=2))
    _, ev = run(sig)
    assert len(ev) == 1


def test_louder_background_needs_a_louder_impact():
    sig = add(background(8, level_db=-35), 4.0, impact(level_db=-15))     # only ~10 dB over the hum
    _, ev = run(sig)
    assert ev == []
    sig = add(background(8, level_db=-35), 4.0, impact(level_db=-3))
    _, ev = run(sig)
    assert len(ev) == 1


def test_deterministic():
    sig = add(background(8), 4.0, impact())
    assert run(sig)[1] == run(sig)[1]


def test_empty_and_tiny_chunks_are_harmless():
    det = E2Detector()
    assert det.feed(0.0, []) == []
    assert det.feed(0.0, np.zeros(5, dtype=np.float32)) == []


def test_dip_in_the_background_then_recovery_is_not_an_impact():
    sig = background(8, level_db=-30)
    sig[int(4.0 * SR):int(4.02 * SR)] *= 0.02          # one frame of near silence, then back to normal
    det, ev = run(sig)
    assert ev == []


def test_fast_swell_then_drop_without_a_sharp_onset_is_rejected():
    n = int(0.55 * SR)
    t = np.arange(n) / SR
    env_db = np.where(t < 0.4, 30.0 * t / 0.4, 30.0 * (1 - (t - 0.4) / 0.1))
    env = 10 ** (np.clip(env_db, 0, None) / 20) * 10 ** (-60 / 20)
    swell = (rng(9).standard_normal(n) * env * 10).astype(np.float32)
    sig = add(background(8, level_db=-50), 4.0, swell)
    det, ev = run(sig)
    assert ev == []


def test_weak_impact_just_over_the_threshold_still_ends_cleanly():
    sig = add(background(8, level_db=-40), 4.0, impact(level_db=-14))
    det, ev = run(sig)
    assert len(ev) == 1 and ev[0].rise_db < 20


def test_own_speaker_started_mid_event_is_ignored():
    sig = add(background(8), 4.0, impact())
    det, ev = run(sig, mutes=[(4.4, 5.0)])
    assert ev == []


def test_impact_followed_by_a_lingering_rumble_still_ends():
    sig = add(background(8, level_db=-40), 4.0, impact(level_db=-14))
    sig = add(sig, 4.0, background(1.5, level_db=-40, seed=4))      # tail keeps the level ~3 dB above the hum
    det, ev = run(sig)
    assert len(ev) == 1
