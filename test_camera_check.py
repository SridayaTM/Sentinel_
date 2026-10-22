"""Tests for the camera self-check on synthetic pictures (no camera needed)."""

from __future__ import annotations

import numpy as np

from camera_check import CamConfig, CameraHealth

W, H = 320, 180


def scene(seed=0, w=W, h=H):
    """A road-like picture: smooth gradient plus lots of small structure (edges, markings)."""
    r = np.random.default_rng(seed)
    base = r.random((h // 4, w // 4)).astype(np.float32)
    big = np.kron(base, np.ones((4, 4), dtype=np.float32))
    img = 60 + 140 * big
    img[h // 2:, :] += 15
    for x in range(0, w, 24):
        img[:, x:x + 2] += 40
    return np.clip(img, 0, 255)


def blur(img, k=15):
    ker = np.ones(k) / k
    out = np.apply_along_axis(lambda m: np.convolve(m, ker, mode="same"), 1, img)
    return np.apply_along_axis(lambda m: np.convolve(m, ker, mode="same"), 0, out)


def traffic(img, t, rng):
    """moving blobs + sensor noise so live frames always differ"""
    f = img.copy()
    for k in range(4):
        x = int((40 * t + 70 * k) % (W - 40))
        y = 40 + 30 * k
        f[y:y + 20, x:x + 40] = 30 + 20 * k
    f += rng.normal(0, 1.5, f.shape)
    return np.clip(f, 0, 255)


def run(frames_fn, seconds, fps=10, cam=None):
    cam = cam or CameraHealth()
    rng = np.random.default_rng(1)
    states = []
    for k in range(int(seconds * fps)):
        t = k / fps
        cam.update(t, frames_fn(t, rng))
        states.append((t, cam.state))
    return cam, states


def state_at(states, t):
    return [s for tt, s in states if tt <= t][-1]


BASE = scene()


def test_normal_traffic_stays_ok_after_warmup_with_no_flicker():
    cam, st = run(lambda t, r: traffic(BASE, t, r), 40)
    assert state_at(st, 5.0) == "ok"
    assert all(s == "ok" for t, s in st if t > 4.0)
    assert [x for _, x in cam.transitions] == ["ok"]


def test_black_lens_cover_is_reported_after_the_debounce_and_clears():
    def f(t, r):
        return np.zeros((H, W)) + r.normal(0, 0.3, (H, W)) if 10 <= t < 16 else traffic(BASE, t, r)
    cam, st = run(f, 30)
    assert state_at(st, 10.5) == "ok"            # not instant: debounce
    assert state_at(st, 13.0) == "covered"
    assert state_at(st, 20.0) == "ok"            # recovered after good_s


def test_white_cover_is_also_covered():
    cam, st = run(lambda t, r: np.full((H, W), 240.0) + r.normal(0, 0.3, (H, W)) if t >= 8 else traffic(BASE, t, r), 16)
    assert state_at(st, 14.0) == "covered"


def test_dark_but_detailed_scene_is_dark_not_covered():
    dark = BASE * 0.12
    cam, st = run(lambda t, r: np.clip(traffic(dark, t, r), 0, 255) if t >= 6 else traffic(BASE, t, r), 16)
    assert state_at(st, 14.0) == "dark"


def test_blur_is_detected_and_a_sharp_scene_is_not_blurred():
    # a smeared lens blurs everything in the picture, vehicles included
    cam, st = run(lambda t, r: blur(traffic(BASE, t, r), 21) if t >= 6 else traffic(BASE, t, r), 16)
    assert state_at(st, 14.0) == "blurred"
    cam2, st2 = run(lambda t, r: traffic(BASE, t, r), 16)
    assert all(s != "blurred" for _, s in st2)


def test_frozen_stream_is_detected_but_a_noisy_static_scene_is_not():
    held = traffic(BASE, 5.0, np.random.default_rng(5))
    cam, st = run(lambda t, r: held if t >= 6 else traffic(BASE, t, r), 14)
    assert state_at(st, 10.5) == "frozen"
    cam2, st2 = run(lambda t, r: np.clip(BASE + r.normal(0, 1.5, BASE.shape), 0, 255), 14)
    assert all(s != "frozen" for _, s in st2)


def test_shifted_view_is_moved_small_jitter_is_not_and_moving_back_recovers():
    def shifted(img, dx, dy):
        return np.roll(np.roll(img, dx, axis=1), dy, axis=0)
    cam, st = run(lambda t, r: traffic(shifted(BASE, 12, 6), t, r) if 8 <= t < 18 else traffic(BASE, t, r), 30)
    assert state_at(st, 7.0) == "ok"
    assert state_at(st, 15.0) == "moved"
    assert state_at(st, 26.0) == "ok"
    cam2, st2 = run(lambda t, r: traffic(shifted(BASE, 1, 0), t, r) if t >= 8 else traffic(BASE, t, r), 24)
    assert all(s != "moved" for _, s in st2)


def test_one_second_glitch_does_not_flip_the_state():
    cam, st = run(lambda t, r: np.zeros((H, W)) + r.normal(0, 0.3, (H, W)) if 10 <= t < 11 else traffic(BASE, t, r), 24)
    assert all(s != "covered" for _, s in st)


def test_same_scene_at_other_resolutions_gives_same_verdicts():
    def make(w, h):
        b = scene(w=w, h=h)
        def f(t, r):
            return np.zeros((h, w)) + r.normal(0, 0.3, (h, w)) if t >= 8 else b + r.normal(0, 1.5, b.shape)
        return f
    for (w, h) in ((320, 180), (640, 360), (1280, 720)):
        cam, st = run(make(w, h), 16, fps=5)
        assert state_at(st, 5.0) == "ok", (w, h)
        assert state_at(st, 14.0) == "covered", (w, h)


def test_rebaseline_accepts_a_new_view():
    def shifted(img, dx, dy):
        return np.roll(np.roll(img, dx, axis=1), dy, axis=0)
    cam = CameraHealth()
    rng = np.random.default_rng(2)
    for k in range(200):
        cam.update(k / 10, traffic(BASE, k / 10, rng))
    assert cam.ok
    cam.rebaseline()
    assert cam.state == "warming"
    for k in range(200, 500):
        cam.update(k / 10, traffic(shifted(BASE, 12, 6), k / 10, rng))
    assert cam.ok


def test_colour_frames_are_accepted():
    rgb = np.stack([BASE, BASE, BASE], axis=2)
    cam = CameraHealth()
    for k in range(60):
        cam.update(k / 10, rgb + np.random.default_rng(k).normal(0, 1.5, rgb.shape))
    assert cam.ok


def test_recovery_needs_a_clear_stretch_so_a_flapping_fault_stays_a_fault():
    # covered 10-16 s, clear for only 0.8 s, covered again: must not report ok in between
    def f(t, r):
        covered = (10 <= t < 16) or (16.8 <= t < 22)
        return np.zeros((H, W)) + r.normal(0, 0.3, (H, W)) if covered else traffic(BASE, t, r)
    cam, st = run(f, 34)
    assert state_at(st, 13.0) == "covered"
    assert state_at(st, 16.5) == "covered"
    assert state_at(st, 21.0) == "covered"
    assert state_at(st, 30.0) == "ok"


def test_small_shift_is_ignored_and_big_shift_is_found_at_high_resolution():
    big = scene(w=1280, h=720)

    def frame(img, dx, r):
        return np.clip(np.roll(img, dx, axis=1) + r.normal(0, 1.5, img.shape), 0, 255)
    cam, st = run(lambda t, r: frame(big, 3, r) if t >= 8 else frame(big, 0, r), 20, fps=5)
    assert all(s != "moved" for _, s in st)          # 3 px of 1280 is nothing
    cam, st = run(lambda t, r: frame(big, 48, r) if t >= 8 else frame(big, 0, r), 20, fps=5)
    assert state_at(st, 18.0) == "moved"
