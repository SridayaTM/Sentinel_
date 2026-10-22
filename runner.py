"""Vision runner: ties the detector/tracker output to E1 (crash) and E2 (impact sound) and writes events.

The pipeline core (`Pipeline`) takes already-tracked boxes, so it is testable without a model or a camera.
`run_video` is the thin wrapper that feeds it from a video file or camera using Ultralytics + ByteTrack.

Event line (one JSON object per line, schema version 1):
  {"v":1,"src":"E1","kind":"collision"|"two_wheeler_fall","t":<capture s>,"t_contact":..,"score":..,"score255":..,
   "at":[x,y],"zone":..,"track_ids":[..],"classes":[..],"score_version":1}
  {"v":1,"src":"E2","kind":"impact","t":<onset s>,"t_fire":..,"score":..,"score255":..,"peak_db":..,"score_version":1}
  {"v":1,"src":"HEALTH","kind":"camera_stalled"|"camera_back"|"mic_<state>"|"mic_ok","t":..}

Times are capture times (seconds on one clock). Keeping the camera and the microphone on the same clock is the
caller's job; this module never reads the wall clock when processing frames.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from typing import Callable, Iterable, Optional

from camera_check import CameraHealth
from e1 import E1Config, E1Detector, E1Event
from e2 import E2Config, E2Detector, E2Event

SCHEMA_VERSION = 1


def _e1_dict(ev: E1Event, zone) -> dict:
    return {"v": SCHEMA_VERSION, "src": "E1", "kind": ev.kind, "t": round(ev.t_fire, 3),
            "t_contact": round(ev.t_contact, 3), "score": ev.score, "score255": ev.score255,
            "at": [round(ev.at[0], 1), round(ev.at[1], 1)], "zone": zone,
            "track_ids": list(ev.track_ids), "classes": list(ev.classes), "score_version": ev.score_version}


def _e2_dict(ev: E2Event) -> dict:
    return {"v": SCHEMA_VERSION, "src": "E2", "kind": "impact", "t": round(ev.t_onset, 3),
            "t_fire": round(ev.t_fire, 3), "score": ev.score, "score255": ev.score255,
            "peak_db": ev.peak_db, "score_version": ev.score_version}


class Pipeline:
    def __init__(
        self,
        e1: Optional[E1Detector] = None,
        e2: Optional[E2Detector] = None,
        zone_of: Optional[Callable[[float, float], object]] = None,
        calibration_ok: Callable[[float], bool] = lambda t: True,
        stall_s: float = 1.0,
        camera: Optional[CameraHealth] = None,
    ):
        self.e1 = e1 or E1Detector(E1Config())
        self.e2 = e2
        self.zone_of = zone_of or (lambda x, y: 0)
        self.calibration_ok = calibration_ok
        self.stall_s = stall_s
        self.camera = camera
        self._last_frame_t: Optional[float] = None
        self._stalled = False
        self._mic_state = "ok"
        self.frames = 0

    def on_frame(self, t: float, tracked: Iterable[tuple], frame=None) -> list[dict]:
        out: list[dict] = []
        if self._last_frame_t is not None:
            if t < self._last_frame_t:
                return out                            # out-of-order frame: ignore, never go back in time
            if t - self._last_frame_t > self.stall_s:
                out.append({"v": SCHEMA_VERSION, "src": "HEALTH", "kind": "camera_stalled",
                            "t": round(self._last_frame_t, 3), "gap_s": round(t - self._last_frame_t, 3)})
                out.append({"v": SCHEMA_VERSION, "src": "HEALTH", "kind": "camera_back", "t": round(t, 3)})
        self._last_frame_t = t
        self.frames += 1
        cam_ok = True
        if self.camera is not None and frame is not None:
            for tt, state in self.camera.update(t, frame):
                if state == "ok":
                    out.append({"v": SCHEMA_VERSION, "src": "HEALTH", "kind": "camera_ok", "t": round(tt, 3)})
                elif state != "warming":
                    out.append({"v": SCHEMA_VERSION, "src": "HEALTH", "kind": "camera_fault", "reason": state,
                                "t": round(tt, 3)})
            cam_ok = self.camera.ok
        elif self.camera is not None:
            cam_ok = self.camera.ok
        for ev in self.e1.update(t, list(tracked), calibration_ok=self.calibration_ok(t) and cam_ok,
                                 zone_of=self.zone_of):
            out.append(_e1_dict(ev, self.zone_of(*ev.at)))
        return out

    def on_audio(self, t0: float, samples) -> list[dict]:
        out: list[dict] = []
        if self.e2 is None:
            return out
        for ev in self.e2.feed(t0, samples):
            out.append(_e2_dict(ev))
        if self.e2.health != self._mic_state:
            self._mic_state = self.e2.health
            kind = "mic_ok" if self._mic_state == "ok" else f"mic_{self._mic_state}"
            out.append({"v": SCHEMA_VERSION, "src": "HEALTH", "kind": kind, "t": round(t0, 3)})
        return out


def run_video(weights: str, source, out_path: str, imgsz: int = 480, conf: float = 0.3,
              stride: int = 1, class_names: Optional[list[str]] = None, max_seconds: Optional[float] = None) -> int:
    """Feed a video file (or camera index) through detector + ByteTrack + Pipeline. Returns number of events.

    For a video file, capture time = frame_index / fps. For a live camera the caller should supply real capture
    timestamps (not done here); that part is untested without hardware.
    """
    import cv2
    from ultralytics import YOLO

    is_file = isinstance(source, str) and not str(source).isdigit()
    fps = 25.0
    if is_file:
        cap = cv2.VideoCapture(source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
    model = YOLO(weights, task="detect")
    names = class_names or [model.names[i] for i in sorted(model.names)]
    pipe = Pipeline(camera=CameraHealth())
    n_events = 0
    n_dets = 0
    max_tracks = 0
    seen_ids: set = set()
    import time as _time
    t_wall = _time.time()
    with open(out_path, "w") as f:
        k = 0
        for r in model.track(source=int(source) if str(source).isdigit() else source, stream=True, imgsz=imgsz,
                             conf=conf, tracker="bytetrack.yaml", persist=True, verbose=False, vid_stride=stride,
                             agnostic_nms=True):
            t = (k * stride) / fps
            k += 1
            if max_seconds is not None and t > max_seconds:
                break
            tracked = []
            if r.boxes is not None and r.boxes.id is not None:
                for tid, c, box in zip(r.boxes.id.int().tolist(), r.boxes.cls.int().tolist(), r.boxes.xyxy.tolist()):
                    tracked.append((tid, names[c], tuple(box)))
            n_dets += len(tracked)
            max_tracks = max(max_tracks, len(tracked))
            seen_ids.update(x[0] for x in tracked)
            for ev in pipe.on_frame(t, tracked, frame=r.orig_img):
                f.write(json.dumps(ev) + "\n")
                f.flush()
                n_events += 1
                print("EVENT", json.dumps(ev), file=sys.stderr, flush=True)
            if k % 100 == 0:
                print(f"frame {k}  video {t:.1f}s  tracks now {len(tracked)}  speed {k / max(_time.time() - t_wall, 1e-9):.1f} fps",
                      file=sys.stderr, flush=True)
    wall = max(_time.time() - t_wall, 1e-9)
    print(f"SUMMARY frames={k} detections={n_dets} distinct_tracks={len(seen_ids)} max_tracks_in_a_frame={max_tracks} "
          f"events={n_events} processing_fps={k / wall:.1f}", file=sys.stderr, flush=True)
    return n_events


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="events.jsonl")
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-seconds", type=float, default=None, help="stop after this much video time")
    a = ap.parse_args()
    n = run_video(a.weights, a.source, a.out, a.imgsz, a.conf, a.stride, max_seconds=a.max_seconds)
    print(f"{n} events -> {a.out}", file=sys.stderr)
