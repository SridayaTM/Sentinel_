#!/usr/bin/env python3
"""CM4 detector benchmark (test P1).

Runs YOLO + ByteTrack on a video file or the live camera for N minutes and logs,
every 10 seconds: frames per second, inference time, CPU, memory, temperature and
throttling. Writes bench_<tag>.csv and prints a summary at the end.

Setup on the CM4 (64-bit Raspberry Pi OS, fan on):
    python3 -m venv venv && . venv/bin/activate
    pip install ultralytics psutil ncnn
    yolo export model=yolo11n.pt format=ncnn imgsz=320     # makes yolo11n_ncnn_model/

Runs (30 minutes each, nothing else running):
    python bench.py --weights yolo11n_ncnn_model --imgsz 320 --source road.mp4 --tag ncnn320
    python bench.py --weights yolo11n.pt          --imgsz 320 --source road.mp4 --tag torch320
    python bench.py --weights yolo11n_ncnn_model --imgsz 320 --source 0        --tag cam320
Short test first:  add  --minutes 1

NOTE: written without access to a CM4. Check the first minute of output before
trusting a long run.
"""

import argparse
import csv
import os
import statistics
import subprocess
import time

import psutil
from ultralytics import YOLO

# COCO ids: person, bicycle, car, motorcycle, bus, truck
CLASSES = [0, 1, 2, 3, 5, 7]


def temp_c():
    try:
        out = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True, text=True, timeout=2).stdout
        return float(out.split("=")[1].split("'")[0])
    except Exception:
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                return int(f.read().strip()) / 1000.0
        except Exception:
            return float("nan")


def throttle_bits():
    """vcgencmd get_throttled -> int. 0x1 undervoltage now, 0x4 throttled now,
    0x10000 undervoltage happened, 0x40000 throttling happened."""
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=2).stdout
        return int(out.strip().split("=")[1], 16)
    except Exception:
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--source", default="road.mp4", help="video file, or 0 for the camera")
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--stride", type=int, default=1, help="video files only: use every Nth frame")
    ap.add_argument("--no-track", action="store_true")
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    if isinstance(source, str) and not os.path.exists(source):
        raise SystemExit(f"source not found: {source}")

    model = YOLO(args.weights, task="detect")
    csv_path = f"bench_{args.tag}.csv"
    t_start = time.time()
    t_end = t_start + args.minutes * 60
    win_start = time.time()
    win_frames = 0
    win_inf = []
    fps_rows, temps, inf_all = [], [], []
    seen_bits = 0
    psutil.cpu_percent(None)  # prime

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["elapsed_s", "fps", "inference_ms_median", "cpu_pct", "mem_mb", "temp_c", "throttled_hex"])
        while time.time() < t_end:
            kw = dict(source=source, stream=True, imgsz=args.imgsz, conf=args.conf,
                      classes=CLASSES, verbose=False)
            if isinstance(source, str):
                kw["vid_stride"] = args.stride
            if args.no_track:
                gen = model.predict(**kw)
            else:
                gen = model.track(tracker="bytetrack.yaml", persist=True, **kw)
            for r in gen:
                win_frames += 1
                inf = r.speed.get("inference", 0.0)
                win_inf.append(inf)
                inf_all.append(inf)
                now = time.time()
                if now - win_start >= 10:
                    fps = win_frames / (now - win_start)
                    tc = temp_c()
                    bits = throttle_bits()
                    seen_bits |= max(bits, 0)
                    row = [round(now - t_start), round(fps, 2), round(statistics.median(win_inf), 1),
                           psutil.cpu_percent(None), round(psutil.Process().memory_info().rss / 1e6),
                           tc, hex(bits)]
                    w.writerow(row)
                    f.flush()
                    print(row, flush=True)
                    fps_rows.append(fps)
                    temps.append(tc)
                    win_start, win_frames, win_inf = now, 0, []
                if now >= t_end:
                    break

    if not fps_rows:
        raise SystemExit("no full 10-second window completed; run longer")
    fps_sorted = sorted(fps_rows)
    p5 = fps_sorted[max(0, int(len(fps_sorted) * 0.05) - 0)]
    print("\n===== SUMMARY", args.tag, "=====")
    print(f"windows           : {len(fps_rows)}")
    print(f"fps median        : {statistics.median(fps_rows):.2f}")
    print(f"fps slowest 5%    : {p5:.2f}")
    print(f"inference ms med  : {statistics.median(inf_all):.1f}")
    print(f"max temp C        : {max(t for t in temps if t == t):.1f}" if any(t == t for t in temps) else "max temp C        : n/a")
    print(f"undervoltage seen : {bool(seen_bits & 0x10001)}")
    print(f"throttling seen   : {bool(seen_bits & 0x40004)}")
    print(f"csv               : {csv_path}")


if __name__ == "__main__":
    main()
