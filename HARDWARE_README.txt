SENTINEL VISION - HAND-OFF FOR THE HARDWARE TEAM
=================================================

WHAT THIS IS
  A small vehicle detector (YOLO11n, 5 classes: car, bus, truck, three_wheeler, two_wheeler)
  trained on Indian road images (UVH-26). Exports are in sentinel_vision_export.zip:
    nano_best.pt            PyTorch weights
    nano_ncnn_320 / nano_ncnn_480   NCNN models (use these on the CM4)
    nano_onnx_320.onnx / nano_onnx_480.onnx   ONNX models (use these on the Omen)
    report.json             accuracy numbers (measured on a held-out set, NOT on a pole camera)

CM4 (64-bit Raspberry Pi OS, fan on, good power supply)
  1. python3 -m venv venv && . venv/bin/activate
  2. pip install ultralytics psutil ncnn lap
     (lap is needed by the tracker. If it fails to install, try: pip install lapx)
     Also check: vcgencmd measure_temp   (must work, otherwise temperature/throttling show as UNKNOWN)
  3. Copy bench.py, a road video (road.mp4) and the nano_ncnn_480 folder to the CM4.
  4. SHORT TEST FIRST (1 minute):
       python bench.py --weights nano_ncnn_480 --imgsz 480 --source road.mp4 --tag ncnn480 --minutes 1
  5. If it runs, the real test (30 minutes):
       python bench.py --weights nano_ncnn_480 --imgsz 480 --source road.mp4 --tag ncnn480
     Also try 320:
       python bench.py --weights nano_ncnn_320 --imgsz 320 --source road.mp4 --tag ncnn320
  6. SEND BACK: the printed SUMMARY block and the bench_<tag>.csv files.
     Important numbers: fps median, fps slowest 5%, max temp, "undervoltage seen", "throttling seen".

TARGET
  At least 8 frames per second sustained with no throttling. If 480 cannot hold it, use 320.

OMEN LAPTOP
  pip install ultralytics onnxruntime
  python bench.py --weights nano_onnx_480.onnx --imgsz 480 --source road.mp4 --tag onnx480 --minutes 1

NOTES
  - Nothing here has been measured on a CM4 yet. These are the first real numbers.
  - The vehicle detector is only one part. The crash rule (e1.py), sound rule (e2.py) and runner (runner.py)
    run on top of it and are tested separately.
  - Do not edit thresholds in e1.py / e2.py.
