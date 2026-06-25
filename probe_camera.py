from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vitals_cam.camera import AUTO_CAMERA, list_cameras, measure_camera_fps, parse_camera_spec, probe_camera


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe webcam brightness and FPS for rPPG/PPG use.")
    parser.add_argument("--camera", default=AUTO_CAMERA, help="Camera index, or auto/best/-1.")
    parser.add_argument("--max-index", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    args = parser.parse_args()

    camera = parse_camera_spec(args.camera)
    if camera == AUTO_CAMERA:
        probes = list_cameras(args.max_index, width=args.width, height=args.height)
        usable = False
        for result in probes:
            usable = usable or result.usable
            status = "OK" if result.usable else "WARN" if result.opened else "NO"
            print(f"{result.index}: {status} - {result.detail()}")
        selected, frames, fps = measure_camera_fps(AUTO_CAMERA, args.seconds, args.width, args.height)
        print(f"selected={selected} frames={frames} fps={fps:.2f}")
        return 0 if usable and fps >= 10.0 else 2

    result = probe_camera(int(camera), width=args.width, height=args.height, seconds=args.seconds)
    status = "OK" if result.usable else "WARN" if result.opened else "NO"
    print(f"{result.index}: {status} - {result.detail()}")
    return 0 if result.usable else 2


if __name__ == "__main__":
    raise SystemExit(main())
