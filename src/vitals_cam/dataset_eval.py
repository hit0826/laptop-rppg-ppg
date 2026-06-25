from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .app import compute_estimates
from .roi import FaceRoiDetector, finger_roi


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate rPPG/PPG estimates on an offline video.")
    parser.add_argument("--video", required=True, help="Path to a video file.")
    parser.add_argument("--mode", choices=["face", "finger"], default="face")
    parser.add_argument("--gt-bpm", type=float, default=float("nan"), help="Optional ground-truth BPM for error reporting.")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--no-mediapipe", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return 2

    timestamps, rgb = extract_trace(video_path, args.mode, not args.no_mediapipe, args.max_frames)
    if timestamps.size < 90:
        print(f"Not enough usable samples: {timestamps.size}")
        return 3
    estimate = compute_estimates(timestamps, rgb, args.mode, 45.0, 180.0)
    hr = estimate["heart_rate"]
    print(f"samples: {timestamps.size}")
    print(f"estimated_hr_bpm: {hr['value']:.2f}")
    print(f"quality: {hr['quality']:.2f}")
    print(f"status: {hr['status']}")
    if np.isfinite(args.gt_bpm):
        print(f"gt_bpm: {args.gt_bpm:.2f}")
        print(f"abs_error_bpm: {abs(hr['value'] - args.gt_bpm):.2f}")
    return 0


def extract_trace(
    video_path: Path,
    mode: str,
    use_mediapipe: bool,
    max_frames: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    detector = FaceRoiDetector(use_mediapipe=use_mediapipe)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    timestamps = []
    rgb_samples = []
    frame_idx = 0
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            roi = detector.detect(frame) if mode == "face" else finger_roi(frame)
            if roi.ok:
                timestamps.append(frame_idx / fps)
                rgb_samples.append(roi.rgb)
            frame_idx += 1
            if max_frames and frame_idx >= max_frames:
                break
    finally:
        detector.close()
        cap.release()
    return np.asarray(timestamps, dtype=float), np.asarray(rgb_samples, dtype=float)


if __name__ == "__main__":
    raise SystemExit(main())

