from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .camera import AUTO_CAMERA, open_selected_camera
from .roi import FaceRoiDetector, finger_roi
from .session import Sample, SessionRecorder
from .signal_processing import (
    PPG_ALGORITHMS,
    RPPG_ALGORITHMS,
    estimate_bpm_from_ppg_rgb,
    estimate_bpm_from_rgb,
    estimate_hrv_rmssd,
    estimate_respiration_rate,
    estimate_spo2_experimental,
    finger_ppg,
    pos_rppg,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local webcam rPPG/PPG vital-sign estimator.")
    parser.add_argument("--mode", choices=["face", "finger"], default="face")
    parser.add_argument("--camera", default=AUTO_CAMERA, help="Camera index, or auto/best.")
    parser.add_argument("--max-auto-camera-index", type=int, default=4)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--window-sec", type=float, default=24.0)
    parser.add_argument("--min-bpm", type=float, default=45.0)
    parser.add_argument("--max-bpm", type=float, default=180.0)
    parser.add_argument("--rppg-algorithm", choices=RPPG_ALGORITHMS, default="auto")
    parser.add_argument("--ppg-algorithm", choices=PPG_ALGORITHMS, default="auto")
    parser.add_argument("--save-dir", default="sessions")
    parser.add_argument("--no-mediapipe", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--duration", type=float, default=0.0, help="Optional auto-stop duration in seconds.")
    parser.add_argument("--headless", action="store_true", help="Run without an OpenCV window. Uses 10s if no duration is set.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run(args)


def run(args: argparse.Namespace) -> int:
    cap, selected_camera, probes = open_selected_camera(
        args.camera,
        args.width,
        args.height,
        30,
        max_auto_index=args.max_auto_camera_index,
    )
    _print_camera_selection(args.camera, selected_camera, probes)
    if cap is None or selected_camera is None or not cap.isOpened():
        print(f"Could not open camera {args.camera}.")
        return 2

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    detector = FaceRoiDetector(use_mediapipe=not args.no_mediapipe)
    recorder = SessionRecorder()
    recorder.recording = bool(args.record)
    mode = args.mode
    started = time.perf_counter()
    last_estimate = 0.0
    estimate = {}
    if args.headless and args.duration <= 0:
        args.duration = 10.0
    timestamps: deque[float] = deque()
    rgb_samples: deque[tuple[float, float, float]] = deque()
    last_saved = ""
    win_name = "Laptop rPPG/PPG Vital Signs"

    try:
        while True:
            ok, frame = cap.read()
            now = time.perf_counter()
            if not ok or frame is None:
                print("Camera returned no frame.")
                return 3

            roi = detector.detect(frame) if mode == "face" else finger_roi(frame)
            recorder.add(Sample(now, mode, roi.rgb[0], roi.rgb[1], roi.rgb[2], roi.ok, roi.quality_hint))
            if roi.ok:
                timestamps.append(now)
                rgb_samples.append(roi.rgb)

            while timestamps and now - timestamps[0] > args.window_sec:
                timestamps.popleft()
                rgb_samples.popleft()

            if now - last_estimate > 1.0 and len(timestamps) >= 90:
                estimate = compute_estimates(
                    np.array(timestamps),
                    np.array(rgb_samples),
                    mode,
                    args.min_bpm,
                    args.max_bpm,
                    args.rppg_algorithm,
                    args.ppg_algorithm,
                )
                recorder.summary = estimate
                last_estimate = now

            key = 255
            if not args.headless:
                display = draw_ui(
                    frame,
                    roi.rects,
                    mode,
                    recorder.recording,
                    timestamps,
                    rgb_samples,
                    estimate,
                    roi.status,
                    last_saved,
                    roi.tracking_rect,
                )
                cv2.imshow(win_name, display)
                key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("1"):
                mode = "face"
                timestamps.clear()
                rgb_samples.clear()
                estimate = {}
            elif key == ord("2"):
                mode = "finger"
                timestamps.clear()
                rgb_samples.clear()
                estimate = {}
            elif key == ord("r"):
                recorder.recording = not recorder.recording
            elif key == ord("s"):
                csv_path, json_path = recorder.save(Path(args.save_dir), estimate)
                last_saved = f"saved {csv_path.name}, {json_path.name}"

            if args.duration > 0 and now - started >= args.duration:
                break
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()

    if recorder.samples and recorder.recording:
        csv_path, json_path = recorder.save(Path(args.save_dir), estimate)
        print(f"Saved session: {csv_path} / {json_path}")
    if args.headless:
        print(json.dumps({"mode": mode, "camera": selected_camera, "samples": len(timestamps), "estimate": estimate}, indent=2))
    return 0


def compute_estimates(
    timestamps: np.ndarray,
    rgb: np.ndarray,
    mode: str,
    min_bpm: float,
    max_bpm: float,
    rppg_algorithm: str = "auto",
    ppg_algorithm: str = "auto",
) -> dict[str, object]:
    if mode == "face":
        hr = estimate_bpm_from_rgb(rgb, timestamps, method=rppg_algorithm, min_bpm=min_bpm, max_bpm=max_bpm)
        pulse = pos_rppg(rgb)
        respiration_source = rgb[:, 1]
    else:
        hr = estimate_bpm_from_ppg_rgb(rgb, timestamps, method=ppg_algorithm, min_bpm=min_bpm, max_bpm=max_bpm)
        pulse = finger_ppg(rgb, method=hr.method or "green")
        respiration_source = rgb[:, 1]

    rr = estimate_respiration_rate(respiration_source, timestamps)
    hrv = estimate_hrv_rmssd(pulse, timestamps, hr.value)
    spo2, spo2_status = estimate_spo2_experimental(rgb) if mode == "finger" else (float("nan"), "face_mode_disabled")
    return {
        "heart_rate": asdict(hr),
        "respiration_rate": asdict(rr),
        "hrv_rmssd": asdict(hrv),
        "spo2_experimental": {"value": spo2, "status": spo2_status},
    }


def draw_ui(
    frame: np.ndarray,
    rects: list[tuple[int, int, int, int]],
    mode: str,
    recording: bool,
    timestamps: deque[float],
    rgb_samples: deque[tuple[float, float, float]],
    estimate: dict[str, object],
    roi_status: str,
    last_saved: str,
    tracking_rect: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    display = frame.copy()
    panel_h = 138
    panel_top = display.shape[0] - panel_h
    if mode == "face" and tracking_rect is not None:
        x, y, w, h = tracking_rect
        cv2.rectangle(display, (x, y), (x + w, y + h), (35, 35, 255), 3)
        hr = estimate.get("heart_rate", {}) if estimate else {}
        label = _face_box_label(hr)
        if label:
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            preferred_y = y + h + 22
            if preferred_y + baseline + 4 >= panel_top:
                label_y = max(label_size[1] + baseline + 4, y - 8)
            else:
                label_y = preferred_y
            bg_y0 = max(0, label_y - label_size[1] - baseline - 4)
            bg_y1 = min(display.shape[0] - 1, label_y + baseline + 4)
            bg_x1 = min(display.shape[1] - 1, x + label_size[0] + 12)
            cv2.rectangle(display, (x, bg_y0), (bg_x1, bg_y1), (245, 245, 245), -1)
            cv2.putText(display, label, (x + 6, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (35, 35, 255), 2, cv2.LINE_AA)

    if mode != "face" or tracking_rect is None:
        for rect in rects:
            x, y, w, h = rect
            cv2.rectangle(display, (x, y), (x + w, y + h), (60, 220, 90), 2)

    h, w = display.shape[:2]
    cv2.rectangle(display, (0, h - panel_h), (w, h), (15, 15, 15), -1)
    draw_waveform(display, timestamps, rgb_samples, (220, h - panel_h + 60, w - 235, 64), mode)

    lines = [
        f"mode: {mode}   samples: {len(timestamps)}   roi: {roi_status}",
        "keys: 1 face  2 finger  r record  s save  q quit",
        f"recording: {'on' if recording else 'off'}",
    ]
    hr = estimate.get("heart_rate", {}) if estimate else {}
    rr = estimate.get("respiration_rate", {}) if estimate else {}
    hrv = estimate.get("hrv_rmssd", {}) if estimate else {}
    spo2 = estimate.get("spo2_experimental", {}) if estimate else {}
    if hr:
        method = hr.get("method") or "--"
        lines.append(
            f"HR: {_fmt(hr.get('value'))} bpm  quality: {_fmt(hr.get('quality'))}  "
            f"status: {hr.get('status')}  method: {method}"
        )
    if rr:
        lines.append(f"Resp: {_fmt(rr.get('value'))} br/min  quality: {_fmt(rr.get('quality'))}")
    if hrv:
        lines.append(f"RMSSD: {_fmt(hrv.get('rmssd_ms'))} ms  beats: {hrv.get('beats')}  {hrv.get('status')}")
    if mode == "finger" and spo2:
        lines.append(f"SpO2 exp: {_fmt(spo2.get('value'))}%  {spo2.get('status')}")
    if last_saved:
        lines.append(last_saved)

    y = h - panel_h + 22
    for idx, line in enumerate(lines[:7]):
        color = (220, 220, 220)
        if idx == 2 and recording:
            color = (80, 80, 255)
        cv2.putText(display, line, (12, y + 18 * idx), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    return display


def draw_waveform(
    image: np.ndarray,
    timestamps: deque[float],
    rgb_samples: deque[tuple[float, float, float]],
    rect: tuple[int, int, int, int],
    mode: str,
) -> None:
    x, y, w, h = rect
    cv2.rectangle(image, (x, y), (x + w, y + h), (45, 45, 45), 1)
    if len(timestamps) < 5:
        return
    rgb = np.array(rgb_samples)
    pulse = pos_rppg(rgb) if mode == "face" else finger_ppg(rgb)
    if pulse.size < 2 or np.std(pulse) < 1e-6:
        return
    values = pulse[-min(pulse.size, 240) :]
    values = (values - np.min(values)) / (np.ptp(values) + 1e-9)
    xs = np.linspace(x + 2, x + w - 2, values.size).astype(int)
    ys = (y + h - 3 - values * (h - 6)).astype(int)
    pts = np.column_stack([xs, ys]).reshape((-1, 1, 2))
    cv2.polylines(image, [pts], isClosed=False, color=(80, 220, 180), thickness=1)


def _fmt(value: object) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(val):
        return "--"
    return f"{val:.1f}"


def _face_box_label(hr: dict[str, object]) -> str:
    try:
        value = float(hr.get("value"))
    except (AttributeError, TypeError, ValueError):
        return ""
    if not np.isfinite(value):
        return ""
    return f"{value:.0f} bpm"


def _print_camera_selection(camera: str, selected: int | None, probes) -> None:
    if selected is None:
        print("Camera probe results:")
        for probe in probes:
            print(f"  {probe.index}: {probe.detail()}")
        return

    if str(camera).strip().lower() in {AUTO_CAMERA, "best", "-1"}:
        print(f"Using camera {selected} (auto selected).")
        for probe in probes:
            print(f"  probe {probe.index}: {probe.detail()}")
        return

    print(f"Using camera {selected}.")
    if probes and not probes[0].usable:
        print(
            f"WARN: camera {selected} may not be usable: {probes[0].detail()}. "
            "Try --camera auto if the preview is black or too slow."
        )


if __name__ == "__main__":
    raise SystemExit(main())
