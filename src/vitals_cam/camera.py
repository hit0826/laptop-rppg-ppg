from __future__ import annotations

from dataclasses import dataclass
import time

import cv2
import numpy as np


AUTO_CAMERA = "auto"
DEFAULT_AUTO_MAX_INDEX = 4
MIN_USABLE_FPS = 10.0
MIN_USABLE_MEAN_LUMA = 8.0
MIN_USABLE_MAX_LUMA = 16


try:
    cv2.setLogLevel(0)
except AttributeError:
    pass


@dataclass(frozen=True)
class FrameStats:
    width: int
    height: int
    mean_luma: float
    std_luma: float
    min_luma: int
    max_luma: int

    @property
    def looks_black(self) -> bool:
        return self.mean_luma < MIN_USABLE_MEAN_LUMA or self.max_luma < MIN_USABLE_MAX_LUMA


@dataclass(frozen=True)
class CameraProbe:
    index: int
    opened: bool
    frames: int = 0
    fps: float = 0.0
    stats: FrameStats | None = None
    message: str = "not opened"

    @property
    def usable(self) -> bool:
        if not self.opened or self.frames <= 0 or self.stats is None:
            return False
        return self.fps >= MIN_USABLE_FPS and not self.stats.looks_black

    def detail(self) -> str:
        if self.stats is None:
            return self.message
        return (
            f"{self.stats.width}x{self.stats.height}, fps={self.fps:.1f}, "
            f"luma mean={self.stats.mean_luma:.1f}, min={self.stats.min_luma}, "
            f"max={self.stats.max_luma}, std={self.stats.std_luma:.1f} ({self.message})"
        )


def parse_camera_spec(value: int | str) -> int | str:
    if isinstance(value, int):
        return value
    normalized = str(value).strip().lower()
    if normalized in {AUTO_CAMERA, "best", "-1"}:
        return AUTO_CAMERA
    try:
        return int(normalized)
    except ValueError as exc:
        raise ValueError(f"camera must be an integer index or '{AUTO_CAMERA}', got {value!r}") from exc


def open_camera(index: int, width: int, height: int, fps: int, attempts: int = 3) -> cv2.VideoCapture:
    capture: cv2.VideoCapture | None = None
    for attempt in range(max(1, attempts)):
        if capture is not None:
            capture.release()
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(index)
        _configure_capture(capture, width, height, fps)
        if capture.isOpened():
            return capture
        time.sleep(0.4 + attempt * 0.3)
    return capture if capture is not None else cv2.VideoCapture()


def list_cameras(max_index: int = DEFAULT_AUTO_MAX_INDEX, width: int = 320, height: int = 240) -> list[CameraProbe]:
    return [probe_camera(index, width=width, height=height) for index in range(max_index + 1)]


def probe_camera(
    index: int,
    width: int = 320,
    height: int = 240,
    fps: int = 30,
    seconds: float = 2.2,
    max_frames: int = 60,
) -> CameraProbe:
    capture = open_camera(index, width, height, fps, attempts=1)
    if not capture.isOpened():
        capture.release()
        return CameraProbe(index=index, opened=False, message="not opened")

    frames: list[tuple[float, np.ndarray]] = []
    started = time.perf_counter()
    while len(frames) < max_frames and time.perf_counter() - started < seconds:
        ok, frame = capture.read()
        if ok and frame is not None:
            frames.append((time.perf_counter(), frame))
    elapsed = max(time.perf_counter() - started, 1e-6)
    capture.release()

    if not frames:
        return CameraProbe(index=index, opened=True, frames=0, fps=0.0, message="opened, no frame")

    stats = max((frame_stats(frame) for _timestamp, frame in frames), key=lambda item: item.mean_luma)
    if len(frames) >= 2:
        active_elapsed = max(frames[-1][0] - frames[0][0], 1e-6)
        measured_fps = (len(frames) - 1) / active_elapsed
    else:
        measured_fps = len(frames) / elapsed

    message = "ok"
    if stats.looks_black:
        message = "near-black frame"
    elif measured_fps < MIN_USABLE_FPS:
        message = "too slow"

    return CameraProbe(
        index=index,
        opened=True,
        frames=len(frames),
        fps=measured_fps,
        stats=stats,
        message=message,
    )


def frame_stats(frame: np.ndarray) -> FrameStats:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    return FrameStats(
        width=width,
        height=height,
        mean_luma=float(np.mean(gray)),
        std_luma=float(np.std(gray)),
        min_luma=int(np.min(gray)),
        max_luma=int(np.max(gray)),
    )


def open_selected_camera(
    camera: int | str,
    width: int,
    height: int,
    fps: int,
    max_auto_index: int = DEFAULT_AUTO_MAX_INDEX,
) -> tuple[cv2.VideoCapture | None, int | None, list[CameraProbe]]:
    spec = parse_camera_spec(camera)
    if spec != AUTO_CAMERA:
        index = int(spec)
        probe = probe_camera(index, width, height, fps)
        if not probe.opened or probe.frames <= 0:
            time.sleep(1.0)
            probe = probe_camera(index, width, height, fps)
        return open_camera(index, width, height, fps), index, [probe]

    probes: list[CameraProbe] = []
    fallback_index: int | None = None
    fallback_score = -1.0
    for index in range(max_auto_index + 1):
        probe = probe_camera(index, width, height, fps)
        probes.append(probe)
        if probe.usable:
            return open_camera(index, width, height, fps), index, probes
        if probe.opened and probe.frames > 0 and probe.stats is not None:
            score = probe.fps + probe.stats.mean_luma / 100.0
            if score > fallback_score:
                fallback_score = score
                fallback_index = index

    if fallback_index is None:
        return None, None, probes
    return open_camera(fallback_index, width, height, fps), fallback_index, probes


def measure_camera_fps(
    camera: int | str = AUTO_CAMERA,
    seconds: float = 5.0,
    width: int = 320,
    height: int = 240,
) -> tuple[int | None, int, float]:
    capture, selected_index, _probes = open_selected_camera(camera, width, height, 30)
    if capture is None or not capture.isOpened():
        return selected_index, 0, 0.0
    count = 0
    started = time.perf_counter()
    while time.perf_counter() - started < seconds:
        ok, _frame = capture.read()
        if ok:
            count += 1
    elapsed = max(time.perf_counter() - started, 1e-6)
    capture.release()
    return selected_index, count, count / elapsed


def _configure_capture(capture: cv2.VideoCapture, width: int, height: int, fps: int) -> None:
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
