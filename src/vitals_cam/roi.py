from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass
class RoiResult:
    rgb: tuple[float, float, float]
    rects: list[tuple[int, int, int, int]]
    ok: bool
    quality_hint: float
    status: str
    tracking_rect: tuple[int, int, int, int] | None = None


class FaceRoiDetector:
    def __init__(self, use_mediapipe: bool = True) -> None:
        self.mp_face_detection = None
        self.detector = None
        self.use_mediapipe = False
        if use_mediapipe:
            try:
                import mediapipe as mp

                self.mp_face_detection = mp.solutions.face_detection
                self.detector = self.mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.55)
                self.use_mediapipe = True
            except Exception:
                self.detector = None
        self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def close(self) -> None:
        if self.detector is not None and hasattr(self.detector, "close"):
            self.detector.close()

    def detect(self, frame_bgr: np.ndarray) -> RoiResult:
        h, w = frame_bgr.shape[:2]
        face = self._detect_face(frame_bgr)
        if face is None:
            return RoiResult((0.0, 0.0, 0.0), [], False, 0.0, "no_face")
        x, y, fw, fh = _clip_rect(face, w, h)
        tracking_rect = _expand_rect((x, y, fw, fh), w, h)
        if fw < 40 or fh < 40:
            return RoiResult((0.0, 0.0, 0.0), [], False, 0.0, "face_too_small", tracking_rect)

        forehead = _clip_rect((x + int(0.25 * fw), y + int(0.12 * fh), int(0.50 * fw), int(0.16 * fh)), w, h)
        left_cheek = _clip_rect((x + int(0.18 * fw), y + int(0.42 * fh), int(0.22 * fw), int(0.20 * fh)), w, h)
        right_cheek = _clip_rect((x + int(0.60 * fw), y + int(0.42 * fh), int(0.22 * fw), int(0.20 * fh)), w, h)
        rects = [forehead, left_cheek, right_cheek]
        rgb, quality = mean_rgb_from_rects(frame_bgr, rects, skin_filter=True)
        ok = all(math.isfinite(v) for v in rgb) and quality > 0.15
        return RoiResult(rgb, rects, ok, quality, "ok" if ok else "bad_skin_roi", tracking_rect)

    def _detect_face(self, frame_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
        h, w = frame_bgr.shape[:2]
        if self.detector is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = self.detector.process(rgb)
            detections = getattr(result, "detections", None)
            if detections:
                best = max(detections, key=lambda d: d.score[0] if d.score else 0.0)
                box = best.location_data.relative_bounding_box
                return (
                    int(box.xmin * w),
                    int(box.ymin * h),
                    int(box.width * w),
                    int(box.height * h),
                )

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        if len(faces) == 0:
            return None
        return tuple(max(faces, key=lambda r: r[2] * r[3]))


def finger_roi(frame_bgr: np.ndarray) -> RoiResult:
    h, w = frame_bgr.shape[:2]
    size = int(min(w, h) * 0.42)
    rect = _clip_rect((w // 2 - size // 2, h // 2 - size // 2, size, size), w, h)
    rgb, quality = mean_rgb_from_rects(frame_bgr, [rect], skin_filter=False)
    red, green, blue = rgb
    brightness = (red + green + blue) / 3.0
    x, y, rw, rh = rect
    roi = frame_bgr[y : y + rh, x : x + rw]
    skin_ratio = float(np.mean(skin_mask(roi) > 0)) if roi.size else 0.0
    red_ratio = red / max(red + green + blue, 1.0)
    red_score = max(0.0, min(1.0, (red_ratio - 0.34) / 0.12))
    contact_score = max(min(1.0, skin_ratio * 2.0), red_score)
    ok = brightness > 10 and contact_score > 0.25
    status = "ok" if ok else "no_finger_contact" if brightness > 10 else "too_dark"
    return RoiResult(rgb, [rect], ok, float(contact_score), status)


def mean_rgb_from_rects(
    frame_bgr: np.ndarray,
    rects: list[tuple[int, int, int, int]],
    skin_filter: bool = True,
) -> tuple[tuple[float, float, float], float]:
    samples = []
    kept_pixels = 0
    total_pixels = 0
    for x, y, w, h in rects:
        if w <= 1 or h <= 1:
            continue
        roi = frame_bgr[y : y + h, x : x + w]
        if roi.size == 0:
            continue
        total_pixels += roi.shape[0] * roi.shape[1]
        if skin_filter:
            mask = skin_mask(roi)
            selected = roi[mask > 0]
            if selected.shape[0] < max(20, total_pixels * 0.05):
                selected = roi.reshape(-1, 3)
            else:
                kept_pixels += selected.shape[0]
        else:
            selected = roi.reshape(-1, 3)
            kept_pixels += selected.shape[0]
        samples.append(selected)
    if not samples:
        return (float("nan"), float("nan"), float("nan")), 0.0
    pixels = np.vstack(samples)
    bgr = np.mean(pixels, axis=0)
    quality = min(1.0, kept_pixels / max(total_pixels, 1))
    return (float(bgr[2]), float(bgr[1]), float(bgr[0])), float(quality)


def skin_mask(roi_bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def _clip_rect(rect: tuple[int, int, int, int], max_w: int, max_h: int) -> tuple[int, int, int, int]:
    x, y, w, h = rect
    x = max(0, min(int(x), max_w - 1))
    y = max(0, min(int(y), max_h - 1))
    w = max(1, min(int(w), max_w - x))
    h = max(1, min(int(h), max_h - y))
    return x, y, w, h


def _expand_rect(rect: tuple[int, int, int, int], max_w: int, max_h: int) -> tuple[int, int, int, int]:
    x, y, w, h = rect
    pad_x = int(w * 0.10)
    pad_top = int(h * 0.14)
    pad_bottom = int(h * 0.10)
    return _clip_rect((x - pad_x, y - pad_top, w + pad_x * 2, h + pad_top + pad_bottom), max_w, max_h)
