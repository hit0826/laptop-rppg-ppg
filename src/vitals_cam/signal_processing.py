from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import signal


RPPG_ALGORITHMS = ("auto", "pos", "chrom", "green")
RPPG_AUTO_CANDIDATES = ("pos", "chrom", "green")
PPG_ALGORITHMS = ("auto", "green", "red", "blue", "luma", "gminusrb", "pos", "chrom")
PPG_AUTO_CANDIDATES = ("green", "red", "luma", "gminusrb", "pos", "chrom")


@dataclass(frozen=True)
class RateEstimate:
    value: float
    quality: float
    peak_hz: float
    samples: int
    seconds: float
    status: str
    method: str = ""

    @property
    def ok(self) -> bool:
        return math.isfinite(self.value) and self.quality >= 1.5 and self.status == "ok"


@dataclass(frozen=True)
class HrvEstimate:
    rmssd_ms: float
    beats: int
    status: str


def _empty_rate(status: str, samples: int = 0, seconds: float = 0.0, method: str = "") -> RateEstimate:
    return RateEstimate(float("nan"), 0.0, float("nan"), samples, seconds, status, method)


def sampling_rate(timestamps: np.ndarray) -> float:
    timestamps = np.asarray(timestamps, dtype=float)
    if timestamps.size < 2:
        return float("nan")
    duration = timestamps[-1] - timestamps[0]
    if duration <= 0:
        return float("nan")
    return (timestamps.size - 1) / duration


def resample_uniform(values: np.ndarray, timestamps: np.ndarray, target_fs: float | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    values, timestamps = _clean_series(values, timestamps)
    if values.size < 3:
        return values.copy(), timestamps.copy(), float("nan")

    fs = sampling_rate(timestamps)
    if not math.isfinite(fs) or fs <= 1:
        return values.copy(), timestamps.copy(), float("nan")
    fs = float(target_fs or min(max(fs, 10.0), 30.0))
    start = float(timestamps[0])
    stop = float(timestamps[-1])
    uniform_t = np.arange(start, stop, 1.0 / fs)
    if uniform_t.size < 3:
        return values.copy(), timestamps.copy(), fs
    uniform_v = np.interp(uniform_t, timestamps, values)
    return uniform_v, uniform_t, fs


def detrend_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return values.copy()
    cleaned = signal.detrend(values, type="linear") if values.size > 2 else values - np.mean(values)
    std = np.std(cleaned)
    if std < 1e-9:
        return cleaned * 0.0
    return cleaned / std


def bandpass(values: np.ndarray, fs: float, low_hz: float, high_hz: float, order: int = 3) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size < 8 or not math.isfinite(fs) or fs <= high_hz * 2:
        return detrend_normalize(values)
    nyquist = fs / 2.0
    low = max(0.001, low_hz / nyquist)
    high = min(0.999, high_hz / nyquist)
    if low >= high:
        return detrend_normalize(values)
    sos = signal.butter(order, [low, high], btype="bandpass", output="sos")
    padlen = min(values.size - 1, 3 * (2 * len(sos) + 1))
    if padlen < 6:
        return signal.sosfilt(sos, detrend_normalize(values))
    return signal.sosfiltfilt(sos, detrend_normalize(values), padlen=padlen)


def pos_rppg(rgb: np.ndarray) -> np.ndarray:
    """Project RGB traces to a pulse signal using the POS projection."""
    return detrend_normalize(rgb_pulse(rgb, "pos"))


def chrom_rppg(rgb: np.ndarray) -> np.ndarray:
    """Project RGB traces to a pulse signal using the CHROM projection."""
    return detrend_normalize(rgb_pulse(rgb, "chrom"))


def finger_ppg(rgb: np.ndarray, method: str = "green") -> np.ndarray:
    return detrend_normalize(ppg_pulse(rgb, method))


def rgb_pulse(rgb: np.ndarray, method: str = "pos") -> np.ndarray:
    rgb = _clean_rgb(rgb)
    if rgb.size == 0:
        return np.array([], dtype=float)

    method = method.lower()
    if method not in RPPG_ALGORITHMS or method == "auto":
        raise ValueError(f"unknown rPPG method: {method!r}")
    if method == "green":
        return rgb[:, 1]

    mean_rgb = np.mean(rgb, axis=0)
    mean_rgb = np.where(np.abs(mean_rgb) < 1e-6, 1.0, mean_rgb)
    normalized = (rgb / mean_rgb) - 1.0
    red, green, blue = normalized[:, 0], normalized[:, 1], normalized[:, 2]

    if method == "pos":
        x = green - blue
        y = green + blue - 2.0 * red
        return x + _safe_std_ratio(x, y) * y
    if method == "chrom":
        x = 3.0 * red - 2.0 * green
        y = 1.5 * red + green - 1.5 * blue
        return x - _safe_std_ratio(x, y) * y
    raise ValueError(f"unknown rPPG method: {method!r}")


def ppg_pulse(rgb: np.ndarray, method: str = "green") -> np.ndarray:
    rgb = _clean_rgb(rgb)
    if rgb.size == 0:
        return np.array([], dtype=float)

    method = method.lower()
    if method not in PPG_ALGORITHMS or method == "auto":
        raise ValueError(f"unknown PPG method: {method!r}")

    red, green, blue = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    if method == "green":
        return green
    if method == "red":
        return red
    if method == "blue":
        return blue
    if method == "luma":
        return 0.299 * red + 0.587 * green + 0.114 * blue
    if method == "gminusrb":
        return green - 0.5 * (red + blue)
    if method in {"pos", "chrom"}:
        return rgb_pulse(rgb, method)
    raise ValueError(f"unknown PPG method: {method!r}")


def estimate_rate(
    values: np.ndarray,
    timestamps: np.ndarray,
    min_per_min: float,
    max_per_min: float,
    label: str,
    method: str = "",
) -> RateEstimate:
    values, timestamps = _clean_series(values, timestamps)
    samples = int(values.size)
    seconds = float(timestamps[-1] - timestamps[0]) if timestamps.size > 1 else 0.0
    if samples < 30 or seconds < 6.0:
        return _empty_rate(f"need_more_{label}_samples", samples, seconds, method)

    uniform_v, _uniform_t, fs = resample_uniform(values, timestamps)
    if uniform_v.size < 30 or not math.isfinite(fs):
        return _empty_rate("bad_timing", samples, seconds, method)

    low_hz = min_per_min / 60.0
    high_hz = max_per_min / 60.0
    filtered = bandpass(uniform_v, fs, low_hz, high_hz)
    if np.std(filtered) < 1e-6:
        return _empty_rate("flat_signal", samples, seconds, method)

    nperseg = min(filtered.size, max(128, int(fs * min(20.0, max(8.0, seconds)))))
    freqs, power = signal.welch(filtered, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(mask):
        return _empty_rate("empty_band", samples, seconds, method)

    band_freqs = freqs[mask]
    band_power = power[mask]
    peak_idx = int(np.argmax(band_power))
    peak_hz = float(band_freqs[peak_idx])
    peak_power = float(band_power[peak_idx])
    median_power = float(np.median(band_power) + 1e-12)
    quality = peak_power / median_power
    rate = peak_hz * 60.0
    return RateEstimate(rate, quality, peak_hz, samples, seconds, "ok", method)


def estimate_bpm(
    values: np.ndarray,
    timestamps: np.ndarray,
    min_bpm: float = 45.0,
    max_bpm: float = 180.0,
    method: str = "",
) -> RateEstimate:
    return estimate_rate(values, timestamps, min_bpm, max_bpm, "bpm", method)


def estimate_bpm_from_rgb(
    rgb: np.ndarray,
    timestamps: np.ndarray,
    method: str = "auto",
    min_bpm: float = 45.0,
    max_bpm: float = 180.0,
) -> RateEstimate:
    method = method.lower()
    if method not in RPPG_ALGORITHMS:
        raise ValueError(f"unknown rPPG method: {method!r}")
    methods = RPPG_AUTO_CANDIDATES if method == "auto" else (method,)
    estimates = [
        estimate_bpm(detrend_normalize(rgb_pulse(rgb, candidate)), timestamps, min_bpm, max_bpm, candidate)
        for candidate in methods
    ]
    return max(estimates, key=_rate_sort_key) if estimates else _empty_rate("no_candidate", method=method)


def estimate_bpm_from_ppg_rgb(
    rgb: np.ndarray,
    timestamps: np.ndarray,
    method: str = "auto",
    min_bpm: float = 45.0,
    max_bpm: float = 180.0,
) -> RateEstimate:
    method = method.lower()
    if method not in PPG_ALGORITHMS:
        raise ValueError(f"unknown PPG method: {method!r}")
    methods = PPG_AUTO_CANDIDATES if method == "auto" else (method,)
    estimates = [
        estimate_bpm(detrend_normalize(ppg_pulse(rgb, candidate)), timestamps, min_bpm, max_bpm, candidate)
        for candidate in methods
    ]
    return max(estimates, key=_rate_sort_key) if estimates else _empty_rate("no_candidate", method=method)


def estimate_respiration_rate(values: np.ndarray, timestamps: np.ndarray) -> RateEstimate:
    return estimate_rate(values, timestamps, 6.0, 30.0, "respiration", "respiration")


def estimate_hrv_rmssd(pulse: np.ndarray, timestamps: np.ndarray, bpm: float) -> HrvEstimate:
    if not math.isfinite(bpm) or bpm <= 0:
        return HrvEstimate(float("nan"), 0, "missing_bpm")
    uniform_v, uniform_t, fs = resample_uniform(pulse, timestamps)
    if uniform_v.size < 120 or not math.isfinite(fs):
        return HrvEstimate(float("nan"), 0, "need_more_samples")
    filtered = bandpass(uniform_v, fs, 0.7, 3.0)
    min_distance = max(1, int(fs * 60.0 / min(180.0, max(45.0, bpm * 1.4))))
    peaks, _ = signal.find_peaks(filtered, distance=min_distance, prominence=max(0.15, np.std(filtered) * 0.25))
    if peaks.size < 4:
        return HrvEstimate(float("nan"), int(peaks.size), "need_more_beats")
    beat_t = uniform_t[peaks]
    ibi_ms = np.diff(beat_t) * 1000.0
    ibi_ms = ibi_ms[(ibi_ms > 300.0) & (ibi_ms < 1500.0)]
    if ibi_ms.size < 3:
        return HrvEstimate(float("nan"), int(peaks.size), "bad_intervals")
    rmssd = float(np.sqrt(np.mean(np.diff(ibi_ms) ** 2)))
    return HrvEstimate(rmssd, int(peaks.size), "ok")


def estimate_spo2_experimental(rgb: np.ndarray) -> tuple[float, str]:
    rgb = _clean_rgb(rgb)
    if rgb.shape[0] < 90:
        return float("nan"), "need_more_samples"
    red = rgb[:, 0]
    green = rgb[:, 1]
    red_dc = float(np.mean(red))
    green_dc = float(np.mean(green))
    if red_dc <= 1 or green_dc <= 1:
        return float("nan"), "dark_or_invalid"
    red_ac = float(np.std(signal.detrend(red, type="linear")) / max(red_dc, 1e-6))
    green_ac = float(np.std(signal.detrend(green, type="linear")) / max(green_dc, 1e-6))
    if green_ac <= 1e-9:
        return float("nan"), "flat_signal"
    ratio = red_ac / green_ac
    spo2 = float(np.clip(110.0 - 25.0 * ratio, 70.0, 100.0))
    return spo2, "experimental"


def _clean_series(values: np.ndarray, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)
    if values.size != timestamps.size:
        return np.array([], dtype=float), np.array([], dtype=float)
    finite = np.isfinite(values) & np.isfinite(timestamps)
    values = values[finite]
    timestamps = timestamps[finite]
    if timestamps.size == 0:
        return values, timestamps
    order = np.argsort(timestamps)
    return values[order], timestamps[order]


def _clean_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=float)
    if rgb.ndim != 2 or rgb.shape[1] != 3 or rgb.shape[0] == 0:
        return np.empty((0, 3), dtype=float)
    finite = np.all(np.isfinite(rgb), axis=1)
    return rgb[finite]


def _safe_std_ratio(numerator: np.ndarray, denominator: np.ndarray) -> float:
    den = float(np.std(denominator))
    if den < 1e-12:
        return 0.0
    return float(np.std(numerator) / den)


def _rate_sort_key(estimate: RateEstimate) -> tuple[int, float]:
    usable = 1 if estimate.status == "ok" and math.isfinite(estimate.value) else 0
    return (usable, estimate.quality)
