from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np

from .signal_processing import (
    PPG_ALGORITHMS,
    PPG_AUTO_CANDIDATES,
    RPPG_ALGORITHMS,
    RPPG_AUTO_CANDIDATES,
    RateEstimate,
    estimate_bpm_from_ppg_rgb,
    estimate_bpm_from_rgb,
)


MIN_DURATION_S = 30.0
MIN_FPS = 10.0
MIN_QUALITY = 1.5
GOOD_QUALITY = 6.0
MIN_ROI_COVERAGE = 0.80
MIN_FINGER_CONTACT_QUALITY = 0.25
MAX_STABLE_BPM_SPAN = 18.0


@dataclass(frozen=True)
class StabilitySummary:
    windows: int
    bpm_min: float | None
    bpm_max: float | None
    bpm_span: float | None
    quality_median: float | None

    @property
    def is_stable(self) -> bool:
        return self.bpm_span is not None and self.bpm_span <= MAX_STABLE_BPM_SPAN


@dataclass(frozen=True)
class CsvAnalysis:
    path: Path
    mode: str
    rows: int
    duration_s: float
    fps: float
    roi_coverage: float
    roi_quality_median: float | None
    estimate: RateEstimate
    stability: StabilitySummary


@dataclass(frozen=True)
class CandidateEstimate:
    method: str
    estimate: RateEstimate
    stability: StabilitySummary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze a saved rPPG/PPG session CSV.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--mode", choices=["auto", "face", "finger"], default="auto")
    parser.add_argument("--rppg-algorithm", choices=RPPG_ALGORITHMS, default="auto")
    parser.add_argument("--ppg-algorithm", choices=PPG_ALGORITHMS, default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return print_csv_analysis(args.csv, args.mode, args.rppg_algorithm, args.ppg_algorithm)


def analyze_csv(
    path: Path,
    mode: str = "auto",
    rppg_algorithm: str = "auto",
    ppg_algorithm: str = "auto",
) -> CsvAnalysis:
    rows = _read_rows(path)
    timestamps_raw = np.asarray([_float(row.get("timestamp")) for row in rows], dtype=float)
    finite_t = timestamps_raw[np.isfinite(timestamps_raw)]
    start = float(finite_t[0]) if finite_t.size else 0.0
    timestamps = timestamps_raw - start
    rgb = np.asarray(
        [(_float(row.get("red")), _float(row.get("green")), _float(row.get("blue"))) for row in rows],
        dtype=float,
    )
    roi_ok = [_bool(row.get("roi_ok")) for row in rows]
    quality_values = [_float(row.get("quality_hint")) for row in rows]
    finite_quality = [value for value in quality_values if np.isfinite(value)]
    inferred_mode = _infer_mode(rows, mode)
    usable = np.asarray(roi_ok, dtype=bool) & np.isfinite(timestamps) & np.all(np.isfinite(rgb), axis=1)
    signal_timestamps = timestamps[usable]
    signal_rgb = rgb[usable]

    if inferred_mode == "face" and rppg_algorithm == "auto":
        candidate = _select_auto_candidate(signal_timestamps, signal_rgb, inferred_mode, RPPG_AUTO_CANDIDATES)
        estimate = candidate.estimate
        stability = candidate.stability
    elif inferred_mode == "face":
        estimate = estimate_bpm_from_rgb(signal_rgb, signal_timestamps, method=rppg_algorithm)
        stability = _stability_summary(signal_timestamps, signal_rgb, inferred_mode, rppg_algorithm, ppg_algorithm)
    elif ppg_algorithm == "auto":
        candidate = _select_auto_candidate(signal_timestamps, signal_rgb, inferred_mode, PPG_AUTO_CANDIDATES)
        estimate = candidate.estimate
        stability = candidate.stability
    else:
        estimate = estimate_bpm_from_ppg_rgb(signal_rgb, signal_timestamps, method=ppg_algorithm)
        stability = _stability_summary(signal_timestamps, signal_rgb, inferred_mode, rppg_algorithm, ppg_algorithm)

    duration = float(np.nanmax(timestamps) - np.nanmin(timestamps)) if timestamps.size else 0.0
    fps = float(len(rows) / duration) if duration > 0 else 0.0
    roi_coverage = sum(roi_ok) / len(roi_ok) if roi_ok else 0.0
    return CsvAnalysis(
        path=path,
        mode=inferred_mode,
        rows=len(rows),
        duration_s=duration,
        fps=fps,
        roi_coverage=roi_coverage,
        roi_quality_median=median(finite_quality) if finite_quality else None,
        estimate=estimate,
        stability=stability,
    )


def print_csv_analysis(
    path: Path,
    mode: str = "auto",
    rppg_algorithm: str = "auto",
    ppg_algorithm: str = "auto",
) -> int:
    result = analyze_csv(path, mode, rppg_algorithm, ppg_algorithm)
    verdict, notes = validation_verdict(result)
    bpm = "--" if not np.isfinite(result.estimate.value) else f"{result.estimate.value:.1f}"
    roi_quality = "--" if result.roi_quality_median is None else f"{result.roi_quality_median:.2f}"
    print(f"csv: {result.path}")
    print(f"mode={result.mode} method={result.estimate.method or '--'}")
    print(f"samples={result.rows} duration={result.duration_s:.1f}s fps~={result.fps:.1f}")
    print(f"roi_coverage={result.roi_coverage * 100.0:.1f}% roi_quality_median={roi_quality}")
    print(
        f"bpm={bpm} quality={result.estimate.quality:.2f} "
        f"status={result.estimate.status}"
    )
    print(_format_stability(result.stability))
    print(f"validation={verdict}")
    for note in notes:
        print(f"- {note}")
    return 0


def validation_verdict(result: CsvAnalysis) -> tuple[str, list[str]]:
    failures: list[str] = []
    warnings: list[str] = []

    if result.duration_s < MIN_DURATION_S:
        failures.append(f"duration {result.duration_s:.1f}s is below {MIN_DURATION_S:.0f}s minimum")
    if result.fps < MIN_FPS:
        failures.append(f"fps {result.fps:.1f} is below {MIN_FPS:.0f}fps minimum")
    if result.roi_coverage < MIN_ROI_COVERAGE:
        failures.append(f"ROI coverage {result.roi_coverage * 100.0:.1f}% is below {MIN_ROI_COVERAGE * 100.0:.0f}%")
    if (
        result.mode == "finger"
        and result.roi_quality_median is not None
        and result.roi_quality_median < MIN_FINGER_CONTACT_QUALITY
    ):
        failures.append(
            f"finger contact quality {result.roi_quality_median:.2f} is below {MIN_FINGER_CONTACT_QUALITY:.2f}"
        )
    if not np.isfinite(result.estimate.value):
        failures.append(f"no BPM estimate: {result.estimate.status}")
    elif result.estimate.quality < MIN_QUALITY:
        failures.append(f"quality {result.estimate.quality:.2f} is below {MIN_QUALITY:.1f}")
    elif result.estimate.quality < GOOD_QUALITY:
        warnings.append(f"quality {result.estimate.quality:.2f} is experimental, not strong")

    if result.stability.windows < 3:
        warnings.append("not enough rolling windows to prove BPM stability")
    elif not result.stability.is_stable:
        failures.append(f"rolling BPM span {result.stability.bpm_span:.1f} exceeds {MAX_STABLE_BPM_SPAN:.0f} BPM")

    notes = failures + warnings
    if not notes:
        notes.append("meets experimental quality gates; compare against a reference pulse before trusting it")
    return ("REMEASURE" if failures else "USABLE_EXPERIMENTAL", notes)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def _bool(value: str | None) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _infer_mode(rows: list[dict[str, str]], requested: str) -> str:
    if requested != "auto":
        return requested
    counts: dict[str, int] = {}
    for row in rows:
        mode = (row.get("mode") or "").strip().lower()
        if mode:
            counts[mode] = counts.get(mode, 0) + 1
    if counts:
        return max(counts.items(), key=lambda item: item[1])[0]
    name = str(rows[0].get("source", "") if rows else "").lower()
    return "finger" if "finger" in name or "ppg" in name else "face"


def _stability_summary(
    timestamps: np.ndarray,
    rgb: np.ndarray,
    mode: str,
    rppg_algorithm: str,
    ppg_algorithm: str,
    window_s: float = 20.0,
    step_s: float = 5.0,
) -> StabilitySummary:
    if timestamps.size < 2:
        return StabilitySummary(0, None, None, None, None)

    end_time = float(np.nanmax(timestamps))
    window_end = window_s
    bpms: list[float] = []
    qualities: list[float] = []
    while window_end <= end_time + 0.001:
        window_start = max(0.0, window_end - window_s)
        indexes = np.where((timestamps >= window_start) & (timestamps <= window_end))[0]
        if indexes.size:
            if mode == "face":
                estimate = estimate_bpm_from_rgb(rgb[indexes], timestamps[indexes], method=rppg_algorithm)
            else:
                estimate = estimate_bpm_from_ppg_rgb(rgb[indexes], timestamps[indexes], method=ppg_algorithm)
            if np.isfinite(estimate.value):
                bpms.append(estimate.value)
                qualities.append(estimate.quality)
        window_end += step_s

    if not bpms:
        return StabilitySummary(0, None, None, None, None)
    bpm_min = min(bpms)
    bpm_max = max(bpms)
    return StabilitySummary(
        windows=len(bpms),
        bpm_min=bpm_min,
        bpm_max=bpm_max,
        bpm_span=bpm_max - bpm_min,
        quality_median=median(qualities) if qualities else None,
    )


def _select_auto_candidate(
    timestamps: np.ndarray,
    rgb: np.ndarray,
    mode: str,
    methods: tuple[str, ...],
) -> CandidateEstimate:
    candidates: list[CandidateEstimate] = []
    for method in methods:
        if mode == "face":
            estimate = estimate_bpm_from_rgb(rgb, timestamps, method=method)
            stability = _stability_summary(timestamps, rgb, mode, method, "green")
        else:
            estimate = estimate_bpm_from_ppg_rgb(rgb, timestamps, method=method)
            stability = _stability_summary(timestamps, rgb, mode, "pos", method)
        candidates.append(CandidateEstimate(method, estimate, stability))
    return max(candidates, key=_candidate_sort_key)


def _candidate_sort_key(candidate: CandidateEstimate) -> tuple[int, int, float, float]:
    usable = int(np.isfinite(candidate.estimate.value) and candidate.estimate.quality >= MIN_QUALITY)
    stable = int(candidate.stability.windows >= 3 and candidate.stability.is_stable)
    span = candidate.stability.bpm_span if candidate.stability.bpm_span is not None else float("inf")
    return (usable, stable, -span, candidate.estimate.quality)


def _format_stability(summary: StabilitySummary) -> str:
    if summary.windows == 0:
        return "stability=not enough data"
    bpm_min = "--" if summary.bpm_min is None else f"{summary.bpm_min:.1f}"
    bpm_max = "--" if summary.bpm_max is None else f"{summary.bpm_max:.1f}"
    bpm_span = "--" if summary.bpm_span is None else f"{summary.bpm_span:.1f}"
    quality = "--" if summary.quality_median is None else f"{summary.quality_median:.2f}"
    label = "stable" if summary.is_stable else "unstable"
    return (
        f"stability={label} windows={summary.windows} "
        f"bpm_range={bpm_min}-{bpm_max} span={bpm_span} median_quality={quality}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
