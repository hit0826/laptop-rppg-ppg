from pathlib import Path
import argparse
import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vitals_cam.analysis import analyze_csv
from vitals_cam.signal_processing import detrend_normalize, ppg_pulse, rgb_pulse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot a saved rPPG/PPG session CSV.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--mode", choices=["auto", "face", "finger"], default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rppg-algorithm", default="auto")
    parser.add_argument("--ppg-algorithm", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    csv_path = args.csv.resolve()
    output = args.output or csv_path.with_suffix(".png")

    rows = _read_rows(csv_path)
    if not rows:
        raise SystemExit(f"No rows in {csv_path}")

    timestamps = np.asarray([_float(row.get("timestamp")) for row in rows], dtype=float)
    finite = np.isfinite(timestamps)
    if not np.any(finite):
        raise SystemExit(f"No valid timestamps in {csv_path}")
    t = timestamps - float(timestamps[finite][0])
    rgb = np.asarray(
        [(_float(row.get("red")), _float(row.get("green")), _float(row.get("blue"))) for row in rows],
        dtype=float,
    )
    roi_ok = np.asarray([_bool(row.get("roi_ok")) for row in rows], dtype=bool)
    quality = np.asarray([_float(row.get("quality_hint")) for row in rows], dtype=float)
    result = analyze_csv(csv_path, args.mode, args.rppg_algorithm, args.ppg_algorithm)

    signal_t = t[roi_ok & np.isfinite(t) & np.all(np.isfinite(rgb), axis=1)]
    signal_rgb = rgb[roi_ok & np.isfinite(t) & np.all(np.isfinite(rgb), axis=1)]
    pulse = _pulse_trace(signal_rgb, result.mode, result.estimate.method)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(
        (
            f"{csv_path.name} | mode={result.mode} | "
            f"BPM={_fmt(result.estimate.value)} | quality={result.estimate.quality:.2f} | "
            f"ROI={result.roi_coverage * 100:.1f}%"
        ),
        fontsize=12,
    )

    axes[0].plot(t, rgb[:, 0], color="#c0392b", linewidth=0.8, label="red")
    axes[0].plot(t, rgb[:, 1], color="#2e7d32", linewidth=0.8, label="green")
    axes[0].plot(t, rgb[:, 2], color="#1565c0", linewidth=0.8, label="blue")
    axes[0].set_ylabel("RGB mean")
    axes[0].legend(loc="upper right", ncols=3, fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].plot(t, np.nan_to_num(quality, nan=0.0), color="#6a4c93", linewidth=0.9, label="ROI quality")
    axes[1].fill_between(t, 0, 1, where=roi_ok, color="#2ca25f", alpha=0.18, transform=axes[1].get_xaxis_transform(), label="ROI OK")
    axes[1].set_ylabel("ROI quality")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.25)

    if pulse.size:
        axes[2].plot(signal_t[: pulse.size], pulse, color="#111111", linewidth=0.9)
    axes[2].set_ylabel("pulse trace")
    axes[2].set_xlabel("seconds")
    axes[2].grid(alpha=0.25)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved graph: {output}")
    return 0


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def _bool(value: str | None) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _pulse_trace(rgb: np.ndarray, mode: str, method: str) -> np.ndarray:
    if rgb.size == 0:
        return np.array([], dtype=float)
    selected = method if method and method != "auto" else ("pos" if mode == "face" else "green")
    if mode == "face":
        return detrend_normalize(rgb_pulse(rgb, selected if selected in {"pos", "chrom", "green"} else "pos"))
    return detrend_normalize(ppg_pulse(rgb, selected if selected != "auto" else "green"))


def _fmt(value: float) -> str:
    return "--" if not np.isfinite(value) else f"{value:.1f}"


if __name__ == "__main__":
    raise SystemExit(main())
