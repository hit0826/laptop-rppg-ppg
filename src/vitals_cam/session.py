from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import csv
import json


@dataclass
class Sample:
    timestamp: float
    mode: str
    red: float
    green: float
    blue: float
    roi_ok: bool
    quality_hint: float


class SessionRecorder:
    def __init__(self) -> None:
        self.samples: list[Sample] = []
        self.recording = False
        self.summary: dict[str, object] = {}

    def add(self, sample: Sample) -> None:
        if self.recording:
            self.samples.append(sample)

    def clear(self) -> None:
        self.samples.clear()
        self.summary.clear()

    def save(self, out_dir: str | Path, summary: dict[str, object] | None = None) -> tuple[Path, Path]:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        mode_label, mode_counts = _mode_label(self.samples)
        csv_path = out_path / f"session-{mode_label}-{stamp}.csv"
        json_path = out_path / f"session-{mode_label}-{stamp}.json"

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(Sample(0, "", 0, 0, 0, False, 0)).keys()))
            writer.writeheader()
            for sample in self.samples:
                writer.writerow(asdict(sample))

        payload = {
            "created_at": stamp,
            "capture_mode": mode_label,
            "mode_counts": mode_counts,
            "sample_count": len(self.samples),
            "summary": summary or self.summary,
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return csv_path, json_path


def _mode_label(samples: list[Sample]) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    for sample in samples:
        mode = sample.mode.strip().lower()
        if mode:
            counts[mode] = counts.get(mode, 0) + 1

    if set(counts) == {"face"}:
        return "rppg-face", counts
    if set(counts) == {"finger"}:
        return "ppg-finger", counts
    if {"face", "finger"}.issubset(counts):
        return "mixed-rppg-ppg", counts
    return "unknown", counts
