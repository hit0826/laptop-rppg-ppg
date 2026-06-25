import csv
import json
import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vitals_cam.analysis import analyze_csv, validation_verdict
from vitals_cam.camera import AUTO_CAMERA, frame_stats, parse_camera_spec
from vitals_cam.session import Sample, SessionRecorder
from vitals_cam.signal_processing import (
    estimate_bpm,
    estimate_bpm_from_ppg_rgb,
    estimate_bpm_from_rgb,
    estimate_respiration_rate,
    finger_ppg,
    pos_rppg,
)
from vitals_cam.synthetic import synthetic_rgb_trace


class SignalProcessingTests(unittest.TestCase):
    def test_pos_rppg_recovers_synthetic_heart_rate(self):
        timestamps, rgb = synthetic_rgb_trace(duration_s=30.0, fps=30.0, heart_bpm=72.0, resp_bpm=15.0)
        pulse = pos_rppg(rgb)
        estimate = estimate_bpm(pulse, timestamps)
        self.assertEqual(estimate.status, "ok")
        self.assertLess(abs(estimate.value - 72.0), 3.0)
        self.assertGreater(estimate.quality, 2.0)

    def test_rppg_auto_recovers_synthetic_heart_rate(self):
        timestamps, rgb = synthetic_rgb_trace(duration_s=30.0, fps=30.0, heart_bpm=68.0, resp_bpm=15.0)
        estimate = estimate_bpm_from_rgb(rgb, timestamps, method="auto")
        self.assertEqual(estimate.status, "ok")
        self.assertLess(abs(estimate.value - 68.0), 3.0)
        self.assertIn(estimate.method, {"pos", "chrom", "green"})

    def test_finger_ppg_recovers_synthetic_heart_rate(self):
        timestamps, rgb = synthetic_rgb_trace(duration_s=30.0, fps=30.0, heart_bpm=84.0, resp_bpm=12.0)
        pulse = finger_ppg(rgb)
        estimate = estimate_bpm(pulse, timestamps)
        self.assertEqual(estimate.status, "ok")
        self.assertLess(abs(estimate.value - 84.0), 4.0)

    def test_ppg_auto_recovers_synthetic_heart_rate(self):
        timestamps, rgb = synthetic_rgb_trace(duration_s=30.0, fps=30.0, heart_bpm=76.0, resp_bpm=12.0)
        estimate = estimate_bpm_from_ppg_rgb(rgb, timestamps, method="auto")
        self.assertEqual(estimate.status, "ok")
        self.assertLess(abs(estimate.value - 76.0), 4.0)
        self.assertIn(estimate.method, {"green", "red", "luma", "gminusrb", "pos", "chrom"})

    def test_respiration_rate_recovers_synthetic_rate(self):
        timestamps, rgb = synthetic_rgb_trace(duration_s=40.0, fps=30.0, heart_bpm=72.0, resp_bpm=18.0)
        estimate = estimate_respiration_rate(rgb[:, 1], timestamps)
        self.assertEqual(estimate.status, "ok")
        self.assertLess(abs(estimate.value - 18.0), 3.0)

    def test_short_signal_reports_need_more_samples(self):
        timestamps = np.linspace(0.0, 2.0, 20)
        estimate = estimate_bpm(np.ones_like(timestamps), timestamps)
        self.assertTrue(estimate.status.startswith("need_more"))


class CameraUtilityTests(unittest.TestCase):
    def test_parse_camera_spec_accepts_auto_and_integer_indexes(self):
        self.assertEqual(parse_camera_spec("auto"), AUTO_CAMERA)
        self.assertEqual(parse_camera_spec("best"), AUTO_CAMERA)
        self.assertEqual(parse_camera_spec("-1"), AUTO_CAMERA)
        self.assertEqual(parse_camera_spec("0"), 0)
        self.assertEqual(parse_camera_spec(1), 1)

    def test_frame_stats_flags_near_black_frames(self):
        black = np.zeros((12, 16, 3), dtype=np.uint8)
        bright = np.full((12, 16, 3), 120, dtype=np.uint8)
        self.assertTrue(frame_stats(black).looks_black)
        self.assertFalse(frame_stats(bright).looks_black)


class SessionAnalysisTests(unittest.TestCase):
    def test_analyze_csv_accepts_good_synthetic_face_session(self):
        timestamps, rgb = synthetic_rgb_trace(duration_s=35.0, fps=30.0, heart_bpm=72.0, resp_bpm=15.0)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "face.csv"
            self._write_session_csv(path, "face", timestamps, rgb)
            result = analyze_csv(path, mode="face")
            verdict, notes = validation_verdict(result)

        self.assertEqual(verdict, "USABLE_EXPERIMENTAL", notes)
        self.assertLess(abs(result.estimate.value - 72.0), 3.0)
        self.assertGreaterEqual(result.stability.windows, 3)

    @staticmethod
    def _write_session_csv(path: Path, mode: str, timestamps: np.ndarray, rgb: np.ndarray) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["timestamp", "mode", "red", "green", "blue", "roi_ok", "quality_hint"],
            )
            writer.writeheader()
            for timestamp, sample in zip(timestamps, rgb):
                writer.writerow(
                    {
                        "timestamp": timestamp,
                        "mode": mode,
                        "red": sample[0],
                        "green": sample[1],
                        "blue": sample[2],
                        "roi_ok": "True",
                        "quality_hint": "1.0",
                    }
                )


class SessionRecorderTests(unittest.TestCase):
    def test_saved_face_session_names_outputs_as_rppg(self):
        recorder = SessionRecorder()
        recorder.samples.append(Sample(1.0, "face", 10.0, 20.0, 30.0, True, 0.9))

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path, json_path = recorder.save(tmp_dir)
            payload = json.loads(json_path.read_text(encoding="utf-8"))

        self.assertRegex(csv_path.name, r"^session-rppg-face-\d{8}-\d{6}\.csv$")
        self.assertEqual(json_path.name, csv_path.with_suffix(".json").name)
        self.assertEqual(payload["capture_mode"], "rppg-face")
        self.assertEqual(payload["mode_counts"], {"face": 1})

    def test_saved_finger_session_names_outputs_as_ppg(self):
        recorder = SessionRecorder()
        recorder.samples.append(Sample(1.0, "finger", 10.0, 20.0, 30.0, True, 0.9))

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path, json_path = recorder.save(tmp_dir)
            payload = json.loads(json_path.read_text(encoding="utf-8"))

        self.assertRegex(csv_path.name, r"^session-ppg-finger-\d{8}-\d{6}\.csv$")
        self.assertEqual(json_path.name, csv_path.with_suffix(".json").name)
        self.assertEqual(payload["capture_mode"], "ppg-finger")
        self.assertEqual(payload["mode_counts"], {"finger": 1})


if __name__ == "__main__":
    unittest.main()
