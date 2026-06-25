from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vitals_cam.signal_processing import estimate_bpm, estimate_respiration_rate, pos_rppg
from vitals_cam.synthetic import synthetic_rgb_trace


def main() -> int:
    timestamps, rgb = synthetic_rgb_trace(duration_s=30.0, fps=30.0, heart_bpm=72.0, resp_bpm=15.0)
    pulse = pos_rppg(rgb)
    hr = estimate_bpm(pulse, timestamps)
    rr = estimate_respiration_rate(rgb[:, 1], timestamps)

    print(f"synthetic heart rate: {hr.value:.1f} bpm quality={hr.quality:.2f}")
    print(f"synthetic respiration: {rr.value:.1f} br/min quality={rr.quality:.2f}")
    return 0 if abs(hr.value - 72.0) < 3.0 and abs(rr.value - 15.0) < 3.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

