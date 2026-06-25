from __future__ import annotations

import numpy as np


def synthetic_rgb_trace(
    duration_s: float = 30.0,
    fps: float = 30.0,
    heart_bpm: float = 72.0,
    resp_bpm: float = 15.0,
    noise: float = 0.02,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    timestamps = np.arange(0.0, duration_s, 1.0 / fps)
    heart_hz = heart_bpm / 60.0
    resp_hz = resp_bpm / 60.0
    pulse = np.sin(2.0 * np.pi * heart_hz * timestamps)
    second_harmonic = 0.25 * np.sin(4.0 * np.pi * heart_hz * timestamps + 0.5)
    respiration = np.sin(2.0 * np.pi * resp_hz * timestamps)
    motion = 0.15 * np.sin(2.0 * np.pi * 0.07 * timestamps)

    rgb = np.column_stack(
        [
            112.0 + 0.55 * pulse + 0.20 * second_harmonic + 1.00 * respiration + motion,
            92.0 + 1.45 * pulse + 0.35 * second_harmonic + 0.70 * respiration + motion,
            76.0 + 0.35 * pulse + 0.15 * second_harmonic + 0.45 * respiration + motion,
        ]
    )
    rgb += rng.normal(0.0, noise, size=rgb.shape)
    return timestamps, rgb
