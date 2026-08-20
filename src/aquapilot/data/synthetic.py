from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EnvironmentSpec:
    name: str
    paths: int
    max_delay_bins: float
    doppler_cycles_per_frame: float
    delay_drift_bins: float
    ar: float
    absorption_slope: float
    notch_strength: float = 0.0
    phase_noise: float = 0.0


DEFAULT_ENVIRONMENTS: dict[str, EnvironmentSpec] = {
    "calm_fjord": EnvironmentSpec("calm_fjord", 4, 8.0, 0.010, 0.10, 0.995, 0.10, 0.00, 0.002),
    "continental_shelf": EnvironmentSpec("continental_shelf", 6, 14.0, 0.025, 0.25, 0.985, 0.18, 0.05, 0.005),
    "busy_harbor": EnvironmentSpec("busy_harbor", 7, 18.0, 0.045, 0.45, 0.970, 0.25, 0.25, 0.010),
    "fast_tow": EnvironmentSpec("fast_tow", 8, 22.0, 0.085, 0.90, 0.945, 0.32, 0.35, 0.020),
}


def _complex_normal(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def generate_episode(
    spec: EnvironmentSpec,
    n_frames: int,
    n_subcarriers: int,
    seed: int,
) -> np.ndarray:
    """Generate a controlled doubly-spread complex CSI episode.

    The generator is deliberately modest: a sum of continuously drifting paths,
    colored AR amplitudes, common clock/phase drift, smooth absorption, and an
    optional moving spectral notch. It is a sanity-check source, not a substitute
    for Watermark or at-sea data.
    """
    if n_frames < 4 or n_subcarriers < 8:
        raise ValueError("Synthetic episodes require at least 4 frames and 8 subcarriers.")
    rng = np.random.default_rng(seed)
    k = np.arange(n_subcarriers, dtype=np.float64)
    f = (k - (n_subcarriers - 1) / 2.0) / n_subcarriers

    base_delays = np.sort(rng.uniform(0.0, spec.max_delay_bins, size=spec.paths))
    path_power = np.exp(-base_delays / max(spec.max_delay_bins / 2.5, 1.0))
    path_power /= path_power.sum()
    amplitudes = np.zeros((n_frames, spec.paths), dtype=np.complex128)
    amplitudes[0] = _complex_normal(rng, (spec.paths,)) * np.sqrt(path_power)
    fd = rng.uniform(-1.0, 1.0, size=spec.paths) * spec.doppler_cycles_per_frame
    drift_phase = rng.uniform(0.0, 2.0 * np.pi, size=spec.paths)
    drift_rate = rng.uniform(0.6, 1.4, size=spec.paths)

    innovation_scale = np.sqrt(max(1.0 - spec.ar**2, 1e-8))
    for t in range(1, n_frames):
        rotation = np.exp(1j * 2.0 * np.pi * fd)
        amplitudes[t] = (
            spec.ar * amplitudes[t - 1] * rotation
            + innovation_scale * _complex_normal(rng, (spec.paths,)) * np.sqrt(path_power)
        )

    absorption = np.exp(-spec.absorption_slope * (f + 0.5))
    h = np.empty((n_frames, n_subcarriers), dtype=np.complex128)
    common_phase = 0.0
    for t in range(n_frames):
        tau = base_delays + spec.delay_drift_bins * np.sin(
            2.0 * np.pi * drift_rate * t / max(n_frames, 1) + drift_phase
        )
        steering = np.exp(-1j * 2.0 * np.pi * np.outer(tau, k) / n_subcarriers)
        frame = (amplitudes[t, :, None] * steering).sum(axis=0) * absorption
        common_phase += rng.normal(0.0, spec.phase_noise)
        frame *= np.exp(1j * common_phase)

        if spec.notch_strength > 0.0:
            center = 0.22 * np.sin(2.0 * np.pi * t / max(n_frames - 1, 1))
            width = 0.045 + 0.015 * np.cos(2.0 * np.pi * t / max(n_frames - 1, 1))
            notch = 1.0 - spec.notch_strength * np.exp(-0.5 * ((f - center) / width) ** 2)
            frame *= notch
        h[t] = frame

    rms = np.sqrt(np.mean(np.abs(h) ** 2))
    h /= max(rms, 1e-12)
    return h.astype(np.complex64)


def generate_synthetic_splits(cfg: dict[str, Any], seed: int) -> dict[str, list[dict[str, Any]]]:
    """Generate train/validation/test sequence dictionaries from configuration."""
    n_subcarriers = int(cfg.get("n_subcarriers", 64))
    n_frames = int(cfg.get("frames_per_sequence", 160))
    split_envs = cfg.get(
        "environments",
        {
            "train": ["calm_fjord", "continental_shelf"],
            "val": ["busy_harbor"],
            "test": ["fast_tow"],
        },
    )
    per_split = cfg.get("sequences_per_environment", {"train": 4, "val": 2, "test": 3})
    result: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    serial = 0
    for split in ("train", "val", "test"):
        env_names = split_envs.get(split, [])
        count = int(per_split.get(split, 1))
        for env_name in env_names:
            if env_name not in DEFAULT_ENVIRONMENTS:
                raise KeyError(f"Unknown synthetic environment '{env_name}'.")
            for rep in range(count):
                episode_seed = seed + 1009 * serial + 37 * rep
                csi = generate_episode(DEFAULT_ENVIRONMENTS[env_name], n_frames, n_subcarriers, episode_seed)
                result[split].append(
                    {
                        "id": f"{split}_{env_name}_{rep:02d}",
                        "environment": env_name,
                        "csi": csi,
                        "source": "synthetic_sanity_check",
                    }
                )
                serial += 1
    return result
