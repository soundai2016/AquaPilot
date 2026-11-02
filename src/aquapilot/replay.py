"""Shared definitions for measured-channel replay and its reference estimators."""
from __future__ import annotations

from typing import Any

import numpy as np

from aquapilot.data.prepared import interpolate_complex
from aquapilot.models.classical import interpolate_observation
from aquapilot.provenance import PROTOCOL_ID


BASELINE_STAGES = {
    "zero": "analytic reference; no observations",
    "persistence_prior": "pre-acquisition; previous noisy interpolated frame",
    "current_interpolation": "post-acquisition; current noisy pilots only",
    "persistence_innovation": "post-acquisition; previous frame plus interpolated current innovation",
}


def validate_replay_config(cfg: dict[str, Any]) -> None:
    if int(cfg["data"].get("horizon", 1)) != 1:
        raise ValueError("Closed-loop replay supports horizon=1 only.")
    if int(cfg["data"].get("history", 8)) < 2:
        raise ValueError("History must contain at least two frames.")
    convention = cfg["pilot"].get("update_noise_convention", "component")
    if convention not in {"component", "complex_rms_as_component"}:
        raise ValueError("pilot.update_noise_convention must be 'component' or 'complex_rms_as_component'.")
    storage = cfg.get("eval", {}).get("prediction_storage", "compact")
    if storage not in {"none", "compact", "full"}:
        raise ValueError("eval.prediction_storage must be 'none', 'compact', or 'full'.")


def update_component_sigma(complex_rms: float, history_scale: float, convention: str) -> float:
    """Noise standard deviation used in each real-component Gaussian update."""
    if not np.isfinite(complex_rms) or complex_rms < 0 or not np.isfinite(history_scale) or history_scale <= 0:
        raise ValueError("Noise RMS and history scale must be finite and nonnegative/positive.")
    if convention == "component":
        return float(complex_rms / (np.sqrt(2.0) * history_scale))
    if convention == "complex_rms_as_component":
        return float(complex_rms / history_scale)
    raise ValueError(f"Unknown update-noise convention: {convention!r}")


def as_real_imag(array: np.ndarray) -> np.ndarray:
    return np.stack((array.real, array.imag), axis=-1).astype(np.float32)


def acquire_pilots(reference: np.ndarray, noise: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return only acquired noisy responses. Zeros elsewhere are placeholders."""
    reference, noise, mask = np.asarray(reference), np.asarray(noise), np.asarray(mask)
    if reference.shape != mask.shape or noise.shape != mask.shape or mask.dtype != np.bool_:
        raise ValueError("Reference, noise and boolean pilot mask must have the same shape.")
    observed = np.zeros(mask.shape, dtype=np.result_type(reference, noise))
    observed[mask] = reference[mask] + noise[mask]
    return observed


def matched_information_baselines(
    observed_history: np.ndarray,
    mask_history: np.ndarray,
    current_observation: np.ndarray,
    current_mask: np.ndarray,
    history_scale: float,
) -> dict[str, np.ndarray]:
    """No reference CSI or fitted test residuals enter these estimators.

    Inputs are in physical channel units. Returned arrays use the same scalar
    normalization as the learned estimator. Linear interpolation operates on
    real/imaginary parts and holds the nearest pilot value outside their span.
    """
    previous = interpolate_complex(observed_history[-1:], mask_history[-1:])[0]
    current = interpolate_observation(current_observation, current_mask)
    innovation = interpolate_observation(current_observation - previous, current_mask)
    estimates = {
        "zero": np.zeros_like(previous),
        "persistence_prior": previous,
        "current_interpolation": current,
        "persistence_innovation": previous + innovation,
    }
    return {name: as_real_imag(value / history_scale) for name, value in estimates.items()}


def protocol_metadata(cfg: dict[str, Any]) -> dict[str, Any]:
    convention = cfg["pilot"].get("update_noise_convention", "component")
    return {
        "id": PROTOCOL_ID,
        "update_noise_convention": convention,
        "observation_sigma_definition": "complex noise RMS divided by full reference-sequence RMS",
        "injected_component_variance": "(observation_sigma * max(sequence_RMS,1e-6))^2 / 2",
        "assumed_component_variance": "injected component variance" if convention == "component" else "complex RMS used as a component standard deviation (twice the injected component variance)",
        "noise_variance_information": "Known injected noise level supplied by the replay; not estimated from received pilots",
        "noise_generator": "NumPy default_rng; complete sequence real normals then imaginary normals; divide by sqrt(2)",
        "noise_seed_rule": "evaluation_seed + 4241 * ordered_sequence_index",
        "initial_history": "same rotating comb and physical noise for every policy",
        "receiver_observations": "Frequency-aligned arrays carry noisy pilot values and zeros elsewhere, with a separate mask; full reference responses are reserved for the simulator and scoring.",
        "normalization": "RMS of interpolated noisy sparse history; same scalar for prediction and reference",
        "nonpilot_metrics": "computed only on the complement of the current policy's pilot mask",
        "baseline_stages": BASELINE_STAGES,
        "record_independence": "not assumed; recording identities and source paths are preserved",
        "prediction_storage": cfg.get("eval", {}).get("prediction_storage", "compact"),
        "run_link_proxy": bool(cfg.get("eval", {}).get("run_link_proxy", False)),
        "link_proxy_scope": "optional uncoded QPSK one-tap simulation; no feedback, synchronization, packet protocol, or measured goodput",
    }


def check_checkpoint_protocol(bundle: dict[str, Any], cfg: dict[str, Any], n_subcarriers: int) -> None:
    if int(bundle["n_subcarriers"]) != int(n_subcarriers):
        raise ValueError("Checkpoint and evaluation data use different subcarrier counts.")
    original = bundle.get("experiment_config", {}).get("data", {})
    if "history" not in original or "horizon" not in original:
        raise ValueError("Checkpoint lacks recorded history/horizon; supply a checkpoint with its original experiment_config.")
    if int(original["history"]) != int(cfg["data"].get("history", 8)):
        raise ValueError("Checkpoint history length and evaluation history differ.")
    if int(original["horizon"]) != 1 or int(cfg["data"].get("horizon", 1)) != 1:
        raise ValueError("Checkpoint and closed-loop evaluation must both use horizon=1.")
