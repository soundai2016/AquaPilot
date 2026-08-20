from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class ComplexFrequencyPrior:
    mean: np.ndarray
    covariance: np.ndarray
    shrinkage: float


def fit_frequency_prior(
    sequences: Iterable[dict[str, Any]],
    shrinkage: float = 0.05,
    max_frames: int | None = 20_000,
) -> ComplexFrequencyPrior:
    """Fit a training-only proper-complex frequency covariance.

    Each contiguous sequence is RMS-normalized once. This removes arbitrary
    recording gain while preserving frame-to-frame fades that a current-frame
    LMMSE receiver must represent. A small diagonal shrinkage keeps pilot
    submatrices invertible.
    """
    frames: list[np.ndarray] = []
    for sequence in sequences:
        csi = np.asarray(sequence["csi"])
        if csi.ndim != 2:
            raise ValueError("Each sequence must contain csi with shape [time, frequency].")
        scale = max(float(np.sqrt(np.mean(np.abs(csi) ** 2))), 1e-8)
        frames.extend(np.asarray(frame, dtype=np.complex128) / scale for frame in csi)
        if max_frames is not None and len(frames) >= max_frames:
            frames = frames[:max_frames]
            break
    if len(frames) < 2:
        raise ValueError("At least two training frames are required to fit an LMMSE prior.")
    matrix = np.stack(frames, axis=0)
    mean = matrix.mean(axis=0)
    centered = matrix - mean[None, :]
    covariance = centered.conj().T @ centered / max(matrix.shape[0] - 1, 1)
    covariance = 0.5 * (covariance + covariance.conj().T)
    alpha = float(np.clip(shrinkage, 0.0, 1.0))
    average_variance = max(float(np.real(np.trace(covariance)) / covariance.shape[0]), 1e-8)
    covariance = (1.0 - alpha) * covariance + alpha * average_variance * np.eye(covariance.shape[0])
    return ComplexFrequencyPrior(mean=mean.astype(np.complex128), covariance=covariance, shrinkage=alpha)


def interpolate_observation(observation: np.ndarray, mask: np.ndarray) -> np.ndarray:
    observation = np.asarray(observation, dtype=np.complex128)
    mask = np.asarray(mask, dtype=bool)
    if observation.ndim != 1 or observation.shape != mask.shape:
        raise ValueError("observation and mask must be one-dimensional and shape-matched.")
    index = np.flatnonzero(mask)
    if index.size == 0:
        return np.zeros_like(observation)
    if index.size == 1:
        return np.full_like(observation, observation[index[0]])
    coordinate = np.arange(observation.size)
    real = np.interp(coordinate, index, observation[index].real)
    imag = np.interp(coordinate, index, observation[index].imag)
    return real + 1j * imag


def lmmse_estimate(
    observation: np.ndarray,
    mask: np.ndarray,
    prior: ComplexFrequencyPrior,
    observation_variance: float,
) -> np.ndarray:
    observation = np.asarray(observation, dtype=np.complex128)
    mask = np.asarray(mask, dtype=bool)
    pilot = np.flatnonzero(mask)
    if pilot.size == 0:
        return prior.mean.copy()
    covariance = prior.covariance
    if covariance.shape != (observation.size, observation.size):
        raise ValueError("Prior and observation frequency dimensions differ.")
    cross = covariance[:, pilot]
    system = covariance[np.ix_(pilot, pilot)] + max(float(observation_variance), 1e-10) * np.eye(pilot.size)
    innovation = observation[pilot] - prior.mean[pilot]
    try:
        weights = np.linalg.solve(system, innovation)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(system, rcond=1e-8) @ innovation
    return prior.mean + cross @ weights


def omp_delay_estimate(
    observation: np.ndarray,
    mask: np.ndarray,
    delay_bins: int = 24,
    sparsity: int = 6,
    ridge: float = 1e-6,
) -> np.ndarray:
    """Sparse delay-domain OMP reconstruction from arbitrary pilot subcarriers."""
    observation = np.asarray(observation, dtype=np.complex128)
    mask = np.asarray(mask, dtype=bool)
    n_freq = observation.size
    pilot = np.flatnonzero(mask)
    if pilot.size == 0:
        return np.zeros_like(observation)
    delay_bins = max(1, min(int(delay_bins), n_freq))
    sparsity = max(1, min(int(sparsity), delay_bins, pilot.size))
    frequency = np.arange(n_freq, dtype=np.float64)
    delay = np.arange(delay_bins, dtype=np.float64)
    dictionary = np.exp(-2j * np.pi * frequency[:, None] * delay[None, :] / n_freq) / np.sqrt(n_freq)
    sensing = dictionary[pilot]
    y = observation[pilot]
    residual = y.copy()
    support: list[int] = []
    coefficients = np.empty(0, dtype=np.complex128)
    for _ in range(sparsity):
        correlation = sensing.conj().T @ residual
        if support:
            correlation[np.asarray(support)] = 0.0
        chosen = int(np.argmax(np.abs(correlation)))
        support.append(chosen)
        selected = sensing[:, support]
        gram = selected.conj().T @ selected + max(float(ridge), 0.0) * np.eye(len(support))
        coefficients = np.linalg.solve(gram, selected.conj().T @ y)
        residual = y - selected @ coefficients
        if np.linalg.norm(residual) <= 1e-8 * max(np.linalg.norm(y), 1e-12):
            break
    impulse = np.zeros(delay_bins, dtype=np.complex128)
    impulse[np.asarray(support)] = coefficients
    return dictionary @ impulse
