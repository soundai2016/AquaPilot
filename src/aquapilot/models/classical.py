from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class ComplexFrequencyPrior:
    mean: np.ndarray
    covariance: np.ndarray
    shrinkage: float
    fit_records: tuple[dict[str, Any], ...] = ()
    fit_indices: tuple[np.ndarray, ...] = ()


def _training_allocation(lengths: np.ndarray, max_frames: int | None) -> np.ndarray:
    """Guarantee one frame per record; apportion the rest by residual capacity."""
    available = int(lengths.sum())
    if max_frames is not None and (not isinstance(max_frames, (int, np.integer)) or max_frames < 2):
        raise ValueError("max_frames must be None or an integer of at least two.")
    budget = available if max_frames is None else min(int(max_frames), available)
    if budget < 2:
        raise ValueError("At least two training frames are required to fit an LMMSE prior.")
    if budget < len(lengths):
        raise ValueError("max_covariance_frames must cover at least one frame from each training record.")
    if budget == available:
        return lengths.copy()
    capacity = lengths - 1
    numerator = (budget - len(lengths)) * capacity
    shares, remainder = np.divmod(numerator, capacity.sum())
    counts = 1 + shares
    extra = budget - int(counts.sum())
    counts[np.argsort(-remainder, kind="stable")[:extra]] += 1
    return counts


def fit_frequency_prior(
    sequences: Iterable[dict[str, Any]],
    shrinkage: float = 0.05,
    max_frames: int | None = 20_000,
) -> ComplexFrequencyPrior:
    """Fit a training-only proper-complex frequency covariance.

    Normalize each training record by its RMS. When the cap binds, sample
    across every record and throughout its duration, with one draw per time
    stratum. Sampling depends on the record identity, not the network seed.
    """
    records = []
    for i, sequence in enumerate(sequences):
        csi = np.asarray(sequence["csi"])
        if csi.ndim != 2 or min(csi.shape) < 1:
            raise ValueError("Each training sequence must contain nonempty [time, frequency] CSI.")
        record_id = str(sequence.get("id", f"record_{i}"))
        records.append((str(sequence.get("environment", "unspecified")), record_id, csi))
    records.sort(key=lambda row: row[:2])
    if len({row[:2] for row in records}) != len(records):
        raise ValueError("LMMSE fitting requires distinct training record identities within an environment.")
    if len({row[2].shape[1] for row in records}) > 1:
        raise ValueError("Training records must use the same frequency grid.")
    lengths = np.array([len(row[2]) for row in records], dtype=np.int64)
    counts = _training_allocation(lengths, max_frames)
    frames, fit_records, fit_indices = [], [], []
    for (environment, record_id, csi), length, count in zip(records, lengths, counts):
        if count == length:
            indices = np.arange(length, dtype=np.int64)
        else:
            key = f"{environment}\0{record_id}".encode("utf-8")
            seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
            edges = (np.arange(count + 1, dtype=np.int64) * length) // count
            indices = np.random.default_rng(seed).integers(edges[:-1], edges[1:], dtype=np.int64)
        scale = max(float(np.sqrt(np.mean(np.abs(csi) ** 2))), 1e-8)
        frames.append(np.asarray(csi[indices], dtype=np.complex128) / scale)
        fit_records.append({"environment": environment, "record_id": record_id,
                            "available_frames": int(length), "used_frames": int(count)})
        fit_indices.append(indices)
    matrix = np.concatenate(frames, axis=0)
    mean = matrix.mean(axis=0)
    centered = matrix - mean[None, :]
    covariance = centered.T @ centered.conj() / max(matrix.shape[0] - 1, 1)
    covariance = 0.5 * (covariance + covariance.conj().T)
    alpha = float(np.clip(shrinkage, 0.0, 1.0))
    average_variance = max(float(np.real(np.trace(covariance)) / covariance.shape[0]), 1e-8)
    covariance = (1.0 - alpha) * covariance + alpha * average_variance * np.eye(covariance.shape[0])
    return ComplexFrequencyPrior(mean=mean, covariance=covariance, shrinkage=alpha,
                                 fit_records=tuple(fit_records), fit_indices=tuple(fit_indices))


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
