from __future__ import annotations

import numpy as np
import torch


def comb_mask(n_subcarriers: int, budget: int, offset: int = 0) -> np.ndarray:
    """Return an approximately uniform circular mask with an exact budget."""
    if not 1 <= budget <= n_subcarriers:
        raise ValueError("budget must be between 1 and n_subcarriers.")
    positions = np.floor((np.arange(budget, dtype=np.float64) + 0.5) * n_subcarriers / budget).astype(int)
    positions = (positions + int(offset)) % n_subcarriers
    if np.unique(positions).size != budget:
        raise RuntimeError("Internal comb construction produced duplicate pilot positions.")
    mask = np.zeros(n_subcarriers, dtype=bool)
    mask[positions] = True
    return mask


def random_mask(n_subcarriers: int, budget: int, rng: np.random.Generator) -> np.ndarray:
    if not 1 <= budget <= n_subcarriers:
        raise ValueError("budget must be between 1 and n_subcarriers.")
    mask = np.zeros(n_subcarriers, dtype=bool)
    mask[rng.choice(n_subcarriers, size=budget, replace=False)] = True
    return mask


def _rbf_correlation(n_freq: int, length_scale: float, *, dtype=np.float64) -> np.ndarray:
    positions = np.arange(n_freq, dtype=dtype)
    distance = positions[:, None] - positions[None, :]
    return np.exp(-0.5 * np.square(distance / max(float(length_scale), 1e-3)))


def _eligible_candidates(n_freq: int, selected: list[int], min_spacing: int) -> np.ndarray:
    candidates = np.ones(n_freq, dtype=bool)
    if selected:
        index = np.arange(n_freq)
        for chosen in selected:
            candidates &= np.abs(index - chosen) >= max(1, int(min_spacing))
    candidates[np.asarray(selected, dtype=int)] = False
    return candidates


def active_variance_reduction_mask(
    mean_ri: np.ndarray,
    sigma_ri: np.ndarray,
    budget: int,
    min_spacing: int = 2,
    length_scale: float = 4.0,
    observation_sigma: float = 0.03,
    gradient_weight: float = 0.0,
) -> np.ndarray:
    """Greedy A-optimal selection; observation_sigma is per real component.

    A circular complex noise RMS must be divided by sqrt(2) before calling.
    The caller may explicitly retain the original larger variance for replay.
    """
    mean_ri = np.asarray(mean_ri, dtype=np.float64)
    sigma_ri = np.asarray(sigma_ri, dtype=np.float64)
    if mean_ri.shape != sigma_ri.shape or mean_ri.ndim != 2 or mean_ri.shape[1] != 2:
        raise ValueError("mean_ri and sigma_ri must both have shape [frequency, 2].")
    n_freq = mean_ri.shape[0]
    if not 1 <= int(budget) <= n_freq:
        raise ValueError("budget must be between 1 and the number of subcarriers.")
    sigma_ri = np.maximum(sigma_ri, 1e-6)
    correlation = _rbf_correlation(n_freq, length_scale)
    covariances = [
        np.outer(sigma_ri[:, channel], sigma_ri[:, channel]) * correlation
        for channel in range(2)
    ]
    obs_var = max(float(observation_sigma) ** 2, 1e-10)

    channel = mean_ri[:, 0] + 1j * mean_ri[:, 1]
    gradient = np.abs(np.gradient(channel))
    gradient = gradient / max(float(gradient.max()), 1e-12)

    selected: list[int] = []
    for _ in range(int(budget)):
        eligible = _eligible_candidates(n_freq, selected, min_spacing)
        if not eligible.any():
            eligible = _eligible_candidates(n_freq, selected, 1)
        scores = np.full(n_freq, -np.inf, dtype=np.float64)
        candidate_indices = np.flatnonzero(eligible)
        for index in candidate_indices:
            reduction = 0.0
            for covariance in covariances:
                column = covariance[:, index]
                reduction += float(np.dot(column, column) / (covariance[index, index] + obs_var))
            scores[index] = reduction + float(gradient_weight) * gradient[index]
        chosen = int(np.argmax(scores))
        selected.append(chosen)

        for channel_index, covariance in enumerate(covariances):
            column = covariance[:, chosen].copy()
            denominator = float(covariance[chosen, chosen] + obs_var)
            updated = covariance - np.outer(column, column) / max(denominator, 1e-12)
            updated = 0.5 * (updated + updated.T)
            diagonal = np.maximum(np.diag(updated), 1e-12)
            np.fill_diagonal(updated, diagonal)
            covariances[channel_index] = updated

    mask = np.zeros(n_freq, dtype=bool)
    mask[np.asarray(selected, dtype=int)] = True
    return mask


def uncertainty_mask(
    mean_ri: np.ndarray,
    sigma_ri: np.ndarray,
    budget: int,
    min_spacing: int = 2,
    gradient_weight: float = 0.35,
    length_scale: float = 4.0,
    observation_sigma: float = 0.03,
) -> np.ndarray:
    """Alias for A-optimal active selection."""
    return active_variance_reduction_mask(
        mean_ri,
        sigma_ri,
        budget=budget,
        min_spacing=min_spacing,
        length_scale=length_scale,
        observation_sigma=observation_sigma,
        gradient_weight=gradient_weight,
    )



def uncertainty_top_mask(
    sigma_ri: np.ndarray,
    budget: int,
    min_spacing: int = 2,
) -> np.ndarray:
    """Select the largest marginal variances with an exact budget."""
    sigma_ri = np.asarray(sigma_ri, dtype=np.float64)
    if sigma_ri.ndim != 2 or sigma_ri.shape[1] != 2:
        raise ValueError("sigma_ri must have shape [frequency, 2].")
    n_freq = sigma_ri.shape[0]
    if not 1 <= int(budget) <= n_freq:
        raise ValueError("budget must be between 1 and the number of subcarriers.")
    score = np.square(np.maximum(sigma_ri, 0.0)).sum(axis=1)
    selected: list[int] = []
    for _ in range(int(budget)):
        eligible = _eligible_candidates(n_freq, selected, min_spacing)
        if not eligible.any():
            eligible = _eligible_candidates(n_freq, selected, 1)
        candidates = np.flatnonzero(eligible)
        chosen = int(candidates[np.argmax(score[candidates])])
        selected.append(chosen)
    mask = np.zeros(n_freq, dtype=bool)
    mask[np.asarray(selected)] = True
    return mask

def posterior_kernel_update(
    prior_mean: torch.Tensor,
    prior_sigma: torch.Tensor,
    observation: torch.Tensor,
    mask: torch.Tensor,
    observation_sigma: float = 0.03,
    length_scale: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the common Gaussian update and return component marginals.

    observation_sigma is the standard deviation of each real/imaginary
    observation component, in the same normalized units as prior_mean.
    """
    if prior_mean.ndim != 3 or prior_mean.shape[-1] != 2:
        raise ValueError("prior_mean must have shape [B,F,2].")
    if prior_sigma.shape != prior_mean.shape or observation.shape != prior_mean.shape:
        raise ValueError("prior_sigma and observation must match prior_mean.")
    batch, n_freq, _ = prior_mean.shape
    mask = mask.to(device=prior_mean.device, dtype=torch.bool)
    if mask.shape != (batch, n_freq):
        raise ValueError("mask must have shape [B,F].")

    positions = torch.arange(n_freq, device=prior_mean.device, dtype=prior_mean.dtype)
    distance = positions[:, None] - positions[None, :]
    correlation = torch.exp(-0.5 * (distance / max(float(length_scale), 1e-3)).square())
    obs_var = max(float(observation_sigma) ** 2, 1e-10)
    jitter = max(obs_var * 1e-4, 1e-8)

    posterior_mean = prior_mean.clone()
    posterior_var = prior_sigma.square().clamp_min(1e-10).clone()
    for batch_index in range(batch):
        pilot_indices = torch.nonzero(mask[batch_index], as_tuple=False).squeeze(-1)
        if pilot_indices.numel() == 0:
            continue
        for component in range(2):
            sigma = prior_sigma[batch_index, :, component].clamp_min(1e-6)
            covariance = sigma[:, None] * correlation * sigma[None, :]
            cross = covariance[:, pilot_indices]
            pilot_covariance = covariance[pilot_indices][:, pilot_indices]
            system = pilot_covariance + (obs_var + jitter) * torch.eye(
                pilot_indices.numel(), device=prior_mean.device, dtype=prior_mean.dtype
            )
            innovation = (
                observation[batch_index, pilot_indices, component]
                - prior_mean[batch_index, pilot_indices, component]
            )
            solved_innovation = torch.linalg.solve(system, innovation.unsqueeze(-1)).squeeze(-1)
            posterior_mean[batch_index, :, component] = (
                prior_mean[batch_index, :, component] + cross @ solved_innovation
            )
            solved_cross = torch.linalg.solve(system, cross.transpose(0, 1))
            reduction = (cross * solved_cross.transpose(0, 1)).sum(dim=1)
            posterior_var[batch_index, :, component] = (
                torch.diagonal(covariance) - reduction
            ).clamp_min(1e-10)

    return posterior_mean, posterior_var.sqrt()
