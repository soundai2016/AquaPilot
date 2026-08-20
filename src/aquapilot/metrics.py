from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr


def _complex(array_ri: torch.Tensor) -> torch.Tensor:
    return torch.complex(array_ri[..., 0], array_ri[..., 1])


def nmse_db(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    error = (_complex(pred) - _complex(target)).abs().square().sum(dim=-1)
    power = _complex(target).abs().square().sum(dim=-1).clamp_min(1e-10)
    return 10.0 * torch.log10((error / power).clamp_min(1e-12))


def evm_percent(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    error = (_complex(pred) - _complex(target)).abs().square().mean(dim=-1)
    power = _complex(target).abs().square().mean(dim=-1).clamp_min(1e-10)
    return 100.0 * torch.sqrt(error / power)


def interval_coverage(mean: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor, z: float = 1.6448536269514722) -> torch.Tensor:
    inside = (target >= mean - z * sigma) & (target <= mean + z * sigma)
    return inside.float().mean(dim=(-2, -1))


def uncertainty_error_spearman(mean: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor) -> float:
    uncertainty = sigma.square().sum(dim=-1).sqrt().detach().cpu().numpy().reshape(-1)
    error = (mean - target).square().sum(dim=-1).sqrt().detach().cpu().numpy().reshape(-1)
    finite = np.isfinite(uncertainty) & np.isfinite(error)
    if finite.sum() < 2:
        return 0.0
    u = uncertainty[finite]
    e = error[finite]
    if np.ptp(u) <= 1e-12 or np.ptp(e) <= 1e-12:
        return 0.0
    value = spearmanr(u, e).statistic
    return float(value) if np.isfinite(value) else 0.0


def summarize_tensors(
    mean: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, Any]:
    nmse = nmse_db(mean, target)
    evm = evm_percent(mean, target)
    coverage = interval_coverage(mean, sigma, target)
    return {
        "nmse_db_mean": float(nmse.mean().cpu()),
        "nmse_db_std": float(nmse.std(unbiased=False).cpu()),
        "evm_percent_mean": float(evm.mean().cpu()),
        "coverage_90": float(coverage.mean().cpu()),
        "coverage_error_90": float(abs(coverage.mean().cpu().item() - 0.90)),
        "uncertainty_error_spearman": uncertainty_error_spearman(mean, sigma, target),
        "num_frames": int(mean.shape[0]),
    }
