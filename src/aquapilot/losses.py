from __future__ import annotations

from typing import Any

import torch


def gaussian_nll(mean: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    variance = sigma.square().clamp_min(1e-8)
    return (0.5 * (target - mean).square() / variance + torch.log(sigma.clamp_min(1e-8))).mean()


def normalized_mse(mean: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = (mean - target).square().sum(dim=(-2, -1))
    denominator = target.square().sum(dim=(-2, -1)).clamp_min(1e-8)
    return (numerator / denominator).mean()


def causal_delay_penalty(mean: torch.Tensor, max_delay_bins: int) -> torch.Tensor:
    """Penalize circular-CIR energy outside the assumed causal delay support."""
    complex_mean = torch.complex(mean[..., 0], mean[..., 1])
    cir = torch.fft.ifft(complex_mean, dim=-1)
    support = max(1, min(int(max_delay_bins), cir.shape[-1]))
    if support == cir.shape[-1]:
        return mean.new_tensor(0.0)
    tail = cir[..., support:]
    return tail.abs().square().mean()


def second_difference_penalty(component: torch.Tensor) -> torch.Tensor:
    if component.shape[-2] < 3:
        return component.new_tensor(0.0)
    second = component[:, 2:, :] - 2.0 * component[:, 1:-1, :] + component[:, :-2, :]
    return second.square().mean()


def total_loss(outputs: dict[str, torch.Tensor], target: torch.Tensor, cfg: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
    nll = gaussian_nll(outputs["mean"], outputs["sigma"], target)
    nmse = normalized_mse(outputs["mean"], target)
    causal = causal_delay_penalty(outputs["mean"], int(cfg.get("max_delay_bins", 16)))
    smooth = second_difference_penalty(outputs["frequency_component"])
    rank_l2 = outputs["coefficients"].square().mean() if outputs["coefficients"].numel() else target.new_tensor(0.0)
    loss = (
        float(cfg.get("nll_weight", 1.0)) * nll
        + float(cfg.get("nmse_weight", 0.25)) * nmse
        + float(cfg.get("causal_weight", 0.02)) * causal
        + float(cfg.get("smooth_weight", 0.005)) * smooth
        + float(cfg.get("rank_weight", 1e-4)) * rank_l2
    )
    parts = {
        "loss": float(loss.detach().cpu()),
        "nll": float(nll.detach().cpu()),
        "nmse_linear": float(nmse.detach().cpu()),
        "causal": float(causal.detach().cpu()),
        "smooth": float(smooth.detach().cpu()),
        "rank_l2": float(rank_l2.detach().cpu()),
    }
    return loss, parts
