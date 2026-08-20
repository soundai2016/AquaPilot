from __future__ import annotations

import torch


def persistence_forecast(x: torch.Tensor) -> torch.Tensor:
    """Use the last interpolated sparse-CSI frame as the next-frame prior."""
    if x.ndim != 4 or x.shape[-1] < 2:
        raise ValueError("x must have shape [B,T,F,C>=2].")
    return x[:, -1, :, :2]


def linear_time_extrapolation(x: torch.Tensor, damping: float = 0.7) -> torch.Tensor:
    """Damped linear extrapolation from the final two interpolated frames."""
    if x.shape[1] < 2:
        return persistence_forecast(x)
    previous = x[:, -2, :, :2]
    current = x[:, -1, :, :2]
    return current + float(damping) * (current - previous)
