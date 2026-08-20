from __future__ import annotations

from typing import Any

import torch
from torch import nn


class UnstructuredForecastNet(nn.Module):
    """Parameter-controlled complex-CSI forecaster without additive/low-rank heads.

    It intentionally shares the frame encoder and temporal context pattern used by
    AquaPilot, but decodes the complete frequency response through one generic
    nonlinear head. This isolates the value of the proposed physical structure.
    """

    def __init__(
        self,
        n_subcarriers: int,
        hidden_dim: int = 48,
        conv_kernel: int = 5,
        dropout: float = 0.05,
        min_sigma: float = 1e-3,
        fixed_sigma: float | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.n_subcarriers = int(n_subcarriers)
        self.hidden_dim = int(hidden_dim)
        self.rank = 0
        self.min_sigma = float(min_sigma)
        self.fixed_sigma = None if fixed_sigma is None else float(fixed_sigma)
        if self.fixed_sigma is not None and self.fixed_sigma <= 0.0:
            raise ValueError("fixed_sigma must be positive when provided.")
        padding = int(conv_kernel) // 2
        self.frame_encoder = nn.Sequential(
            nn.Conv1d(3, hidden_dim, int(conv_kernel), padding=padding),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, int(conv_kernel), padding=padding),
            nn.GELU(),
        )
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.context_norm = nn.LayerNorm(hidden_dim)
        decoder_input = 2 * hidden_dim + 4
        output_dim = 2 if self.fixed_sigma is not None else 4
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.register_buffer("frequency_coordinate", torch.linspace(-1.0, 1.0, self.n_subcarriers))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 4 or x.shape[-1] != 3:
            raise ValueError("Expected x with shape [batch, history, frequency, 3].")
        batch, history, n_freq, _ = x.shape
        if n_freq != self.n_subcarriers:
            raise ValueError(f"Configured for {self.n_subcarriers} subcarriers, received {n_freq}.")

        pooled_frames: list[torch.Tensor] = []
        local_last: torch.Tensor | None = None
        for index in range(history):
            encoded = self.frame_encoder(x[:, index].transpose(1, 2))
            observed_weight = 0.25 + 0.75 * x[:, index, :, 2].unsqueeze(1)
            pooled = (encoded * observed_weight).sum(dim=-1) / observed_weight.sum(dim=-1).clamp_min(1.0)
            pooled_frames.append(pooled)
            if index == history - 1:
                local_last = encoded.transpose(1, 2)
        assert local_last is not None
        _, hidden = self.temporal(torch.stack(pooled_frames, dim=1))
        context = self.context_norm(hidden[-1])
        context_expanded = context[:, None, :].expand(batch, n_freq, self.hidden_dim)
        coordinate = self.frequency_coordinate[None, :, None].expand(batch, n_freq, 1)
        last_frame = x[:, -1, :, :]
        decoded = self.decoder(torch.cat((context_expanded, local_last, last_frame, coordinate), dim=-1))
        mean = decoded[..., :2]
        if self.fixed_sigma is None:
            log_sigma_raw = decoded[..., 2:].clamp(-7.0, 3.0)
            sigma = torch.nn.functional.softplus(log_sigma_raw) + self.min_sigma
        else:
            sigma = torch.full_like(mean, float(self.fixed_sigma))
            log_sigma_raw = torch.log(sigma.clamp_min(1e-8))
        zeros = torch.zeros_like(mean)
        coefficients = mean.new_zeros((batch, 2, 0))
        return {
            "mean": mean,
            "sigma": sigma,
            "log_sigma_raw": log_sigma_raw,
            "time_component": zeros,
            "frequency_component": zeros,
            "low_rank_component": zeros,
            "skip_component": zeros,
            "coefficients": coefficients,
        }

    @classmethod
    def from_config(cls, n_subcarriers: int, cfg: dict[str, Any]) -> "UnstructuredForecastNet":
        return cls(
            n_subcarriers=n_subcarriers,
            hidden_dim=int(cfg.get("hidden_dim", 48)),
            conv_kernel=int(cfg.get("conv_kernel", 5)),
            dropout=float(cfg.get("dropout", 0.05)),
            min_sigma=float(cfg.get("min_sigma", 1e-3)),
            fixed_sigma=(None if cfg.get("fixed_sigma") is None else float(cfg["fixed_sigma"])),
        )
