from __future__ import annotations

from typing import Any

import torch
from torch import nn


class AquaPilotNet(nn.Module):
    """Structured complex-CSI predictor with heteroscedastic uncertainty."""

    def __init__(
        self,
        n_subcarriers: int,
        hidden_dim: int = 48,
        rank: int = 2,
        conv_kernel: int = 5,
        dropout: float = 0.05,
        min_sigma: float = 1e-3,
        fixed_sigma: float | None = None,
    ) -> None:
        super().__init__()
        self.n_subcarriers = int(n_subcarriers)
        self.hidden_dim = int(hidden_dim)
        self.rank = int(rank)
        self.min_sigma = float(min_sigma)
        self.fixed_sigma = None if fixed_sigma is None else float(fixed_sigma)
        if self.fixed_sigma is not None and self.fixed_sigma <= 0.0:
            raise ValueError("fixed_sigma must be positive when provided.")
        padding = conv_kernel // 2

        self.frame_encoder = nn.Sequential(
            nn.Conv1d(3, hidden_dim, conv_kernel, padding=padding),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, conv_kernel, padding=padding),
            nn.GELU(),
        )
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.time_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2))
        self.skip_gate = nn.Sequential(nn.Linear(hidden_dim, 2), nn.Sigmoid())

        marginal_input = 2 * hidden_dim + 1
        self.frequency_head = nn.Sequential(
            nn.Linear(marginal_input, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.sigma_head = None if self.fixed_sigma is not None else nn.Sequential(
            nn.Linear(marginal_input, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

        if rank > 0:
            self.coefficient_head = nn.Linear(hidden_dim, 2 * rank)
            frequency = torch.linspace(-1.0, 1.0, self.n_subcarriers)
            initial = []
            for channel in range(2):
                bases = []
                for component in range(rank):
                    harmonic = component // 2 + 1
                    if (component + channel) % 2 == 0:
                        bases.append(torch.sin(torch.pi * harmonic * frequency))
                    else:
                        bases.append(torch.cos(torch.pi * harmonic * frequency))
                initial.append(torch.stack(bases, dim=0))
            self.basis = nn.Parameter(torch.stack(initial, dim=0))
        else:
            self.coefficient_head = None
            self.register_parameter("basis", None)

        self.register_buffer("frequency_coordinate", torch.linspace(-1.0, 1.0, self.n_subcarriers))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 4 or x.shape[-1] != 3:
            raise ValueError("Expected x with shape [batch, history, frequency, 3].")
        batch, history, n_freq, _ = x.shape
        if n_freq != self.n_subcarriers:
            raise ValueError(f"Configured for {self.n_subcarriers} subcarriers, received {n_freq}.")

        encoded_frames = []
        local_last = None
        for t in range(history):
            encoded = self.frame_encoder(x[:, t].transpose(1, 2))
            observed_weight = 0.25 + 0.75 * x[:, t, :, 2].unsqueeze(1)
            pooled = (encoded * observed_weight).sum(dim=-1) / observed_weight.sum(dim=-1).clamp_min(1.0)
            encoded_frames.append(pooled)
            if t == history - 1:
                local_last = encoded.transpose(1, 2)
        sequence_features = torch.stack(encoded_frames, dim=1)
        _, hidden = self.temporal(sequence_features)
        context = self.context_norm(hidden[-1])
        assert local_last is not None

        context_expanded = context[:, None, :].expand(batch, n_freq, self.hidden_dim)
        coordinate = self.frequency_coordinate[None, :, None].expand(batch, n_freq, 1)
        marginal_input = torch.cat((context_expanded, local_last, coordinate), dim=-1)

        time_component = self.time_head(context)[:, None, :].expand(batch, n_freq, 2)
        frequency_component = self.frequency_head(marginal_input)
        last_frame = x[:, -1, :, :2]
        skip_component = self.skip_gate(context)[:, None, :] * last_frame

        if self.rank > 0:
            coefficients = self.coefficient_head(context).view(batch, 2, self.rank)
            low_rank = torch.einsum("bck,ckf->bfc", coefficients, self.basis)
        else:
            coefficients = x.new_zeros((batch, 2, 0))
            low_rank = x.new_zeros((batch, n_freq, 2))

        mean = time_component + frequency_component + low_rank + skip_component
        if self.sigma_head is None:
            sigma = torch.full_like(mean, float(self.fixed_sigma))
            log_sigma = torch.log(sigma.clamp_min(1e-8))
        else:
            log_sigma = self.sigma_head(marginal_input).clamp(-7.0, 3.0)
            sigma = torch.nn.functional.softplus(log_sigma) + self.min_sigma
        return {
            "mean": mean,
            "sigma": sigma,
            "log_sigma_raw": log_sigma,
            "time_component": time_component,
            "frequency_component": frequency_component,
            "low_rank_component": low_rank,
            "skip_component": skip_component,
            "coefficients": coefficients,
        }

    def extra_repr(self) -> str:
        return f"n_subcarriers={self.n_subcarriers}, hidden_dim={self.hidden_dim}, rank={self.rank}"

    @classmethod
    def from_config(cls, n_subcarriers: int, cfg: dict[str, Any]) -> "AquaPilotNet":
        return cls(
            n_subcarriers=n_subcarriers,
            hidden_dim=int(cfg.get("hidden_dim", 48)),
            rank=int(cfg.get("rank", 2)),
            conv_kernel=int(cfg.get("conv_kernel", 5)),
            dropout=float(cfg.get("dropout", 0.05)),
            min_sigma=float(cfg.get("min_sigma", 1e-3)),
            fixed_sigma=(None if cfg.get("fixed_sigma") is None else float(cfg["fixed_sigma"])),
        )
