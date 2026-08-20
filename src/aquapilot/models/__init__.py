from __future__ import annotations

from typing import Any

from torch import nn

from .baselines import linear_time_extrapolation, persistence_forecast
from .classical import (
    ComplexFrequencyPrior,
    fit_frequency_prior,
    interpolate_observation,
    lmmse_estimate,
    omp_delay_estimate,
)
from .phymask import AquaPilotNet
from .unstructured import UnstructuredForecastNet


def build_forecast_model(n_subcarriers: int, cfg: dict[str, Any]) -> nn.Module:
    kind = str(cfg.get("kind", "aquapilot")).lower()
    if kind in {"aquapilot", "structured", "phymask"}:
        return AquaPilotNet.from_config(n_subcarriers, cfg)
    if kind in {"unstructured", "dense", "neural_baseline"}:
        return UnstructuredForecastNet.from_config(n_subcarriers, cfg)
    raise ValueError(f"Unknown model kind: {kind}")


__all__ = [
    "AquaPilotNet",
    "UnstructuredForecastNet",
    "build_forecast_model",
    "persistence_forecast",
    "linear_time_extrapolation",
    "ComplexFrequencyPrior",
    "fit_frequency_prior",
    "interpolate_observation",
    "lmmse_estimate",
    "omp_delay_estimate",
]
