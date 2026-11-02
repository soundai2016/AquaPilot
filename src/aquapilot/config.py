from __future__ import annotations

import math
from numbers import Integral
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


def validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Reject malformed settings before training or creating result files.

    Missing optional settings keep their documented implementation defaults.
    Dataset dimensions and checkpoint compatibility are checked after loading.
    """
    if not isinstance(cfg, dict):
        raise ConfigError("The YAML root must be a mapping.")
    for key in ("seed", "output_dir", "data", "model", "train", "pilot", "loss", "eval"):
        if key not in cfg:
            raise ConfigError(f"Missing required configuration section: {key}")
    for key in ("data", "model", "train", "pilot", "loss", "eval", "classical"):
        if key in cfg and not isinstance(cfg[key], dict):
            raise ConfigError(f"{key} must be a mapping.")

    def number(section: dict[str, Any], key: str, label: str, *, minimum: float | None = None,
               maximum: float | None = None, integer: bool = False, positive: bool = False) -> None:
        if key not in section or section[key] is None:
            return
        value = section[key]
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{label} must be a finite number.") from exc
        if isinstance(value, bool) or not math.isfinite(numeric):
            raise ConfigError(f"{label} must be a finite number.")
        if integer and not isinstance(value, Integral):
            raise ConfigError(f"{label} must be an integer.")
        if positive and numeric <= 0:
            raise ConfigError(f"{label} must be positive.")
        if minimum is not None and numeric < minimum:
            raise ConfigError(f"{label} must be >= {minimum:g}.")
        if maximum is not None and numeric > maximum:
            raise ConfigError(f"{label} must be <= {maximum:g}.")

    number(cfg, "seed", "seed", minimum=0, integer=True)
    if cfg["seed"] is None:
        raise ConfigError("seed must be a nonnegative integer.")
    if not isinstance(cfg["output_dir"], (str, Path)) or not str(cfg["output_dir"]).strip():
        raise ConfigError("output_dir must be a nonempty path.")
    if cfg["data"].get("kind") not in {"synthetic", "prepared"}:
        raise ConfigError("data.kind must be 'synthetic' or 'prepared'.")
    data = cfg["data"]
    number(data, "history", "data.history", minimum=2, integer=True)
    number(data, "horizon", "data.horizon", minimum=1, integer=True)
    number(data, "n_subcarriers", "data.n_subcarriers", minimum=2, integer=True)
    if data["kind"] == "prepared":
        if not isinstance(data.get("root"), (str, Path)) or not str(data["root"]).strip():
            raise ConfigError("Prepared data require a nonempty data.root path.")
    else:
        number(data, "n_subcarriers", "data.n_subcarriers", minimum=8, integer=True)
        number(data, "frames_per_sequence", "data.frames_per_sequence", minimum=4, integer=True)
        if int(data.get("frames_per_sequence", 160)) < int(data.get("history", 8)) + int(data.get("horizon", 1)):
            raise ConfigError("Synthetic sequences must contain at least history + horizon frames.")

    model = cfg["model"]
    if str(model.get("kind", "aquapilot")).lower() not in {"aquapilot", "structured", "phymask", "unstructured", "dense", "neural_baseline"}:
        raise ConfigError("model.kind must name the AquaPilot or unstructured forecaster.")
    for key, lower in (("hidden_dim", 1), ("rank", 0), ("conv_kernel", 1)):
        number(model, key, f"model.{key}", minimum=lower, integer=True)
    if int(model.get("conv_kernel", 5)) % 2 == 0:
        raise ConfigError("model.conv_kernel must be odd to preserve the frequency dimension.")
    number(model, "dropout", "model.dropout", minimum=0, maximum=1)
    for key in ("min_sigma", "fixed_sigma"):
        number(model, key, f"model.{key}", positive=True)

    train = cfg["train"]
    for key in ("batch_size", "epochs", "early_stopping_patience", "scheduler_patience", "torch_threads"):
        number(train, key, f"train.{key}", minimum=1, integer=True)
    number(train, "num_workers", "train.num_workers", minimum=0, integer=True)
    number(train, "noise_cache_mb", "train.noise_cache_mb", positive=True)
    number(train, "log_every_batches", "train.log_every_batches", minimum=1, integer=True)
    for key in ("learning_rate", "gradient_clip"):
        number(train, key, f"train.{key}", positive=True)
    for key in ("weight_decay", "min_delta"):
        number(train, key, f"train.{key}", minimum=0)

    pilot = cfg["pilot"]
    number(pilot, "ratio", "pilot.ratio", positive=True, maximum=1)
    number(pilot, "observation_sigma", "pilot.observation_sigma", minimum=0)
    number(pilot, "min_spacing", "pilot.min_spacing", minimum=1, integer=True)
    number(pilot, "update_length_scale", "pilot.update_length_scale", positive=True)
    number(pilot, "gradient_weight", "pilot.gradient_weight", minimum=0)
    number(pilot, "extra_block_mask", "pilot.extra_block_mask", minimum=0, maximum=1)
    if pilot.get("training_mask_mode", "mixed") not in {"comb", "random", "mixed"}:
        raise ConfigError("pilot.training_mask_mode must be 'comb', 'random', or 'mixed'.")
    if pilot.get("update_noise_convention", "component") not in {"component", "complex_rms_as_component"}:
        raise ConfigError("pilot.update_noise_convention must be 'component' or 'complex_rms_as_component'.")

    for key in cfg["loss"]:
        if key.endswith("_weight"):
            number(cfg["loss"], key, f"loss.{key}", minimum=0)
    number(cfg["loss"], "max_delay_bins", "loss.max_delay_bins", minimum=1, integer=True)
    classical = cfg.get("classical", {})
    number(classical, "covariance_shrinkage", "classical.covariance_shrinkage", minimum=0, maximum=1)
    for key in ("max_covariance_frames", "omp_delay_bins", "omp_sparsity"):
        number(classical, key, f"classical.{key}", minimum=1, integer=True)
    number(classical, "omp_ridge", "classical.omp_ridge", minimum=0)

    evaluation = cfg["eval"]
    if evaluation.get("prediction_storage", "compact") not in {"none", "compact", "full"}:
        raise ConfigError("eval.prediction_storage must be 'none', 'compact', or 'full'.")
    for key in ("run_link_proxy", "run_classical"):
        if key in evaluation and not isinstance(evaluation[key], bool):
            raise ConfigError(f"eval.{key} must be a YAML boolean.")
    if "mask_strategies" in evaluation:
        policies = evaluation["mask_strategies"]
        if not isinstance(policies, list) or not policies or any(p not in {"comb", "random", "uncertainty", "active"} for p in policies):
            raise ConfigError("eval.mask_strategies must list comb, random, uncertainty, or active.")
    return cfg


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate an experiment YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return validate_config(yaml.safe_load(handle))


def resolve_path(value: str | Path, base: str | Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute() or base is None:
        return path
    return (Path(base) / path).resolve()
