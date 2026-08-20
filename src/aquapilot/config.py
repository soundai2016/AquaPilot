from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration and perform minimal structural validation."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ConfigError("The YAML root must be a mapping.")
    for key in ("seed", "output_dir", "data", "model", "train", "pilot", "loss", "eval"):
        if key not in cfg:
            raise ConfigError(f"Missing required configuration section: {key}")
    if cfg["data"].get("kind") not in {"synthetic", "prepared"}:
        raise ConfigError("data.kind must be 'synthetic' or 'prepared'.")
    return cfg


def resolve_path(value: str | Path, base: str | Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute() or base is None:
        return path
    return (Path(base) / path).resolve()
