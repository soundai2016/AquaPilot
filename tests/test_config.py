import copy
from pathlib import Path

import pytest

from aquapilot.config import ConfigError, load_config, validate_config


def test_bundled_experiment_configs_are_valid():
    root = Path(__file__).resolve().parents[1]
    for path in (root / "configs").glob("*.yaml"):
        load_config(path)


@pytest.mark.parametrize("section,key,value", [
    ("pilot", "observation_sigma", float("nan")),
    ("pilot", "ratio", 0),
    ("pilot", "ratio", 1.01),
    ("pilot", "update_noise_convention", "unknown"),
    ("data", "history", 1),
    ("data", "frames_per_sequence", 5),
    ("model", "conv_kernel", 4),
    ("train", "epochs", 0),
    ("train", "batch_size", 1.5),
    ("train", "noise_cache_mb", 0),
    ("train", "noise_cache_mb", float("inf")),
    ("train", "log_every_batches", 0),
    ("eval", "run_link_proxy", "false"),
])
def test_invalid_config_fails_before_execution(section, key, value):
    path = Path(__file__).resolve().parents[1] / "configs/smoke.yaml"
    cfg = copy.deepcopy(load_config(path))
    cfg[section][key] = value
    with pytest.raises(ConfigError):
        validate_config(cfg)
