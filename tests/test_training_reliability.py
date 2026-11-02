"""Small numerical fixtures verify caching and preservation of completed epochs."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from aquapilot.data import prepared
from aquapilot.engine import train_model
from aquapilot.utils import save_torch_checkpoint


def records(count=5):
    return [{"id": f"synthetic_{i}", "csi": np.full((12, 8), 1 + 0.1j * i, np.complex64)}
            for i in range(count)]


def dataset(items, budget_mb):
    return prepared.ChannelWindowDataset(items, history=3, horizon=1, pilot_ratio=1,
        mask_mode="comb", observation_sigma=0.03, noise_seed=17, noise_cache_mb=budget_mb)


def test_cache_retains_more_than_two_records_without_changing_noise(monkeypatch):
    items = records()
    expected = [prepared.sequence_observation_noise(item["csi"], 0.03, 17 + 4241 * i)[0]
                for i, item in enumerate(items)]
    data = dataset(items, 1)
    calls = []
    generate = prepared._noise_from_rms

    def counted(*args):
        calls.append(args[2])
        return generate(*args)

    monkeypatch.setattr(prepared, "_noise_from_rms", counted)
    for _ in range(3):
        for i in (3, 0, 4, 1, 2):
            np.testing.assert_array_equal(data._sequence_noise(i), expected[i])
    assert len(calls) == 5
    assert len(data._noise_cache) == 5
    assert data.noise_cache_bytes == sum(array.nbytes for array in expected)
    assert data.noise_cache_bytes <= data.noise_cache_limit_bytes


def test_byte_limit_and_eviction_preserve_window_noise():
    items = records()
    one = prepared.sequence_observation_noise(items[0]["csi"], 0.03, 17)[0]
    data = dataset(items, 2 * one.nbytes / (1024 * 1024))
    first = data[0]
    for i in range(len(items)):
        data._sequence_noise(i)
        assert data.noise_cache_bytes <= data.noise_cache_limit_bytes
    assert list(data._noise_cache) == [3, 4]
    np.testing.assert_array_equal(data[0]["x"].numpy(), first["x"].numpy())
    assert list(data._noise_cache) == [4, 0]
    too_small = dataset(items, 1 / (1024 * 1024))
    np.testing.assert_array_equal(too_small._sequence_noise(0), one)
    assert too_small.noise_cache_bytes == 0 and not too_small._noise_cache


def test_failed_checkpoint_serialization_preserves_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.pt"
    save_torch_checkpoint(path, {"weight": torch.tensor([2.0])})
    original = path.read_bytes()

    def fail(payload, handle):
        handle.write(b"incomplete checkpoint")
        raise OSError("synthetic disk failure")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk failure"):
        save_torch_checkpoint(path, {"weight": torch.tensor([3.0])})
    assert path.read_bytes() == original
    assert sorted(p.name for p in tmp_path.iterdir()) == ["checkpoint.pt"]


class TinyForecast(nn.Module):
    n_subcarriers = 2

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return {"mean": x * self.weight}


def training_fixture(monkeypatch):
    def loss(outputs, target, cfg):
        value = (outputs["mean"] - target).square().mean()
        return value, {"loss": float(value.detach())}

    monkeypatch.setattr("aquapilot.engine.total_loss", loss)
    loader = DataLoader([{"x": torch.ones(2, 2), "target": torch.ones(2, 2)}], batch_size=1)
    cfg = {"train": {"epochs": 3, "learning_rate": 0.1, "early_stopping_patience": 3,
                     "log_every_batches": 1}, "model": {"kind": "synthetic_test"}, "loss": {}}
    return TinyForecast(), loader, cfg


def test_best_epoch_is_saved_before_a_later_interruption(tmp_path, monkeypatch):
    model, loader, cfg = training_fixture(monkeypatch)
    calls = 0

    def validation(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            saved = torch.load(tmp_path / "checkpoint.pt", weights_only=False)
            assert saved["best_epoch"] == 1
            assert len(json.loads((tmp_path / "training_history.json").read_text())) == 1
            raise KeyboardInterrupt("synthetic interruption")
        return {"loss": 1.0}, {}

    monkeypatch.setattr("aquapilot.engine._evaluate_model_loader", validation)
    with pytest.raises(KeyboardInterrupt):
        train_model(model, loader, loader, cfg, tmp_path, torch.device("cpu"))
    saved = torch.load(tmp_path / "checkpoint.pt", weights_only=False)
    assert saved["best_epoch"] == 1 and saved["model_state"]["weight"] > 0
    status = json.loads((tmp_path / "training_status.json").read_text())
    assert status["state"] == "interrupted"
    assert status["epochs_completed"] == status["best_epoch"] == 1
    assert not (tmp_path / "complete.json").exists()


def test_training_returns_validation_selected_weights(tmp_path, monkeypatch):
    model, loader, cfg = training_fixture(monkeypatch)
    validation_losses = iter((1.0, 0.5, 0.8))
    monkeypatch.setattr("aquapilot.engine._evaluate_model_loader",
        lambda *args: ({"loss": next(validation_losses)}, {}))
    selected, history = train_model(model, loader, loader, cfg, tmp_path, torch.device("cpu"))
    saved = torch.load(tmp_path / "checkpoint.pt", weights_only=False)
    assert len(history) == 3 and saved["best_epoch"] == 2
    torch.testing.assert_close(selected.weight, saved["model_state"]["weight"])
    status = json.loads((tmp_path / "training_status.json").read_text())
    assert status["state"] == "training_complete" and status["epochs_completed"] == 3


def test_nonfinite_validation_does_not_create_checkpoint(tmp_path, monkeypatch):
    model, loader, cfg = training_fixture(monkeypatch)
    monkeypatch.setattr("aquapilot.engine._evaluate_model_loader", lambda *args: ({"loss": float("nan")}, {}))
    with pytest.raises(FloatingPointError, match="validation loss"):
        train_model(model, loader, loader, cfg, tmp_path, torch.device("cpu"))
    assert not (tmp_path / "checkpoint.pt").exists()
    status = json.loads((tmp_path / "training_status.json").read_text())
    assert status["state"] == "failed" and status["best_epoch"] is None


def test_nonfinite_training_loss_stops_before_optimizer_step(tmp_path, monkeypatch):
    model, _, cfg = training_fixture(monkeypatch)
    loader = DataLoader([{"x": torch.ones(2, 2), "target": torch.full((2, 2), float("nan"))}], batch_size=1)
    with pytest.raises(FloatingPointError, match="training loss"):
        train_model(model, loader, loader, cfg, tmp_path, torch.device("cpu"))
    assert model.weight.item() == 0.0 and model.weight.grad is None
    assert not (tmp_path / "checkpoint.pt").exists()
