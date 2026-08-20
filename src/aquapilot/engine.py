from __future__ import annotations

import copy
import csv
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from aquapilot.config import load_config
from aquapilot.data import ChannelWindowDataset, generate_synthetic_splits, load_prepared_sequences
from aquapilot.data.prepared import interpolate_complex
from aquapilot.losses import total_loss
from aquapilot.metrics import summarize_tensors
from aquapilot.models import (
    ComplexFrequencyPrior,
    build_forecast_model,
    fit_frequency_prior,
    interpolate_observation,
    linear_time_extrapolation,
    lmmse_estimate,
    omp_delay_estimate,
    persistence_forecast,
)
from aquapilot.ofdm import simulate_qpsk_link
from aquapilot.pilots import (
    comb_mask,
    posterior_kernel_update,
    random_mask,
    uncertainty_mask,
    uncertainty_top_mask,
)
from aquapilot.utils import (
    choose_device,
    configure_torch_runtime,
    count_parameters,
    ensure_dir,
    save_json,
    seed_everything,
)


def _build_sequences(cfg: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    data_cfg = cfg["data"]
    if data_cfg["kind"] == "synthetic":
        return generate_synthetic_splits(data_cfg, int(cfg["seed"]))
    root = Path(data_cfg["root"])
    return {split: load_prepared_sequences(root, split) for split in ("train", "val", "test")}


def _build_datasets(cfg: dict[str, Any], sequences: dict[str, list[dict[str, Any]]]) -> dict[str, ChannelWindowDataset]:
    data_cfg = cfg["data"]
    pilot_cfg = cfg["pilot"]
    datasets: dict[str, ChannelWindowDataset] = {}
    for split in ("train", "val", "test"):
        datasets[split] = ChannelWindowDataset(
            sequences[split],
            history=int(data_cfg.get("history", 8)),
            horizon=int(data_cfg.get("horizon", 1)),
            pilot_ratio=float(pilot_cfg.get("ratio", 0.125)),
            mask_mode=str(pilot_cfg.get("training_mask_mode", "mixed" if split == "train" else "comb")),
            extra_block_mask=float(pilot_cfg.get("extra_block_mask", 0.25 if split == "train" else 0.0)),
            seed=int(cfg["seed"]) + {"train": 0, "val": 1, "test": 2}[split] * 100_003,
        )
    return datasets


def _loader(dataset: ChannelWindowDataset, cfg: dict[str, Any], shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(cfg["train"].get("batch_size", 32)),
        shuffle=shuffle,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        pin_memory=False,
        drop_last=False,
    )


def _evaluate_model_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    model.eval()
    losses: list[float] = []
    means: list[torch.Tensor] = []
    sigmas: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            target = batch["target"].to(device)
            outputs = model(x)
            loss, _ = total_loss(outputs, target, loss_cfg)
            losses.append(float(loss.cpu()))
            means.append(outputs["mean"].cpu())
            sigmas.append(outputs["sigma"].cpu())
            targets.append(target.cpu())
    elapsed = time.perf_counter() - start
    mean = torch.cat(means, dim=0)
    sigma = torch.cat(sigmas, dim=0)
    target = torch.cat(targets, dim=0)
    metrics = summarize_tensors(mean, sigma, target)
    metrics["loss"] = float(np.mean(losses))
    metrics["latency_ms_per_frame"] = 1000.0 * elapsed / max(1, mean.shape[0])
    return metrics, {"mean": mean, "sigma": sigma, "target": target}


def _evaluate_baseline_loader(
    loader: DataLoader,
    predictor: Callable[[torch.Tensor], torch.Tensor],
    name: str,
) -> dict[str, Any]:
    means: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            mean = predictor(batch["x"])
            means.append(mean)
            targets.append(batch["target"])
    elapsed = time.perf_counter() - start
    mean = torch.cat(means, dim=0)
    target = torch.cat(targets, dim=0)
    residual_scale = (mean - target).square().mean().sqrt().clamp_min(1e-3)
    sigma = torch.full_like(mean, residual_scale)
    metrics = summarize_tensors(mean, sigma, target)
    metrics["name"] = name
    metrics["latency_ms_per_frame"] = 1000.0 * elapsed / max(1, mean.shape[0])
    return metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, float]]]:
    train_cfg = cfg["train"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 2e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.6,
        patience=max(1, int(train_cfg.get("scheduler_patience", 3))),
    )
    max_epochs = int(train_cfg.get("epochs", 20))
    patience = int(train_cfg.get("early_stopping_patience", 6))
    clip = float(train_cfg.get("gradient_clip", 1.0))
    best_val = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, float]] = []

    model.to(device)
    for epoch in range(1, max_epochs + 1):
        model.train()
        running = defaultdict(float)
        batches = 0
        for batch in train_loader:
            x = batch["x"].to(device)
            target = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(x)
            loss, parts = total_loss(outputs, target, cfg["loss"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            for key, value in parts.items():
                running[key] += float(value)
            batches += 1

        val_metrics, _ = _evaluate_model_loader(model, val_loader, device, cfg["loss"])
        scheduler.step(val_metrics["loss"])
        row = {"epoch": float(epoch), "learning_rate": float(optimizer.param_groups[0]["lr"])}
        for key, value in running.items():
            row[f"train_{key}"] = value / max(batches, 1)
        row.update({f"val_{key}": float(value) for key, value in val_metrics.items() if isinstance(value, (int, float))})
        history.append(row)

        if val_metrics["loss"] < best_val - float(train_cfg.get("min_delta", 1e-4)):
            best_val = float(val_metrics["loss"])
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint.")
    model.load_state_dict(best_state)
    torch.save(
        {
            "model_state": best_state,
            "model_config": cfg["model"],
            "n_subcarriers": model.n_subcarriers,
            "experiment_config": cfg,
            "best_val_loss": best_val,
        },
        output_dir / "checkpoint.pt",
    )
    save_json(output_dir / "training_history.json", history)
    return model, history


def _make_input(observed_history: np.ndarray, mask_history: np.ndarray) -> tuple[torch.Tensor, float]:
    filled = interpolate_complex(observed_history, mask_history)
    scale = float(np.sqrt(np.mean(np.abs(filled) ** 2)))
    scale = max(scale, 1e-6)
    normalized = filled / scale
    features = np.stack(
        (normalized.real, normalized.imag, mask_history.astype(np.float32)), axis=-1
    ).astype(np.float32)
    return torch.from_numpy(features).unsqueeze(0), scale


def _closed_loop_sequence(
    model: nn.Module,
    sequence: dict[str, Any],
    cfg: dict[str, Any],
    device: torch.device,
    strategy: str,
    seed: int,
) -> dict[str, Any]:
    csi = np.asarray(sequence["csi"], dtype=np.complex64)
    history = int(cfg["data"].get("history", 8))
    if csi.shape[0] <= history:
        raise ValueError(f"Sequence {sequence['id']} is too short for closed-loop evaluation.")
    budget = max(2, int(round(float(cfg["pilot"].get("ratio", 0.125)) * csi.shape[1])))
    obs_sigma_fraction = float(cfg["pilot"].get("observation_sigma", 0.03))
    sequence_rms = float(np.sqrt(np.mean(np.abs(csi) ** 2)))
    rng = np.random.default_rng(seed)
    standardized_noise = (
        rng.standard_normal(csi.shape) + 1j * rng.standard_normal(csi.shape)
    ).astype(np.complex64) / np.sqrt(2.0)
    raw_noise = obs_sigma_fraction * max(sequence_rms, 1e-6) * standardized_noise

    mask_history = np.stack(
        [comb_mask(csi.shape[1], budget, offset=t % max(1, csi.shape[1] // budget)) for t in range(history)],
        axis=0,
    )
    observed_history = np.where(mask_history, csi[:history] + raw_noise[:history], 0.0 + 0.0j)

    prior_mean_list: list[np.ndarray] = []
    prior_sigma_list: list[np.ndarray] = []
    posterior_list: list[np.ndarray] = []
    posterior_sigma_list: list[np.ndarray] = []
    target_list: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for t in range(history, csi.shape[0]):
            x, scale = _make_input(observed_history, mask_history)
            outputs = model(x.to(device))
            prior_mean = outputs["mean"].cpu()[0]
            prior_sigma = outputs["sigma"].cpu()[0]
            normalized_observation_sigma = obs_sigma_fraction * sequence_rms / scale
            if strategy == "active":
                current_mask = uncertainty_mask(
                    prior_mean.numpy(),
                    prior_sigma.numpy(),
                    budget=budget,
                    min_spacing=int(cfg["pilot"].get("min_spacing", 2)),
                    gradient_weight=float(cfg["pilot"].get("gradient_weight", 0.0)),
                    length_scale=float(cfg["pilot"].get("update_length_scale", 4.0)),
                    observation_sigma=normalized_observation_sigma,
                )
            elif strategy == "uncertainty":
                current_mask = uncertainty_top_mask(
                    prior_sigma.numpy(),
                    budget=budget,
                    min_spacing=int(cfg["pilot"].get("min_spacing", 2)),
                )
            elif strategy == "random":
                current_mask = random_mask(csi.shape[1], budget, rng)
            elif strategy == "comb":
                current_mask = comb_mask(csi.shape[1], budget, offset=t % max(1, csi.shape[1] // budget))
            else:
                raise ValueError(f"Unknown closed-loop strategy: {strategy}")

            target = csi[t] / scale
            observation_complex = (csi[t] + raw_noise[t]) / scale
            observation_ri = np.stack((observation_complex.real, observation_complex.imag), axis=-1).astype(np.float32)
            observation = torch.from_numpy(observation_ri).unsqueeze(0)
            mask_tensor = torch.from_numpy(current_mask[None, :])
            posterior_mean, posterior_sigma = posterior_kernel_update(
                prior_mean.unsqueeze(0),
                prior_sigma.unsqueeze(0),
                observation,
                mask_tensor,
                observation_sigma=normalized_observation_sigma,
                length_scale=float(cfg["pilot"].get("update_length_scale", 4.0)),
            )

            prior_mean_list.append(prior_mean.numpy())
            prior_sigma_list.append(prior_sigma.numpy())
            posterior_list.append(posterior_mean[0].numpy())
            posterior_sigma_list.append(posterior_sigma[0].numpy())
            target_list.append(np.stack((target.real, target.imag), axis=-1).astype(np.float32))
            masks.append(current_mask)

            new_observation = np.where(current_mask, csi[t] + raw_noise[t], 0.0 + 0.0j)
            observed_history = np.concatenate((observed_history[1:], new_observation[None, :]), axis=0)
            mask_history = np.concatenate((mask_history[1:], current_mask[None, :]), axis=0)

    return {
        "sequence_id": sequence["id"],
        "environment": sequence.get("environment", "unknown"),
        "prior_mean": np.stack(prior_mean_list),
        "prior_sigma": np.stack(prior_sigma_list),
        "posterior_mean": np.stack(posterior_list),
        "posterior_sigma": np.stack(posterior_sigma_list),
        "target": np.stack(target_list),
        "mask": np.stack(masks),
    }


def evaluate_closed_loop(
    model: nn.Module,
    test_sequences: list[dict[str, Any]],
    cfg: dict[str, Any],
    device: torch.device,
    strategy: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    records = []
    for index, sequence in enumerate(test_sequences):
        records.append(
            _closed_loop_sequence(
                model,
                sequence,
                cfg,
                device,
                strategy,
                seed=int(cfg["seed"]) + 4241 * index,
            )
        )
    prior_mean = np.concatenate([item["prior_mean"] for item in records], axis=0)
    prior_sigma = np.concatenate([item["prior_sigma"] for item in records], axis=0)
    posterior_mean = np.concatenate([item["posterior_mean"] for item in records], axis=0)
    posterior_sigma = np.concatenate([item["posterior_sigma"] for item in records], axis=0)
    target = np.concatenate([item["target"] for item in records], axis=0)
    masks = np.concatenate([item["mask"] for item in records], axis=0)

    prior_metrics = summarize_tensors(
        torch.from_numpy(prior_mean), torch.from_numpy(prior_sigma), torch.from_numpy(target)
    )
    posterior_metrics = summarize_tensors(
        torch.from_numpy(posterior_mean), torch.from_numpy(posterior_sigma), torch.from_numpy(target)
    )
    true_complex = target[..., 0] + 1j * target[..., 1]
    posterior_complex = posterior_mean[..., 0] + 1j * posterior_mean[..., 1]
    link_results = {}
    for snr in cfg["eval"].get("snr_db", [5, 10, 15]):
        link_results[str(snr)] = simulate_qpsk_link(
            true_complex,
            posterior_complex,
            masks,
            snr_db=float(snr),
            trials=int(cfg["eval"].get("link_trials", 4)),
            block_subcarriers=int(cfg["eval"].get("block_subcarriers", 16)),
            code_rate=float(cfg["eval"].get("code_rate", 0.5)),
            seed=int(cfg["seed"]) + int(float(snr) * 17),
        )
    metrics = {
        "strategy": strategy,
        "prior": prior_metrics,
        "posterior": posterior_metrics,
        "pilot_ratio": float(masks.mean()),
        "link": link_results,
        "num_sequences": len(records),
    }
    arrays = {
        "prior_mean": prior_mean,
        "prior_sigma": prior_sigma,
        "posterior_mean": posterior_mean,
        "posterior_sigma": posterior_sigma,
        "target": target,
        "mask": masks,
    }
    return metrics, arrays




def _classical_closed_loop_sequence(
    sequence: dict[str, Any],
    cfg: dict[str, Any],
    receiver: str,
    frequency_prior: ComplexFrequencyPrior | None,
    mask_strategy: str,
    seed: int,
) -> dict[str, Any]:
    """Evaluate a non-neural current-frame estimator at the same pilot budget."""
    csi = np.asarray(sequence["csi"], dtype=np.complex64)
    history = int(cfg["data"].get("history", 8))
    if csi.shape[0] <= history:
        raise ValueError(f"Sequence {sequence['id']} is too short for closed-loop evaluation.")
    budget = max(2, int(round(float(cfg["pilot"].get("ratio", 0.125)) * csi.shape[1])))
    obs_sigma_fraction = float(cfg["pilot"].get("observation_sigma", 0.03))
    sequence_rms = float(np.sqrt(np.mean(np.abs(csi) ** 2)))
    rng = np.random.default_rng(seed)
    standardized_noise = (
        rng.standard_normal(csi.shape) + 1j * rng.standard_normal(csi.shape)
    ).astype(np.complex64) / np.sqrt(2.0)
    raw_noise = obs_sigma_fraction * max(sequence_rms, 1e-6) * standardized_noise

    mask_history = np.stack(
        [comb_mask(csi.shape[1], budget, offset=t % max(1, csi.shape[1] // budget)) for t in range(history)],
        axis=0,
    )
    observed_history = np.where(mask_history, csi[:history] + raw_noise[:history], 0.0 + 0.0j)
    estimates: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    for t in range(history, csi.shape[0]):
        _, scale = _make_input(observed_history, mask_history)
        if mask_strategy == "comb":
            current_mask = comb_mask(csi.shape[1], budget, offset=t % max(1, csi.shape[1] // budget))
        elif mask_strategy == "random":
            current_mask = random_mask(csi.shape[1], budget, rng)
        else:
            raise ValueError(f"Classical receivers support comb/random masks, got {mask_strategy!r}.")
        observation = (csi[t] + raw_noise[t]) / scale
        normalized_observation_sigma = obs_sigma_fraction * sequence_rms / scale
        if receiver == "ls_interpolation":
            estimate = interpolate_observation(observation, current_mask)
        elif receiver == "lmmse":
            if frequency_prior is None:
                raise ValueError("LMMSE requires a training-only frequency prior.")
            estimate = lmmse_estimate(
                observation,
                current_mask,
                frequency_prior,
                observation_variance=normalized_observation_sigma**2,
            )
        elif receiver == "omp":
            classical_cfg = cfg.get("classical", {})
            estimate = omp_delay_estimate(
                observation,
                current_mask,
                delay_bins=int(classical_cfg.get("omp_delay_bins", min(24, csi.shape[1]))),
                sparsity=int(classical_cfg.get("omp_sparsity", 6)),
                ridge=float(classical_cfg.get("omp_ridge", 1e-6)),
            )
        else:
            raise ValueError(f"Unknown classical receiver: {receiver}")
        target = csi[t] / scale
        estimates.append(np.stack((estimate.real, estimate.imag), axis=-1).astype(np.float32))
        targets.append(np.stack((target.real, target.imag), axis=-1).astype(np.float32))
        masks.append(current_mask)
        new_observation = np.where(current_mask, csi[t] + raw_noise[t], 0.0 + 0.0j)
        observed_history = np.concatenate((observed_history[1:], new_observation[None, :]), axis=0)
        mask_history = np.concatenate((mask_history[1:], current_mask[None, :]), axis=0)
    return {
        "estimate": np.stack(estimates),
        "target": np.stack(targets),
        "mask": np.stack(masks),
    }


def evaluate_classical_receiver(
    test_sequences: list[dict[str, Any]],
    cfg: dict[str, Any],
    receiver: str,
    frequency_prior: ComplexFrequencyPrior | None,
    mask_strategy: str = "comb",
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    start = time.perf_counter()
    records = [
        _classical_closed_loop_sequence(
            sequence,
            cfg,
            receiver=receiver,
            frequency_prior=frequency_prior,
            mask_strategy=mask_strategy,
            seed=int(cfg["seed"]) + 4241 * index,
        )
        for index, sequence in enumerate(test_sequences)
    ]
    estimate = np.concatenate([item["estimate"] for item in records], axis=0)
    target = np.concatenate([item["target"] for item in records], axis=0)
    masks = np.concatenate([item["mask"] for item in records], axis=0)
    estimate_tensor = torch.from_numpy(estimate)
    target_tensor = torch.from_numpy(target)
    residual_scale = (estimate_tensor - target_tensor).square().mean().sqrt().clamp_min(1e-3)
    sigma = torch.full_like(estimate_tensor, residual_scale)
    metrics = summarize_tensors(estimate_tensor, sigma, target_tensor)
    metrics.update(
        {
            "receiver": receiver,
            "mask_strategy": mask_strategy,
            "pilot_ratio": float(masks.mean()),
            "uncertainty_applicable": False,
            "latency_ms_per_frame": 1000.0 * (time.perf_counter() - start) / max(1, estimate.shape[0]),
        }
    )
    true_complex = target[..., 0] + 1j * target[..., 1]
    estimate_complex = estimate[..., 0] + 1j * estimate[..., 1]
    metrics["link"] = {
        str(snr): simulate_qpsk_link(
            true_complex,
            estimate_complex,
            masks,
            snr_db=float(snr),
            trials=int(cfg["eval"].get("link_trials", 4)),
            block_subcarriers=int(cfg["eval"].get("block_subcarriers", 16)),
            code_rate=float(cfg["eval"].get("code_rate", 0.5)),
            seed=int(cfg["seed"]) + int(float(snr) * 17),
        )
        for snr in cfg["eval"].get("snr_db", [5, 10, 15])
    }
    return metrics, {"estimate": estimate, "target": target, "mask": masks}

def _write_training_csv(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    keys = sorted({key for row in history for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def _plot_outputs(
    output_dir: Path,
    history: list[dict[str, float]],
    summary: dict[str, Any],
    active_arrays: dict[str, np.ndarray],
) -> None:
    if history:
        epochs = [row["epoch"] for row in history]
        train_loss = [row.get("train_loss", np.nan) for row in history]
        val_loss = [row.get("val_loss", np.nan) for row in history]
        plt.figure(figsize=(5.4, 3.4))
        plt.plot(epochs, train_loss, marker="o", label="train")
        plt.plot(epochs, val_loss, marker="s", label="validation")
        plt.xlabel("Epoch")
        plt.ylabel("Objective")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "training_curve.pdf")
        plt.savefig(output_dir / "training_curve.png", dpi=180)
        plt.close()

    labels = ["Persistence", "Linear", "AquaPilot prior", "Comb posterior", "Active posterior"]
    values = [
        summary["baselines"]["persistence"]["nmse_db_mean"],
        summary["baselines"]["linear"]["nmse_db_mean"],
        summary["model_test"]["nmse_db_mean"],
        summary["closed_loop"]["comb"]["posterior"]["nmse_db_mean"],
        summary["closed_loop"]["active"]["posterior"]["nmse_db_mean"],
    ]
    plt.figure(figsize=(6.4, 3.6))
    positions = np.arange(len(labels))
    plt.bar(positions, values)
    plt.xticks(positions, labels, rotation=20, ha="right")
    plt.ylabel("CSI NMSE (dB; lower is better)")
    plt.tight_layout()
    plt.savefig(output_dir / "nmse_summary.pdf")
    plt.savefig(output_dir / "nmse_summary.png", dpi=180)
    plt.close()

    target = active_arrays["target"]
    estimate = active_arrays["posterior_mean"]
    index = min(4, target.shape[0] - 1)
    target_complex = target[index, :, 0] + 1j * target[index, :, 1]
    estimate_complex = estimate[index, :, 0] + 1j * estimate[index, :, 1]
    pilot_indices = np.flatnonzero(active_arrays["mask"][index])
    x = np.arange(target.shape[1])
    plt.figure(figsize=(6.2, 3.4))
    plt.plot(x, 20.0 * np.log10(np.abs(target_complex) + 1e-5), label="true")
    plt.plot(x, 20.0 * np.log10(np.abs(estimate_complex) + 1e-5), label="posterior")
    plt.scatter(pilot_indices, 20.0 * np.log10(np.abs(target_complex[pilot_indices]) + 1e-5), marker="x", label="pilots")
    plt.xlabel("Subcarrier")
    plt.ylabel("Magnitude (dB, normalized)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "example_reconstruction.pdf")
    plt.savefig(output_dir / "example_reconstruction.png", dpi=180)
    plt.close()

    mask_view = active_arrays["mask"][: min(80, active_arrays["mask"].shape[0])]
    plt.figure(figsize=(6.2, 3.2))
    plt.imshow(mask_view, aspect="auto", interpolation="nearest", origin="lower")
    plt.xlabel("Subcarrier")
    plt.ylabel("Closed-loop frame")
    plt.colorbar(label="Pilot selected")
    plt.tight_layout()
    plt.savefig(output_dir / "active_pilot_map.pdf")
    plt.savefig(output_dir / "active_pilot_map.png", dpi=180)
    plt.close()

    snrs = [float(value) for value in summary["closed_loop"]["active"]["link"].keys()]
    snrs.sort()
    active_goodput = [summary["closed_loop"]["active"]["link"][str(int(s)) if float(s).is_integer() else str(s)]["normalized_goodput"] for s in snrs]
    comb_goodput = [summary["closed_loop"]["comb"]["link"][str(int(s)) if float(s).is_integer() else str(s)]["normalized_goodput"] for s in snrs]
    plt.figure(figsize=(5.4, 3.4))
    plt.plot(snrs, comb_goodput, marker="o", label="fixed comb")
    plt.plot(snrs, active_goodput, marker="s", label="active")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Normalized goodput")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "goodput.pdf")
    plt.savefig(output_dir / "goodput.png", dpi=180)
    plt.close()


def _load_state_dict(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.6
        return torch.load(path, map_location="cpu")


def run_experiment(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    cfg = load_config(config) if not isinstance(config, dict) else copy.deepcopy(config)
    configure_torch_runtime(cfg["train"].get("torch_threads"))
    seed_everything(int(cfg["seed"]))
    output_dir = ensure_dir(cfg["output_dir"])
    save_json(output_dir / "resolved_config.json", cfg)
    device = choose_device(str(cfg["train"].get("device", "auto")))

    sequences = _build_sequences(cfg)
    datasets = _build_datasets(cfg, sequences)
    loaders = {
        "train": _loader(datasets["train"], cfg, shuffle=True),
        "val": _loader(datasets["val"], cfg, shuffle=False),
        "test": _loader(datasets["test"], cfg, shuffle=False),
    }
    model = build_forecast_model(datasets["train"].n_subcarriers, cfg["model"])
    init_checkpoint = cfg["train"].get("init_checkpoint")
    if init_checkpoint:
        bundle = _load_state_dict(init_checkpoint)
        if int(bundle["n_subcarriers"]) != int(datasets["train"].n_subcarriers):
            raise ValueError("Initialization checkpoint and data use different subcarrier counts.")
        model.load_state_dict(bundle["model_state"], strict=True)
    model, history = train_model(model, loaders["train"], loaders["val"], cfg, output_dir, device)
    _write_training_csv(output_dir / "training_history.csv", history)

    val_metrics, _ = _evaluate_model_loader(model, loaders["val"], device, cfg["loss"])
    test_metrics, predictions = _evaluate_model_loader(model, loaders["test"], device, cfg["loss"])
    persistence = _evaluate_baseline_loader(loaders["test"], persistence_forecast, "persistence")
    linear = _evaluate_baseline_loader(loaders["test"], linear_time_extrapolation, "linear_time_extrapolation")

    strategies = list(cfg["eval"].get("mask_strategies", ["comb", "random", "uncertainty", "active"]))
    for required in ("comb", "active"):
        if required not in strategies:
            strategies.append(required)
    closed_loop: dict[str, Any] = {}
    closed_loop_arrays: dict[str, dict[str, np.ndarray]] = {}
    for strategy in strategies:
        metrics, arrays = evaluate_closed_loop(model, sequences["test"], cfg, device, str(strategy))
        closed_loop[str(strategy)] = metrics
        closed_loop_arrays[str(strategy)] = arrays
        np.savez_compressed(output_dir / f"closed_loop_{strategy}.npz", **arrays)

    classical_results: dict[str, Any] = {}
    if bool(cfg["eval"].get("run_classical", True)):
        classical_cfg = cfg.get("classical", {})
        frequency_prior = fit_frequency_prior(
            sequences["train"],
            shrinkage=float(classical_cfg.get("covariance_shrinkage", 0.05)),
            max_frames=int(classical_cfg.get("max_covariance_frames", 20_000)),
        )
        np.savez_compressed(
            output_dir / "classical_frequency_prior.npz",
            mean_real=frequency_prior.mean.real,
            mean_imag=frequency_prior.mean.imag,
            covariance_real=frequency_prior.covariance.real,
            covariance_imag=frequency_prior.covariance.imag,
            shrinkage=np.asarray(frequency_prior.shrinkage),
        )
        for receiver in cfg["eval"].get("classical_receivers", ["ls_interpolation", "lmmse", "omp"]):
            metrics, arrays = evaluate_classical_receiver(
                sequences["test"], cfg, str(receiver), frequency_prior, mask_strategy="comb"
            )
            classical_results[str(receiver)] = metrics
            np.savez_compressed(output_dir / f"classical_{receiver}.npz", **arrays)

    summary = {
        "artifact_status": "synthetic sanity check" if cfg["data"]["kind"] == "synthetic" else "public-data experiment",
        "device": str(device),
        "model_kind": str(cfg["model"].get("kind", "aquapilot")),
        "model_parameters": count_parameters(model),
        "initialized_from": str(init_checkpoint) if init_checkpoint else None,
        "data": {
            split: {
                "sequences": len(sequences[split]),
                "windows": len(datasets[split]),
                "environments": sorted({str(item.get("environment", "unknown")) for item in sequences[split]}),
            }
            for split in ("train", "val", "test")
        },
        "model_validation": val_metrics,
        "model_test": test_metrics,
        "baselines": {"persistence": persistence, "linear": linear},
        "closed_loop": closed_loop,
        "classical_closed_loop": classical_results,
    }
    save_json(output_dir / "metrics.json", summary)
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        mean=predictions["mean"].numpy(),
        sigma=predictions["sigma"].numpy(),
        target=predictions["target"].numpy(),
    )
    _plot_outputs(output_dir, history, summary, closed_loop_arrays["active"])
    return summary


def evaluate_checkpoint(
    config: str | Path | dict[str, Any],
    checkpoint: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Zero-shot evaluation of a trained Watermark model on an external test root."""
    cfg = load_config(config) if not isinstance(config, dict) else copy.deepcopy(config)
    configure_torch_runtime(cfg["train"].get("torch_threads"))
    seed_everything(int(cfg["seed"]))
    destination = ensure_dir(output_dir or cfg.get("output_dir", "results/external_zero_shot"))
    device = choose_device(str(cfg["train"].get("device", "auto")))
    bundle = _load_state_dict(checkpoint)
    n_subcarriers = int(bundle["n_subcarriers"])
    model = build_forecast_model(n_subcarriers, bundle["model_config"])
    model.load_state_dict(bundle["model_state"], strict=True)
    model.to(device)

    if cfg["data"].get("kind") != "prepared":
        raise ValueError("Checkpoint evaluation expects data.kind=prepared.")
    test_sequences = load_prepared_sequences(Path(cfg["data"]["root"]), "test")
    if any(np.asarray(item["csi"]).shape[1] != n_subcarriers for item in test_sequences):
        raise ValueError(
            f"External CSI must be resampled to {n_subcarriers} subcarriers. "
            "Use prepare_acommset.py --n-subcarriers accordingly."
        )
    test_dataset = ChannelWindowDataset(
        test_sequences,
        history=int(cfg["data"].get("history", 8)),
        horizon=int(cfg["data"].get("horizon", 1)),
        pilot_ratio=float(cfg["pilot"].get("ratio", 0.125)),
        mask_mode=str(cfg["pilot"].get("training_mask_mode", "mixed")),
        extra_block_mask=0.0,
        seed=int(cfg["seed"]) + 200_006,
    )
    test_loader = _loader(test_dataset, cfg, shuffle=False)
    test_metrics, predictions = _evaluate_model_loader(model, test_loader, device, cfg["loss"])
    persistence = _evaluate_baseline_loader(test_loader, persistence_forecast, "persistence")
    linear = _evaluate_baseline_loader(test_loader, linear_time_extrapolation, "linear_time_extrapolation")
    strategies = list(cfg["eval"].get("mask_strategies", ["comb", "random", "uncertainty", "active"]))
    for required in ("comb", "active"):
        if required not in strategies:
            strategies.append(required)
    closed_loop: dict[str, Any] = {}
    for strategy in strategies:
        metrics, arrays = evaluate_closed_loop(model, test_sequences, cfg, device, str(strategy))
        closed_loop[str(strategy)] = metrics
        np.savez_compressed(destination / f"closed_loop_{strategy}.npz", **arrays)
    summary = {
        "artifact_status": "external zero-shot evaluation",
        "checkpoint": str(checkpoint),
        "device": str(device),
        "model_kind": str(bundle["model_config"].get("kind", "aquapilot")),
        "model_parameters": count_parameters(model),
        "data": {
            "test_sequences": len(test_sequences),
            "test_windows": len(test_dataset),
            "environments": sorted({str(item.get("environment", "unknown")) for item in test_sequences}),
        },
        "model_test": test_metrics,
        "baselines": {"persistence": persistence, "linear": linear},
        "closed_loop": closed_loop,
    }
    save_json(destination / "resolved_config.json", cfg)
    save_json(destination / "metrics.json", summary)
    np.savez_compressed(
        destination / "test_predictions.npz",
        mean=predictions["mean"].numpy(),
        sigma=predictions["sigma"].numpy(),
        target=predictions["target"].numpy(),
    )
    return summary


def one_standard_error_select(rows: list[dict[str, float]]) -> dict[str, float]:
    """Select the smallest rank within one standard error of the best mean NMSE."""
    if not rows:
        raise ValueError("rows must not be empty.")
    best = min(rows, key=lambda row: row["mean_nmse_db"])
    threshold = best["mean_nmse_db"] + best["se_nmse_db"]
    eligible = [row for row in rows if row["mean_nmse_db"] <= threshold]
    return min(eligible, key=lambda row: int(row["rank"]))


def run_rank_search(config_path: str | Path, ranks: list[int], seeds: list[int]) -> dict[str, Any]:
    base = load_config(config_path)
    root = Path(base["output_dir"]).parent / (Path(base["output_dir"]).name + "_rank_search")
    rows = []
    for rank in ranks:
        values = []
        for seed in seeds:
            cfg = copy.deepcopy(base)
            cfg["seed"] = int(seed)
            cfg["model"]["rank"] = int(rank)
            cfg["output_dir"] = str(root / f"rank_{rank:02d}" / f"seed_{seed}")
            result = run_experiment(cfg)
            values.append(float(result["model_validation"]["nmse_db_mean"]))
        mean = float(np.mean(values))
        se = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        rows.append({"rank": float(rank), "mean_nmse_db": mean, "se_nmse_db": se})
    selected = one_standard_error_select(rows)
    payload = {"rows": rows, "selected": selected}
    save_json(root / "rank_selection.json", payload)
    return payload
