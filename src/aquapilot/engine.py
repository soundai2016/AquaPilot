from __future__ import annotations

import copy
import csv
import hashlib
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

from aquapilot.config import load_config, validate_config
from aquapilot.data import ChannelWindowDataset, generate_synthetic_splits, load_prepared_sequences
from aquapilot.data.prepared import channel_sequence_rms, interpolate_complex, sequence_observation_noise
from aquapilot.losses import total_loss
from aquapilot.metrics import DiagnosticsAccumulator, framewise_diagnostics, summarize_diagnostics
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
from aquapilot.placement import combine_mask_summaries, summarize_masks
from aquapilot.pilots import (
    comb_mask,
    posterior_kernel_update,
    random_mask,
    uncertainty_mask,
    uncertainty_top_mask,
)
from aquapilot.provenance import begin_run, finish_run, resolve_config_paths
from aquapilot.replay import (
    BASELINE_STAGES,
    acquire_pilots,
    as_real_imag,
    check_checkpoint_protocol,
    matched_information_baselines,
    protocol_metadata,
    update_component_sigma,
    validate_replay_config,
)
from aquapilot.utils import (
    choose_device,
    configure_torch_runtime,
    count_parameters,
    save_json,
    save_torch_checkpoint,
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
            mask_mode=str(pilot_cfg.get("training_mask_mode", "mixed")) if split == "train" else "comb",
            extra_block_mask=float(pilot_cfg.get("extra_block_mask", 0.0)) if split == "train" else 0.0,
            seed=int(cfg["seed"]) + {"train": 0, "val": 1, "test": 2}[split] * 100_003,
            observation_sigma=float(pilot_cfg.get("observation_sigma", 0.03)),
            noise_seed=int(cfg["seed"]) + {"train": 1, "val": 2, "test": 0}[split] * 100_003,
            noise_cache_mb=float(cfg.get("train", {}).get("noise_cache_mb", 128.0)),
        )
    return datasets


def _loader(dataset: ChannelWindowDataset, cfg: dict[str, Any], shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(cfg["train"].get("batch_size", 32)),
        shuffle=shuffle,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        pin_memory=choose_device(str(cfg["train"].get("device", "auto"))).type == "cuda",
        drop_last=False,
    )


def _evaluate_model_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    model.eval()
    accumulator = DiagnosticsAccumulator()
    loss_sum, frames = 0.0, 0
    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            target = batch["target"].to(device)
            outputs = model(x)
            loss, _ = total_loss(outputs, target, loss_cfg)
            batch_frames = int(target.shape[0])
            loss_sum += float(loss.cpu()) * batch_frames
            frames += batch_frames
            accumulator.update(framewise_diagnostics(
                outputs["mean"], outputs["sigma"], target,
                normalization_scale=batch["scale"], compute_ranking=False,
            ))
    elapsed = time.perf_counter() - start
    metrics = accumulator.summary()
    metrics["loss"] = loss_sum / max(frames, 1)
    metrics["latency_ms_per_frame"] = 1000.0 * elapsed / max(1, frames)
    metrics["stage"] = "pre-acquisition forecast on deterministic noisy-comb windows"
    return metrics, {}


def _evaluate_baseline_loader(
    loader: DataLoader,
    predictor: Callable[[torch.Tensor], torch.Tensor],
    name: str,
) -> dict[str, Any]:
    accumulator = DiagnosticsAccumulator()
    frames = 0
    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            mean = predictor(batch["x"])
            frames += int(mean.shape[0])
            accumulator.update(framewise_diagnostics(
                mean, None, batch["target"], normalization_scale=batch["scale"], compute_ranking=False,
            ))
    elapsed = time.perf_counter() - start
    metrics = accumulator.summary()
    metrics["name"] = name
    metrics["stage"] = "pre-acquisition forecast on deterministic noisy-comb windows"
    metrics["latency_ms_per_frame"] = 1000.0 * elapsed / max(1, frames)
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
    log_every = int(train_cfg.get("log_every_batches", 100))
    if log_every < 1:
        raise ValueError("train.log_every_batches must be positive.")
    best_val = math.inf
    best_epoch: int | None = None
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, float]] = []
    training_start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    def write_status(state: str, epoch: int, error: str | None = None) -> None:
        save_json(output_dir / "training_status.json", {
            "state": state,
            "epoch": epoch,
            "epochs_completed": len(history),
            "max_epochs": max_epochs,
            "best_epoch": best_epoch,
            "best_val_loss": best_val if math.isfinite(best_val) else None,
            "elapsed_seconds": time.perf_counter() - training_start,
            "checkpoint": "checkpoint.pt" if best_state is not None else None,
            "checkpoint_role": "best validation weights; no optimizer resume state",
            "error": error,
        })

    model.to(device)
    epoch = 0
    try:
        save_json(output_dir / "training_history.json", history)
        write_status("training", epoch)
        print(f"Training on {device}; epochs={max_epochs}; output={output_dir}", flush=True)
        for epoch in range(1, max_epochs + 1):
            epoch_start = time.perf_counter()
            write_status("training", epoch)
            print(f"epoch={epoch}/{max_epochs} phase=train batches={len(train_loader)}", flush=True)
            model.train()
            running = defaultdict(float)
            batches = 0
            for batch in train_loader:
                x = batch["x"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(x)
                loss, parts = total_loss(outputs, target, cfg["loss"])
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}, batch {batches + 1}.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
                optimizer.step()
                for key, value in parts.items():
                    running[key] += float(value)
                batches += 1
                if batches % log_every == 0:
                    print(f"epoch={epoch}/{max_epochs} batch={batches}/{len(train_loader)} loss={float(loss.detach().cpu()):.6g}", flush=True)

            if batches == 0:
                raise RuntimeError("The training loader contains no batches.")
            write_status("validation", epoch)
            print(f"epoch={epoch}/{max_epochs} phase=validation", flush=True)
            val_metrics, _ = _evaluate_model_loader(model, val_loader, device, cfg["loss"])
            val_loss = float(val_metrics["loss"])
            if not math.isfinite(val_loss):
                raise FloatingPointError(f"Nonfinite validation loss at epoch {epoch}.")
            scheduler.step(val_loss)
            row = {"epoch": float(epoch), "learning_rate": float(optimizer.param_groups[0]["lr"]),
                   "epoch_seconds": time.perf_counter() - epoch_start}
            for key, value in running.items():
                row[f"train_{key}"] = value / batches
            row.update({f"val_{key}": float(value) for key, value in val_metrics.items() if isinstance(value, (int, float))})
            history.append(row)

            improved = val_loss < best_val - float(train_cfg.get("min_delta", 1e-4))
            if improved:
                candidate_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                save_torch_checkpoint(output_dir / "checkpoint.pt", {
                    "model_state": candidate_state,
                    "model_config": cfg["model"],
                    "n_subcarriers": model.n_subcarriers,
                    "experiment_config": cfg,
                    "best_val_loss": val_loss,
                    "best_epoch": epoch,
                    "checkpoint_role": "best_validation_weights",
                })
                best_state = candidate_state
                best_val = val_loss
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
            save_json(output_dir / "training_history.json", history)
            write_status("epoch_complete", epoch)
            print(f"epoch={epoch}/{max_epochs} val_loss={val_loss:.6g} best={best_val:.6g} best_epoch={best_epoch} lr={row['learning_rate']:.6g} seconds={row['epoch_seconds']:.1f} checkpoint_updated={improved}", flush=True)
            if stale >= patience:
                print(f"Early stopping after epoch {epoch}; selected epoch {best_epoch}.", flush=True)
                break

        if best_state is None:
            raise RuntimeError("Training produced no finite validation checkpoint.")
        model.load_state_dict(best_state)
        write_status("training_complete", epoch)
    except BaseException as exc:
        try:
            write_status("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", epoch,
                         f"{type(exc).__name__}: {exc}")
        except OSError:
            print("Could not update training_status.json after the training error.", flush=True)
        raise
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


def _initial_replay(sequence: dict[str, Any], cfg: dict[str, Any], seed: int) -> tuple:
    csi = np.asarray(sequence["csi"], dtype=np.complex64)
    history = int(cfg["data"].get("history", 8))
    if csi.ndim != 2 or csi.shape[0] <= history or csi.shape[1] < 2:
        raise ValueError(f"Sequence {sequence['id']} is incompatible with the configured history/frequency grid.")
    ratio = float(cfg["pilot"].get("ratio", 0.125))
    if not 0 < ratio <= 1:
        raise ValueError("Pilot ratio must be in (0,1].")
    budget = max(2, int(round(ratio * csi.shape[1])))
    sequence_rms = channel_sequence_rms(csi)
    fraction = float(cfg["pilot"].get("observation_sigma", 0.03))
    noise, rng = sequence_observation_noise(csi, fraction, seed)
    complex_noise_rms = fraction * max(sequence_rms, 1e-6)
    mask_history = np.stack([
        comb_mask(csi.shape[1], budget, offset=t % max(1, csi.shape[1] // budget))
        for t in range(history)
    ])
    observed_history = acquire_pilots(csi[:history], noise[:history], mask_history)
    return csi, history, budget, complex_noise_rms, noise, rng, mask_history, observed_history


def _sequence_metadata(sequence: dict[str, Any], csi: np.ndarray, history: int, seed: int) -> dict[str, Any]:
    frame_index = np.arange(history, csi.shape[0], dtype=np.int64)
    reference_sha256 = hashlib.sha256(np.ascontiguousarray(csi).view(np.uint8)).hexdigest()
    identity = hashlib.sha256()
    for value in (str(sequence["id"]), str(sequence.get("environment", "unknown")), reference_sha256):
        identity.update(value.encode("utf-8") + b"\0")
    identity.update(frame_index.tobytes())
    return {
        "sequence_id": str(sequence["id"]),
        "environment": str(sequence.get("environment", "unknown")),
        "source": str(sequence.get("source", "unspecified")),
        "noise_seed": int(seed), "frame_index": frame_index,
        "frame_identity_sha256": identity.hexdigest(),
        "reference_sha256": reference_sha256,
    }


def _closed_loop_sequence(
    model: nn.Module, sequence: dict[str, Any], cfg: dict[str, Any],
    device: torch.device, strategy: str, seed: int,
) -> dict[str, Any]:
    csi, history, budget, noise_rms, noise, rng, mask_history, observed_history = _initial_replay(sequence, cfg, seed)
    convention = str(cfg["pilot"].get("update_noise_convention", "component"))
    values: dict[str, list[np.ndarray]] = {name: [] for name in (
        "prior_mean", "prior_sigma", "posterior_mean", "posterior_sigma", "target", "mask",
        *[f"baseline_{name}_mean" for name in BASELINE_STAGES],
    )}
    scales: list[float] = []
    component_sigmas: list[float] = []
    model.eval()
    with torch.no_grad():
        for frame in range(history, csi.shape[0]):
            x, scale = _make_input(observed_history, mask_history)
            outputs = model(x.to(device))
            prior_mean, prior_sigma = outputs["mean"].cpu()[0], outputs["sigma"].cpu()[0]
            component_sigma = update_component_sigma(noise_rms, scale, convention)
            if strategy == "active":
                current_mask = uncertainty_mask(
                    prior_mean.numpy(), prior_sigma.numpy(), budget=budget,
                    min_spacing=int(cfg["pilot"].get("min_spacing", 2)),
                    gradient_weight=float(cfg["pilot"].get("gradient_weight", 0.0)),
                    length_scale=float(cfg["pilot"].get("update_length_scale", 4.0)),
                    observation_sigma=component_sigma,
                )
            elif strategy == "uncertainty":
                current_mask = uncertainty_top_mask(prior_sigma.numpy(), budget=budget, min_spacing=int(cfg["pilot"].get("min_spacing", 2)))
            elif strategy == "random":
                current_mask = random_mask(csi.shape[1], budget, rng)
            elif strategy == "comb":
                current_mask = comb_mask(csi.shape[1], budget, offset=frame % max(1, csi.shape[1] // budget))
            else:
                raise ValueError(f"Unknown closed-loop strategy: {strategy}")
            physical_observation = acquire_pilots(csi[frame], noise[frame], current_mask)
            observation = torch.from_numpy(as_real_imag(physical_observation / scale)).unsqueeze(0)
            posterior_mean, posterior_sigma = posterior_kernel_update(
                prior_mean.unsqueeze(0), prior_sigma.unsqueeze(0), observation,
                torch.from_numpy(current_mask[None, :]), observation_sigma=component_sigma,
                length_scale=float(cfg["pilot"].get("update_length_scale", 4.0)),
            )
            baselines = matched_information_baselines(
                observed_history, mask_history, physical_observation, current_mask, scale,
            )
            values["prior_mean"].append(prior_mean.numpy())
            values["prior_sigma"].append(prior_sigma.numpy())
            values["posterior_mean"].append(posterior_mean[0].numpy())
            values["posterior_sigma"].append(posterior_sigma[0].numpy())
            values["target"].append(as_real_imag(csi[frame] / scale))
            values["mask"].append(current_mask)
            for name, estimate in baselines.items():
                values[f"baseline_{name}_mean"].append(estimate)
            scales.append(scale)
            component_sigmas.append(component_sigma)
            observed_history = np.concatenate((observed_history[1:], physical_observation[None]), axis=0)
            mask_history = np.concatenate((mask_history[1:], current_mask[None]), axis=0)
    return {
        **_sequence_metadata(sequence, csi, history, seed),
        **{name: np.stack(items) for name, items in values.items()},
        "history_scale": np.asarray(scales),
        "update_component_sigma": np.asarray(component_sigmas),
        "injected_complex_noise_rms": float(noise_rms),
    }


def _diagnose_record(record: dict[str, Any], learned: bool = True) -> dict[str, dict[str, np.ndarray]]:
    common = {"observed_mask": record["mask"], "normalization_scale": record["history_scale"]}
    if not learned:
        return {"estimate": framewise_diagnostics(record["estimate"], None, record["target"], **common, compute_ranking=False)}
    diagnostics = {
        "prior": framewise_diagnostics(record["prior_mean"], record["prior_sigma"], record["target"], **common, compute_ranking=True),
        "posterior": framewise_diagnostics(record["posterior_mean"], record["posterior_sigma"], record["target"], **common, compute_ranking=False),
    }
    diagnostics.update({
        f"baseline_{name}": framewise_diagnostics(record[f"baseline_{name}_mean"], None, record["target"], **common, compute_ranking=False)
        for name in BASELINE_STAGES
    })
    return diagnostics


def _ranking_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    rank = dict(summary["rank"])
    if "nonpilot" in summary:
        rank["nonpilot"] = dict(summary["nonpilot"]["rank"])
    return rank


def _record_summary(record: dict[str, Any], diagnostics: dict[str, dict[str, np.ndarray]], frame_start: int) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "record_id": record["sequence_id"], "environment": record["environment"], "source": record["source"],
        "noise_seed": record["noise_seed"], "reference_sha256": record["reference_sha256"],
        "frame_identity_sha256": record["frame_identity_sha256"],
        "frame_start": frame_start, "frame_stop": frame_start + len(record["frame_index"]),
        "first_frame_index": int(record["frame_index"][0]), "num_frames": len(record["frame_index"]),
    }
    if "prior" in diagnostics:
        summary["prior"] = summarize_diagnostics(diagnostics["prior"])
        summary["posterior"] = summarize_diagnostics(diagnostics["posterior"])
        summary["prior_ranking"] = _ranking_from_summary(summary["prior"])
        summary["same_information_baselines"] = {
            name: {**summarize_diagnostics(diagnostics[f"baseline_{name}"]), "stage": stage}
            for name, stage in BASELINE_STAGES.items()
        }
    else:
        summary["estimate"] = summarize_diagnostics(diagnostics["estimate"])
    return summary


def _save_record_shard(
    output_dir: Path | None, group: str, index: int, storage: str,
    record: dict[str, Any], diagnostics: dict[str, dict[str, np.ndarray]],
) -> str | None:
    if output_dir is None or storage == "none":
        return None
    directory = output_dir / group
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"record_{index:06d}.npz"
    if path.exists():
        raise FileExistsError(f"Refusing to replace replay shard: {path}")
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(2), "prediction_storage": np.asarray(storage),
        "record_id": np.asarray(record["sequence_id"]), "environment": np.asarray(record["environment"]),
        "source": np.asarray(record["source"]), "noise_seed": np.asarray(record["noise_seed"]),
        "frame_identity_sha256": np.asarray(record["frame_identity_sha256"]),
        "reference_sha256": np.asarray(record["reference_sha256"]),
        "frame_index": record["frame_index"], "history_scale": record["history_scale"],
        "num_subcarriers": np.asarray(record["target"].shape[1]),
        "pilot_count": record["mask"].sum(axis=1),
    }
    compact_keys = {
        "nmse_db", "nmse_linear", "nonpilot_nmse_db", "nonpilot_nmse_linear",
        "rank_spearman", "rank_valid_frame", "nonpilot_rank_spearman", "nonpilot_rank_valid_frame",
        "coverage_90", "nonpilot_coverage_90",
    }
    for name, diagnostic in diagnostics.items():
        arrays.update({f"{name}_{key}": value for key, value in diagnostic.items() if storage == "full" or key in compact_keys})
    if storage == "full":
        for key in ("prior_mean", "prior_sigma", "posterior_mean", "posterior_sigma", "estimate", "target", "mask", "update_component_sigma"):
            if key in record:
                arrays[key] = record[key]
        arrays.update({key: value for key, value in record.items() if key.startswith("baseline_")})
    np.savez_compressed(path, **arrays)
    return str(path.relative_to(output_dir))


def _record_link_proxy(record: dict[str, Any], estimate: np.ndarray, cfg: dict[str, Any], physical_reference: np.ndarray) -> dict[str, Any]:
    if not bool(cfg.get("eval", {}).get("run_link_proxy", False)):
        return {}
    predicted = (estimate[..., 0] + 1j * estimate[..., 1]) * record["history_scale"][:, None]
    results = {}
    for snr in cfg["eval"].get("snr_db", [5, 10, 15]):
        result = simulate_qpsk_link(
            physical_reference, predicted, record["mask"], snr_db=float(snr),
            trials=int(cfg["eval"].get("link_trials", 4)),
            block_subcarriers=int(cfg["eval"].get("block_subcarriers", 16)),
            code_rate=float(cfg["eval"].get("code_rate", 0.5)),
            seed=(int(record["noise_seed"]) + int(float(snr) * 17)) % (2**32),
        )
        result["normalized_goodput_proxy"] = result.pop("normalized_goodput")
        results[str(snr)] = result
    return results


def _aggregate_link_proxy(parts: list[dict[str, Any]], cfg: dict[str, Any], pilot_ratio: float) -> dict[str, Any]:
    if not parts:
        return {}
    result = {"scope": "uncoded one-tap QPSK proxy; excludes feedback, synchronization, Doppler correction and packet protocol",
              "snr_reference": "mean physical channel power over the evaluated frames and all subcarriers in each record",
              "randomization": "bits and AWGN drawn on the complete frequency grid before masking; common across policies",
              "payload_accounting": "code_rate times bits in error-free blocks per total grid slot, including partial final blocks by their actual length",
              "pilot_ratio": pilot_ratio, "snr_results": {}}
    for snr in parts[0]:
        total_bits = sum(part[snr]["total_bits"] for part in parts)
        bit_errors = sum(part[snr]["bit_errors"] for part in parts)
        blocks = sum(part[snr]["blocks"] for part in parts)
        block_errors = sum(part[snr]["block_errors"] for part in parts)
        delivered = sum(part[snr]["error_free_block_bits"] for part in parts)
        slots = sum(part[snr]["grid_slots"] for part in parts)
        ber = bit_errors / total_bits if total_bits else None
        bler = block_errors / blocks if blocks else None
        result["snr_results"][snr] = {
            "ber": ber, "block_error_proxy": bler, "total_bits": total_bits, "blocks": blocks,
            "bit_errors": bit_errors, "block_errors": block_errors,
            "error_free_block_bits": delivered, "grid_slots": slots,
            "normalized_goodput_proxy": float(cfg["eval"].get("code_rate", 0.5)) * delivered / slots,
            "code_rate_is_accounting_factor_only": True,
        }
    return result


def _validate_sequence_identities(sequences: list[dict[str, Any]]) -> None:
    keys = [(str(item.get("environment", "unknown")), str(item["id"])) for item in sequences]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("Replay requires nonempty test data and unique (environment, original record ID) pairs.")


def evaluate_closed_loop(
    model: nn.Module, test_sequences: list[dict[str, Any]], cfg: dict[str, Any],
    device: torch.device, strategy: str, output_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate one record at a time; return a small first-record plot preview."""
    validate_replay_config(cfg)
    _validate_sequence_identities(test_sequences)
    storage = str(cfg.get("eval", {}).get("prediction_storage", "compact"))
    accumulators = {name: DiagnosticsAccumulator() for name in ("prior", "posterior", *[f"baseline_{name}" for name in BASELINE_STAGES])}
    records, link_parts = [], []
    preview: dict[str, np.ndarray] = {}
    frame_offset, pilot_count, bin_count = 0, 0, 0
    print(f"replay policy={strategy} records={len(test_sequences)}", flush=True)
    for index, sequence in enumerate(test_sequences):
        if index % max(1, len(test_sequences) // 10) == 0:
            print(f"replay policy={strategy} record={index + 1}/{len(test_sequences)} id={sequence['id']}", flush=True)
        record = _closed_loop_sequence(model, sequence, cfg, device, strategy, seed=int(cfg["seed"]) + 4241 * index)
        diagnostics = _diagnose_record(record)
        for name, diagnostic in diagnostics.items():
            accumulators[name].update(diagnostic)
        summary = _record_summary(record, diagnostics, frame_offset)
        summary["placement_diagnostics"] = summarize_masks(record["mask"])
        shard = _save_record_shard(output_dir, f"closed_loop_{strategy}", index, storage, record, diagnostics)
        if shard is not None:
            summary["arrays_file"] = shard
        records.append(summary)
        frame_offset += len(record["frame_index"])
        pilot_count += int(record["mask"].sum())
        bin_count += int(record["mask"].size)
        if not preview:
            preview = {key: np.array(record[key][:80], copy=True) for key in (
                "prior_mean", "prior_sigma", "posterior_mean", "posterior_sigma", "target", "mask", "frame_index", "history_scale",
            )}
            preview["record_id"] = np.asarray(record["sequence_id"])
            preview["environment"] = np.asarray(record["environment"])
        proxy = _record_link_proxy(record, record["posterior_mean"], cfg, np.asarray(sequence["csi"])[record["frame_index"][0]:])
        if proxy:
            link_parts.append(proxy)
        del record, diagnostics
    summaries = {name: accumulator.summary() for name, accumulator in accumulators.items()}
    result = {
        "strategy": strategy, "prior": summaries["prior"], "posterior": summaries["posterior"],
        "prior_ranking": _ranking_from_summary(summaries["prior"]),
        "same_information_baselines": {name: {**summaries[f"baseline_{name}"], "stage": stage} for name, stage in BASELINE_STAGES.items()},
        "pilot_ratio": pilot_count / bin_count, "num_sequences": len(records), "records": records,
        "placement_diagnostics": combine_mask_summaries([r["placement_diagnostics"] for r in records]),
        "arrays_format": "record_shards", "prediction_storage": storage,
        "nonpilot_comparison_scope": "The nonpilot set is policy dependent; full-band NMSE is the primary paired policy comparison.",
        "protocol": protocol_metadata(cfg),
    }
    if link_parts:
        result["link_proxy"] = _aggregate_link_proxy(link_parts, cfg, result["pilot_ratio"])
    return result, preview


def _classical_closed_loop_sequence(
    sequence: dict[str, Any], cfg: dict[str, Any], receiver: str,
    frequency_prior: ComplexFrequencyPrior | None, mask_strategy: str, seed: int,
) -> dict[str, Any]:
    csi, history, budget, noise_rms, noise, rng, mask_history, observed_history = _initial_replay(sequence, cfg, seed)
    estimates, targets, masks, scales = [], [], [], []
    for frame in range(history, csi.shape[0]):
        _, scale = _make_input(observed_history, mask_history)
        if mask_strategy == "comb":
            current_mask = comb_mask(csi.shape[1], budget, offset=frame % max(1, csi.shape[1] // budget))
        elif mask_strategy == "random":
            current_mask = random_mask(csi.shape[1], budget, rng)
        else:
            raise ValueError(f"Classical receivers support comb/random masks, got {mask_strategy!r}.")
        physical_observation = acquire_pilots(csi[frame], noise[frame], current_mask)
        observation = physical_observation / scale
        if receiver == "ls_interpolation":
            estimate = interpolate_observation(observation, current_mask)
        elif receiver == "lmmse":
            if frequency_prior is None:
                raise ValueError("LMMSE requires a training-only frequency prior.")
            estimate = lmmse_estimate(observation, current_mask, frequency_prior, observation_variance=(noise_rms / scale)**2)
        elif receiver == "omp":
            classical = cfg.get("classical", {})
            estimate = omp_delay_estimate(
                observation, current_mask, delay_bins=int(classical.get("omp_delay_bins", min(24, csi.shape[1]))),
                sparsity=int(classical.get("omp_sparsity", 6)), ridge=float(classical.get("omp_ridge", 1e-6)),
            )
        else:
            raise ValueError(f"Unknown classical receiver: {receiver}")
        estimates.append(as_real_imag(estimate))
        targets.append(as_real_imag(csi[frame] / scale))
        masks.append(current_mask)
        scales.append(scale)
        observed_history = np.concatenate((observed_history[1:], physical_observation[None]), axis=0)
        mask_history = np.concatenate((mask_history[1:], current_mask[None]), axis=0)
    return {
        **_sequence_metadata(sequence, csi, history, seed),
        "estimate": np.stack(estimates), "target": np.stack(targets), "mask": np.stack(masks),
        "history_scale": np.asarray(scales),
    }


def evaluate_classical_receiver(
    test_sequences: list[dict[str, Any]], cfg: dict[str, Any], receiver: str,
    frequency_prior: ComplexFrequencyPrior | None, mask_strategy: str = "comb", output_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    validate_replay_config(cfg)
    _validate_sequence_identities(test_sequences)
    start = time.perf_counter()
    accumulator = DiagnosticsAccumulator()
    records, link_parts = [], []
    frame_offset, pilot_count, bin_count = 0, 0, 0
    storage = str(cfg.get("eval", {}).get("prediction_storage", "compact"))
    print(f"reference receiver={receiver} records={len(test_sequences)}", flush=True)
    for index, sequence in enumerate(test_sequences):
        if index % max(1, len(test_sequences) // 10) == 0:
            print(f"reference receiver={receiver} record={index + 1}/{len(test_sequences)} id={sequence['id']}", flush=True)
        record = _classical_closed_loop_sequence(sequence, cfg, receiver, frequency_prior, mask_strategy, seed=int(cfg["seed"]) + 4241 * index)
        diagnostics = _diagnose_record(record, learned=False)
        accumulator.update(diagnostics["estimate"])
        summary = _record_summary(record, diagnostics, frame_offset)
        shard = _save_record_shard(output_dir, f"classical_{receiver}", index, storage, record, diagnostics)
        if shard is not None:
            summary["arrays_file"] = shard
        records.append(summary)
        frame_offset += len(record["frame_index"])
        pilot_count += int(record["mask"].sum())
        bin_count += int(record["mask"].size)
        proxy = _record_link_proxy(record, record["estimate"], cfg, np.asarray(sequence["csi"])[record["frame_index"][0]:])
        if proxy:
            link_parts.append(proxy)
        del record, diagnostics
    metrics = accumulator.summary()
    metrics.update({
        "receiver": receiver, "mask_strategy": mask_strategy, "pilot_ratio": pilot_count / bin_count,
        "stage": "post-acquisition; current noisy pilot observations",
        "uncertainty_applicable": False, "latency_ms_per_frame": 1000 * (time.perf_counter()-start) / max(frame_offset, 1),
        "records": records, "arrays_format": "record_shards", "prediction_storage": storage,
    })
    if receiver == "lmmse":
        metrics["prior_scope"] = "Empirical training-only proper-complex covariance; training records normalized by their RMS; test observations by sparse-history RMS. This is a model-based reference, not an oracle bound."
        metrics["prior_fit"] = {
            "sampling": "At least one frame per training record; remaining slots apportioned by record length minus one; one seeded draw per time stratum.",
            "available_frames": sum(r["available_frames"] for r in frequency_prior.fit_records),
            "used_frames": sum(r["used_frames"] for r in frequency_prior.fit_records),
            "records": list(frequency_prior.fit_records),
        }
    if link_parts:
        metrics["link_proxy"] = _aggregate_link_proxy(link_parts, cfg, metrics["pilot_ratio"])
    return metrics, {}


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

    comb = summary["closed_loop"]["comb"]
    labels = ["Learned + comb", "Interpolation + comb", "Persistence innovation + comb"]
    values = [comb["posterior"]["nmse_db_mean"],
              comb["same_information_baselines"]["current_interpolation"]["nmse_db_mean"],
              comb["same_information_baselines"]["persistence_innovation"]["nmse_db_mean"]]
    for policy, label in (("uncertainty", "Learned + top-B"), ("active", "Learned + A-optimal")):
        if policy in summary["closed_loop"]:
            labels.append(label)
            values.append(summary["closed_loop"][policy]["posterior"]["nmse_db_mean"])
    plt.figure(figsize=(6.4, 3.6))
    positions = np.arange(len(labels))
    plt.bar(positions, [np.nan if value is None else value for value in values])
    plt.axhline(0, color="0.3", linewidth=0.8, linestyle="--", label="Zero-channel reference")
    plt.xticks(positions, labels, rotation=20, ha="right")
    plt.ylabel("Channel NMSE (dB; lower is better)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "nmse_summary.pdf")
    plt.savefig(output_dir / "nmse_summary.png", dpi=180)
    plt.close()

    if not active_arrays:
        return
    target = active_arrays["target"]
    estimate = active_arrays["posterior_mean"]
    index = min(4, target.shape[0] - 1)
    target_complex = target[index, :, 0] + 1j * target[index, :, 1]
    estimate_complex = estimate[index, :, 0] + 1j * estimate[index, :, 1]
    pilot_indices = np.flatnonzero(active_arrays["mask"][index])
    x = np.arange(target.shape[1])
    plt.figure(figsize=(6.2, 3.4))
    plt.plot(x, 20.0 * np.log10(np.abs(target_complex) + 1e-5), label="Reference")
    plt.plot(x, 20.0 * np.log10(np.abs(estimate_complex) + 1e-5), label="posterior")
    plt.scatter(pilot_indices, 20.0 * np.log10(np.abs(target_complex[pilot_indices]) + 1e-5), marker="x", label="Reference at pilot locations")
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

    if "link_proxy" in summary["closed_loop"]["active"]:
        active = summary["closed_loop"]["active"]["link_proxy"]["snr_results"]
        comb_proxy = summary["closed_loop"]["comb"]["link_proxy"]["snr_results"]
        keys = sorted(active, key=float)
        plt.figure(figsize=(5.4, 3.4))
        plt.plot([float(key) for key in keys], [comb_proxy[key]["normalized_goodput_proxy"] for key in keys], marker="o", label="comb")
        plt.plot([float(key) for key in keys], [active[key]["normalized_goodput_proxy"] for key in keys], marker="s", label="A-optimal")
        plt.xlabel("SNR (dB)")
        plt.ylabel("Uncoded QPSK goodput proxy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "link_proxy.pdf")
        plt.close()


def _load_state_dict(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.6
        return torch.load(path, map_location="cpu")


def _policies(cfg: dict[str, Any]) -> list[str]:
    policies = list(dict.fromkeys(str(value) for value in cfg["eval"].get("mask_strategies", ["comb", "random", "uncertainty", "active"])))
    for required in ("comb", "active"):
        if required not in policies:
            policies.append(required)
    unknown = set(policies) - {"comb", "random", "uncertainty", "active"}
    if unknown:
        raise ValueError(f"Unknown placement policies: {sorted(unknown)}")
    return policies


def _evaluate_policies(
    model: nn.Module, test_sequences: list[dict[str, Any]], cfg: dict[str, Any],
    device: torch.device, output_dir: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    results = {}
    active_preview: dict[str, np.ndarray] = {}
    for strategy in _policies(cfg):
        metrics, preview = evaluate_closed_loop(model, test_sequences, cfg, device, strategy, output_dir=output_dir)
        results[strategy] = metrics
        if strategy == "active":
            active_preview = preview
        if preview:
            np.savez_compressed(output_dir / f"preview_{strategy}.npz", **preview)
    return results, active_preview


def _classical_receivers(cfg: dict[str, Any]) -> list[str]:
    if not bool(cfg.get("eval", {}).get("run_classical", True)):
        return []
    names = list(dict.fromkeys(str(name) for name in cfg["eval"].get("classical_receivers", ["ls_interpolation", "lmmse", "omp"])))
    if set(names) - {"ls_interpolation", "lmmse", "omp"}:
        raise ValueError(f"Unknown classical receivers: {names}")
    return names


def _evaluate_classical_suite(
    test_sequences: list[dict[str, Any]], training_sequences: list[dict[str, Any]] | None,
    cfg: dict[str, Any], output_dir: Path,
) -> dict[str, Any]:
    names = _classical_receivers(cfg)
    prior = None
    if "lmmse" in names:
        if not training_sequences:
            raise ValueError("LMMSE needs a real training split; set classical.training_root or remove lmmse from eval.classical_receivers.")
        frequency_count = np.asarray(test_sequences[0]["csi"]).shape[1]
        if any(np.asarray(item["csi"]).shape[1] != frequency_count for item in training_sequences):
            raise ValueError("LMMSE training and test frequency grids have different dimensions.")
        classical = cfg.get("classical", {})
        prior = fit_frequency_prior(training_sequences, shrinkage=float(classical.get("covariance_shrinkage", 0.05)), max_frames=int(classical.get("max_covariance_frames", 20_000)))
        np.savez_compressed(output_dir / "classical_frequency_prior.npz", mean_real=prior.mean.real, mean_imag=prior.mean.imag,
                            covariance_real=prior.covariance.real, covariance_imag=prior.covariance.imag, shrinkage=np.asarray(prior.shrinkage),
                            training_environments=np.asarray([r["environment"] for r in prior.fit_records]),
                            training_record_ids=np.asarray([r["record_id"] for r in prior.fit_records]),
                            frame_offsets=np.cumsum([0, *[len(idx) for idx in prior.fit_indices]]),
                            frame_indices=np.concatenate(prior.fit_indices))
    return {name: evaluate_classical_receiver(test_sequences, cfg, name, prior, output_dir=output_dir)[0] for name in names}


def run_experiment(config: str | Path | dict[str, Any], *, skip_existing: bool = False) -> dict[str, Any]:
    cfg = resolve_config_paths(load_config(config) if not isinstance(config, dict) else copy.deepcopy(config))
    validate_config(cfg)
    validate_replay_config(cfg)
    configure_torch_runtime(cfg["train"].get("torch_threads"))
    seed_everything(int(cfg["seed"]))
    init_checkpoint = cfg["train"].get("init_checkpoint")
    context = begin_run(cfg, mode="fine_tuning" if init_checkpoint else "fresh_training", checkpoint=init_checkpoint, skip_existing=skip_existing)
    if context["cached_result"] is not None:
        return context["cached_result"]
    output_dir = context["output_dir"]
    device = choose_device(str(cfg["train"].get("device", "auto")))
    sequences = _build_sequences(cfg)
    datasets = _build_datasets(cfg, sequences)
    loaders = {split: _loader(dataset, cfg, shuffle=split == "train") for split, dataset in datasets.items()}
    model = build_forecast_model(datasets["train"].n_subcarriers, cfg["model"])
    if init_checkpoint:
        bundle = _load_state_dict(init_checkpoint)
        check_checkpoint_protocol(bundle, cfg, datasets["train"].n_subcarriers)
        model.load_state_dict(bundle["model_state"], strict=True)
    model, history = train_model(model, loaders["train"], loaders["val"], cfg, output_dir, device)
    _write_training_csv(output_dir / "training_history.csv", history)
    val_metrics, _ = _evaluate_model_loader(model, loaders["val"], device, cfg["loss"])
    test_metrics, _ = _evaluate_model_loader(model, loaders["test"], device, cfg["loss"])
    persistence = _evaluate_baseline_loader(loaders["test"], persistence_forecast, "persistence")
    linear = _evaluate_baseline_loader(loaders["test"], linear_time_extrapolation, "linear_time_extrapolation")
    closed_loop, preview = _evaluate_policies(model, sequences["test"], cfg, device, output_dir)
    classical_results = _evaluate_classical_suite(sequences["test"], sequences["train"], cfg, output_dir)
    summary = {
        "artifact_status": "synthetic sanity check" if cfg["data"]["kind"] == "synthetic" or any(str(item.get("source", "")).startswith("synthetic_") for item in sequences["test"]) else "measured-channel experiment",
        "device": str(device), "model_kind": str(cfg["model"].get("kind", "aquapilot")),
        "model_parameters": count_parameters(model), "initialized_from": str(init_checkpoint) if init_checkpoint else None,
        "protocol": protocol_metadata(cfg),
        "training_history_protocol": {"noise_complex_rms_fraction": float(cfg["pilot"].get("observation_sigma", 0.03)),
                                      "training_masks": cfg["pilot"].get("training_mask_mode", "mixed"),
                                      "heldout_masks": "deterministic rotating comb; no block augmentation", "targets": "clean reference CSI"},
        "data": {split: {"sequences": len(sequences[split]), "windows": len(datasets[split]),
                         "environments": sorted({str(item.get("environment", "unknown")) for item in sequences[split]})}
                 for split in ("train", "val", "test")},
        "model_validation": val_metrics, "model_test": test_metrics,
        "baselines": {"persistence": persistence, "linear": linear},
        "closed_loop": closed_loop, "classical_closed_loop": classical_results,
    }
    _plot_outputs(output_dir, history, summary, preview)
    return finish_run(context, summary)


def evaluate_checkpoint(
    config: str | Path | dict[str, Any], checkpoint: str | Path,
    output_dir: str | Path | None = None, *, skip_existing: bool = False,
) -> dict[str, Any]:
    """Replay a fixed checkpoint, optionally including classical estimators."""
    raw = load_config(config) if not isinstance(config, dict) else copy.deepcopy(config)
    if output_dir is not None:
        raw["output_dir"] = str(output_dir)
    cfg = resolve_config_paths(raw)
    checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(__file__).resolve().parents[2] / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    bundle = _load_state_dict(checkpoint_path)
    cfg["model"] = copy.deepcopy(bundle["model_config"])
    validate_config(cfg)
    validate_replay_config(cfg)
    check_checkpoint_protocol(bundle, cfg, int(bundle["n_subcarriers"]))
    if cfg["data"].get("kind") != "prepared":
        raise ValueError("Checkpoint reevaluation expects data.kind=prepared.")
    configure_torch_runtime(cfg["train"].get("torch_threads"))
    seed_everything(int(cfg["seed"]))
    classical_names = _classical_receivers(cfg)
    training_root = Path(cfg.get("classical", {}).get("training_root", cfg["data"]["root"]))
    splits = ["test"]
    if "lmmse" in classical_names and training_root.resolve() == Path(cfg["data"]["root"]).resolve():
        splits.append("train")
    context = begin_run(cfg, mode="checkpoint_reevaluation", checkpoint=checkpoint_path, skip_existing=skip_existing, input_splits=splits)
    if context["cached_result"] is not None:
        return context["cached_result"]
    destination = context["output_dir"]
    device = choose_device(str(cfg["train"].get("device", "auto")))
    test_sequences = load_prepared_sequences(cfg["data"]["root"], "test")
    for item in test_sequences:
        check_checkpoint_protocol(bundle, cfg, np.asarray(item["csi"]).shape[1])
    model = build_forecast_model(int(bundle["n_subcarriers"]), bundle["model_config"])
    model.load_state_dict(bundle["model_state"], strict=True)
    model.to(device)
    test_dataset = ChannelWindowDataset(
        test_sequences, history=int(cfg["data"]["history"]), horizon=1,
        pilot_ratio=float(cfg["pilot"].get("ratio", 0.125)), mask_mode="comb", extra_block_mask=0,
        seed=int(cfg["seed"]) + 200_006, noise_seed=int(cfg["seed"]),
        observation_sigma=float(cfg["pilot"].get("observation_sigma", 0.03)),
        noise_cache_mb=float(cfg["train"].get("noise_cache_mb", 128.0)),
    )
    test_loader = _loader(test_dataset, cfg, shuffle=False)
    test_metrics, _ = _evaluate_model_loader(model, test_loader, device, cfg["loss"])
    persistence = _evaluate_baseline_loader(test_loader, persistence_forecast, "persistence")
    linear = _evaluate_baseline_loader(test_loader, linear_time_extrapolation, "linear_time_extrapolation")
    closed_loop, preview = _evaluate_policies(model, test_sequences, cfg, device, destination)
    training_sequences = load_prepared_sequences(training_root, "train") if "lmmse" in classical_names else None
    classical_results = _evaluate_classical_suite(test_sequences, training_sequences, cfg, destination)
    summary = {
        "artifact_status": "synthetic sanity check" if any(str(item.get("source", "")).startswith("synthetic_") for item in test_sequences) else "fixed-checkpoint measured-channel reevaluation", "checkpoint": str(checkpoint_path),
        "checkpoint_training_config": bundle.get("experiment_config"), "device": str(device),
        "model_kind": str(cfg["model"].get("kind", "aquapilot")), "model_parameters": count_parameters(model),
        "protocol": protocol_metadata(cfg),
        "data": {"test_sequences": len(test_sequences), "test_windows": len(test_dataset),
                 "environments": sorted({str(item.get("environment", "unknown")) for item in test_sequences})},
        "model_test": test_metrics, "baselines": {"persistence": persistence, "linear": linear},
        "closed_loop": closed_loop, "classical_closed_loop": classical_results,
    }
    _plot_outputs(destination, [], summary, preview)
    return finish_run(context, summary)


def one_standard_error_select(rows: list[dict[str, float]]) -> dict[str, float]:
    """Select the smallest rank within one standard error of the best mean NMSE."""
    if not rows:
        raise ValueError("rows must not be empty.")
    best = min(rows, key=lambda row: row["mean_nmse_db"])
    threshold = best["mean_nmse_db"] + best["se_nmse_db"]
    eligible = [row for row in rows if row["mean_nmse_db"] <= threshold]
    return min(eligible, key=lambda row: int(row["rank"]))


def run_rank_search(config_path: str | Path, ranks: list[int], seeds: list[int]) -> dict[str, Any]:
    base = resolve_config_paths(load_config(config_path))
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
