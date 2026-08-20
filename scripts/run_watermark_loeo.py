#!/usr/bin/env python3
"""Run and aggregate leakage-safe Watermark leave-one-environment-out experiments."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any

from aquapilot.config import load_config
from aquapilot.engine import run_experiment

DEFAULT_FOLDS = ("fold_test_NOF", "fold_test_NCS", "fold_test_BCH", "fold_test_KAU")


def flatten_run(
    result: dict[str, Any], fold: str, ratio: float, seed: int, output_dir: Path
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "fold": fold,
        "pilot_ratio_requested": ratio,
        "seed": seed,
        "output_dir": str(output_dir),
        "model_kind": result.get("model_kind"),
        "parameters": result.get("model_parameters"),
        "forecast_nmse_db": result["model_test"]["nmse_db_mean"],
        "forecast_evm_percent": result["model_test"]["evm_percent_mean"],
        "forecast_latency_ms": result["model_test"]["latency_ms_per_frame"],
    }
    for name, metrics in result.get("baselines", {}).items():
        row[f"baseline_{name}_nmse_db"] = metrics.get("nmse_db_mean")
    for strategy, metrics in result.get("closed_loop", {}).items():
        prefix = f"closed_{strategy}"
        row[f"{prefix}_pilot_ratio"] = metrics.get("pilot_ratio")
        row[f"{prefix}_nmse_db"] = metrics["posterior"].get("nmse_db_mean")
        row[f"{prefix}_evm_percent"] = metrics["posterior"].get("evm_percent_mean")
        row[f"{prefix}_coverage90"] = metrics["posterior"].get("coverage_90")
        for snr, link in metrics.get("link", {}).items():
            row[f"{prefix}_ber_snr{snr}"] = link.get("ber")
            row[f"{prefix}_bler_snr{snr}"] = link.get("bler")
            row[f"{prefix}_goodput_snr{snr}"] = link.get("normalized_goodput")
    for receiver, metrics in result.get("classical_closed_loop", {}).items():
        prefix = f"classical_{receiver}"
        row[f"{prefix}_nmse_db"] = metrics.get("nmse_db_mean")
        row[f"{prefix}_evm_percent"] = metrics.get("evm_percent_mean")
        for snr, link in metrics.get("link", {}).items():
            row[f"{prefix}_ber_snr{snr}"] = link.get("ber")
            row[f"{prefix}_bler_snr{snr}"] = link.get("bler")
            row[f"{prefix}_goodput_snr{snr}"] = link.get("normalized_goodput")
    return row


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/watermark.yaml"))
    parser.add_argument(
        "--prepared-root", type=Path, default=Path("data/prepared/watermark")
    )
    parser.add_argument("--results-root", type=Path, default=Path("results/watermark"))
    parser.add_argument("--folds", nargs="+", default=list(DEFAULT_FOLDS))
    parser.add_argument("--pilot-ratios", nargs="+", type=float, default=[0.25, 0.125, 0.0625])
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 37, 57])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = load_config(args.config)
    model_kind = str(base["model"].get("kind", "aquapilot"))
    rows: list[dict[str, Any]] = []
    plan: list[dict[str, Any]] = []

    for fold in args.folds:
        fold_root = args.prepared_root / fold
        for split in ("train", "val", "test"):
            if not (fold_root / split).exists() and not args.dry_run:
                raise FileNotFoundError(
                    f"Missing {fold_root / split}. Run scripts/prepare_watermark_loeo.py first."
                )
        for ratio in args.pilot_ratios:
            if not 0.0 < ratio <= 1.0:
                raise ValueError(f"Invalid pilot ratio: {ratio}")
            ratio_tag = f"ratio_{ratio:.4f}".rstrip("0").rstrip(".")
            for seed in args.seeds:
                output = args.results_root / model_kind / fold / ratio_tag / f"seed_{seed}"
                run_plan = {
                    "fold": fold,
                    "root": str(fold_root),
                    "pilot_ratio": ratio,
                    "seed": seed,
                    "output_dir": str(output),
                    "config": str(args.config),
                }
                plan.append(run_plan)
                if args.dry_run:
                    continue
                metrics_path = output / "metrics.json"
                if args.skip_existing and metrics_path.exists():
                    result = json.loads(metrics_path.read_text(encoding="utf-8"))
                else:
                    cfg = copy.deepcopy(base)
                    cfg["seed"] = int(seed)
                    cfg["data"]["root"] = str(fold_root)
                    cfg["pilot"]["ratio"] = float(ratio)
                    cfg["output_dir"] = str(output)
                    result = run_experiment(cfg)
                rows.append(flatten_run(result, fold, ratio, seed, output))
                write_rows(args.results_root / model_kind / "aggregate_runs.csv", rows)
                (args.results_root / model_kind / "run_plan.json").write_text(
                    json.dumps(plan, indent=2), encoding="utf-8"
                )

    plan_path = args.results_root / model_kind / "run_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    if rows:
        write_rows(args.results_root / model_kind / "aggregate_runs.csv", rows)
    print(json.dumps({"runs": len(plan), "executed": len(rows), "plan": str(plan_path)}, indent=2))


if __name__ == "__main__":
    main()
