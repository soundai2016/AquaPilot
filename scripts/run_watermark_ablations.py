#!/usr/bin/env python3
"""Run optional Watermark ablations on one prepared LOEO fold."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from aquapilot.config import load_config
from aquapilot.engine import run_experiment
from aquapilot.utils import save_json

ABLATIONS = ("full", "unstructured", "rank0", "rank8", "no_delay", "homoscedastic")


def apply_ablation(cfg: dict[str, Any], name: str, fixed_sigma: float) -> None:
    if name == "full":
        return
    if name == "unstructured":
        cfg["model"]["kind"] = "unstructured"
        cfg["model"]["rank"] = 0
        cfg["loss"]["causal_weight"] = 0.0
        cfg["loss"]["smooth_weight"] = 0.0
        cfg["loss"]["rank_weight"] = 0.0
        cfg["eval"]["run_classical"] = False
        return
    if name == "rank0":
        cfg["model"]["rank"] = 0
        return
    if name == "rank8":
        cfg["model"]["rank"] = 8
        return
    if name == "no_delay":
        cfg["loss"]["causal_weight"] = 0.0
        return
    if name == "homoscedastic":
        cfg["model"]["fixed_sigma"] = float(fixed_sigma)
        return
    raise ValueError(f"Unknown ablation: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/watermark.yaml"))
    parser.add_argument(
        "--fold-root", type=Path, default=Path("data/prepared/watermark/fold_test_KAU")
    )
    parser.add_argument("--output-root", type=Path, default=Path("results/watermark_ablations"))
    parser.add_argument("--ablations", nargs="+", choices=ABLATIONS, default=list(ABLATIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 37, 57])
    parser.add_argument("--pilot-ratio", type=float, default=0.125)
    parser.add_argument(
        "--fixed-sigma",
        type=float,
        default=0.5,
        help="Homoscedastic normalized standard deviation; select on validation data only",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run:
        for split in ("train", "val", "test"):
            if not (args.fold_root / split).exists():
                raise FileNotFoundError(args.fold_root / split)
    base = load_config(args.config)
    index: list[dict[str, Any]] = []
    for ablation in args.ablations:
        for seed in args.seeds:
            output = args.output_root / args.fold_root.name / ablation / f"seed_{seed}"
            record: dict[str, Any] = {
                "ablation": ablation,
                "seed": seed,
                "pilot_ratio": args.pilot_ratio,
                "fold_root": str(args.fold_root),
                "output_dir": str(output),
            }
            if not args.dry_run:
                metrics_path = output / "metrics.json"
                if args.skip_existing and metrics_path.exists():
                    result = json.loads(metrics_path.read_text(encoding="utf-8"))
                else:
                    cfg = copy.deepcopy(base)
                    cfg["seed"] = seed
                    cfg["data"]["root"] = str(args.fold_root)
                    cfg["pilot"]["ratio"] = args.pilot_ratio
                    cfg["output_dir"] = str(output)
                    apply_ablation(cfg, ablation, args.fixed_sigma)
                    result = run_experiment(cfg)
                record.update(
                    {
                        "forecast_nmse_db": result["model_test"]["nmse_db_mean"],
                        "comb_nmse_db": result["closed_loop"]["comb"]["posterior"]["nmse_db_mean"],
                        "active_nmse_db": result["closed_loop"]["active"]["posterior"]["nmse_db_mean"],
                        "coverage90": result["model_test"]["coverage_90"],
                        "parameters": result["model_parameters"],
                    }
                )
            index.append(record)
            save_json(args.output_root / "ablation_index.json", index)
    print(json.dumps({"planned_runs": len(index), "index": str(args.output_root / 'ablation_index.json')}, indent=2))


if __name__ == "__main__":
    main()
