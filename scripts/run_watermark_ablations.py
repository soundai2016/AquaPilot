#!/usr/bin/env python3
"""Run optional Watermark ablations on one prepared LOEO fold."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from aquapilot.config import load_config
from aquapilot.provenance import resolve_config_paths

ABLATIONS = ("full", "unstructured", "rank0", "rank8", "no_delay", "homoscedastic")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


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
        "--fold-root", type=Path, default=Path("data/prepared/watermark_inband/fold_test_KAU")
    )
    parser.add_argument("--output-root", type=Path, default=Path("results/watermark_inband_ablations"))
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
    args.config = project_path(args.config)
    args.fold_root = project_path(args.fold_root)
    args.output_root = project_path(args.output_root)
    if len(args.ablations) != len(set(args.ablations)) or len(args.seeds) != len(set(args.seeds)):
        parser.error("Duplicate ablations or seeds would reuse an output directory.")
    if not 0.0 < args.pilot_ratio <= 1.0:
        parser.error("--pilot-ratio must be in (0, 1].")
    base = load_config(args.config)
    if not args.dry_run:
        from aquapilot.utils import save_json
        plan_path = args.output_root / "ablation_plan.json"
        plan = {"config": base, "fold_root": str(args.fold_root), "ablations": args.ablations,
                "seeds": args.seeds, "pilot_ratio": args.pilot_ratio, "fixed_sigma": args.fixed_sigma}
        if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError(f"{plan_path} describes a different experiment. Choose a new --output-root.")
        if not plan_path.exists() and (args.output_root / "ablation_index.json").exists():
            raise ValueError(f"{args.output_root} contains an older ablation index. Choose a new --output-root.")
        save_json(plan_path, plan)
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
                from aquapilot.engine import run_experiment
                cfg = copy.deepcopy(base)
                cfg["seed"] = seed
                cfg["data"]["root"] = str(args.fold_root)
                cfg["pilot"]["ratio"] = args.pilot_ratio
                cfg["output_dir"] = str(output)
                apply_ablation(cfg, ablation, args.fixed_sigma)
                result = run_experiment(resolve_config_paths(cfg, PROJECT_ROOT), skip_existing=args.skip_existing)
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
            if not args.dry_run:
                from aquapilot.utils import save_json
                save_json(args.output_root / "ablation_index.json", index)
    payload = {"planned_runs": len(index), "plan": index} if args.dry_run else {
        "planned_runs": len(index), "index": str(args.output_root / 'ablation_index.json')}
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
