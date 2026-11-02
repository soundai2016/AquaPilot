#!/usr/bin/env python3
"""Reevaluate saved Watermark checkpoints without changing their training runs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from aquapilot.config import load_config
from scripts.run_watermark_loeo import (add_matrix_arguments, build_plan, config_for_run,
                                        flatten_run, postprocess, project_path, ratio_tag, save_plan, write_rows)


def checkpoint_for_run(root: Path, kind: str, item: dict, *, dry_run: bool = False) -> Path:
    suffix = Path(item["fold"]) / ratio_tag(item["pilot_ratio"]) / f"seed_{item['seed']}" / "checkpoint.pt"
    candidates = [root / kind / suffix, root / suffix]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) > 1:
        raise ValueError(f"Two checkpoints match {suffix}; pass --checkpoint-root ending in the intended model directory.")
    if existing:
        return existing[0]
    if dry_run:
        return candidates[0]
    raise FileNotFoundError(f"No checkpoint for {suffix}. Checked {candidates}. Narrow --folds/--pilot-ratios/--seeds to saved runs, or use fresh training.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_matrix_arguments(parser, results_root="results/watermark_inband_reeval")
    parser.add_argument("--checkpoint-root", type=Path, default=Path("results/watermark_inband"),
                        help="Existing root with optional model-kind directory above fold/ratio/seed")
    args = parser.parse_args()
    base = load_config(project_path(args.config))
    if base["data"]["kind"] != "prepared":
        parser.error("Checkpoint reevaluation requires prepared Watermark folds.")
    kind = str(base["model"].get("kind", "aquapilot"))
    plan = build_plan(args, base)
    checkpoint_root = project_path(args.checkpoint_root)
    for item in plan:
        checkpoint = checkpoint_for_run(checkpoint_root, kind, item, dry_run=args.dry_run)
        item["checkpoint"] = str(checkpoint)
        if Path(item["output_dir"]).resolve() == checkpoint.parent.resolve():
            parser.error("Reevaluation output must be separate from the original checkpoint directory.")
    if args.dry_run:
        print(json.dumps({"mode": "checkpoint_reevaluation", "runs": len(plan), "plan": plan}, indent=2))
        return
    from aquapilot.engine import evaluate_checkpoint
    destination = project_path(args.results_root) / kind
    save_plan(destination / "run_plan.json", plan)
    rows = []
    for index, item in enumerate(plan, 1):
        print(f"[{index}/{len(plan)}] {item['fold']} ratio={item['pilot_ratio']} seed={item['seed']}", flush=True)
        cfg = config_for_run(base, item)
        result = evaluate_checkpoint(cfg, item["checkpoint"], item["output_dir"], skip_existing=args.skip_existing)
        rows.append(flatten_run(result, item["fold"], item["pilot_ratio"], item["seed"], Path(item["output_dir"])))
        write_rows(destination / "aggregate_runs.csv", rows)
    postprocess(destination, plan, export_tables=False)
    print(json.dumps({"runs": len(rows), "analysis": str(destination / "analysis")}, indent=2))


if __name__ == "__main__":
    main()
