#!/usr/bin/env python3
"""Validate the existing prepared Watermark folds without preprocessing them."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aquapilot.config import load_config
from aquapilot.data.audit import audit_watermark, project_path
from aquapilot.utils import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, default=Path("data/prepared/watermark_inband"))
    parser.add_argument("--config", type=Path, default=Path("configs/watermark.yaml"))
    parser.add_argument("--output", type=Path, help="Optional audit JSON outside the data directory")
    args = parser.parse_args()
    cfg = load_config(project_path(args.config))
    root = project_path(args.prepared_root)
    output = project_path(args.output) if args.output is not None else None
    protected = [root, project_path("data")]
    data_parent = next((parent for parent in root.parents if parent.name == "data"), None)
    if data_parent is not None:
        protected.append(data_parent)
    if output is not None:
        if output.suffix.lower() != ".json":
            parser.error("The audit report must use a .json filename.")
        if any(output.is_relative_to(parent) for parent in protected):
            parser.error("The audit report must be outside data; for example results/data_audit.json.")
    try:
        report = audit_watermark(root, history=int(cfg["data"].get("history", 16)),
                                 horizon=int(cfg["data"].get("horizon", 1)),
                                 n_subcarriers=int(cfg["data"].get("n_subcarriers", 128)))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Data audit failed: {exc}\n")
    if output is not None:
        save_json(output, report)
    compact = {key: value for key, value in report.items() if key not in {"folds", "manifest_fingerprints"}}
    compact["folds"] = {name: fold["splits"] for name, fold in report["folds"].items()}
    if output is not None:
        compact["report"] = str(output)
    print(json.dumps(compact, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
