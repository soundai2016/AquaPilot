#!/usr/bin/env python3
"""Build four leakage-safe Watermark leave-one-environment-out folds."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

GROUPS = ("NOF", "NCS", "BCH", "KAU")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--n-subcarriers", type=int, default=128)
    parser.add_argument("--band-fraction", type=float, default=0.8)
    parser.add_argument("--time-axis", type=int, choices=[0, 1], default=None)
    args = parser.parse_args()

    script = Path(__file__).with_name("prepare_watermark.py")
    args.output_root.mkdir(parents=True, exist_ok=True)
    folds: dict[str, dict[str, object]] = {}
    for test_index, test_group in enumerate(GROUPS):
        val_group = GROUPS[(test_index + 1) % len(GROUPS)]
        train_groups = [group for group in GROUPS if group not in {test_group, val_group}]
        fold_name = f"fold_test_{test_group}"
        output = args.output_root / fold_name
        command = [
            sys.executable,
            str(script),
            "--input",
            str(args.input),
            "--output",
            str(output),
            "--n-subcarriers",
            str(args.n_subcarriers),
            "--band-fraction",
            str(args.band_fraction),
            "--train-env",
            *train_groups,
            "--val-env",
            val_group,
            "--test-env",
            test_group,
        ]
        if args.time_axis is not None:
            command.extend(["--time-axis", str(args.time_axis)])
        subprocess.run(command, check=True)
        folds[fold_name] = {
            "train": train_groups,
            "val": [val_group],
            "test": [test_group],
            "root": str(output),
        }
    manifest = {
        "protocol": "Watermark four-fold leave-one-geographical-environment-out",
        "groups": list(GROUPS),
        "note": "KAU1 and KAU2 are inseparable and both map to KAU.",
        "folds": folds,
    }
    path = args.output_root / "folds.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
