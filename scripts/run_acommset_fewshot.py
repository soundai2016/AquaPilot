#!/usr/bin/env python3
"""Fine-tune a Watermark checkpoint on chronological ACommSet adaptation data."""
from __future__ import annotations

import argparse
from pathlib import Path

from aquapilot.config import load_config
from aquapilot.engine import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/acommset_few_shot.yaml"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    cfg = load_config(args.config)
    cfg["train"]["init_checkpoint"] = str(args.checkpoint)
    if args.output is not None:
        cfg["output_dir"] = str(args.output)
    result = run_experiment(cfg)
    print(result["model_test"])


if __name__ == "__main__":
    main()
