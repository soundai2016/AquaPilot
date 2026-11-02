#!/usr/bin/env python3
"""Fine-tune a Watermark checkpoint on chronological ACommSet adaptation data."""
from __future__ import annotations

import argparse
from pathlib import Path

from aquapilot.config import load_config
from aquapilot.provenance import resolve_config_paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/acommset_few_shot.yaml"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--skip-existing", action="store_true", help="Validate a completed identical run before reuse")
    args = parser.parse_args()

    checkpoint = project_path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    cfg = load_config(project_path(args.config))
    cfg["train"]["init_checkpoint"] = str(checkpoint)
    if args.output is not None:
        cfg["output_dir"] = str(args.output)
    from aquapilot.engine import run_experiment
    result = run_experiment(resolve_config_paths(cfg, PROJECT_ROOT), skip_existing=args.skip_existing)
    print(result["model_test"])


if __name__ == "__main__":
    main()
