from __future__ import annotations

import argparse
import json
from pathlib import Path

from aquapilot.engine import evaluate_checkpoint, run_experiment, run_rank_search


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aquapilot", description="AquaPilot experiment runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="train and evaluate one configuration")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--skip-existing", action="store_true", help="Reuse only a completed run with matching inputs and evaluator")

    evaluate = subparsers.add_parser("evaluate", help="zero-shot evaluation of a saved checkpoint")
    evaluate.add_argument("--config", required=True, type=Path)
    evaluate.add_argument("--checkpoint", required=True, type=Path)
    evaluate.add_argument("--output", required=False, type=Path, default=None)
    evaluate.add_argument("--skip-existing", action="store_true", help="Reuse only a completed run with matching inputs and evaluator")

    rank = subparsers.add_parser("rank-search", help="one-standard-error low-rank selection")
    rank.add_argument("--config", required=True, type=Path)
    rank.add_argument("--ranks", nargs="+", type=int, default=[0, 1, 2, 4, 8])
    rank.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27])
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "run":
        result = run_experiment(args.config, skip_existing=args.skip_existing)
    elif args.command == "evaluate":
        result = evaluate_checkpoint(args.config, args.checkpoint, args.output, skip_existing=args.skip_existing)
    else:
        result = run_rank_search(args.config, args.ranks, args.seeds)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
