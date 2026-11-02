#!/usr/bin/env python3
"""Prepare four Watermark folds, holding out one environment at a time."""
from __future__ import annotations

import argparse
from pathlib import Path

if __package__:
    from .prepare_watermark import WATERMARK_GROUPS as GROUPS, prepare_watermark
else:
    from prepare_watermark import WATERMARK_GROUPS as GROUPS, prepare_watermark


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--n-subcarriers', type=int, default=128)
    parser.add_argument('--band-fraction', type=float, default=0.8)
    parser.add_argument('--time-axis', type=int, choices=[0,1], default=0, help='Slow-time axis in MATLAB subscript order (WatermarkV1: 0).')
    args = parser.parse_args(argv)
    layouts = {}
    for i, test in enumerate(GROUPS):
        val = GROUPS[(i+1) % len(GROUPS)]
        layouts['fold_test_'+test] = dict(train=[g for g in GROUPS if g not in (val,test)], val=[val], test=[test])
    prepare_watermark(args.input, args.output_root, layouts, n_subcarriers=args.n_subcarriers,
                      band_fraction=args.band_fraction, time_axis=args.time_axis)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc))
