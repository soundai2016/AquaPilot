#!/usr/bin/env python3
"""Create descriptive tables and a figure from one Watermark matrix."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aquapilot.provenance import PROTOCOL_ID

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICIES = {"comb": "Comb", "random": "Random", "uncertainty": "Uncertainty top-B", "active": "A-optimal"}


def project_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def mean_std(series: pd.Series) -> tuple[float, float]:
    values = pd.to_numeric(series, errors="raise").to_numpy(dtype=float)
    # An undefined run must not disappear from a seemingly complete mean.
    if not len(values) or not np.isfinite(values).all():
        return float("nan"), float("nan")
    return float(values.mean()), float(values.std(ddof=0))


def validate_matrix(frame: pd.DataFrame, allow_partial: bool = False) -> None:
    keys = ["fold", "pilot_ratio_requested", "seed"]
    required = set(keys + ["protocol_id", "protocol_fingerprint", "noise_convention", "run_mode"])
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing current-protocol columns: {sorted(missing)}. Generate a new run table before aggregation.")
    if frame.empty or frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError("The run table is empty or contains missing/duplicate fold-budget-seed rows.")
    if set(frame["protocol_id"]) != {PROTOCOL_ID}:
        raise ValueError("The run table was produced by a different evaluation protocol.")
    for key in ("protocol_fingerprint", "noise_convention", "run_mode"):
        if frame[key].isna().any() or frame[key].nunique() != 1:
            raise ValueError(f"Do not combine runs with different {key} values.")
    if not allow_partial:
        expected = {(fold, ratio, seed)
                    for fold in ("fold_test_NOF", "fold_test_NCS", "fold_test_BCH", "fold_test_KAU")
                    for ratio in (0.0625, 0.125, 0.25) for seed in (17, 37, 57)}
        observed = set(frame[keys].itertuples(index=False, name=None))
        if observed != expected:
            raise ValueError(f"Expected the complete 36-run matrix; found {len(observed)} rows. Use --allow-partial only for a labeled subset.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/watermark_inband/aquapilot/aggregate_runs.csv"))
    parser.add_argument("--output", type=Path, default=None, help="Defaults to summary beside the input CSV")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    source = project_path(args.input)
    destination = project_path(args.output) if args.output else source.parent / "summary"
    frame = pd.read_csv(source)
    validate_matrix(frame, args.allow_partial)
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    metrics = {"nmse_db": "nmse_db", "channel_nrmse_percent": "channel_nrmse_percent",
               "coverage90": "post_update_coverage90", "nonpilot_nmse_db": "nonpilot_nmse_db",
               "ranking_mean_frame_spearman": "prior_rank_spearman"}
    for ratio, group in frame.groupby("pilot_ratio_requested", sort=True):
        for key, label in POLICIES.items():
            row = {"pilot_ratio": ratio, "policy": label, "num_runs": len(group), "sd_ddof": 0}
            for suffix, name in metrics.items():
                column = f"closed_{key}_{suffix}"
                if column in group:
                    mean, std = mean_std(group[column])
                    row[f"{name}_mean"] = mean
                    row[f"{name}_sd"] = std
                    values = pd.to_numeric(group[column], errors="raise").to_numpy(dtype=float)
                    row[f"{name}_valid_runs"] = int(np.isfinite(values).sum())
                    row[f"{name}_undefined_runs"] = int((~np.isfinite(values)).sum())
                else:
                    row[f"{name}_mean"] = float("nan")
                    row[f"{name}_sd"] = float("nan")
                    row[f"{name}_valid_runs"] = 0
                    row[f"{name}_undefined_runs"] = len(group)
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(destination / "aggregate.csv", index=False)

    # Paired comparisons need recording/frame identities and matched validity,
    # which the flattened run CSV cannot establish.
    (destination / "environment_differences.csv").unlink(missing_ok=True)
    notes = ["SD uses divisor n. Undefined cells and their counts are retained in aggregate.csv.",
             "Paired differences use original frame matching in scripts/analyze_results.py."]
    if args.allow_partial:
        notes.append("Partial Watermark matrix.")
    (destination / "summary_notes.txt").write_text("\n".join(notes)+"\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    for label in POLICIES.values():
        subset = summary[summary["policy"] == label].sort_values("pilot_ratio")
        subset = subset[np.isfinite(subset["nmse_db_mean"]) & np.isfinite(subset["nmse_db_sd"])]
        if subset.empty:
            continue
        ax.errorbar(subset["pilot_ratio"], subset["nmse_db_mean"], yerr=subset["nmse_db_sd"],
                    marker="o", capsize=2, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks([0.0625, 0.125, 0.25], ["1/16", "1/8", "1/4"])
    ax.set_xlabel("Pilot fraction")
    ax.set_ylabel("Post-update NMSE (dB)")
    if args.allow_partial:
        ax.set_title("Partial matrix: descriptive outcomes", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(destination / "nmse_vs_pilot.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved descriptive summaries to {destination}.")


if __name__ == "__main__":
    main()
