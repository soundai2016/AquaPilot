#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

POLICIES = {
    "comb": "Fixed comb",
    "random": "Random",
    "uncertainty": "Uncertainty top-B",
    "active": "A-optimal",
}


def mean_std(series: pd.Series) -> tuple[float, float]:
    values = pd.to_numeric(series, errors="raise")
    return float(values.mean()), float(values.std(ddof=0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/watermark/aquapilot/aggregate_runs.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/watermark/aquapilot/paper_summary"),
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    required = {"fold", "pilot_ratio_requested", "seed"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")

    combinations = frame[["fold", "pilot_ratio_requested", "seed"]].drop_duplicates()
    if len(combinations) != 36:
        raise ValueError(f"Expected 36 fold-budget-seed configurations, found {len(combinations)}")

    rows: list[dict[str, float | str]] = []
    for ratio in sorted(frame["pilot_ratio_requested"].unique()):
        group = frame[frame["pilot_ratio_requested"] == ratio]
        for key, label in POLICIES.items():
            nmse_mean, nmse_std = mean_std(group[f"closed_{key}_nmse_db"])
            evm_mean, evm_std = mean_std(group[f"closed_{key}_evm_percent"])
            cov_mean, cov_std = mean_std(group[f"closed_{key}_coverage90"])
            rows.append(
                {
                    "pilot_ratio": ratio,
                    "policy": label,
                    "nmse_mean_db": nmse_mean,
                    "nmse_std_db": nmse_std,
                    "evm_mean_percent": evm_mean,
                    "evm_std_percent": evm_std,
                    "post_update_coverage90_mean": cov_mean,
                    "post_update_coverage90_std": cov_std,
                }
            )

    args.output.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output / "aggregate.csv", index=False)

    quarter = frame[frame["pilot_ratio_requested"].round(8) == 0.25]
    env_rows = []
    for fold, group in quarter.groupby("fold", sort=True):
        comb_mean, comb_std = mean_std(group["closed_comb_nmse_db"])
        active_mean, active_std = mean_std(group["closed_active_nmse_db"])
        env_rows.append(
            {
                "fold": fold,
                "comb_mean_db": comb_mean,
                "comb_std_db": comb_std,
                "a_optimal_mean_db": active_mean,
                "a_optimal_std_db": active_std,
                "delta_db": active_mean - comb_mean,
            }
        )
    pd.DataFrame(env_rows).to_csv(args.output / "environment_025.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    for label in POLICIES.values():
        subset = summary[summary["policy"] == label].sort_values("pilot_ratio")
        ax.errorbar(
            subset["pilot_ratio"],
            subset["nmse_mean_db"],
            yerr=subset["nmse_std_db"],
            marker="o",
            capsize=2,
            label=label,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks([0.0625, 0.125, 0.25], ["1/16", "1/8", "1/4"])
    ax.set_xlabel("Pilot fraction")
    ax.set_ylabel("Posterior NMSE (dB)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(args.output / "nmse_vs_pilot.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
