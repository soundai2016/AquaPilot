#!/usr/bin/env python3
"""Report the corrected-grid, 32-of-128 Top-B versus rotating-comb comparison."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import fmean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if not __package__:
    sys.path[:0] = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)]

from aquapilot.analysis import ENVIRONMENTS, SEEDS, analyze_results, read_json
from aquapilot.data.audit import project_path
from aquapilot.export_io import _fold_inputs, _source_table
from aquapilot.provenance import file_fingerprint
from aquapilot.watermark_grid import BANKS
from scripts.run_watermark_loeo import DEFAULT_RESULTS_ROOT, QUARTER_RESULTS_ROOT
from aquapilot.verification import FRAMES, RECORDS, verify_axes


def historical_note() -> str:
    reference = PROJECT_ROOT / "results" / "historical"
    with (reference / "first_manuscript" / "watermark_table2.csv").open(newline="", encoding="utf-8") as f:
        old = {r["policy_label"]: float(r["posterior_nmse_db_mean"]) for r in csv.DictReader(f)
               if float(r["pilot_fraction"]) == .25}
    with (reference / "quarter_budget" / "paired_top_b_025.csv").open(newline="", encoding="utf-8") as f:
        earlier = [float(r["delta_db"]) for r in csv.DictReader(f)]
    top, comb = old["Uncertainty top-B"], old["Fixed comb"]
    return (f"The original submission reported {top:.2f} dB for Top-B and {comb:.2f} dB for comb "
            f"(difference of rounded means {top-comb:+.2f} dB). The archived rerun on the earlier grid "
            f"gave {fmean(earlier):+.4f} dB with {sum(d < 0 for d in earlier)} wins in {len(earlier)} pairs. "
            "The earlier grid used the wrong transform axis in NCS and included out-of-band tones in the other "
            "banks. These historical values have been excluded from the corrected estimate above. "
            "Source tables are in `results/historical/first_manuscript/` and `results/historical/quarter_budget/`.")


def write_quarter_report(model_root: Path, axes: Path, *, output_parent: Path | None = None) -> Path:
    """Check all twelve original runs against the axis receipt and publish the paired results."""
    model_root, axes = Path(model_root).resolve(), Path(axes).resolve()
    receipt = read_json(axes)
    paths = {(env, seed): model_root / f"fold_test_{env}" / "ratio_0.25" / f"seed_{seed}" / "metrics.json"
             for env in ENVIRONMENTS for seed in SEEDS}
    missing = [str(path.relative_to(model_root)) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError("Incomplete quarter-budget matrix. Missing " + ", ".join(missing))

    parent = Path(output_parent).resolve() if output_parent is not None else model_root
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".quarter-report-", dir=parent) as work:
        analysis = Path(work) / "analysis"
        summary = analyze_results(list(paths.values()), analysis)
        if (summary["run_mode"] != "fresh_training" or summary["noise_convention"] != "component"
                or summary["input_data_kinds"] != ["prepared"]
                or summary["declared_source_statuses"] != ["measured-channel experiment"]
                or summary["num_runs"] != 12 or summary["num_environment_seed_budget_cells"] != 12):
            raise ValueError("Expected 12 freshly trained, measured-channel runs with component noise.")

        pair_file = analysis / "paired_top_b_025.csv"
        with pair_file.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if len(rows) != 12 or {(r["environment"], int(r["seed"])) for r in rows} != set(paths):
            raise ValueError("Expected exactly one Top-B/comb pair for every environment and seed.")
        scored_records = defaultdict(dict)
        with (analysis / "paired_by_record.csv").open(newline="", encoding="utf-8") as f:
            for record in csv.DictReader(f):
                if record["policy"] == "uncertainty":
                    scored_records[record["source_metrics"]][record["record_id"]] = int(record["num_matched_valid_frames"])
        fitted_by_fold = {}
        for row in rows:
            env, seed = row["environment"], int(row["seed"])
            path = paths[env, seed]
            if (row["source_metrics"] != str(path) or float(row["pilot_fraction"]) != .25
                    or row["policy"] != "uncertainty" or row["reference_policy"] != "comb"):
                raise ValueError(f"{path}: stored run identity differs from its selected comparison.")
            manifest = read_json(path.parent / "run_manifest.json")
            cfg = manifest["resolved_config"]
            if (cfg["seed"] != seed or cfg["pilot"]["ratio"] != .25
                    or cfg["model"].get("kind", "aquapilot") != "aquapilot" or cfg["data"].get("history", 8) != 16
                    or cfg["data"].get("horizon", 1) != 1 or cfg["data"].get("n_subcarriers", 128) != 128):
                raise ValueError(f"{path}: unexpected model, seed, history, horizon or pilot budget.")
            fitted = _fold_inputs(manifest, env)
            if fitted_by_fold.setdefault(env, fitted) != fitted:
                raise ValueError(f"{env}: seeds used different training, validation or test inputs.")
            count, frames = RECORDS[env], FRAMES[env]
            test_files = manifest["data"]["main"]["splits"]["test"]
            record_ids = [Path(item["path"]).stem for item in test_files]
            expected_records = {rid: BANKS[rid.split("_", 1)[0]][0][0] - cfg["data"]["history"] for rid in record_ids}
            if scored_records[str(path)] != expected_records:
                raise ValueError(f"{path}: scored record IDs or matched frame counts differ from the held-out Watermark inputs.")
            if (row["support_known"] != "True" or int(row["num_records"]) != count
                    or int(row["num_frames"]) != frames or int(row["num_matched_valid_frames"]) != frames
                    or int(row["num_excluded_frames"]) != 0):
                raise ValueError(f"{path}: the comparison lacks complete matched-frame support.")
            top, comb, delta = (float(row[k]) for k in ("policy_nmse_db", "comb_nmse_db", "delta_db"))
            if not all(map(math.isfinite, (top, comb, delta))) or not math.isclose(top-comb, delta, abs_tol=1e-9):
                raise ValueError(f"{path}: undefined or inconsistent paired NMSE.")

        verify_axes({"prepared_input_manifest_by_fold": fitted_by_fold}, receipt)
        with (analysis / "paired_summary.json").open(encoding="utf-8") as f:
            total, = (r for r in json.load(f) if r["policy"] == "uncertainty" and r["pilot_fraction"] == .25)
        delta = total["equal_environment_equal_seed_delta_db"]
        ci = total["conditional_seed_interval"]
        wins, losses = total["improved_pair_count"], total["worse_pair_count"]
        top_mean = fmean(float(r["policy_nmse_db"]) for r in rows)
        comb_mean = fmean(float(r["comb_nmse_db"]) for r in rows)
        if ci["t_upper"] < 0:
            interval_note = "The entire interval favors Top-B."
        elif ci["t_lower"] > 0:
            interval_note = "The entire interval favors the comb."
        else:
            interval_note = "The repeatability interval includes zero."
        lines = [
            "# Corrected-grid quarter-budget comparison", "",
            "Watermark closed-loop replay with 32 pilots on 128 in-band tones. Each paired outcome uses the same "
            "held-out recordings, initial history, observation noise and fitted checkpoint for both policies.", "",
            f"Mean posterior channel NMSE was {top_mean:+.4f} dB for Top-B and {comb_mean:+.4f} dB for the rotating comb. "
            f"The paired difference (Top-B minus comb) was **{delta:+.4f} dB**. Negative values favor Top-B.", "",
            f"The 95% Student t interval was [{ci['t_lower']:+.4f}, {ci['t_upper']:+.4f}] dB "
            "(three seed blocks, two degrees of freedom). Each seed block gives equal weight to the four held-out environments. "
            f"{interval_note} The interval describes variation over training and replay seeds on these environments.", "",
            f"Top-B had lower NMSE in {wins} of 12 pairs, higher NMSE in {losses}, and identical NMSE in {12-wins-losses}.", "",
            "| Held-out environment | Seed 17 (dB) | Seed 37 (dB) | Seed 57 (dB) | Mean difference (dB) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for env in ENVIRONMENTS:
            values = [float(r["delta_db"]) for r in sorted(rows, key=lambda r: int(r["seed"])) if r["environment"] == env]
            lines.append("| " + env + " | " + " | ".join(f"{v:+.4f}" for v in [*values, fmean(values)]) + " |")
        lines.extend(["", "Each cell averages full-band per-sounding dB NMSE over matched frames. "
                      "The accompanying CSV lists frame counts and the SHA-256 of the source metrics for each pair.", "",
                      "## Historical comparison", "", historical_note(), "",
                      f"Verified axis receipt SHA-256: `{file_fingerprint(axes)['sha256']}`.", ""])
        staged = Path(work) / "quarter_budget"
        staged.mkdir()
        (staged / "report.md").write_text("\n".join(lines), encoding="utf-8")
        sources = dict(zip(summary["source_metrics"], summary["source_summary_sha256"], strict=True))
        (staged / pair_file.name).write_bytes(_source_table(pair_file, sources))
        destination = parent / "quarter_budget"
        if destination.exists():
            if destination.is_symlink() or not destination.is_dir() or any(
                    not (destination / f.name).is_file() or (destination / f.name).read_bytes() != f.read_bytes()
                    for f in staged.iterdir()):
                raise FileExistsError(f"Existing {destination} differs from this analysis. Preserve it before exporting.")
        else:
            staged.rename(destination)
        return destination / "report.md"


def default_quarter_receipt(model: Path, project: Path) -> Path:
    """Prefer the run's receipt. Verification still binds each training split to its digest."""
    run_root = model.resolve().parent
    candidates = [run_root / "axes.json"]
    if run_root.name == Path(QUARTER_RESULTS_ROOT).name:
        candidates.append(run_root.parent / Path(DEFAULT_RESULTS_ROOT).name / "axes.json")
    candidates.append(project / "results/axes.json")
    for path in candidates:
        if path.exists() or path.is_symlink():
            return path
    raise FileNotFoundError("No axis receipt available. Pass --axes for the prepared inputs used by this run.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(DEFAULT_RESULTS_ROOT) / "aquapilot",
                        help="Model directory containing fold_test_* runs")
    parser.add_argument("--axes", type=Path, help="Verified axis receipt; default: the run receipt, then the packaged results/axes.json")
    args = parser.parse_args()
    model = project_path(args.results)
    try:
        axes = project_path(args.axes) if args.axes is not None else default_quarter_receipt(model, PROJECT_ROOT)
        print(write_quarter_report(model, axes))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Quarter-budget comparison failed: {exc}\n")


if __name__ == "__main__":
    main()
