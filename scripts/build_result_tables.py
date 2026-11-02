#!/usr/bin/env python3
"""Export numerical CSV summaries from the complete measured Watermark experiment."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
import tempfile
from pathlib import Path
from typing import Any

from aquapilot.provenance import PROTOCOL_ID, file_fingerprint
from aquapilot.analysis import load_run, conditional_seed_interval, paired_record, pool_pairs
from aquapilot.data.audit import _inventory, EXPECTED_COUNTS
from aquapilot.utils import save_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENTS = ("NOF", "NCS", "BCH", "KAU")
SEEDS = (17, 37, 57)
FRACTIONS = (0.0625, 0.125, 0.25)
POLICIES = ("comb", "random", "uncertainty", "active")
MEASURED_STATUS = {"fresh_training": "measured-channel experiment",
                   "checkpoint_reevaluation": "fixed-checkpoint measured-channel reevaluation"}
INPUT_FILES = ("analysis_summary.json", "configuration_metrics.csv", "aggregate_metrics.csv",
               "paired_top_b_025.csv", "paired_by_environment.csv", "paired_summary.json",
               "prior_ranking_by_environment.csv", "prior_ranking_by_run.csv")


def read_json(path: Path) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"{path}: nonstandard JSON number {value}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"{path}: empty or malformed CSV")
    return rows


def number(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None or isinstance(value, bool) or value == "":
        raise ValueError(f"Missing numeric {key} in {row}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric {key}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite {key}: {value!r}; resolve the analysis before exporting tables")
    return result


def integer(row: dict[str, Any], key: str, minimum: int = 0) -> int:
    result = number(row, key)
    if not result.is_integer() or result < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}")
    return int(result)


def close(actual: float, expected: float, description: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-9):
        raise ValueError(f"Inconsistent {description}: {actual} versus {expected}. Regenerate the analysis.")


def indexed(rows: list[dict[str, Any]], keys: tuple[str, ...], expected: set[tuple], name: str) -> dict[tuple, dict]:
    result = {}
    for row in rows:
        key = tuple(integer(row, field) if field == "seed" else number(row, field)
                    if field == "pilot_fraction" else row.get(field) for field in keys)
        if key in result:
            raise ValueError(f"Duplicate {name} row: {key}")
        result[key] = row
    if set(result) != expected:
        raise ValueError(f"Incomplete or unexpected {name} rows: expected {len(expected)}, found {len(result)}")
    return result


def validate_sources(analysis: Path, summary: dict[str, Any]) -> dict[str, Any]:
    if summary.get("analysis_schema_version") != 2 or summary.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Expected a summary from the current analysis protocol.")
    if summary.get("expected_watermark_matrix") is not True or summary.get("matrix_complete") is not True:
        raise ValueError("Select the complete Watermark matrix to build the result tables.")
    if summary.get("num_environment_seed_budget_cells") != 36 or summary.get("num_runs") != 36 or summary.get("missing_cells") != []:
        raise ValueError("Watermark tables require all 36 Watermark environment-seed-budget cells and runs.")
    mode = summary.get("run_mode")
    status = MEASURED_STATUS.get(mode)
    if summary.get("data_kind") != "prepared" or summary.get("input_data_kinds") != ["prepared"]:
        raise ValueError("Watermark tables require prepared measured-channel data, not synthetic checks.")
    if status is None or summary.get("artifact_status") != status or summary.get("declared_source_statuses") != [status]:
        raise ValueError("The analysis lacks a uniform measured-channel experiment marker.")
    if summary.get("noise_convention") != "component":
        raise ValueError("Watermark tables require the component noise variance used in the Watermark experiment.")
    artifacts = summary.get("analysis_artifacts", {})
    for name in INPUT_FILES:
        if name != "analysis_summary.json" and (name not in artifacts or file_fingerprint(analysis / name) != artifacts[name]):
            raise ValueError(f"Analysis artifact hash mismatch: {name}; regenerate the analysis before exporting tables.")
    paths, hashes = summary.get("source_metrics"), summary.get("source_summary_sha256")
    if not isinstance(paths, list) or not isinstance(hashes, list) or len(paths) != 36 or len(hashes) != 36:
        raise ValueError("Expected 36 source metric paths and 36 corresponding SHA256 values.")
    sources = {}
    for value, digest in zip(paths, hashes):
        path = Path(value)
        if not path.is_absolute():
            raise ValueError("Source metric paths must be the absolute paths recorded by the analyzer.")
        path = path.resolve()
        if str(path) in sources:
            raise ValueError(f"Duplicate source metric path: {path}")
        if file_fingerprint(path)["sha256"] != digest:
            raise ValueError(f"Source metrics SHA256 mismatch: {path}; regenerate the analysis.")
        run = load_run(path)
        if run.manifest["resolved_config"]["data"]["kind"] != "prepared" or run.result.get("artifact_status") != status:
            raise ValueError(f"{path}: source is not the declared measured-channel experiment.")
        for key in ("protocol_fingerprint", "run_mode", "noise_convention"):
            if run.result.get(key) != summary.get(key):
                raise ValueError(f"{path}: analysis/source {key} mismatch")
        sources[str(path)] = run
    if len({run.manifest["evaluator_source_fingerprint"] for run in sources.values()}) != 1:
        raise ValueError("Cannot combine Watermark tables from different evaluator sources.")
    return sources


def build_tables(analysis: str | Path, output: str | Path) -> dict[str, Any]:
    analysis, output = Path(analysis).resolve(), Path(output).resolve()
    if output == analysis or analysis.is_relative_to(output):
        raise ValueError("Use a separate output directory for the generated tables.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} is not empty; choose a new output directory.")
    summary = read_json(analysis / "analysis_summary.json")
    sources = validate_sources(analysis, summary)
    if any(output.is_relative_to(run.path.parent) for run in sources.values()):
        raise ValueError("Table export must not write inside a completed source run.")
    cells = set(itertools.product(ENVIRONMENTS, SEEDS, FRACTIONS))
    configuration = indexed(read_csv(analysis / "configuration_metrics.csv"),
        ("environment", "seed", "pilot_fraction", "policy"), {(*cell, policy) for cell in cells for policy in POLICIES}, "configuration")
    used_sources: dict[str, set[tuple]] = {path: set() for path in sources}
    inventory = _inventory(EXPECTED_COUNTS)
    excluded_errors = 0
    for (env, seed, fraction, policy), row in configuration.items():
        source = str(Path(row.get("source_metrics", "")).resolve())
        if source not in sources:
            raise ValueError(f"Configuration refers to an unverified source: {source}")
        run = sources[source]
        if run.seed != seed or run.fraction != fraction or row.get("run_mode") != summary["run_mode"]:
            raise ValueError("Configuration identity differs from its source run.")
        used_sources[source].add((env, seed, fraction))
        count = integer(row, "num_frames", 1)
        valid = integer(row, "num_valid_error_frames", 1)
        source_result = run.result["closed_loop"][policy]
        expected_ids = {record_id for record_id, (group, _) in inventory.items() if group == env}
        evaluated = [(r["record_id"], r["environment"]) for r in source_result["records"]]
        if len(evaluated) != len(expected_ids) or set(evaluated) != {(name, env) for name in expected_ids}:
            raise ValueError(f"Source record inventory disagrees with the held-out {env} fold for {policy}.")
        close(number(source_result, "pilot_ratio"), fraction, "realized pilot fraction")
        if (count, valid) != tuple(integer(source_result["posterior"], k) for k in ("num_frames", "num_valid_error_frames")):
            raise ValueError("Configuration frame counts differ from the source receiver.")
        if valid > count:
            raise ValueError("Valid error-frame count exceeds total frames.")
        excluded_errors += count - valid
        for key in ("nmse_db_mean", "channel_nrmse_percent_mean"):
            close(number(row, key), number(run.result["closed_loop"][policy]["posterior"], key), f"source {policy} {key}")
        if number(row, "channel_nrmse_percent_mean") < 0:
            raise ValueError("Channel NRMSE cannot be negative.")
    if any(len(value) != 1 for value in used_sources.values()):
        raise ValueError("Expected exactly one environment-seed-budget cell per Watermark source run.")

    aggregates = indexed(read_csv(analysis / "aggregate_metrics.csv"), ("pilot_fraction", "policy"),
                         set(itertools.product(FRACTIONS, POLICIES)), "aggregate")
    main_rows = []
    for fraction in FRACTIONS:
        for policy in POLICIES:
            row = aggregates[(fraction, policy)]
            if integer(row, "num_environment_seed_cells") != 12:
                raise ValueError("Each aggregate must contain all 12 environment-seed cells.")
            group = [configuration[(env, seed, fraction, policy)] for env in ENVIRONMENTS for seed in SEEDS]
            nmse = [number(item, "nmse_db_mean") for item in group]
            nrmse = [number(item, "channel_nrmse_percent_mean") for item in group]
            mean, sd = statistics.mean(nmse), statistics.pstdev(nmse)
            close(number(row, "equal_environment_seed_nmse_db"), mean, "aggregate NMSE mean")
            close(number(row, "descriptive_sd_nmse_db"), sd, "aggregate NMSE SD")
            close(number(row, "equal_environment_seed_channel_nrmse_percent"), statistics.mean(nrmse), "aggregate NRMSE")
            main_rows.append({
                "pilot_fraction": fraction, "policy": policy, "num_environment_seed_cells": len(group),
                "nmse_db_mean": mean, "nmse_db_sd": sd,
                "channel_nrmse_percent_mean": statistics.mean(nrmse),
                "channel_nrmse_percent_sd": statistics.pstdev(nrmse),
            })

    pairs = indexed(read_csv(analysis / "paired_top_b_025.csv"), ("environment", "seed"),
                    set(itertools.product(ENVIRONMENTS, SEEDS)), "top-B paired")
    environment_rows = [row for row in read_csv(analysis / "paired_by_environment.csv")
                        if row.get("policy") == "uncertainty" and number(row, "pilot_fraction") == 0.25]
    environment_pairs = indexed(environment_rows, ("environment",), {(env,) for env in ENVIRONMENTS}, "paired environment")
    pair_rows, deltas, excluded_pairs = [], {}, 0
    for env in ENVIRONMENTS:
        values = []
        for seed in SEEDS:
            row = pairs[(env, seed)]
            if row.get("policy") != "uncertainty" or row.get("reference_policy") != "comb" or number(row, "pilot_fraction") != 0.25:
                raise ValueError("The paired 1/4 table contains the wrong policy, reference or pilot fraction.")
            source = str(Path(row.get("source_metrics", "")).resolve())
            if source != str(Path(configuration[(env, seed, 0.25, "comb")]["source_metrics"]).resolve()):
                raise ValueError("Paired row refers to a different source run.")
            value = number(row, "delta_db")
            close(value, number(row, "policy_nmse_db") - number(row, "comb_nmse_db"), "paired difference")
            count = integer(row, "num_frames", 1)
            valid = integer(row, "num_matched_valid_frames", 1)
            excluded = integer(row, "num_excluded_frames")
            if valid + excluded != count:
                raise ValueError("Paired frame counts do not sum to the total.")
            excluded_pairs += excluded
            cr = configuration[(env, seed, 0.25, "comb")]
            tr = configuration[(env, seed, 0.25, "uncertainty")]
            if any(count != integer(source_row, "num_frames", 1) or valid > integer(source_row, "num_valid_error_frames")
                   for source_row in (cr, tr)):
                raise ValueError("Paired support counts disagree with the source receivers.")
            if excluded:
                run = sources[source]
                source_records = {p: {r["record_id"]: r for r in run.result["closed_loop"][p]["records"]}
                                  for p in ("comb", "uncertainty")}
                paired = pool_pairs([paired_record(run, record, source_records["uncertainty"][k])
                                     for k, record in source_records["comb"].items()])
                if not paired["support_known"] or paired["num_matched_valid_frames"] != valid:
                    raise ValueError("Cannot verify partial paired support from the source records.")
                close(number(row, "comb_nmse_db"), number(paired, "reference_nmse_db"), "matched-frame comb mean")
                close(number(row, "policy_nmse_db"), number(paired, "candidate_nmse_db"), "matched-frame top-B mean")
            else:
                close(number(row, "comb_nmse_db"), number(configuration[(env, seed, 0.25, "comb")], "nmse_db_mean"), "paired comb mean")
                close(number(row, "policy_nmse_db"), number(configuration[(env, seed, 0.25, "uncertainty")], "nmse_db_mean"), "paired top-B mean")
            values.append(value)
            deltas[(env, seed)] = value
        env_row = environment_pairs[(env,)]
        if integer(env_row, "num_seeds") != 3 or integer(env_row, "undefined_seed_count") != 0:
            raise ValueError("Each environment must have three defined paired seed outcomes.")
        close(number(env_row, "mean_delta_db"), statistics.mean(values), "paired environment mean")
        pair_rows.append({"environment": env,
                          **{f"seed_{seed}_delta_db": value for seed, value in zip(SEEDS, values)},
                          "mean_delta_db": statistics.mean(values)})
    totals = read_json(analysis / "paired_summary.json")
    chosen = [row for row in totals if row.get("policy") == "uncertainty" and number(row, "pilot_fraction") == 0.25]
    if len(chosen) != 1:
        raise ValueError("Expected one top-B 1/4 paired summary.")
    total = chosen[0]
    if total.get("complete_environment_seed_grid") is not True or tuple(integer(total, key) for key in ("num_environments", "num_seeds", "num_paired_outcomes")) != (4, 3, 12):
        raise ValueError("Incomplete paired interval grid.")
    interval = total.get("conditional_seed_interval", {})
    if integer(interval, "num_seed_blocks") != 3 or integer(interval, "degrees_of_freedom") != 2:
        raise ValueError("Expected the conditional interval over three seed blocks.")
    for seed in SEEDS:
        close(number(interval.get("seed_block_means_db", {}), str(seed)),
              statistics.mean(deltas[(env, seed)] for env in ENVIRONMENTS), "seed-block mean")
    overall = statistics.mean(deltas.values())
    close(number(total, "equal_environment_equal_seed_delta_db"), overall, "overall paired difference")
    low, high = number(interval, "t_lower"), number(interval, "t_upper")
    substantiated = conditional_seed_interval({s: statistics.mean(deltas[(e, s)] for e in ENVIRONMENTS) for s in SEEDS})
    close(low, substantiated["t_lower"], "conditional interval lower endpoint")
    close(high, substantiated["t_upper"], "conditional interval upper endpoint")

    ranking_rows = [row for row in read_csv(analysis / "prior_ranking_by_environment.csv") if row.get("policy") == "uncertainty"]
    rankings = indexed(ranking_rows, ("environment", "pilot_fraction"), set(itertools.product(ENVIRONMENTS, FRACTIONS)), "ranking environment")
    run_rank_rows = [row for row in read_csv(analysis / "prior_ranking_by_run.csv") if row.get("policy") == "uncertainty"]
    run_rankings = indexed(run_rank_rows, ("environment", "seed", "pilot_fraction"), cells, "ranking run")
    ranking_rows_out, undefined_ranks = [], 0
    for env in ENVIRONMENTS:
        for fraction in FRACTIONS:
            row = rankings[(env, fraction)]
            if integer(row, "num_seeds") != 3:
                raise ValueError("Each ranking entry must contain all three seeds.")
            group = [run_rankings[(env, seed, fraction)] for seed in SEEDS]
            values = []
            for seed, item in zip(SEEDS, group):
                if str(Path(item.get("source_metrics", "")).resolve()) != str(Path(configuration[(env, seed, fraction, "uncertainty")]["source_metrics"]).resolve()):
                    raise ValueError("Ranking row refers to a different source run.")
                n, valid, undefined = (integer(item, k) for k in ("num_frames", "num_valid_frames", "num_undefined_frames"))
                if not n or valid + undefined != n:
                    raise ValueError("Ranking frame counts do not sum to the total.")
                run = sources[str(Path(item["source_metrics"]).resolve())]
                source_rank = run.result["closed_loop"]["uncertainty"]["prior_ranking"]
                if (n, valid, undefined) != tuple(integer(source_rank, k) for k in ("num_frames", "num_valid_frames", "num_undefined_frames")):
                    raise ValueError("Ranking frame counts differ from the source run.")
                if valid:
                    value = number(item, "mean_frame_spearman")
                    if not -1 <= value <= 1:
                        raise ValueError("Spearman correlation lies outside [-1, 1].")
                    close(value, number(source_rank, "mean_frame_spearman"), "source prior ranking")
                    values.append(value)
                elif item.get("mean_frame_spearman") not in ("", None) or source_rank.get("mean_frame_spearman") is not None:
                    raise ValueError("A run with no defined frame correlations must have a null mean.")
            if len(values) == len(SEEDS):
                rho = number(row, "equal_seed_mean_frame_spearman")
                close(rho, statistics.mean(values), "environment ranking mean")
            elif row.get("equal_seed_mean_frame_spearman") not in ("", None):
                raise ValueError("A three-seed ranking cannot omit a seed with no defined frame correlations.")
            else:
                rho = None
            undefined = integer(row, "num_undefined_frames")
            if undefined != sum(integer(item, "num_undefined_frames") for item in group):
                raise ValueError("Ranking undefined-frame counts differ across summaries.")
            undefined_ranks += undefined
            ranking_rows_out.append({
                "environment": env, "pilot_fraction": fraction,
                "equal_seed_mean_frame_spearman": rho, "num_defined_seeds": len(values),
                "num_undefined_frames": undefined,
            })

    paired_summary = {
        "pilot_fraction": 0.25, "policy": "uncertainty", "reference_policy": "comb",
        "mean_delta_db": overall, "t95_lower_db": low, "t95_upper_db": high,
        "num_seed_blocks": len(SEEDS), "degrees_of_freedom": len(SEEDS) - 1,
        "num_pairs": len(deltas), "wins": sum(value < 0 for value in deltas.values()),
        "ties": sum(value == 0 for value in deltas.values()),
        "losses": sum(value > 0 for value in deltas.values()), "excluded_frames": excluded_pairs,
        **{f"seed_{seed}_mean_delta_db": substantiated["seed_block_means_db"][seed] for seed in SEEDS},
    }
    tables = {"main_results.csv": main_rows, "paired_top_b_025.csv": pair_rows,
              "paired_summary_025.csv": [paired_summary], "ranking_by_environment.csv": ranking_rows_out}
    manifest = {"table_schema_version": 2, "protocol_id": PROTOCOL_ID, "protocol_fingerprint": summary["protocol_fingerprint"],
                "run_mode": summary["run_mode"], "data_kind": "prepared", "artifact_status": summary["artifact_status"],
                "num_cells": 36, "analysis_directory": str(analysis),
                "analysis_files": {name: file_fingerprint(analysis / name) for name in INPUT_FILES},
                "source_metrics": [{"path": path, "sha256": digest} for path, digest in zip(summary["source_metrics"], summary["source_summary_sha256"])],
                "builder": file_fingerprint(Path(__file__)), "main_sd_ddof": 0,
                "undefined_error_frames": excluded_errors, "undefined_paired_frames": excluded_pairs,
                "undefined_rank_frames": undefined_ranks}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".watermark-tables-", dir=output.parent) as directory:
        stage = Path(directory) / "tables"
        stage.mkdir()
        for name, rows in tables.items():
            with (stage / name).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
        manifest["outputs"] = {name: file_fingerprint(stage / name) for name in tables}
        save_json(stage / "tables_manifest.json", manifest)
        stage.replace(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New or empty directory; existing tables are never overwritten")
    args = parser.parse_args()
    analysis = args.analysis if args.analysis.is_absolute() else PROJECT_ROOT / args.analysis
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    manifest = build_tables(analysis, output)
    print(json.dumps({"output": str(output.resolve()), "tables": list(manifest["outputs"]), "num_cells": 36}, indent=2))


if __name__ == "__main__":
    main()
