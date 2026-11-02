"""Constructed fixtures exercise export guards; these are not measured results."""
from __future__ import annotations

import csv
import itertools
import json
import statistics
from pathlib import Path

import pytest

from aquapilot.data.audit import EXPECTED_COUNTS, _inventory
from aquapilot.provenance import PROTOCOL_ID, file_fingerprint, fingerprint
from aquapilot.analysis import conditional_seed_interval
from scripts.build_result_tables import ENVIRONMENTS, FRACTIONS, INPUT_FILES, POLICIES, SEEDS, build_tables


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def refresh_analysis_hash(analysis: Path, name: str) -> None:
    summary = json.loads((analysis / "analysis_summary.json").read_text())
    summary["analysis_artifacts"][name] = file_fingerprint(analysis / name)
    write_json(analysis / "analysis_summary.json", summary)


@pytest.fixture
def analysis_fixture(tmp_path):
    # Deliberately artificial numbers and metadata, confined to the test directory.
    analysis = tmp_path / "constructed_analysis_fixture"
    analysis.mkdir()
    configuration, pairs, rank_runs, source_paths, source_hashes = [], [], [], [], []
    for env_index, env in enumerate(ENVIRONMENTS):
        names = [name for name, (group, _) in _inventory(EXPECTED_COUNTS).items() if group == env]
        n = len(names)
        for fraction, seed in itertools.product(FRACTIONS, SEEDS):
            seed_index = SEEDS.index(seed)
            run_dir = tmp_path / "constructed_sources" / f"{env}_{fraction}_{seed}"
            run_dir.mkdir(parents=True)
            cfg = {"seed": seed, "data": {"kind": "prepared"}, "pilot": {"ratio": fraction}}
            identity = {"protocol_id": PROTOCOL_ID, "protocol_fingerprint": "a" * 64,
                        "run_mode": "fresh_training", "config_fingerprint": fingerprint(cfg),
                        "evaluator_source_fingerprint": "b" * 64, "runtime_fingerprint": "c" * 64,
                        "data_fingerprint": fingerprint(env), "test_sequence_fingerprint": fingerprint(env),
                        "checkpoint_fingerprint": "d" * 64}
            result = {**identity, "result_schema_version": 2, "noise_convention": "component",
                      "artifact_status": "measured-channel experiment", "closed_loop": {},
                      "fixture_notice": "ARTIFICIAL UNIT-TEST FIXTURE; NOT AN EXPERIMENT"}
            path = run_dir / "metrics.json"
            comb = env_index * 0.1 + seed_index * 0.01 + fraction
            rho = -0.2 + env_index * 0.1 + fraction + seed_index * 0.01
            values = {"comb": comb, "random": comb + 0.5,
                      "uncertainty": comb - 0.2 - seed_index * 0.01, "active": comb - 0.1}
            for policy, nmse in values.items():
                metric = {"nmse_db_mean": nmse, "channel_nrmse_percent_mean": 100 + 5 * nmse}
                result["closed_loop"][policy] = {"posterior": {**metric, "num_frames": n, "num_valid_error_frames": n}, "pilot_ratio": fraction,
                    "prior_ranking": {"mean_frame_spearman": rho, "num_valid_frames": n, "num_undefined_frames": 0, "num_frames": n},
                    "records": [{"record_id": name, "environment": env} for name in names]}
                configuration.append({"environment": env, "seed": seed, "pilot_fraction": fraction,
                    "policy": policy, "source_metrics": str(path), "run_mode": "fresh_training",
                    "num_frames": n, "num_valid_error_frames": n, **metric})
            rank_runs.append({"environment": env, "seed": seed, "pilot_fraction": fraction,
                "policy": "uncertainty", "source_metrics": str(path), "mean_frame_spearman": rho,
                "num_frames": n, "num_valid_frames": n, "num_undefined_frames": 0})
            if fraction == 0.25:
                pairs.append({"environment": env, "seed": seed, "pilot_fraction": fraction,
                    "policy": "uncertainty", "reference_policy": "comb", "source_metrics": str(path),
                    "comb_nmse_db": comb, "policy_nmse_db": values["uncertainty"],
                    "delta_db": values["uncertainty"] - comb, "num_frames": n,
                    "num_matched_valid_frames": n, "num_excluded_frames": 0})
            manifest = {**identity, "resolved_config": cfg}
            write_json(run_dir / "resolved_config.json", cfg)
            write_json(run_dir / "run_manifest.json", manifest)
            write_json(path, result)
            complete = {"identity_fingerprint": fingerprint(identity), "artifacts": {
                name: file_fingerprint(run_dir / name)
                for name in ("metrics.json", "run_manifest.json", "resolved_config.json")}}
            write_json(run_dir / "complete.json", complete)
            source_paths.append(str(path))
            source_hashes.append(file_fingerprint(path)["sha256"])
    aggregate = []
    for fraction, policy in itertools.product(FRACTIONS, POLICIES):
        group = [row for row in configuration if row["pilot_fraction"] == fraction and row["policy"] == policy]
        nmse = [row["nmse_db_mean"] for row in group]
        aggregate.append({"pilot_fraction": fraction, "policy": policy, "num_environment_seed_cells": 12,
            "equal_environment_seed_nmse_db": statistics.mean(nmse), "descriptive_sd_nmse_db": statistics.pstdev(nmse),
            "equal_environment_seed_channel_nrmse_percent": statistics.mean(row["channel_nrmse_percent_mean"] for row in group)})
    env_pairs = [{"environment": env, "pilot_fraction": 0.25, "policy": "uncertainty", "num_seeds": 3,
        "mean_delta_db": statistics.mean(row["delta_db"] for row in pairs if row["environment"] == env),
        "undefined_seed_count": 0} for env in ENVIRONMENTS]
    rank_env = [{"environment": env, "pilot_fraction": fraction, "policy": "uncertainty", "num_seeds": 3,
        "equal_seed_mean_frame_spearman": statistics.mean(row["mean_frame_spearman"] for row in rank_runs
            if row["environment"] == env and row["pilot_fraction"] == fraction), "num_undefined_frames": 0}
        for env, fraction in itertools.product(ENVIRONMENTS, FRACTIONS)]
    seed_means = {seed: statistics.mean(row["delta_db"] for row in pairs if row["seed"] == seed) for seed in SEEDS}
    totals = [{"pilot_fraction": 0.25, "policy": "uncertainty", "complete_environment_seed_grid": True,
        "num_environments": 4, "num_seeds": 3, "num_paired_outcomes": 12,
        "equal_environment_equal_seed_delta_db": statistics.mean(seed_means.values()),
        "conditional_seed_interval": conditional_seed_interval(seed_means)}]
    csv_files = {"configuration_metrics.csv": configuration, "aggregate_metrics.csv": aggregate,
        "paired_top_b_025.csv": pairs, "paired_by_environment.csv": env_pairs,
        "prior_ranking_by_environment.csv": rank_env, "prior_ranking_by_run.csv": rank_runs}
    for name, rows in csv_files.items():
        write_csv(analysis / name, rows)
    write_json(analysis / "paired_summary.json", totals)
    summary = {"analysis_schema_version": 2, "protocol_id": PROTOCOL_ID, "protocol_fingerprint": "a" * 64,
        "run_mode": "fresh_training", "noise_convention": "component", "num_runs": 36,
        "num_environment_seed_budget_cells": 36, "data_kind": "prepared", "input_data_kinds": ["prepared"],
        "artifact_status": "measured-channel experiment", "declared_source_statuses": ["measured-channel experiment"],
        "expected_watermark_matrix": True, "matrix_complete": True, "missing_cells": [],
        "source_metrics": source_paths, "source_summary_sha256": source_hashes,
        "analysis_artifacts": {name: file_fingerprint(analysis / name) for name in INPUT_FILES if name != "analysis_summary.json"},
        "fixture_notice": "ARTIFICIAL UNIT-TEST FIXTURE; NOT AN EXPERIMENT"}
    write_json(analysis / "analysis_summary.json", summary)
    return analysis


def test_full_constructed_fixture_exports_numbers_and_provenance(analysis_fixture, tmp_path):
    output = tmp_path / "fixture_tables"
    manifest = build_tables(analysis_fixture, output)
    main = read_csv(output / "main_results.csv")
    assert len(main) == 12
    comb = next(row for row in main if row["policy"] == "comb" and float(row["pilot_fraction"]) == 0.0625)
    raw = [index * 0.1 + seed_index * 0.01 + 0.0625 for index in range(4) for seed_index in range(3)]
    assert float(comb["nmse_db_mean"]) == statistics.mean(raw)
    assert float(comb["channel_nrmse_percent_sd"]) == statistics.pstdev(100 + 5 * v for v in raw)
    paired = read_csv(output / "paired_top_b_025.csv")
    nof = next(row for row in paired if row["environment"] == "NOF")
    assert [float(nof[f"seed_{seed}_delta_db"]) for seed in SEEDS] == pytest.approx([-0.20, -0.21, -0.22])
    summary, = read_csv(output / "paired_summary_025.csv")
    assert (summary["num_pairs"], summary["wins"], summary["num_seed_blocks"]) == ("12", "12", "3")
    expected = conditional_seed_interval(dict(zip(SEEDS, [-0.20, -0.21, -0.22])))
    assert float(summary["t95_lower_db"]) == pytest.approx(expected["t_lower"])
    ranks = read_csv(output / "ranking_by_environment.csv")
    assert float(ranks[0]["equal_seed_mean_frame_spearman"]) == pytest.approx(-0.1275)
    assert (ranks[0]["num_undefined_frames"], ranks[0]["num_defined_seeds"]) == ("0", "3")
    assert len(manifest["source_metrics"]) == 36
    assert manifest["main_sd_ddof"] == 0
    assert {p.name for p in output.iterdir()} == {*manifest["outputs"], "tables_manifest.json"}
    assert all(name.endswith(".csv") and file_fingerprint(output / name) == identity
               for name, identity in manifest["outputs"].items())


@pytest.mark.parametrize("change", [
    {"expected_watermark_matrix": False}, {"matrix_complete": False},
    {"num_environment_seed_budget_cells": 35},
    {"data_kind": "synthetic", "input_data_kinds": ["synthetic"]},
    {"artifact_status": "synthetic sanity check"}, {"noise_convention": "complex_rms_as_component"},
])
def test_incompatible_or_partial_inputs_are_rejected(analysis_fixture, tmp_path, change):
    path = analysis_fixture / "analysis_summary.json"
    summary = json.loads(path.read_text())
    summary.update(change)
    write_json(path, summary)
    output = tmp_path / "rejected_tables"
    with pytest.raises(ValueError):
        build_tables(analysis_fixture, output)
    assert not output.exists()


def test_changed_source_summary_is_rejected(analysis_fixture, tmp_path):
    summary = json.loads((analysis_fixture / "analysis_summary.json").read_text())
    path = Path(summary["source_metrics"][0])
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="Source metrics SHA256 mismatch"):
        build_tables(analysis_fixture, tmp_path / "rejected_tables")


def test_edited_analysis_csv_is_rejected(analysis_fixture, tmp_path):
    path = analysis_fixture / "aggregate_metrics.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="Analysis artifact hash mismatch"):
        build_tables(analysis_fixture, tmp_path / "rejected_tables")


def test_missing_paired_outcome_is_rejected_even_if_declared_complete(analysis_fixture, tmp_path):
    path = analysis_fixture / "paired_top_b_025.csv"
    write_csv(path, read_csv(path)[:-1])
    refresh_analysis_hash(analysis_fixture, path.name)
    with pytest.raises(ValueError, match="Incomplete.*paired"):
        build_tables(analysis_fixture, tmp_path / "rejected_tables")


def test_nonfinite_ranking_is_not_printed_as_blank(analysis_fixture, tmp_path):
    path = analysis_fixture / "prior_ranking_by_run.csv"
    rows = read_csv(path)
    rows[0]["mean_frame_spearman"] = "nan"
    write_csv(path, rows)
    refresh_analysis_hash(analysis_fixture, path.name)
    with pytest.raises(ValueError, match="Nonfinite"):
        build_tables(analysis_fixture, tmp_path / "rejected_tables")


def test_existing_table_directory_is_preserved(analysis_fixture, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    original = output / "main_results.csv"
    original.write_text("existing results")
    with pytest.raises(FileExistsError):
        build_tables(analysis_fixture, output)
    assert original.read_text() == "existing results"


def change_completed_result(analysis, path, update):
    result = json.loads(path.read_text())
    update(result)
    write_json(path, result)
    complete_path = path.parent/'complete.json'
    complete = json.loads(complete_path.read_text())
    complete['artifacts']['metrics.json'] = file_fingerprint(path)
    write_json(complete_path, complete)
    summary_path = analysis/'analysis_summary.json'
    summary = json.loads(summary_path.read_text())
    summary['source_summary_sha256'][summary['source_metrics'].index(str(path))] = file_fingerprint(path)['sha256']
    write_json(summary_path, summary)


def test_interval_endpoints_must_follow_from_the_three_seed_blocks(analysis_fixture, tmp_path):
    path = analysis_fixture/'paired_summary.json'
    rows = json.loads(path.read_text())
    rows[0]['conditional_seed_interval']['t_lower'] -= 1
    write_json(path, rows)
    refresh_analysis_hash(analysis_fixture, path.name)
    with pytest.raises(ValueError, match='interval lower endpoint'):
        build_tables(analysis_fixture, tmp_path/'rejected')


def test_missing_held_out_record_refuses_complete_matrix_label(analysis_fixture, tmp_path):
    path = Path(json.loads((analysis_fixture/'analysis_summary.json').read_text())['source_metrics'][0])
    change_completed_result(analysis_fixture, path, lambda d: d['closed_loop']['comb']['records'].pop())
    with pytest.raises(ValueError, match='Source record inventory'):
        build_tables(analysis_fixture, tmp_path/'rejected')


def test_undefined_rank_seed_preserves_other_tables(analysis_fixture, tmp_path):
    path = analysis_fixture/'prior_ranking_by_run.csv'
    rows = read_csv(path)
    item = rows[0]
    n = int(item['num_frames'])
    change_completed_result(analysis_fixture, Path(item['source_metrics']),
        lambda d: d['closed_loop']['uncertainty']['prior_ranking'].update(
            mean_frame_spearman=None, num_valid_frames=0, num_undefined_frames=n))
    item.update(mean_frame_spearman='',num_valid_frames=0,num_undefined_frames=n)
    write_csv(path, rows)
    refresh_analysis_hash(analysis_fixture, path.name)
    path = analysis_fixture/'prior_ranking_by_environment.csv'
    envs = read_csv(path)
    env = next(row for row in envs if row['environment'] == item['environment'] and row['pilot_fraction'] == item['pilot_fraction'])
    env.update(equal_seed_mean_frame_spearman='', num_undefined_frames=n)
    write_csv(path, envs)
    refresh_analysis_hash(analysis_fixture, path.name)
    output = tmp_path/'tables'
    build_tables(analysis_fixture, output)
    rankings = read_csv(output / "ranking_by_environment.csv")
    row = next(r for r in rankings if r["environment"] == item["environment"]
               and r["pilot_fraction"] == item["pilot_fraction"])
    assert row["equal_seed_mean_frame_spearman"] == ""
    assert (row["num_defined_seeds"], row["num_undefined_frames"]) == ("2", str(n))
    assert (output / "paired_top_b_025.csv").is_file()
