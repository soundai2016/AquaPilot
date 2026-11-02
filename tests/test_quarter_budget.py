"""Exercise the report with constructed summary records; no benchmark values are asserted."""
import csv
import json

import pytest

from aquapilot.analysis import ENVIRONMENTS, POLICIES, SEEDS
from aquapilot.data.audit import EXPECTED_COUNTS, _inventory, expected_split_groups
from aquapilot.provenance import PROTOCOL_ID, fingerprint, finish_run
from scripts.quarter_budget import write_quarter_report
from aquapilot.verification import FRAMES


@pytest.fixture
def matrix(tmp_path):
    model = tmp_path / "aquapilot"
    receipt = dict(status="ok", method="independent_delay_axis_fft", frequency_grid="uniform_inner_probe_band",
                   history=16, horizon=1, n_subcarriers=128, band_fraction=.8, num_unique_records=156,
                   total_prepared_files=624, environments=[], ordered_split_fingerprints={})
    inventory = _inventory(EXPECTED_COUNTS)
    for env in ENVIRONMENTS:
        splits = {s: [dict(path=f"{s}/{name}.npz", size_bytes=1, sha256=fingerprint(["test", name]))
                      for name, (group, _) in inventory.items() if group in groups]
                  for s, groups in expected_split_groups(env).items()}
        digests = {s: fingerprint(rows) for s, rows in splits.items()}
        receipt["ordered_split_fingerprints"][env] = digests
        count = EXPECTED_COUNTS[env]
        frames = FRAMES[env] // count
        receipt["environments"].append(dict(environment=env, records=count, scored=FRAMES[env],
                                            snapshots=FRAMES[env]+16*count, max_error_over_peak=0))
        for seed in SEEDS:
            cfg = dict(seed=seed, data=dict(kind="prepared", root=f"fold_test_{env}", history=16, horizon=1,
                                             n_subcarriers=128), pilot=dict(ratio=.25), model=dict(kind="aquapilot"))
            path = model / f"fold_test_{env}" / "ratio_0.25" / f"seed_{seed}"
            path.mkdir(parents=True)
            manifest = dict(resolved_config=cfg, source={"git_commit": None}, protocol_id=PROTOCOL_ID,
                            protocol_fingerprint="test-grid", run_mode="fresh_training", config_fingerprint=fingerprint(cfg),
                            evaluator_source_fingerprint="test-evaluator", runtime_fingerprint="test-runtime",
                            data_fingerprint=fingerprint(splits), test_sequence_fingerprint=digests["test"],
                            checkpoint_fingerprint="none", data={"main": dict(splits=splits, split_fingerprints=digests)})
            (path / "run_manifest.json").write_text(json.dumps(manifest))
            (path / "resolved_config.json").write_text(json.dumps(cfg))
            policies = {}
            for policy in POLICIES:
                db = (-1 if seed != 57 else 2) if policy == "uncertainty" else 0
                metric = dict(schema_version=2, num_frames=frames, num_valid_error_frames=frames,
                              nmse_db_mean=db, nmse_linear_mean=10**(db/10),
                              channel_nrmse_percent_mean=100*10**(db/20), pooled_target_energy=1,
                              pooled_squared_error_energy=10**(db/10))
                ranks = dict(num_frames=frames, num_valid_frames=frames, num_undefined_frames=0, mean_frame_spearman=0)
                baselines = dict.fromkeys(("zero", "persistence_prior", "current_interpolation", "persistence_innovation"), metric)
                records = [dict(record_id=rid, environment=env, noise_seed=seed, num_frames=frames,
                                frame_identity_sha256=fingerprint(rid), prior=metric, posterior=metric,
                                prior_ranking=ranks, same_information_baselines=baselines)
                           for rid, (group, _) in inventory.items() if group == env]
                policies[policy] = dict(records=records)
            finish_run(dict(output_dir=path, manifest=manifest), dict(artifact_status="measured-channel experiment",
                                                                     closed_loop=policies))
    axes = tmp_path / "axes.json"
    axes.write_text(json.dumps(receipt))
    return model, axes


def test_report_uses_only_verified_quarter_runs(matrix):
    model, axes = matrix
    different_budget = model / "fold_test_NCS" / "ratio_0.125" / "seed_17"
    different_budget.mkdir(parents=True)
    (different_budget / "metrics.json").write_text("unrelated budget")
    report = write_quarter_report(model, axes)
    assert report == model / "quarter_budget" / "report.md"
    assert "8 of 12 pairs" in report.read_text()
    assert "two degrees of freedom" in report.read_text()
    with (report.parent / "paired_top_b_025.csv").open() as f:
        pairs = list(csv.DictReader(f))
    assert len(pairs) == 12
    assert "source_metrics" not in pairs[0] and len(pairs[0]["metrics_sha256"]) == 64
    assert write_quarter_report(model, axes) == report


def test_changed_axis_digest_rejected(matrix):
    model, axes = matrix
    receipt = json.loads(axes.read_text())
    receipt["ordered_split_fingerprints"]["NOF"]["val"] = "a"*64
    axes.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="trained inputs differ"):
        write_quarter_report(model, axes)
    assert not (model / "quarter_budget").exists()


def test_one_missing_cell_blocks_result(matrix):
    model, axes = matrix
    (model / "fold_test_NOF" / "ratio_0.25" / "seed_17" / "metrics.json").unlink()
    with pytest.raises(ValueError, match="Incomplete quarter-budget matrix"):
        write_quarter_report(model, axes)


def test_a_run_from_another_seed_cannot_fill_a_missing_cell(matrix):
    model, axes = matrix
    ratio = model / "fold_test_BCH" / "ratio_0.25"
    (ratio / "seed_17").rename(ratio / "swap")
    (ratio / "seed_37").rename(ratio / "seed_17")
    (ratio / "swap").rename(ratio / "seed_37")
    with pytest.raises(ValueError, match="stored run identity"):
        write_quarter_report(model, axes)
    assert not (model / "quarter_budget").exists()
