from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

from aquapilot.provenance import PROTOCOL_ID
from scripts.run_watermark_loeo import build_plan, flatten_run, ratio_tag
from scripts.reevaluate_watermark import checkpoint_for_run
from scripts.summarize_watermark import validate_matrix


def test_flatten_run_keeps_posterior_coverage() -> None:
    result = {
        "result_schema_version": 2,
        "protocol_id": PROTOCOL_ID,
        "model_kind": "aquapilot",
        "model_parameters": 123,
        "model_test": {
            "nmse_db_mean": -1.0,
            "channel_nrmse_percent_mean": 10.0,
            "latency_ms_per_frame": 0.5,
        },
        "baselines": {},
        "closed_loop": {
            "aoptimal": {
                "pilot_ratio": 0.125,
                "posterior": {
                    "nmse_db_mean": -2.0,
                    "channel_nrmse_percent_mean": 8.0,
                    "coverage_90": 0.89,
                },
                "prior_ranking": {"mean_frame_spearman": 0.35, "num_valid_frames": 4},
                "same_information_baselines": {"current_interpolation": {"nmse_db_mean": -1.5}},
            }
        },
        "classical_closed_loop": {},
    }

    row = flatten_run(result, "fold_test_KAU", 0.125, 17, Path("out"))
    assert row["closed_aoptimal_coverage90"] == 0.89
    assert row["closed_aoptimal_channel_nrmse_percent"] == 8.0
    assert row["closed_aoptimal_ranking_mean_frame_spearman"] == 0.35
    assert row["closed_aoptimal_matched_current_interpolation_nmse_db"] == -1.5
    assert not any("evm" in key or "goodput" in key for key in row)


def test_flatten_rejects_missing_protocol() -> None:
    with pytest.raises(ValueError, match="current evaluation protocol"):
        flatten_run({}, "fold_test_KAU", 0.125, 17, Path("out"))


def test_matrix_has_unique_full_paths_independent_of_cwd(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    args = Namespace(config=Path("configs/watermark.yaml"), prepared_root=Path("data/prepared/watermark"),
                     results_root=Path("results/test"),
                     folds=["fold_test_NOF", "fold_test_NCS", "fold_test_BCH", "fold_test_KAU"],
                     pilot_ratios=[0.25, 0.125, 0.0625], seeds=[17, 37, 57])
    plan = build_plan(args, {"model": {"kind": "aquapilot"}})
    assert len(plan) == len({row["output_dir"] for row in plan}) == 36
    assert all(Path(row["root"]).is_absolute() for row in plan)
    assert all(not str(row["root"]).startswith(str(tmp_path)) for row in plan)
    assert ratio_tag(0.0625) == "ratio_0.0625"
    args.seeds = [17, 17]
    with pytest.raises(ValueError, match="Duplicate seeds"):
        build_plan(args, {"model": {"kind": "aquapilot"}})


def test_reevaluation_resolves_checkpoint_without_modification(tmp_path) -> None:
    item = {"fold": "fold_test_KAU", "pilot_ratio": 0.125, "seed": 17}
    path = tmp_path / "aquapilot/fold_test_KAU/ratio_0.125/seed_17/checkpoint.pt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"existing checkpoint")
    assert checkpoint_for_run(tmp_path, "aquapilot", item) == path
    assert path.read_bytes() == b"existing checkpoint"
    second = tmp_path / "fold_test_KAU/ratio_0.125/seed_17/checkpoint.pt"
    second.parent.mkdir(parents=True)
    second.write_bytes(b"other checkpoint")
    with pytest.raises(ValueError, match="Two checkpoints"):
        checkpoint_for_run(tmp_path, "aquapilot", item)


def test_summary_rejects_duplicate_rows_and_mixed_protocols() -> None:
    row = {"fold": "fold_test_NOF", "pilot_ratio_requested": 0.25, "seed": 17,
           "protocol_id": PROTOCOL_ID, "protocol_fingerprint": "same",
           "noise_convention": "component", "run_mode": "checkpoint_reevaluation"}
    with pytest.raises(ValueError, match="duplicate"):
        validate_matrix(pd.DataFrame([row, row]), allow_partial=True)
    with pytest.raises(ValueError, match="protocol_fingerprint"):
        validate_matrix(pd.DataFrame([row, {**row, "seed": 37, "protocol_fingerprint": "different"}]), allow_partial=True)
    validate_matrix(pd.DataFrame([row]), allow_partial=True)
