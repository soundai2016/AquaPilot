from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from aquapilot.provenance import begin_run, finish_run, prepared_fingerprint, resolve_config_paths, source_snapshot


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    root = tmp_path / "project"
    source = root / "src/aquapilot/evaluator.py"
    source.parent.mkdir(parents=True)
    source.write_text("PROTOCOL = 2\n")
    data = root / "data/test/sequence.npz"
    data.parent.mkdir(parents=True)
    data.write_bytes(b"synthetic hash fixture")
    checkpoint = root / "saved/checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic checkpoint fixture")
    cfg = {"seed": 17, "output_dir": "results/evaluation", "data": {"kind": "prepared", "root": "data"},
           "model": {"kind": "aquapilot"}, "train": {}, "pilot": {"ratio": 0.25, "update_noise_convention": "component"},
           "eval": {"run_classical": False}}
    monkeypatch.setattr("aquapilot.provenance.runtime_snapshot", lambda: {"python": "test", "packages": {}})
    return root, cfg, data, checkpoint, source


def run_fixture(experiment):
    root, cfg, _, checkpoint, _ = experiment
    context = begin_run(cfg, mode="checkpoint_reevaluation", checkpoint=checkpoint, project_root=root)
    (context["output_dir"] / "prediction.npz").write_bytes(b"synthetic output fixture")
    result = finish_run(context, {"closed_loop": {}, "value": 1.0})
    return context, result


def test_complete_run_reused_only_with_matching_inputs(experiment):
    root, cfg, _, checkpoint, _ = experiment
    context, result = run_fixture(experiment)
    reused = begin_run(cfg, mode="checkpoint_reevaluation", checkpoint=checkpoint,
                       project_root=root, skip_existing=True)
    assert reused["cached_result"] == result
    assert result["noise_convention"] == "component"
    marker = json.loads((context["output_dir"] / "complete.json").read_text())
    assert {"prediction.npz", "metrics.json", "resolved_config.json", "run_manifest.json"} == set(marker["artifacts"])
    with pytest.raises(FileExistsError):
        begin_run(cfg, mode="checkpoint_reevaluation", checkpoint=checkpoint, project_root=root)


@pytest.mark.parametrize("change", ["data", "checkpoint", "source", "config", "artifact"])
def test_skip_rejects_changed_inputs_or_outputs(experiment, change):
    root, cfg, data, checkpoint, source = experiment
    context, _ = run_fixture(experiment)
    if change == "data":
        data.write_bytes(b"changed data")
    elif change == "checkpoint":
        checkpoint.write_bytes(b"changed checkpoint")
    elif change == "source":
        source.write_text("PROTOCOL = 3\n")
    elif change == "config":
        cfg = copy.deepcopy(cfg)
        cfg["pilot"]["update_noise_convention"] = "complex_rms_as_component"
    else:
        (context["output_dir"] / "prediction.npz").write_bytes(b"corrupted result")
    with pytest.raises(RuntimeError, match="changed|missing"):
        begin_run(cfg, mode="checkpoint_reevaluation", checkpoint=checkpoint,
                  project_root=root, skip_existing=True)


def test_interrupted_run_has_no_completion_and_cannot_be_silently_skipped(experiment):
    root, cfg, _, checkpoint, _ = experiment
    context = begin_run(cfg, mode="checkpoint_reevaluation", checkpoint=checkpoint, project_root=root)
    assert not (context["output_dir"] / "complete.json").exists()
    with pytest.raises(RuntimeError, match="incomplete"):
        begin_run(cfg, mode="checkpoint_reevaluation", checkpoint=checkpoint,
                  project_root=root, skip_existing=True)


def test_failed_json_serialization_does_not_mark_complete(experiment):
    root, cfg, _, checkpoint, _ = experiment
    context = begin_run(cfg, mode="checkpoint_reevaluation", checkpoint=checkpoint, project_root=root)
    with pytest.raises(ValueError):
        finish_run(context, {"invalid": float("nan")})
    assert not (context["output_dir"] / "complete.json").exists()


def test_data_fingerprint_tracks_sequence_order(tmp_path):
    for name in ("a.npz", "b.npz"):
        (tmp_path / name).write_bytes(name.encode())
    manifest = tmp_path / "splits.json"
    manifest.write_text(json.dumps({"test": ["a.npz", "b.npz"]}))
    first = prepared_fingerprint(tmp_path, ["test"])
    manifest.write_text(json.dumps({"test": ["b.npz", "a.npz"]}))
    second = prepared_fingerprint(tmp_path, ["test"])
    assert first["fingerprint"] != second["fingerprint"]


def test_data_and_output_trees_must_be_separate(experiment):
    root, cfg, _, checkpoint, _ = experiment
    cfg["output_dir"] = "data/output"
    with pytest.raises(ValueError, match="separate"):
        begin_run(cfg, mode="checkpoint_reevaluation", checkpoint=checkpoint, project_root=root)


def test_paths_and_source_fingerprint(experiment):
    root, cfg, _, _, source = experiment
    resolved = resolve_config_paths(cfg, root)
    assert resolved["output_dir"] == str(root / "results/evaluation")
    assert cfg["output_dir"] == "results/evaluation"
    before = source_snapshot(root)["evaluator_source_fingerprint"]
    source.write_text("PROTOCOL = 4\n")
    assert source_snapshot(root)["evaluator_source_fingerprint"] != before
