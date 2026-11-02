from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from aquapilot.config import load_config
from scripts import run_watermark_loeo as runner


def make_args(tmp_path, *extra):
    return runner.parser().parse_args([
        "--results-root", str(tmp_path / "results"),
        "--prepared-root", str(tmp_path / "prepared"), *extra,
    ])


def test_fresh_dry_run_has_36_cases_and_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    runner.main(["--dry-run", "--results-root", str(tmp_path / "results")])
    output = json.loads(capsys.readouterr().out)
    assert output["mode"] == "fresh_training" and output["initialized_from"] is None
    assert output["runs"] == len({row["output_dir"] for row in output["plan"]}) == 36
    assert all(Path(row["root"]).is_absolute() for row in output["plan"])
    assert not (tmp_path / "results").exists()


def test_fresh_runner_rejects_previous_weights_before_preflight(tmp_path, monkeypatch):
    cfg = load_config(runner.PROJECT_ROOT / "configs/watermark.yaml")
    cfg["train"]["init_checkpoint"] = "/old/checkpoint.pt"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(runner, "preflight", lambda *a: pytest.fail("Preflight ran before fresh initialization check"))
    with pytest.raises(ValueError, match="no previous model"):
        runner.main(["--config", str(path), "--results-root", str(tmp_path / "results"), "--background"])
    assert not (tmp_path / "results").exists()


def test_background_preflight_failure_does_not_launch_or_create_pid(tmp_path, monkeypatch):
    from aquapilot.data import audit
    monkeypatch.setattr(runner, "runtime_preflight", lambda cfg: {"selected_device": "cpu"})
    monkeypatch.setattr(audit, "audit_watermark", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad prepared data")))
    monkeypatch.setattr(runner, "_launch_background", lambda *a: pytest.fail("Background launched after failed audit"))
    with pytest.raises(ValueError, match="bad prepared data"):
        runner.main(["--results-root", str(tmp_path / "results"), "--prepared-root", str(tmp_path / "prepared"), "--background"])
    assert not (tmp_path / "results").exists()


def test_explicit_cuda_request_cannot_fall_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        runner.runtime_preflight({"train": {"device": "cuda", "torch_threads": 1}})


def test_background_uses_current_interpreter_and_keeps_lock(tmp_path, monkeypatch):
    args = make_args(tmp_path, "--background", "--device", "cpu")
    captured = {}

    def fake_popen(command, **kwargs):
        captured.update(command=command, **kwargs)
        assert kwargs["start_new_session"] is True
        assert os.fstat(kwargs["pass_fds"][0]).st_size == 0
        return SimpleNamespace(pid=12345)

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    argv = ["--results-root", str(tmp_path / "results"), "--background", "--device", "cpu"]
    runner._launch_background(args, argv, {"fixture": "synthetic test only"})
    command = captured["command"]
    assert command[0] == sys.executable
    assert "--background" not in command
    assert "--device" in command and "--session-id" in command and "--lock-fd" in command
    logs = tmp_path / "results/logs"
    assert next(logs.glob("*.pid")).read_text().strip() == "12345"
    assert len(list(logs.glob("*.preflight.json"))) == 1


def test_result_root_lock_prevents_concurrent_matrix(tmp_path):
    args = make_args(tmp_path)
    first = runner._lock(args)
    try:
        with pytest.raises(RuntimeError, match="Another training process"):
            runner._lock(args)
    finally:
        os.close(first)
    second = runner._lock(args)
    os.close(second)


def test_existing_incomplete_run_is_never_skipped(tmp_path):
    args = make_args(tmp_path, "--skip-existing", "--folds", "fold_test_NOF", "--pilot-ratios", "0.25", "--seeds", "17")
    base = runner.load_fresh_config(args)
    plan = runner.build_plan(args, base)
    output = Path(plan[0]["output_dir"])
    output.mkdir(parents=True)
    (output / "checkpoint.pt").write_bytes(b"interrupted synthetic fixture")
    with pytest.raises(RuntimeError, match="Interrupted or incomplete"):
        runner.validate_output_paths(args, base, plan)
    assert (output / "checkpoint.pt").read_bytes() == b"interrupted synthetic fixture"


def test_output_cannot_overlap_prepared_data(tmp_path):
    args = make_args(tmp_path)
    args.results_root = args.prepared_root
    base = runner.load_fresh_config(args)
    plan = runner.build_plan(args, base)
    with pytest.raises(ValueError, match="separate from data"):
        runner.validate_output_paths(args, base, plan)
    assert not args.prepared_root.exists()


@pytest.mark.parametrize("old_option,old_dir", [
    ("--prepared-root", "data/prepared/watermark"),
    ("--results-root", "results/watermark"),
])
def test_prepare_leaves_old_server_paths_untouched(tmp_path, monkeypatch, old_option, old_dir):
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    old = tmp_path / old_dir
    old.mkdir(parents=True)
    survivor = old / "existing"
    survivor.write_bytes(b"original")
    with pytest.raises(SystemExit) as stopped:
        runner.main(["--prepare", old_option, old_dir])
    assert stopped.value.code == 2
    assert list(old.iterdir()) == [survivor] and survivor.read_bytes() == b"original"
    assert not (tmp_path / "results/watermark_inband").exists()


def test_unknown_fold_and_incompatible_launch_modes_are_rejected(tmp_path):
    args = make_args(tmp_path, "--folds", "fold_test_UNKNOWN")
    with pytest.raises(ValueError, match="Watermark folds"):
        runner.build_plan(args, runner.load_fresh_config(args))
    with pytest.raises(SystemExit):
        runner.main(["--dry-run", "--background"])
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize(('section','key','value','error'), [
    ('eval','mask_strategies',['comb','active'],'all four'),
    ('data','horizon',2,'one-snapshot'),
    ('eval','prediction_storage','none','frame-matched'),
])
def test_matrix_validates_observation_settings_before_start(tmp_path, section,key,value,error):
    cfg=load_config(runner.PROJECT_ROOT / 'configs/watermark.yaml'); cfg[section][key]=value
    path=tmp_path/'config.yaml'; path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match=error):
        runner.main(['--config',str(path),'--results-root',str(tmp_path/'results'),'--dry-run'])
    assert not (tmp_path/'results').exists()
