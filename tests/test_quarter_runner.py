import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_watermark_loeo as runner


def test_quarter_dry_run_has_twelve_cells_in_a_separate_root(capsys):
    runner.main(["--quarter-budget", "--dry-run"])
    plan = json.loads(capsys.readouterr().out)["plan"]
    assert len(plan) == 12
    assert {(r["fold"], r["pilot_ratio"], r["seed"]) for r in plan} == {
        (fold, .25, seed) for fold in runner.DEFAULT_FOLDS for seed in runner.DEFAULT_SEEDS
    }
    assert all(Path(r["output_dir"]).is_relative_to(runner.project_path(runner.QUARTER_RESULTS_ROOT)) for r in plan)


@pytest.mark.parametrize("extra", [
    ["--seeds", "17"], ["--folds", "fold_test_NCS"], ["--pilot-ratios", "0.125"],
    ["--export-tables"], ["--results-root", runner.DEFAULT_RESULTS_ROOT],
    ["--results-root", runner.DEFAULT_RESULTS_ROOT + "/subset"],
])
def test_conflicting_quarter_requests_fail_before_output(extra):
    with pytest.raises(SystemExit) as stopped:
        runner.main(["--quarter-budget", "--dry-run", *extra])
    assert stopped.value.code == 2


def test_quarter_prepare_and_background_use_full_data_receipt(tmp_path, monkeypatch):
    receipt = runner.project_path(runner.DEFAULT_RESULTS_ROOT) / "axes.json"
    calls = []
    monkeypatch.setattr(runner, "prepare_inputs", lambda args: calls.append(("prepare", runner.axes_path(args))))
    monkeypatch.setattr(runner, "preflight", lambda *args: {})
    def launch(args, argv, report):
        assert "--quarter-budget" in argv
        calls.append(("launch", runner.axes_path(args)))
    monkeypatch.setattr(runner, "_launch_background", launch)
    for mode in ("--prepare", "--background"):
        runner.main(["--quarter-budget", mode, "--results-root", str(tmp_path / "results")])
    assert calls == [("prepare", receipt), ("launch", receipt)]


def test_quarter_execution_reports_only_after_all_runs(tmp_path, monkeypatch):
    config_path = runner.PROJECT_ROOT / "configs/watermark.yaml"
    root, receipt = tmp_path / "quarter", tmp_path / "custom-axes.json"
    calls = []
    def fit(config, **kwargs):
        calls.append(("fit", config["pilot"]["ratio"], config["seed"]))
        return {"result_schema_version": 2, "protocol_id": runner.PROTOCOL_ID}
    def report(model_root, axes, output):
        calls.append(("report", model_root, axes, output))
        return output
    monkeypatch.setitem(sys.modules, "aquapilot.engine", SimpleNamespace(run_experiment=fit))
    monkeypatch.setitem(sys.modules, "scripts.export_quarter", SimpleNamespace(export_quarter=report))
    monkeypatch.setattr(runner, "preflight", lambda *args: {})
    monkeypatch.setattr(runner, "postprocess", lambda *args: pytest.fail("Called 36-cell postprocessor"))
    runner.main(["--quarter-budget", "--config", str(config_path), "--results-root", str(root), "--axes", str(receipt)])
    assert calls[:-1] == [("fit", .25, seed) for _ in runner.DEFAULT_FOLDS for seed in runner.DEFAULT_SEEDS]
    assert calls[-1] == ("report", root / "aquapilot", receipt, runner.PROJECT_ROOT / "results/AquaPilot_quarter_results.zip")
    assert json.loads(next((root / "logs").glob("*.status.json")).read_text())["state"] == "completed"


def test_unidentified_custom_matrix_is_preserved(tmp_path, monkeypatch):
    dest = tmp_path / "other/aquapilot"
    dest.mkdir(parents=True)
    aggregate = dest / "aggregate_runs.csv"
    aggregate.write_text("full matrix")
    monkeypatch.setattr(runner, "runtime_preflight", lambda _: pytest.fail("Overlapping result root reached runtime"))
    with pytest.raises(ValueError, match="without a matching run plan"):
        runner.main(["--quarter-budget", "--results-root", str(dest.parent)])
    assert aggregate.read_text() == "full matrix"
