import csv
import hashlib
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from aquapilot.quarter_export import PAIR_FILE, member_info, read_quarter_export
from scripts.export_quarter import export_quarter
from test_quarter_budget import matrix


@pytest.fixture
def exported(matrix, tmp_path):
    model, axes = matrix
    # A report from a previous version can already exist and must remain byte-identical.
    old = model / "quarter_budget"
    old.mkdir()
    (old / "report.md").write_text("Original completed version 1.0 report")
    output = tmp_path / "results" / "quarter.zip"
    export_quarter(model, axes, output)
    assert (old / "report.md").read_text() == "Original completed version 1.0 report"
    return model, axes, output


def test_portable_export_reopens_without_installed_packages(exported):
    model, axes, output = exported
    results = read_quarter_export(output)
    assert results["summary"]["wins"] == 8 and results["summary"]["losses"] == 4
    assert results["summary"]["delta_db"] == 0
    assert results["summary"]["ci_low_db"] < 0 < results["summary"]["ci_high_db"]
    assert results["axes_sha256"] == hashlib.sha256(axes.read_bytes()).hexdigest()
    assert len(results["rows"]) == 12
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    assert export_quarter(model, axes, output) == output
    assert hashlib.sha256(output.read_bytes()).hexdigest() == digest
    reader = Path(__file__).resolve().parents[1] / "src/aquapilot/quarter_export.py"
    subprocess.run([sys.executable, "-I", "-S", "-c",
                    "import runpy,sys; assert len(runpy.run_path(sys.argv[1])['read_quarter_export'](sys.argv[2])['rows'])==12",
                    str(reader), str(output)], check=True)


@pytest.mark.parametrize("alteration,match", [
    ("legacy", "corrected-grid"), ("dup", "duplicate members"), ("hash", "SHA-256"),
    ("delta", "signed Top-B"), ("support", "frame counts"), ("wrong_inputs", "does not match"),
    ("summary", "reproduce"), ("source", "run identities"), ("dup_cell", "duplicate environment"),
])
def test_portable_export_refuses_corruption(exported, tmp_path, alteration, match):
    _, _, output = exported
    with zipfile.ZipFile(output) as archive:
        members = {n: archive.read(n) for n in archive.namelist()}
    manifest = json.loads(members["manifest.json"])
    if alteration == "legacy":
        manifest["protocol_id"] = "aquapilot_closed_loop"
    elif alteration == "hash":
        members["report.md"] += b"changed"
    elif alteration in {"delta", "support", "dup_cell"}:
        reader = csv.DictReader(io.StringIO(members[PAIR_FILE].decode()))
        rows = list(reader)
        if alteration == "delta":
            rows[0]["delta_db"] = str(-float(rows[0]["delta_db"]))
        elif alteration == "dup_cell":
            rows[1] = rows[0]
        else:
            rows[0]["num_matched_valid_frames"] = "1"
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        members[PAIR_FILE] = stream.getvalue().encode()
        manifest["members"][PAIR_FILE] = member_info(members[PAIR_FILE])
    elif alteration == "wrong_inputs":
        manifest["trained_split_fingerprints"]["NOF"]["test"] = "a"*64
    elif alteration == "summary":
        manifest["summary"]["wins"] += 1
    elif alteration == "source":
        manifest["source_cells"][0]["metrics_sha256"] = "f"*64
    members["manifest.json"] = json.dumps(manifest).encode()
    changed = tmp_path / "changed.zip"
    with zipfile.ZipFile(changed, "w") as archive:
        for name, contents in members.items():
            archive.writestr(name, contents)
        if alteration == "dup":
            with pytest.warns(UserWarning, match="Duplicate"):
                archive.writestr(PAIR_FILE, members[PAIR_FILE])
    with pytest.raises(ValueError, match=match):
        read_quarter_export(changed)


def test_cli_uses_packaged_receipt_for_external_runs_and_plural_results(matrix, tmp_path, monkeypatch):
    from scripts import export_quarter as module

    source, receipt = matrix
    model = tmp_path / "server_copy/results/watermark_quarter/aquapilot"
    model.parent.mkdir(parents=True)
    source.rename(model)
    checkout = tmp_path / "new_checkout"
    (checkout / "results").mkdir(parents=True)
    (checkout / "results/axes.json").write_bytes(receipt.read_bytes())
    monkeypatch.setattr(module, "PROJECT", checkout)
    module.main(["--results", str(model)])
    output = checkout / "results/AquaPilot_quarter_results.zip"
    assert read_quarter_export(output)["axes_sha256"] == hashlib.sha256(receipt.read_bytes()).hexdigest()
    assert not (checkout / "result").exists()

    # A present run receipt takes precedence. A mismatch is rejected even when a packaged receipt matches.
    local_receipt = model.parent.parent / "watermark_inband/axes.json"
    local_receipt.parent.mkdir()
    changed = json.loads(receipt.read_bytes())
    changed["ordered_split_fingerprints"]["NOF"]["val"] = "0" * 64
    local_receipt.write_text(json.dumps(changed))
    with pytest.raises(SystemExit) as error:
        module.main(["--results", str(model)])
    assert error.value.code == 2
    assert read_quarter_export(output)["axes_sha256"] == hashlib.sha256(receipt.read_bytes()).hexdigest()
