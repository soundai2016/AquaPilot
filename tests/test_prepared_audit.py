"""Constructed channel arrays check data validation; no measured results."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aquapilot.data.audit import EXPECTED_COUNTS, GROUPS, audit_watermark, expected_split_groups
from aquapilot.provenance import file_fingerprint


SMALL_COUNTS = {"NOF": 2, "NCS": 2, "BCH": 1, "KAU": 2}


def write_fixture(root: Path, *, counts=None, n_subcarriers=8, frames=6) -> Path:
    counts = SMALL_COUNTS if counts is None else counts
    channels = {"NOF1": counts["NOF"], "NCS1": counts["NCS"], "BCH1": counts["BCH"],
                "KAU1": counts["KAU"] // 2, "KAU2": counts["KAU"] // 2}
    source = {}
    for serial, (code, count) in enumerate(channels.items()):
        for number in range(1, count + 1):
            rng = np.random.default_rng(100 * serial + number)
            csi = (rng.standard_normal((frames, n_subcarriers)) + 1j * rng.standard_normal((frames, n_subcarriers))).astype(np.complex64)
            source[f"{code}_{code}_{number:03d}"] = (code[:3], code, csi)
    fold_map = {}
    for environment in GROUPS:
        name = f"fold_test_{environment}"
        fold = root / name
        groups = expected_split_groups(environment)
        rows = []
        for identity, (group, code, csi) in source.items():
            split = next(split for split, members in groups.items() if group in members)
            destination = fold / split / f"{identity}.npz"
            destination.parent.mkdir(parents=True, exist_ok=True)
            original = identity.removeprefix(code + "_") + ".mat"
            old_source = f"/unavailable/old_host/{name}/{code}/mat/{original}"
            np.savez_compressed(destination, csi_real=csi.real, csi_imag=csi.imag,
                environment=np.asarray(group), channel_code=np.asarray(code), source=np.asarray("synthetic_fixture"),
                auxiliary_metadata=np.asarray(name))
            rows.append({"prepared": f"{split}/{identity}.npz", "source": old_source,
                         "environment": group, "channel_code": code, "split": split, "shape": list(csi.shape)})
        index = {"format": "AquaPilot prepared CSI v1", "n_subcarriers": n_subcarriers,
                 "band_fraction": 0.8, "split_groups": {split: sorted(members) for split, members in groups.items()},
                 "records": rows, "skipped": []}
        (fold / "index.json").write_text(json.dumps(index))
        fold_map[name] = {**{split: sorted(members) for split, members in groups.items()}, "root": f"/old_mount/{name}"}
    (root / "folds.json").write_text(json.dumps({"groups": list(GROUPS), "folds": fold_map}))
    return root


@pytest.fixture
def dataset(tmp_path):
    return write_fixture(tmp_path / "prepared/watermark")


def audit_small(root):
    return audit_watermark(root, history=3, horizon=1, n_subcarriers=8, expected_counts=SMALL_COUNTS)


def rewrite_npz(path, **changes):
    with np.load(path, allow_pickle=False) as data:
        values = {key: data[key] for key in data.files}
    values.update(changes)
    np.savez_compressed(path, **values)


def state(root):
    return {str(path.relative_to(root)): (file_fingerprint(path), path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file()}


def test_audit_is_read_only_and_compares_logical_csi_not_zip_bytes(dataset):
    before = state(dataset)
    report = audit_small(dataset)
    assert state(dataset) == before
    assert report["status"] == "ok"
    assert report["num_unique_records"] == 7
    assert report["total_prepared_files"] == 28
    assert len(report["data_fingerprint"]) == 64
    assert report["folds"]["fold_test_NOF"]["splits"]["train"]["sequences"] == 3
    copies = [next(row for row in fold["records"] if row["record_id"] == "NOF1_NOF1_001") for fold in report["folds"].values()]
    assert len({row["csi_sha256"] for row in copies}) == 1
    assert len({row["file"]["sha256"] for row in copies}) == 4


def test_default_inventory_matches_the_existing_server_layout(tmp_path):
    dataset = write_fixture(tmp_path / "watermark", counts=EXPECTED_COUNTS, n_subcarriers=128, frames=17)
    report = audit_watermark(dataset)
    assert report["total_prepared_files"] == 624 and report["num_unique_records"] == 156
    expected = {"NOF": (36, 60, 60), "NCS": (92, 4, 60), "BCH": (120, 32, 4), "KAU": (64, 60, 32)}
    for group, numbers in expected.items():
        splits = report["folds"][f"fold_test_{group}"]["splits"]
        assert tuple(splits[split]["sequences"] for split in ("train", "val", "test")) == numbers
        assert all(row["sequences"] == row["windows"] for row in splits.values())


@pytest.mark.parametrize("change,pattern", [
    ("nonfinite", "nonfinite"), ("short", "insufficient"), ("frequency", "expected CSI"),
    ("environment", "environment"), ("shape", "same shape"), ("zero", "zero energy"),
])
def test_bad_record_is_rejected(dataset, change, pattern):
    path = dataset / "fold_test_NOF/test/NOF1_NOF1_001.npz"
    if change == "nonfinite":
        values = np.ones((6, 8), dtype=np.float32)
        values[0, 0] = np.nan
        rewrite_npz(path, csi_real=values)
    elif change == "short":
        rewrite_npz(path, csi_real=np.ones((3, 8)), csi_imag=np.ones((3, 8)))
    elif change == "frequency":
        rewrite_npz(path, csi_real=np.ones((6, 7)), csi_imag=np.ones((6, 7)))
    elif change == "environment":
        rewrite_npz(path, environment=np.asarray("KAU"))
    elif change == "shape":
        rewrite_npz(path, csi_imag=np.ones((6, 7)))
    else:
        rewrite_npz(path, csi_real=np.zeros((6, 8)), csi_imag=np.zeros((6, 8)))
    with pytest.raises(ValueError, match=pattern):
        audit_small(dataset)


def test_missing_or_unindexed_records_fail(dataset):
    (dataset / "fold_test_NOF/test/NOF1_NOF1_001.npz").unlink()
    with pytest.raises(ValueError, match="inventory differs"):
        audit_small(dataset)


def test_duplicate_index_record_fails(dataset):
    path = dataset / "fold_test_NOF/index.json"
    index = json.loads(path.read_text())
    index["records"].append(index["records"][0])
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="duplicate prepared record"):
        audit_small(dataset)


def test_cross_fold_csi_mismatch_fails(dataset):
    path = dataset / "fold_test_KAU/val/NOF1_NOF1_001.npz"
    rewrite_npz(path, csi_real=np.full((6, 8), 2.0))
    with pytest.raises(ValueError, match="differs from another fold"):
        audit_small(dataset)


def test_split_manifest_cannot_override_the_validated_directory(dataset):
    fold = dataset / "fold_test_NOF"
    (fold / "splits.json").write_text(json.dumps({"train": ["test/NOF1_NOF1_001.npz"], "val": [], "test": []}))
    with pytest.raises(ValueError, match="paths disagree"):
        audit_small(dataset)


def test_changed_group_manifest_fails(dataset):
    path = dataset / "folds.json"
    manifest = json.loads(path.read_text())
    manifest["folds"]["fold_test_NOF"]["train"] = ["NOF", "BCH"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="incorrect train/val/test"):
        audit_small(dataset)


def test_cli_refuses_reports_anywhere_inside_the_data_tree(tmp_path, monkeypatch):
    from scripts.check_data import main
    dataset = tmp_path / "data/prepared/watermark"
    destination = tmp_path / "data/raw/report.json"
    monkeypatch.setattr("sys.argv", ["check_data.py", "--prepared-root", str(dataset), "--output", str(destination)])
    monkeypatch.setattr("scripts.check_data.audit_watermark", lambda *args, **kwargs: pytest.fail("Output protection must run before the data audit"))
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert not destination.exists()


def test_held_out_csi_cannot_reappear_under_another_environment_id(dataset):
    first = next(dataset.rglob('NOF1_NOF1_001.npz'))
    with np.load(first) as data:
        values = {key:data[key] for key in ('csi_real','csi_imag')}
    for other in dataset.rglob('NCS1_NCS1_001.npz'):
        rewrite_npz(other, **values)
    with pytest.raises(ValueError, match='Identical CSI under different environment labels'):
        audit_small(dataset)
