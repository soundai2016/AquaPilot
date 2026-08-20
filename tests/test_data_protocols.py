from pathlib import Path

import numpy as np
from scipy.io import savemat

from scripts.prepare_acommset import chronological_segments, resample_frequency
from scripts.prepare_watermark import environment_group
from scripts.prepare_watermark_loeo import GROUPS


def test_watermark_kau_recordings_are_one_environment():
    assert environment_group("KAU1") == "KAU"
    assert environment_group("KAU2") == "KAU"
    assert len(GROUPS) == 4


def test_acommset_continuous_split_has_guard_gaps():
    segments = chronological_segments(
        n_frames=400,
        train_ratio=0.10,
        val_ratio=0.10,
        guard_frames=16,
        min_frames=17,
    )
    train, val, test = segments
    assert train.split == "train" and val.split == "val" and test.split == "test"
    assert val.start - train.stop == 16
    assert test.start - val.stop == 16
    assert all(segment.stop - segment.start >= 17 for segment in segments)


def test_complex_frequency_resampling():
    time = np.arange(10)[:, None]
    frequency = np.linspace(0.0, 1.0, 20)[None, :]
    csi = np.exp(1j * (0.1 * time + 2.0 * frequency)).astype(np.complex64)
    resized = resample_frequency(csi, 32)
    assert resized.shape == (10, 32)
    assert np.isfinite(resized.real).all()
    assert np.isfinite(resized.imag).all()


def test_fake_watermark_source_names_can_be_materialized(tmp_path: Path):
    # A minimal fixture for converter-facing assumptions: complex h and canonical codes.
    for code in ("NOF1", "NCS1", "BCH1", "KAU1", "KAU2"):
        h = np.ones((20, 32), dtype=np.complex64) * (1 + 0.1j)
        savemat(tmp_path / f"{code}_fixture.mat", {"h": h, "fc": 12_000.0})
    assert {environment_group(path.stem[:4]) for path in tmp_path.glob("*.mat")} == {
        "NOF",
        "NCS",
        "BCH",
        "KAU",
    }


def test_watermark_loeo_converter_keeps_kau_together(tmp_path: Path):
    import json
    import subprocess
    import sys

    raw = tmp_path / "raw"
    prepared = tmp_path / "prepared"
    raw.mkdir()
    for index, code in enumerate(("NOF1", "NCS1", "BCH1", "KAU1", "KAU2")):
        rng = np.random.default_rng(index)
        h = (
            rng.standard_normal((20, 32)) + 1j * rng.standard_normal((20, 32))
        ).astype(np.complex64)
        savemat(raw / f"{code}_fixture.mat", {"h": h, "fc": 12_000.0})
    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_watermark_loeo.py",
            "--input",
            str(raw),
            "--output-root",
            str(prepared),
            "--n-subcarriers",
            "16",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((prepared / "folds.json").read_text(encoding="utf-8"))
    assert set(manifest["groups"]) == {"NOF", "NCS", "BCH", "KAU"}
    kau_fold = prepared / "fold_test_KAU"
    test_codes = set()
    for path in (kau_fold / "test").glob("*.npz"):
        with np.load(path, allow_pickle=False) as bundle:
            test_codes.add(str(bundle["channel_code"].item()))
    assert test_codes == {"KAU1", "KAU2"}
    for split in ("train", "val"):
        for path in (kau_fold / split).glob("*.npz"):
            with np.load(path, allow_pickle=False) as bundle:
                assert str(bundle["environment"].item()) != "KAU"
