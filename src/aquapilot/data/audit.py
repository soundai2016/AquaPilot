"""Read-only validation of the prepared four-fold Watermark dataset."""
from __future__ import annotations

import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np

from aquapilot.provenance import PROJECT_ROOT, file_fingerprint, fingerprint

from aquapilot.watermark_grid import BANKS, GRID_NAME, subcarrier_grid

GROUPS = ("NOF", "NCS", "BCH", "KAU")
SPLITS = ("train", "val", "test")
EXPECTED_COUNTS = {"NOF": 60, "NCS": 60, "BCH": 4, "KAU": 32}
CHANNEL_GROUPS = {"NOF1": "NOF", "NCS1": "NCS", "BCH1": "BCH", "KAU1": "KAU", "KAU2": "KAU"}


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing prepared-data manifest: {path}")
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read prepared-data manifest {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"{path}: expected a JSON object.")
    return result


def _group_set(value: Any, context: str) -> set[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{context}: expected a list of environment groups.")
    groups = {CHANNEL_GROUPS.get(item.upper(), item.upper()) for item in value}
    if not groups.issubset(GROUPS):
        raise ValueError(f"{context}: unknown environment group.")
    return groups


def expected_split_groups(test_group: str) -> dict[str, set[str]]:
    index = GROUPS.index(test_group)
    validation = GROUPS[(index + 1) % len(GROUPS)]
    return {"train": set(GROUPS) - {test_group, validation}, "val": {validation}, "test": {test_group}}


def _inventory(counts: Mapping[str, int]) -> dict[str, tuple[str, str]]:
    if set(counts) != set(GROUPS) or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in counts.values()):
        raise ValueError("expected_counts must give positive integer counts for NOF, NCS, BCH and KAU.")
    if counts["KAU"] % 2:
        raise ValueError("KAU counts must include equal numbers of KAU1 and KAU2 records.")
    channel_counts = {"NOF1": counts["NOF"], "NCS1": counts["NCS"], "BCH1": counts["BCH"],
                      "KAU1": counts["KAU"] // 2, "KAU2": counts["KAU"] // 2}
    return {f"{code}_{code}_{index:03d}": (CHANNEL_GROUPS[code], code)
            for code, count in channel_counts.items() for index in range(1, count + 1)}


def _scalar(bundle: Any, name: str, path: Path, *, required: bool = True) -> str | None:
    if name not in bundle:
        if required:
            raise ValueError(f"{path}: missing {name} metadata; inspect the preparation output.")
        return None
    array = np.asarray(bundle[name])
    if array.size != 1 or array.dtype.kind not in "US":
        raise ValueError(f"{path}: {name} must contain one text value.")
    value = array.reshape(-1)[0]
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _read_record(path: Path, *, history: int, horizon: int, n_subcarriers: int) -> dict[str, Any]:
    try:
        with np.load(path, allow_pickle=False) as bundle:
            if "csi_real" in bundle or "csi_imag" in bundle:
                if not {"csi_real", "csi_imag"}.issubset(bundle.files):
                    raise ValueError("both csi_real and csi_imag are required")
                real, imaginary = np.asarray(bundle["csi_real"]), np.asarray(bundle["csi_imag"])
                if real.shape != imaginary.shape or real.dtype.kind not in "fiu" or imaginary.dtype.kind not in "fiu":
                    raise ValueError("csi_real/csi_imag must be real numeric arrays with the same shape")
                with np.errstate(over="ignore", invalid="ignore"):
                    csi = real.astype(np.float32) + 1j * imaginary.astype(np.float32)
            elif "csi" in bundle:
                csi = np.asarray(bundle["csi"])
                if csi.dtype.kind not in "fciu":
                    raise ValueError("csi must be a numeric array")
                with np.errstate(over="ignore", invalid="ignore"):
                    csi = csi.astype(np.complex64)
            else:
                raise ValueError("missing csi or csi_real/csi_imag arrays")
            environment = _scalar(bundle, "environment", path)
            channel_code = _scalar(bundle, "channel_code", path, required=False)
            source = _scalar(bundle, "source", path, required=False)
            axes = {}
            for key, (kind, shape) in {
                'matlab_h_shape':('iu',(2,)), 'slow_time_axis':('iu',()), 'fft_delay_axis':('iu',()),
                'shifted_dft_bin_indices':('iu',(n_subcarriers,)), 'subcarrier_hz':('fiu',(n_subcarriers,)),
                'probe_band_hz':('fiu',(2,)), 'nominal_probe_bandwidth_hz':('fiu',()), 'padded_dft_length':('iu',()),
                'fc':('fiu',()), 'fs_t':('fiu',()), 'fs_tau':('fiu',())}.items():
                if key in bundle:
                    field = np.asarray(bundle[key]);
                    if field.dtype.kind not in kind or field.shape != shape:
                        raise ValueError(f'invalid {key} metadata')
                    axes[key] = field.tolist()
            if 'frequency_grid' in bundle:
                axes['frequency_grid'] = _scalar(bundle,'frequency_grid',path)
    except (OSError, KeyError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Cannot validate {path}: {exc}") from exc
    if csi.ndim != 2 or csi.shape[1] != n_subcarriers:
        raise ValueError(f"{path}: expected CSI [time, {n_subcarriers}], found {csi.shape}.")
    if csi.shape[0] < history + horizon:
        raise ValueError(f"{path}: {csi.shape[0]} frames are insufficient for history={history}, horizon={horizon}.")
    if not np.isfinite(csi).all():
        raise ValueError(f"{path}: CSI contains nonfinite values after the loader's complex64 conversion.")
    if not np.any(csi):
        raise ValueError(f"{path}: the entire reference CSI record has zero energy.")
    if environment not in GROUPS:
        raise ValueError(f"{path}: environment must be one of {GROUPS}, found {environment!r}.")
    return {**axes, "record_id": path.stem, "environment": environment, "channel_code": channel_code,
            "source": source, "shape": list(csi.shape), "frames": int(csi.shape[0]),
            "windows": int(csi.shape[0] - history - horizon + 1),
            "zero_reference_frames": int(np.count_nonzero(~np.any(csi != 0, axis=1))),
            "csi_sha256": hashlib.sha256(np.ascontiguousarray(csi).view(np.uint8)).hexdigest(),
            "file": file_fingerprint(path)}


def validate_standard_axes(record, index_row, channel_code, path, fraction=.8):
    """Reject the wrong transform orientation or an out-of-probe-band frequency grid."""
    shape, band, fs_t, fs_tau = BANKS[channel_code]
    keys = ('matlab_h_shape','slow_time_axis','fft_delay_axis','frequency_grid','nominal_probe_bandwidth_hz','padded_dft_length',
            'probe_band_hz','shifted_dft_bin_indices','subcarrier_hz','fc','fs_tau','fs_t')
    for key in keys:
        if key not in record or index_row.get(key) != record[key]:
            raise ValueError(f'{path}: missing or inconsistent {key}. Rebuild from the original MAT files in a new directory with --time-axis 0.')
    if (record['slow_time_axis'] != 0 or record['fft_delay_axis'] != 1
            or tuple(record['matlab_h_shape']) != shape or record['frames'] != shape[0]):
        raise ValueError(f'{path}: {channel_code} expects MATLAB h{shape}: rows are snapshots; FFT must use the delay columns.')
    if not np.isclose(record['fs_t'],fs_t,rtol=1e-6):
        raise ValueError(f'{path}: unexpected slow-time sampling frequency for {channel_code}.')
    padded,bins,hz = subcarrier_grid(channel_code,shape[1],record['shape'][1],fraction,record['fc'],record['fs_tau'])
    if (record['frequency_grid'] != GRID_NAME or record['padded_dft_length'] != padded or tuple(record['probe_band_hz']) != band
            or record['nominal_probe_bandwidth_hz'] != band[1]-band[0]
            or not np.array_equal(record['shifted_dft_bin_indices'],bins)
            or not np.allclose(record['subcarrier_hz'],hz,rtol=0,atol=1e-6)):
        raise ValueError(f'{path}: frequency samples do not match the specified fraction of the Watermark probe band.')


def _index_rows(path: Path, expected_groups: dict[str, set[str]], n_subcarriers: int) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = _json(path)
    if manifest.get("n_subcarriers") != n_subcarriers:
        raise ValueError(f"{path}: n_subcarriers must be {n_subcarriers}.")
    fraction = manifest.get("band_fraction")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction <= 1:
        raise ValueError(f"{path}: band_fraction must be a finite number in (0, 1].")
    split_groups = manifest.get("split_groups")
    if not isinstance(split_groups, dict) or any(
            _group_set(split_groups.get(split), f"{path}:{split}") != expected_groups[split] for split in SPLITS):
        raise ValueError(f"{path}: split_groups does not match the four-fold Watermark protocol.")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path}: records must be a nonempty list.")
    indexed: dict[str, dict[str, Any]] = {}
    for row in records:
        if not isinstance(row, dict) or not isinstance(row.get("prepared"), str):
            raise ValueError(f"{path}: each record needs a relative prepared path.")
        relative = PurePosixPath(row["prepared"])
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 2 or relative.parts[0] not in SPLITS:
            raise ValueError(f"{path}: invalid prepared record path {row['prepared']!r}.")
        name = relative.as_posix()
        if name in indexed:
            raise ValueError(f"{path}: duplicate prepared record {name}.")
        if row.get("split") != relative.parts[0]:
            raise ValueError(f"{path}: split field disagrees with prepared path {name}.")
        indexed[name] = row
    return manifest, indexed


def audit_watermark(prepared_root: str | Path, *, history: int = 16, horizon: int = 1,
                    n_subcarriers: int = 128, expected_counts: Mapping[str, int] | None = None) -> dict[str, Any]:
    """Check prepared files without rewriting data or relying on old raw paths.

    The production inventory is the supplied WatermarkV1 layout. The optional
    count mapping exists for small constructed software-test fixtures.
    Each recording must occur once per fold, in its prescribed environment
    split, and contain identical CSI across the four copies.
    """
    for value, name, minimum in ((history, "history", 2), (horizon, "horizon", 1), (n_subcarriers, "n_subcarriers", 2)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    root = project_path(prepared_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Prepared Watermark directory does not exist: {root}")
    counts = dict(EXPECTED_COUNTS if expected_counts is None else expected_counts)
    inventory = _inventory(counts)
    fold_names = {f"fold_test_{group}" for group in GROUPS}
    manifest = _json(root / "folds.json")
    if _group_set(manifest.get("groups"), str(root / "folds.json")) != set(GROUPS):
        raise ValueError(f"{root / 'folds.json'}: expected four Watermark groups.")
    fold_map = manifest.get("folds")
    if not isinstance(fold_map, dict) or set(fold_map) != fold_names:
        raise ValueError(f"{root / 'folds.json'}: expected exactly {sorted(fold_names)}.")
    actual_folds = {path.name for path in root.glob("fold_test_*") if path.is_dir()}
    if actual_folds != fold_names:
        raise ValueError(f"{root}: fold directories differ from folds.json; found {sorted(actual_folds)}.")

    reference: dict[str, dict[str, Any]] = {}
    report_folds: dict[str, Any] = {}
    manifest_hashes = {"folds.json": file_fingerprint(root / "folds.json")}
    data_files = []
    band_fraction: Any = None
    for group in GROUPS:
        fold_name = f"fold_test_{group}"
        fold = root / fold_name
        expected_groups = expected_split_groups(group)
        declared = fold_map[fold_name]
        if not isinstance(declared, dict) or any(
                _group_set(declared.get(split), f"folds.json:{fold_name}:{split}") != expected_groups[split]
                for split in SPLITS):
            raise ValueError(f"folds.json: {fold_name} contains incorrect train/val/test groups.")
        index_path = fold / "index.json"
        index, rows = _index_rows(index_path, expected_groups, n_subcarriers)
        if band_fraction is None:
            band_fraction = index.get("band_fraction")
        elif index.get("band_fraction") != band_fraction:
            raise ValueError(f"{index_path}: band_fraction differs between folds.")
        manifest_hashes[f"{fold_name}/index.json"] = file_fingerprint(index_path)
        paths = sorted(fold.rglob("*.npz"))
        actual = {path.relative_to(fold).as_posix() for path in paths}
        if actual != set(rows):
            missing, extra = sorted(set(rows) - actual), sorted(actual - set(rows))
            raise ValueError(f"{index_path}: inventory differs from files; missing={missing[:5]}, unindexed={extra[:5]}.")
        split_path = fold / "splits.json"
        if split_path.exists():
            split_manifest = _json(split_path)
            for split in SPLITS:
                declared_paths = split_manifest.get(split)
                expected_paths = {name for name in actual if name.startswith(split + "/")}
                if (not isinstance(declared_paths, list) or any(not isinstance(name, str) for name in declared_paths)
                        or len(declared_paths) != len(set(declared_paths)) or set(declared_paths) != expected_paths):
                    raise ValueError(f"{split_path}: {split} paths disagree with the validated directory/index inventory.")
            manifest_hashes[f"{fold_name}/splits.json"] = file_fingerprint(split_path)
        if len(paths) != len(inventory):
            raise ValueError(f"{fold}: expected {len(inventory)} records, found {len(paths)}.")
        seen: set[str] = set()
        split_records: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
        for path in paths:
            if not path.resolve().is_relative_to(root):
                raise ValueError(f"{path}: prepared record resolves outside the prepared Watermark root.")
            name = path.relative_to(fold).as_posix()
            split = path.relative_to(fold).parts[0]
            row = rows[name]
            record = _read_record(path, history=history, horizon=horizon, n_subcarriers=n_subcarriers)
            record_id = record["record_id"]
            if record_id not in inventory or record_id in seen:
                raise ValueError(f"{path}: unexpected or duplicated original record ID {record_id!r}.")
            seen.add(record_id)
            environment, channel_code = inventory[record_id]
            if record["environment"] != environment or environment not in expected_groups[split]:
                raise ValueError(f"{path}: environment {record['environment']} is inconsistent with the record or {split} split.")
            if record["channel_code"] is not None and record["channel_code"] != channel_code:
                raise ValueError(f"{path}: channel_code does not match its original record ID.")
            if row.get("environment") != environment or row.get("channel_code") != channel_code or row.get("shape") != record["shape"]:
                raise ValueError(f"{index_path}: environment/channel_code/shape disagrees with {name}.")
            if expected_counts is None and not str(record.get('source') or '').startswith('synthetic_'):
                if index.get('frequency_grid') != GRID_NAME or band_fraction != 0.8 or n_subcarriers != 128:
                    raise ValueError(f'{index_path}: standard measured training requires 128 uniformly spaced tones in the central 80% of the probe band. Rebuild from the original MAT files in a new directory.')
                validate_standard_axes(record, row, channel_code, path, band_fraction)
            for source in (record["source"], row.get("source")):
                # Raw roots may refer to another machine. Their existence is
                # irrelevant; the original measurement filename is retained.
                if source and not str(source).startswith("synthetic_"):
                    original = PurePosixPath(str(source).replace("\\", "/")).name
                    if original != f"{record_id.removeprefix(channel_code + '_')}.mat":
                        raise ValueError(f"{path}: source filename {original!r} disagrees with its original record ID.")
            identity = {key: record[key] for key in ("environment", "shape", "csi_sha256")}
            if record_id in reference and identity != reference[record_id]:
                raise ValueError(f"{path}: CSI or metadata differs from another fold's copy of {record_id}.")
            reference.setdefault(record_id, identity)
            record.update(path=f"{fold_name}/{name}", channel_code=channel_code)
            split_records[split].append(record)
            data_files.append({"path": record["path"], **record["file"]})
        if seen != set(inventory):
            raise ValueError(f"{fold}: missing original record IDs {sorted(set(inventory) - seen)[:5]}.")
        split_summary = {}
        for split, records in split_records.items():
            expected = sum(counts[environment] for environment in expected_groups[split])
            if len(records) != expected:
                raise ValueError(f"{fold / split}: expected {expected} records, found {len(records)}.")
            frames = [record["frames"] for record in records]
            split_summary[split] = {"sequences": len(records), "frames": sum(frames),
                "windows": sum(record["windows"] for record in records),
                "environments": sorted(expected_groups[split]), "min_frames": min(frames), "max_frames": max(frames),
                "zero_reference_frames": sum(record["zero_reference_frames"] for record in records),
                "counts_by_environment": dict(sorted(Counter(record["environment"] for record in records).items()))}
        report_folds[fold_name] = {"splits": split_summary, "records": [record for split in SPLITS for record in split_records[split]]}
    # The legitimate four copies carry the same original ID. A different ID
    # in another held-out group must not repeat the very same complex response.
    content_groups = {}
    for record_id, item in reference.items():
        key = (tuple(item['shape']), item['csi_sha256'])
        owner = content_groups.setdefault(key, (record_id, item['environment']))
        if owner[1] != item['environment']:
            raise ValueError(f'Identical CSI under different environment labels: {owner[0]} and {record_id}.')
    return {"audit_schema_version": 2, "status": "ok", "root": str(root),
        "n_subcarriers": n_subcarriers, "history": history, "horizon": horizon,
        "expected_counts": counts, "num_unique_records": len(reference), "total_prepared_files": len(data_files),
        "band_fraction": band_fraction, "folds": report_folds, "manifest_fingerprints": manifest_hashes,
        "data_fingerprint": fingerprint({"manifests": manifest_hashes, "files": data_files}),
        "source_path_policy": "Raw source paths and fold root metadata are not dereferenced; prepared files are read in place.",
        "write_policy": "No data files are created, moved or modified by this audit."}
