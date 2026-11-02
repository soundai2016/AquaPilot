#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.io import loadmat


if not __package__:  # Standalone invocation from a checkout.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from aquapilot.watermark_grid import BANKS, GRID_NAME, subcarrier_grid


def load_mat(path: Path) -> dict[str, Any]:
    if not h5py.is_hdf5(path):
        return loadmat(path, simplify_cells=True)
    result: dict[str, Any] = {}
    with h5py.File(path, 'r') as handle:
        for key in ('h', 'V0', 'fc', 'fs_t', 'fs_tau'):
            if key not in handle:
                continue
            value = handle[key]
            if not isinstance(value, h5py.Dataset) or 'MATLAB_class' not in value.attrs:
                raise ValueError(f'{path}: {key} is not a numeric MATLAB v7.3 dataset; its axis convention is unknown.')
            array = np.asarray(value)
            if array.dtype.fields and {'real','imag'}.issubset(array.dtype.fields):
                array = array['real'] + 1j * array['imag']
            # MATLAB v7.3 stores reversed dataset dimensions in HDF5. Return the
            # same MATLAB subscript order as scipy.io.loadmat for Level-5 files.
            result[key] = array.T
    return result


def scalar(data: dict[str, Any], key: str) -> float | None:
    if key not in data:
        return None
    value = np.asarray(data[key]).squeeze()
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


WATERMARK_CHANNEL_CODES = ("NOF1", "NCS1", "BCH1", "KAU1", "KAU2")
WATERMARK_GROUPS = ("NOF", "NCS", "BCH", "KAU")


def infer_channel_code(path: Path) -> str:
    codes = {m.upper() for m in re.findall(r"NOF1|NCS1|BCH1|KAU1|KAU2", path.as_posix(), re.IGNORECASE)}
    if len(codes) > 1:
        raise ValueError(f"Conflicting Watermark channel codes in {path}: {sorted(codes)}")
    return next(iter(codes)) if codes else path.parent.name.upper()


def environment_group(channel_code: str) -> str:
    """Keep both Kauai receiver records in the same split."""
    code = channel_code.upper()
    if code.startswith("NOF"):
        return "NOF"
    if code.startswith("NCS"):
        return "NCS"
    if code.startswith("BCH"):
        return "BCH"
    if code.startswith("KAU"):
        return "KAU"
    return code


def convert_h_to_csi(h: np.ndarray, n_subcarriers: int, band_fraction: float, time_axis: int | None, *,
                     channel_code: str | None = None, fc: float | None = None, fs_tau: float | None = None) -> np.ndarray:
    h = np.asarray(h).squeeze()
    if h.ndim != 2:
        raise ValueError(f"Watermark h must be 2-D after squeeze; got {h.shape}")
    if time_axis is None:
        # In WatermarkV1 the first MATLAB subscript is slow time, even for T > L.
        time_axis = 0
    if time_axis not in (0, 1):
        raise ValueError("time_axis must be 0 or 1.")
    if time_axis == 1:
        h = h.T
    n_delay = h.shape[1]
    if not 2 <= n_subcarriers <= n_delay:
        raise ValueError(f"Need 2 to {n_delay} distinct frequency bins on this FFT grid; requested {n_subcarriers}.")
    if h.dtype.kind not in 'fciu' or not np.isfinite(h).all():
        raise ValueError('Expected a finite numeric channel impulse response.')
    if channel_code not in BANKS:
        raise ValueError('A Watermark channel bank is required to identify the probe band.')
    padded_length, selected, _ = subcarrier_grid(channel_code, n_delay, n_subcarriers, band_fraction, fc, fs_tau)
    response = np.fft.fftshift(np.fft.fft(np.asarray(h, dtype=np.complex128), n=padded_length, axis=1), axes=1)
    with np.errstate(over='ignore', invalid='ignore'):
        csi = response[:, selected].astype(np.complex64)
    if not np.isfinite(csi).all() or not np.any(csi):
        raise ValueError('Extracted channel response is nonfinite or has zero energy throughout.')
    return csi


def prepare_watermark(input_root: Path, output: Path, layouts: dict[str, dict[str, list[str]]], *,
                      n_subcarriers: int, band_fraction: float, time_axis: int | None) -> None:
    """Convert each selected recording once and publish the requested folds together."""
    input_root = input_root.resolve()
    if output.is_symlink():
        raise ValueError(f"Output must not be a symbolic link: {output}")
    output = output.resolve()
    if input_root.is_relative_to(output) or output.is_relative_to(input_root):
        raise ValueError("Raw input and prepared output must be separate directories.")
    if not input_root.is_dir():
        raise FileNotFoundError(f"Missing Watermark input directory: {input_root}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Use a new or empty output directory: {output}")
    split_maps = {}
    for name, splits in layouts.items():
        assignment = {}
        for split, names in splits.items():
            groups = {environment_group(group) for group in names}
            if not groups or groups - set(WATERMARK_GROUPS) or groups.intersection(assignment):
                raise ValueError(f"Unknown, empty or overlapping {split} environment group: {names}")
            assignment.update({group: split for group in groups})
        split_maps[name] = assignment
    files = sorted(input_root.rglob('*.mat'))
    selected = []
    skipped = []
    identities = {}
    for path in files:
        relative = path.relative_to(input_root)
        code = infer_channel_code(relative)
        env = environment_group(code)
        if code not in WATERMARK_CHANNEL_CODES or not any(env in m for m in split_maps.values()):
            skipped.append({'file': str(path), 'reason': 'outside selected Watermark channels'})
            continue
        if not path.resolve().is_relative_to(input_root):
            raise ValueError(f"Selected MAT file resolves outside the input directory: {relative}")
        stem = re.sub(r'[^A-Za-z0-9_.-]+', '_', f'{code}_{path.stem}')
        previous = identities.setdefault(stem.casefold(), relative)
        if previous != relative:
            raise ValueError(f"Both {previous} and {relative} map to the same prepared record ID: {stem}")
        selected.append((path, code, env, stem))
    expected_codes = {code for code in WATERMARK_CHANNEL_CODES
                      if any(environment_group(code) in m for m in split_maps.values())}
    missing = expected_codes - {code for _,code,_,_ in selected}
    if missing:
        raise ValueError(f"Missing selected Watermark channels: {', '.join(sorted(missing))}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.watermark-', dir=output.parent) as work:
        stage = Path(work) / 'prepared'
        stage.mkdir()
        indices: dict[str, list[dict[str, Any]]] = {name: [] for name in layouts}
        for path, code, env, stem in selected:
            try:
                data = load_mat(path)
                if 'h' not in data:
                    raise ValueError("missing channel impulse response variable 'h'")
                h = np.asarray(data['h']).squeeze()
                t_axis = 0 if time_axis is None else int(time_axis)
                metadata = {key: scalar(data,key) for key in ('V0','fc','fs_t','fs_tau')}
                if h.shape == BANKS[code][0] and (t_axis != 0 or metadata['fs_t'] is None
                        or not np.isclose(metadata['fs_t'],BANKS[code][2],rtol=1e-6)):
                    raise ValueError(f'{code}: original MATLAB rows must index slow time at {BANKS[code][2]:g} Hz.')
                csi = convert_h_to_csi(h, n_subcarriers, band_fraction, t_axis, channel_code=code, fc=metadata['fc'], fs_tau=metadata['fs_tau'])
                padded_length, selected_bins, hz = subcarrier_grid(code, h.shape[1-t_axis], n_subcarriers, band_fraction, metadata['fc'], metadata['fs_tau'])
                band = BANKS[code][1]
                axes = {'matlab_h_shape': list(h.shape), 'slow_time_axis': t_axis, 'fft_delay_axis': 1-t_axis,
                        'frequency_grid': GRID_NAME, 'padded_dft_length':padded_length, 'probe_band_hz': list(band), 'nominal_probe_bandwidth_hz': band[1]-band[0],
                        'shifted_dft_bin_indices': selected_bins.tolist(), 'subcarrier_hz': hz.tolist()}
            except Exception as exc:
                raise ValueError(f"Cannot convert {path.relative_to(input_root)}: {exc}") from exc
            first = None
            for name, assignment in split_maps.items():
                if env not in assignment:
                    continue
                split = assignment[env]
                relative = Path(split) / (stem + '.npz')
                dest = stage / name / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open('xb') as target:
                    if first is None:
                        np.savez_compressed(target, csi_real=csi.real, csi_imag=csi.imag,
                            environment=np.asarray(env), channel_code=np.asarray(code), source=np.asarray(str(path)),
                            **{key: np.asarray(value) for key,value in axes.items()},
                            **{key: np.asarray(value if value is not None else np.nan) for key,value in metadata.items()})
                        first = dest
                    else:
                        with first.open('rb') as source:
                            shutil.copyfileobj(source,target)
                indices[name].append(dict(source=str(path), prepared=relative.as_posix(),
                    environment=env, channel_code=code, split=split, shape=list(csi.shape), **axes, **metadata))
        fold_index = {}
        for name, assignment in split_maps.items():
            splits = {s: sorted(g for g, owner in assignment.items() if owner==s) for s in ('train','val','test')}
            manifest = dict(format='AquaPilot prepared CSI', n_subcarriers=n_subcarriers, band_fraction=band_fraction, frequency_grid=GRID_NAME,
                            split_groups=splits, records=indices[name], skipped=skipped)
            (stage/name/'index.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n', encoding='utf-8')
            if name:
                fold_index[name] = {**splits, 'root': str(output/name)}
        if fold_index:
            manifest = {'protocol': 'Watermark leave-one-environment-out', 'groups': list(WATERMARK_GROUPS), 'folds': fold_index}
            (stage/'folds.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
        stage.rename(output)
    print(f"Prepared {len(selected)} recordings in {output}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert Watermark impulse responses to complex frequency responses")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n-subcarriers", type=int, default=128)
    parser.add_argument("--band-fraction", type=float, default=0.8)
    parser.add_argument("--time-axis", type=int, choices=[0, 1], default=0, help="Slow-time axis in MATLAB subscript order (WatermarkV1: 0).")
    parser.add_argument("--train-env", nargs="+", default=["NOF", "NCS"])
    parser.add_argument("--val-env", nargs="+", default=["BCH"])
    parser.add_argument("--test-env", nargs="+", default=["KAU"])
    args = parser.parse_args(argv)
    splits = dict(train=args.train_env, val=args.val_env, test=args.test_env)
    prepare_watermark(args.input, args.output, {'': splits}, n_subcarriers=args.n_subcarriers,
                      band_fraction=args.band_fraction, time_axis=args.time_axis)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc))
