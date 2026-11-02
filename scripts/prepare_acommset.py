#!/usr/bin/env python3
"""Prepare complex channel responses from extracted ACommSet files."""
from __future__ import annotations

import argparse
import hashlib
import tempfile
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.io import loadmat

SUPPORTED_SUFFIXES = {".mat", ".npy", ".npz"}


@dataclass(frozen=True)
class Segment:
    split: str
    start: int
    stop: int


def load_container(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".npy":
        return {"array": np.load(path, allow_pickle=False)}
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as bundle:
            return {key: bundle[key] for key in bundle.files}
    try:
        return loadmat(path, simplify_cells=True)
    except NotImplementedError:
        result: dict[str, Any] = {}
        with h5py.File(path, "r") as handle:

            def visit(name: str, obj: h5py.Dataset | h5py.Group) -> None:
                if not isinstance(obj, h5py.Dataset):
                    return
                array = np.asarray(obj)
                if array.dtype.fields and {"real", "imag"}.issubset(array.dtype.fields):
                    array = array["real"] + 1j * array["imag"]
                result[name] = array

            handle.visititems(visit)
        return result


def get_array(
    data: dict[str, Any], key: str | None, real_key: str | None, imag_key: str | None
) -> np.ndarray:
    if real_key or imag_key:
        if not (real_key and imag_key):
            raise ValueError("Both --real-key and --imag-key are required together.")
        if real_key not in data or imag_key not in data:
            raise KeyError(f"Missing real/imag keys; available keys: {sorted(data)}")
        real, imag = np.asarray(data[real_key]), np.asarray(data[imag_key])
        if real.shape != imag.shape or real.dtype.kind not in 'fiu' or imag.dtype.kind not in 'fiu':
            raise ValueError('Real and imaginary arrays must be real-valued with identical shapes.')
        return real + 1j * imag
    if key is None:
        candidates = [
            name
            for name, value in data.items()
            if not name.startswith("__") and np.asarray(value).ndim >= 2
        ]
        if len(candidates) != 1:
            raise KeyError(f"Specify --key. Candidate arrays: {candidates}")
        key = candidates[0]
    if key not in data:
        raise KeyError(f"Key {key!r} not found. Available keys: {sorted(data)}")
    return np.asarray(data[key])


def split_from_path(path: Path) -> str | None:
    lowered = {part.lower() for part in path.parts}
    for split in ("train", "val", "test"):
        if split in lowered or (split == "val" and "validation" in lowered):
            return split
    return None


def convert_array(array: np.ndarray, time_axis: int, frequency_axis: int) -> list[np.ndarray]:
    array = np.asarray(array)
    if array.ndim < 2:
        raise ValueError(f"CSI array must have at least two dimensions; got {array.shape}")
    if any(axis < -array.ndim or axis >= array.ndim for axis in (time_axis, frequency_axis)):
        raise ValueError(f'Time/frequency axes are outside the loaded array shape {array.shape}.')
    time_axis %= array.ndim
    frequency_axis %= array.ndim
    if time_axis == frequency_axis:
        raise ValueError("time_axis and frequency_axis must differ.")
    if not np.iscomplexobj(array) and array.ndim-1 in (time_axis, frequency_axis):
        raise ValueError('For packed real/imaginary data, reserve the last axis for the two components.')
    array = np.moveaxis(array, (time_axis, frequency_axis), (0, 1))
    if not np.iscomplexobj(array):
        if array.ndim >= 3 and array.shape[-1] == 2:
            array = array[..., 0] + 1j * array[..., 1]
        else:
            raise ValueError(
                "Selected CSI array is real. Use --real-key/--imag-key or provide a final RI dimension."
            )
    array = np.asarray(array, dtype=np.complex64)
    if not np.isfinite(array).all() or array.shape[1] < 2:
        raise ValueError('Expected finite complex CSI on at least two source frequency bins.')
    if array.ndim == 2:
        return [array]
    flattened = array.reshape(array.shape[0], array.shape[1], -1)
    return [flattened[:, :, index].astype(np.complex64) for index in range(flattened.shape[-1])]


def resample_frequency(csi: np.ndarray, n_subcarriers: int | None) -> np.ndarray:
    csi = np.asarray(csi, dtype=np.complex64)
    if n_subcarriers is None or csi.shape[1] == int(n_subcarriers):
        return csi
    target = int(n_subcarriers)
    if target < 2:
        raise ValueError("--n-subcarriers must be at least 2.")
    source_coordinate = np.linspace(0.0, 1.0, csi.shape[1])
    target_coordinate = np.linspace(0.0, 1.0, target)
    real = np.stack([np.interp(target_coordinate, source_coordinate, row.real) for row in csi])
    imag = np.stack([np.interp(target_coordinate, source_coordinate, row.imag) for row in csi])
    return (real + 1j * imag).astype(np.complex64)


def chronological_segments(
    n_frames: int,
    train_ratio: float,
    val_ratio: float,
    guard_frames: int,
    min_frames: int,
) -> list[Segment]:
    if not 0.0 < train_ratio < 1.0 or not 0.0 < val_ratio < 1.0:
        raise ValueError("train_ratio and val_ratio must be in (0, 1).")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be smaller than 1.")
    guard = max(0, int(guard_frames))
    train_stop = int(np.floor(n_frames * train_ratio))
    val_boundary = int(np.floor(n_frames * (train_ratio + val_ratio)))
    segments = [
        Segment("train", 0, train_stop),
        Segment("val", min(train_stop + guard, n_frames), val_boundary),
        Segment("test", min(val_boundary + guard, n_frames), n_frames),
    ]
    too_short = [segment for segment in segments if segment.stop - segment.start < min_frames]
    if too_short:
        detail = ", ".join(
            f"{segment.split}={segment.stop - segment.start}" for segment in too_short
        )
        raise ValueError(
            f"Chronological split is too short ({detail}; need >= {min_frames}). "
            "Reduce --history/--guard-frames or combine longer contiguous records."
        )
    return segments


def save_sequence(
    output_root: Path,
    split: str,
    stem: str,
    csi: np.ndarray,
    environment: str,
    source: Path,
    channel_index: int,
    start: int,
    stop: int,
) -> Path:
    destination_dir = output_root / split
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{stem}_seq{channel_index:03d}_t{start:06d}-{stop:06d}.npz"
    with destination.open("xb") as handle:
        np.savez_compressed(
            handle,
            csi_real=csi.real.astype(np.float32),
            csi_imag=csi.imag.astype(np.float32),
            environment=np.asarray(environment),
            source=np.asarray(str(source)),
            source_start=np.asarray(start),
            source_stop=np.asarray(stop),
        )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--key", default=None)
    parser.add_argument("--real-key", default=None)
    parser.add_argument("--imag-key", default=None)
    parser.add_argument("--time-axis", type=int, required=True)
    parser.add_argument("--frequency-axis", type=int, required=True)
    parser.add_argument(
        "--split-mode",
        choices=["all-test", "path", "continuous"],
        default="all-test",
        help=(
            "all-test: external zero-shot set; path: require train/val/test directory names; "
            "continuous: chronological few-shot adaptation with guard frames"
        ),
    )
    parser.add_argument("--n-subcarriers", type=int, default=128)
    parser.add_argument("--train-ratio", type=float, default=0.10)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--guard-frames", type=int, default=16)
    parser.add_argument("--history", type=int, default=16)
    args = parser.parse_args()

    args.input, args.output = args.input.resolve(), args.output.resolve()
    if args.input.is_relative_to(args.output) or args.output.is_relative_to(args.input):
        parser.error('Keep the extracted source and prepared destination in separate directories.')
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error('Use a new or empty output directory; existing prepared data are not overwritten.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.acommset-', dir=args.output.parent) as work:
        staging = Path(work) / 'prepared'
        staging.mkdir()
        counts = _prepare(args, staging)
        # Same-filesystem atomic publication; a failed conversion leaves no usable partial split.
        staging.rename(args.output)
    print(f'Prepared {counts}; manifest: {args.output / "index.json"}')


def _prepare(args: argparse.Namespace, staging: Path) -> dict[str, int]:
    files = sorted(
        path for path in args.input.rglob("*") if path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(f"No MAT/NPY/NPZ files found under {args.input}")
    records: list[dict[str, Any]] = []
    min_frames = int(args.history) + 1

    for path in files:
        relative = path.relative_to(args.input).as_posix()
        record_stem = path.stem + '_' + hashlib.sha256(relative.encode('utf-8')).hexdigest()[:16]
        try:
            data = load_container(path)
            array = get_array(data, args.key, args.real_key, args.imag_key)
            sequences = convert_array(array, args.time_axis, args.frequency_axis)
            environment = path.parent.name
            for channel_index, raw_csi in enumerate(sequences):
                csi = resample_frequency(raw_csi, args.n_subcarriers)
                if args.split_mode == "all-test":
                    segments = [Segment("test", 0, csi.shape[0])]
                elif args.split_mode == "path":
                    split = split_from_path(path)
                    if split is None:
                        raise ValueError(
                            "--split-mode path requires a train/val/test directory component; "
                            f"none found in {path}"
                        )
                    segments = [Segment(split, 0, csi.shape[0])]
                else:
                    segments = chronological_segments(
                        csi.shape[0],
                        args.train_ratio,
                        args.val_ratio,
                        args.guard_frames,
                        min_frames,
                    )

                for segment in segments:
                    segment_csi = csi[segment.start : segment.stop]
                    if segment_csi.shape[0] < min_frames:
                        raise ValueError(
                            f"{segment.split} segment has {segment_csi.shape[0]} frames; "
                            f"need at least {min_frames}."
                        )
                    destination = save_sequence(
                        staging,
                        segment.split,
                        record_stem,
                        segment_csi,
                        environment,
                        path,
                        channel_index,
                        segment.start,
                        segment.stop,
                    )
                    records.append(
                        {
                            "source": str(path),
                            "prepared": str(destination.relative_to(staging)),
                            "split": segment.split,
                            "environment": environment,
                            "channel_index": channel_index,
                            "source_relative_path": relative,
                            "source_frequency_bins": int(raw_csi.shape[1]),
                            "source_interval": [segment.start, segment.stop],
                            "shape": list(segment_csi.shape),
                        }
                    )
        except Exception as exc:  # noqa: BLE001 - independent archive files may differ
            raise ValueError(f"Failed to prepare {relative}: {exc}. No split was published.") from exc

    manifest = {
        "format": "AquaPilot prepared CSI",
        "source_dataset": "ACommSet (user-approved local copy)",
        "split_mode": args.split_mode,
        "n_subcarriers": args.n_subcarriers,
        "history": args.history,
        "variable": {"key": args.key, "real_key": args.real_key, "imag_key": args.imag_key},
        "loaded_array_axes": {"time": args.time_axis, "frequency": args.frequency_axis},
        "frequency_mapping": "linear interpolation of real/imaginary components on normalized band coordinate; not an absolute-Hz alignment",
        "train_ratio": args.train_ratio if args.split_mode == "continuous" else None,
        "val_ratio": args.val_ratio if args.split_mode == "continuous" else None,
        "guard_frames": args.guard_frames if args.split_mode == "continuous" else None,
        "records": records,
        "failures": [],
    }
    index_path = staging / "index.json"
    index_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not records:
        raise RuntimeError(
            f"No ACommSet sequences converted. Inspect {index_path} failures and key/axis arguments."
        )
    counts = {split: sum(record["split"] == split for record in records) for split in ("train", "val", "test")}
    return counts


if __name__ == "__main__":
    main()
