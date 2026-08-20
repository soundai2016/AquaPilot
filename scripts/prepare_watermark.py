#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.io import loadmat


def load_mat(path: Path) -> dict[str, Any]:
    try:
        return loadmat(path, simplify_cells=True)
    except NotImplementedError:
        result: dict[str, Any] = {}
        with h5py.File(path, "r") as handle:
            for key, value in handle.items():
                if isinstance(value, h5py.Dataset):
                    array = np.asarray(value)
                    if array.dtype.fields and {"real", "imag"}.issubset(array.dtype.fields):
                        array = array["real"] + 1j * array["imag"]
                    result[key] = array
        return result


def scalar(data: dict[str, Any], key: str) -> float | None:
    if key not in data:
        return None
    value = np.asarray(data[key]).squeeze()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


WATERMARK_CHANNEL_CODES = ("NOF1", "NCS1", "BCH1", "KAU1", "KAU2")
WATERMARK_GROUPS = ("NOF", "NCS", "BCH", "KAU")


def infer_channel_code(path: Path) -> str:
    match = re.search(r"(NOF1|NCS1|BCH1|KAU1|KAU2)", path.as_posix(), re.IGNORECASE)
    return match.group(1).upper() if match else path.parent.name.upper()


def environment_group(channel_code: str) -> str:
    """Map Watermark recordings to four independent geographical environments.

    KAU1 and KAU2 are recordings from the same KAM11/Kauai probe transmission
    observed at different receivers. They must remain in the same split.
    """
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


def convert_h_to_csi(h: np.ndarray, n_subcarriers: int, band_fraction: float, time_axis: int | None) -> np.ndarray:
    h = np.asarray(h).squeeze()
    if h.ndim != 2:
        raise ValueError(f"Watermark h must be 2-D after squeeze; got {h.shape}")
    if time_axis is None:
        # Manual example is 258 time snapshots x 2048 delay samples.
        time_axis = int(np.argmin(h.shape))
    if time_axis not in (0, 1):
        raise ValueError("time_axis must be 0 or 1.")
    if time_axis == 1:
        h = h.T
    n_delay = h.shape[1]
    response = np.fft.fftshift(np.fft.fft(h, n=n_delay, axis=1), axes=1)
    fraction = float(band_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("band_fraction must be in (0, 1].")
    width = max(n_subcarriers, int(round(fraction * n_delay)))
    width = min(width, n_delay)
    left = (n_delay - width) // 2
    selected = np.linspace(left, left + width - 1, n_subcarriers, dtype=int)
    return response[:, selected].astype(np.complex64)


def _normalize_group_names(values: set[str]) -> set[str]:
    return {environment_group(value) for value in values}


def assign_split(environment: str, train: set[str], val: set[str], test: set[str]) -> str:
    environment = environment_group(environment)
    train = _normalize_group_names(train)
    val = _normalize_group_names(val)
    test = _normalize_group_names(test)
    memberships = [name for name, group in (("train", train), ("val", val), ("test", test)) if environment in group]
    if len(memberships) > 1:
        raise ValueError(f"Environment group {environment} appears in multiple splits: {memberships}")
    return memberships[0] if memberships else "unassigned"


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Watermark TVIR MAT files to AquaPilot CSI sequences")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n-subcarriers", type=int, default=128)
    parser.add_argument("--band-fraction", type=float, default=0.8)
    parser.add_argument("--time-axis", type=int, choices=[0, 1], default=None)
    parser.add_argument(
        "--train-env", nargs="+", default=["NOF", "NCS"],
        help="Independent environment groups or channel codes; KAU1/KAU2 are always grouped as KAU",
    )
    parser.add_argument("--val-env", nargs="+", default=["BCH"])
    parser.add_argument("--test-env", nargs="+", default=["KAU"])
    args = parser.parse_args()

    files = sorted(args.input.rglob("*.mat"))
    if not files:
        raise FileNotFoundError(f"No .mat files found under {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)
    split_sets = tuple(set(value.upper() for value in group) for group in (args.train_env, args.val_env, args.test_env))
    index = []
    skipped = []
    for path in files:
        data = load_mat(path)
        if "h" not in data:
            skipped.append({"file": str(path), "reason": "no h variable"})
            continue
        channel_code = infer_channel_code(path)
        environment = environment_group(channel_code)
        split = assign_split(environment, *split_sets)
        if split == "unassigned":
            skipped.append({"file": str(path), "reason": f"environment group {environment} unassigned"})
            continue
        csi = convert_h_to_csi(data["h"], args.n_subcarriers, args.band_fraction, args.time_axis)
        destination_dir = args.output / split
        destination_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{channel_code}_{path.stem}")
        destination = destination_dir / f"{safe_name}.npz"
        np.savez_compressed(
            destination,
            csi_real=csi.real.astype(np.float32),
            csi_imag=csi.imag.astype(np.float32),
            environment=np.asarray(environment),
            channel_code=np.asarray(channel_code),
            source=np.asarray(str(path)),
            V0=np.asarray(scalar(data, "V0") if scalar(data, "V0") is not None else np.nan),
            fc=np.asarray(scalar(data, "fc") if scalar(data, "fc") is not None else np.nan),
            fs_t=np.asarray(scalar(data, "fs_t") if scalar(data, "fs_t") is not None else np.nan),
            fs_tau=np.asarray(scalar(data, "fs_tau") if scalar(data, "fs_tau") is not None else np.nan),
        )
        index.append(
            {
                "source": str(path),
                "prepared": str(destination.relative_to(args.output)),
                "environment": environment,
                "channel_code": channel_code,
                "split": split,
                "shape": list(csi.shape),
                "V0": scalar(data, "V0"),
                "fc": scalar(data, "fc"),
                "fs_t": scalar(data, "fs_t"),
                "fs_tau": scalar(data, "fs_tau"),
            }
        )
    manifest = {
        "format": "AquaPilot prepared CSI v1",
        "n_subcarriers": args.n_subcarriers,
        "band_fraction": args.band_fraction,
        "split_groups": {
            "train": sorted(_normalize_group_names(split_sets[0])),
            "val": sorted(_normalize_group_names(split_sets[1])),
            "test": sorted(_normalize_group_names(split_sets[2])),
        },
        "records": index,
        "skipped": skipped,
    }
    (args.output / "index.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not index:
        raise RuntimeError("No Watermark channel files were converted. Inspect skipped entries in index.json.")
    print(f"Converted {len(index)} files; skipped {len(skipped)}. Manifest: {args.output / 'index.json'}")


if __name__ == "__main__":
    main()
