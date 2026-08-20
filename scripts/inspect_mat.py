#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat


def describe_hdf5(path: Path) -> None:
    with h5py.File(path, "r") as handle:
        def visit(name: str, obj: h5py.Dataset | h5py.Group) -> None:
            if isinstance(obj, h5py.Dataset):
                print(f"{name}: shape={obj.shape}, dtype={obj.dtype}")
        handle.visititems(visit)


def describe_mat(path: Path) -> None:
    try:
        data = loadmat(path, simplify_cells=False)
    except NotImplementedError:
        describe_hdf5(path)
        return
    for key, value in sorted(data.items()):
        if key.startswith("__"):
            continue
        array = np.asarray(value)
        print(f"{key}: shape={array.shape}, dtype={array.dtype}, complex={np.iscomplexobj(array)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect MATLAB/HDF5 array keys and shapes")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if not args.path.exists():
        raise FileNotFoundError(args.path)
    describe_mat(args.path)


if __name__ == "__main__":
    main()
