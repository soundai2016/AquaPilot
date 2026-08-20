from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from aquapilot.pilots import comb_mask, random_mask


def _load_npz_sequence(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as bundle:
        if "csi_real" in bundle and "csi_imag" in bundle:
            csi = bundle["csi_real"].astype(np.float32) + 1j * bundle["csi_imag"].astype(np.float32)
        elif "csi" in bundle:
            csi = bundle["csi"]
        else:
            raise KeyError(f"{path} lacks csi or csi_real/csi_imag arrays.")
        environment = str(bundle["environment"].item()) if "environment" in bundle else path.parent.name
        source = str(bundle["source"].item()) if "source" in bundle else str(path)
    csi = np.asarray(csi)
    if csi.ndim != 2:
        raise ValueError(f"Expected CSI [time, frequency] in {path}; got {csi.shape}.")
    return {"id": path.stem, "environment": environment, "csi": csi.astype(np.complex64), "source": source}


def load_prepared_sequences(root: str | Path, split: str) -> list[dict[str, Any]]:
    """Load prepared `.npz` sequences.

    Preferred layout: `<root>/<split>/*.npz`. If `splits.json` exists, it may
    instead map split names to relative paths.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Prepared data root does not exist: {root}")
    split_manifest = root / "splits.json"
    paths: list[Path]
    if split_manifest.exists():
        mapping = json.loads(split_manifest.read_text(encoding="utf-8"))
        if split not in mapping:
            raise KeyError(f"Split '{split}' is absent from {split_manifest}.")
        paths = [root / item for item in mapping[split]]
    else:
        paths = sorted((root / split).rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(
            f"No prepared sequences found for split '{split}' under {root}. "
            "Run the relevant preparation script first."
        )
    return [_load_npz_sequence(path) for path in paths]


def interpolate_complex(observed: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Linearly interpolate sparse complex samples along frequency."""
    observed = np.asarray(observed)
    mask = np.asarray(mask, dtype=bool)
    if observed.shape != mask.shape:
        raise ValueError("observed and mask must have the same shape.")
    if observed.ndim != 2:
        raise ValueError("Expected [time, frequency] arrays.")
    n_time, n_freq = observed.shape
    x = np.arange(n_freq)
    filled = np.empty_like(observed, dtype=np.complex64)
    for t in range(n_time):
        idx = np.flatnonzero(mask[t])
        if idx.size == 0:
            filled[t] = 0.0
        elif idx.size == 1:
            filled[t] = observed[t, idx[0]]
        else:
            real = np.interp(x, idx, observed[t, idx].real)
            imag = np.interp(x, idx, observed[t, idx].imag)
            filled[t] = real + 1j * imag
    return filled


class ChannelWindowDataset(Dataset[dict[str, torch.Tensor]]):
    """Leakage-safe windows that never cross sequence boundaries."""

    def __init__(
        self,
        sequences: Iterable[dict[str, Any]],
        history: int,
        horizon: int,
        pilot_ratio: float,
        mask_mode: str = "mixed",
        extra_block_mask: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.sequences = list(sequences)
        self.history = int(history)
        self.horizon = int(horizon)
        self.pilot_ratio = float(pilot_ratio)
        self.mask_mode = mask_mode
        self.extra_block_mask = float(extra_block_mask)
        self.seed = int(seed)
        if self.history < 2 or self.horizon < 1:
            raise ValueError("history must be >=2 and horizon >=1.")
        if not 0.0 < self.pilot_ratio <= 1.0:
            raise ValueError("pilot_ratio must be in (0, 1].")
        self.index: list[tuple[int, int]] = []
        n_freq: int | None = None
        for seq_idx, sequence in enumerate(self.sequences):
            csi = np.asarray(sequence["csi"])
            if csi.ndim != 2:
                raise ValueError(f"Sequence {sequence.get('id')} has shape {csi.shape}, expected [T,F].")
            if n_freq is None:
                n_freq = csi.shape[1]
            elif csi.shape[1] != n_freq:
                raise ValueError("All sequences in a dataset must share the same subcarrier count.")
            max_start = csi.shape[0] - self.history - self.horizon + 1
            self.index.extend((seq_idx, start) for start in range(max(0, max_start)))
        if not self.index:
            raise ValueError("No valid windows; sequences may be shorter than history+horizon.")
        self.n_subcarriers = int(n_freq or 0)

    def __len__(self) -> int:
        return len(self.index)

    def _masks(self, index: int, n_freq: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed + 7919 * index)
        budget = max(2, int(round(self.pilot_ratio * n_freq)))
        masks = np.zeros((self.history, n_freq), dtype=bool)
        for t in range(self.history):
            if self.mask_mode == "comb":
                masks[t] = comb_mask(n_freq, budget, offset=(index + t) % max(1, n_freq // budget))
            elif self.mask_mode == "random":
                masks[t] = random_mask(n_freq, budget, rng)
            elif self.mask_mode == "mixed":
                if rng.random() < 0.5:
                    masks[t] = comb_mask(n_freq, budget, offset=int(rng.integers(0, max(1, n_freq // budget))))
                else:
                    masks[t] = random_mask(n_freq, budget, rng)
            else:
                raise ValueError(f"Unknown mask_mode: {self.mask_mode}")
            if self.extra_block_mask > 0.0 and rng.random() < self.extra_block_mask:
                width = max(1, int(rng.integers(1, max(2, n_freq // 8))))
                left = int(rng.integers(0, max(1, n_freq - width + 1)))
                masks[t, left : left + width] = False
                if masks[t].sum() < 2:
                    masks[t] = comb_mask(n_freq, budget)
        return masks

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        seq_idx, start = self.index[item]
        sequence = self.sequences[seq_idx]
        csi = np.asarray(sequence["csi"], dtype=np.complex64)
        history = csi[start : start + self.history]
        target = csi[start + self.history + self.horizon - 1]
        masks = self._masks(item, csi.shape[1])

        observed = np.where(masks, history, 0.0 + 0.0j)
        filled = interpolate_complex(observed, masks)
        scale = float(np.sqrt(np.mean(np.abs(filled) ** 2)))
        scale = max(scale, 1e-6)
        filled = filled / scale
        target = target / scale

        features = np.stack((filled.real, filled.imag, masks.astype(np.float32)), axis=-1).astype(np.float32)
        target_ri = np.stack((target.real, target.imag), axis=-1).astype(np.float32)
        return {
            "x": torch.from_numpy(features),
            "target": torch.from_numpy(target_ri),
            "mask": torch.from_numpy(masks.astype(np.float32)),
            "scale": torch.tensor(scale, dtype=torch.float32),
            "sequence_index": torch.tensor(seq_idx, dtype=torch.long),
            "start": torch.tensor(start, dtype=torch.long),
        }
