from __future__ import annotations

from typing import Any

import numpy as np


def _qpsk_mod(bits: np.ndarray) -> np.ndarray:
    if bits.shape[-1] != 2:
        raise ValueError("QPSK modulation expects a final dimension of 2 bits.")
    real = 1.0 - 2.0 * bits[..., 0]
    imag = 1.0 - 2.0 * bits[..., 1]
    return (real + 1j * imag) / np.sqrt(2.0)


def _qpsk_demod(symbols: np.ndarray) -> np.ndarray:
    return np.stack((symbols.real < 0.0, symbols.imag < 0.0), axis=-1).astype(np.int8)


def simulate_qpsk_link(
    true_csi: np.ndarray,
    estimated_csi: np.ndarray,
    pilot_mask: np.ndarray,
    snr_db: float,
    trials: int = 8,
    block_subcarriers: int = 16,
    code_rate: float = 0.5,
    seed: int = 0,
) -> dict[str, Any]:
    """Monte-Carlo one-tap equalization with a normalized goodput proxy.

    The simulation is uncoded; `code_rate` is used only in the normalized payload
    accounting. BLER is computed over consecutive groups of data subcarriers.
    """
    true_csi = np.asarray(true_csi)
    estimated_csi = np.asarray(estimated_csi)
    pilot_mask = np.asarray(pilot_mask, dtype=bool)
    if true_csi.shape != estimated_csi.shape or true_csi.ndim != 2:
        raise ValueError("true_csi and estimated_csi must have shape [frames, subcarriers].")
    if pilot_mask.shape != true_csi.shape:
        raise ValueError("pilot_mask must match CSI shape.")
    rng = np.random.default_rng(seed)
    total_bits = 0
    bit_errors = 0
    blocks = 0
    block_errors = 0
    noise_scale = 10.0 ** (-float(snr_db) / 20.0)

    for _ in range(int(trials)):
        for frame in range(true_csi.shape[0]):
            data_indices = np.flatnonzero(~pilot_mask[frame])
            if data_indices.size == 0:
                continue
            bits = rng.integers(0, 2, size=(data_indices.size, 2), dtype=np.int8)
            symbols = _qpsk_mod(bits)
            h = true_csi[frame, data_indices]
            h_est = estimated_csi[frame, data_indices]
            received_power = np.mean(np.abs(h * symbols) ** 2)
            sigma = np.sqrt(max(received_power, 1e-12) / 2.0) * noise_scale
            noise = sigma * (rng.standard_normal(symbols.shape) + 1j * rng.standard_normal(symbols.shape))
            received = h * symbols + noise
            equalized = received / np.where(np.abs(h_est) > 1e-4, h_est, 1e-4 + 0j)
            decoded = _qpsk_demod(equalized)
            errors = decoded != bits
            bit_errors += int(errors.sum())
            total_bits += int(bits.size)

            for left in range(0, data_indices.size, max(1, int(block_subcarriers))):
                right = min(data_indices.size, left + max(1, int(block_subcarriers)))
                blocks += 1
                if errors[left:right].any():
                    block_errors += 1

    ber = bit_errors / max(total_bits, 1)
    bler = block_errors / max(blocks, 1)
    pilot_ratio = float(pilot_mask.mean())
    normalized_goodput = 2.0 * float(code_rate) * (1.0 - pilot_ratio) * (1.0 - bler)
    return {
        "ber": ber,
        "bler": bler,
        "pilot_ratio": pilot_ratio,
        "normalized_goodput": normalized_goodput,
        "snr_db": float(snr_db),
        "total_bits": total_bits,
        "blocks": blocks,
    }
