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

    Input channels are in physical units. A record-wide, full-band SNR
    reference and full-grid random draws are shared by all placement rules.
    The block rule discards an entire block on any error. No decoder is run.
    """
    true_csi = np.asarray(true_csi)
    estimated_csi = np.asarray(estimated_csi)
    pilot_mask = np.asarray(pilot_mask, dtype=bool)
    if true_csi.shape != estimated_csi.shape or true_csi.ndim != 2:
        raise ValueError("true_csi and estimated_csi must have shape [frames, subcarriers].")
    if pilot_mask.shape != true_csi.shape or true_csi.size == 0:
        raise ValueError("pilot_mask must match CSI shape.")
    if not np.isfinite(true_csi).all() or not np.isfinite(estimated_csi).all():
        raise ValueError("QPSK replay requires finite physical channel responses.")
    if not np.isfinite(snr_db) or trials < 1 or block_subcarriers < 1 or not 0 < code_rate <= 1:
        raise ValueError("Invalid QPSK SNR, trial count, block width or code rate.")
    rng = np.random.default_rng(seed)
    total_bits = 0
    bit_errors = 0
    blocks = 0
    block_errors = 0
    delivered_bits = 0
    reference_power = float(np.mean(np.abs(true_csi.astype(np.complex128)) ** 2))
    if reference_power == 0:
        raise ValueError("QPSK replay SNR is undefined for a zero-power reference record.")
    sigma = np.sqrt(reference_power / 2.0) * 10.0 ** (-float(snr_db) / 20.0)
    n_freq = true_csi.shape[1]

    for _ in range(int(trials)):
        for frame in range(true_csi.shape[0]):
            data_indices = np.flatnonzero(~pilot_mask[frame])
            bits_grid = rng.integers(0, 2, size=(n_freq, 2), dtype=np.int8)
            noise_grid = sigma * (rng.standard_normal(n_freq) + 1j * rng.standard_normal(n_freq))
            if data_indices.size == 0:
                continue
            bits = bits_grid[data_indices]
            symbols = _qpsk_mod(bits)
            h = true_csi[frame, data_indices]
            h_est = estimated_csi[frame, data_indices]
            received = h * symbols + noise_grid[data_indices]
            # Hard QPSK decisions depend only on the phase after equalization.
            decoded = _qpsk_demod(received * h_est.conj())
            errors = decoded != bits
            bit_errors += int(errors.sum())
            total_bits += int(bits.size)

            for left in range(0, data_indices.size, max(1, int(block_subcarriers))):
                right = min(data_indices.size, left + max(1, int(block_subcarriers)))
                blocks += 1
                if errors[left:right].any():
                    block_errors += 1
                else:
                    delivered_bits += 2 * (right - left)

    ber = bit_errors / total_bits if total_bits else None
    bler = block_errors / blocks if blocks else None
    pilot_ratio = float(pilot_mask.mean())
    grid_slots = int(trials) * true_csi.size
    normalized_goodput = float(code_rate) * delivered_bits / grid_slots
    return {
        "ber": ber,
        "bler": bler,
        "pilot_ratio": pilot_ratio,
        "normalized_goodput": normalized_goodput,
        "snr_db": float(snr_db),
        "total_bits": total_bits,
        "blocks": blocks,
        "bit_errors": bit_errors, "block_errors": block_errors,
        "error_free_block_bits": delivered_bits, "grid_slots": grid_slots,
        "record_reference_power": reference_power,
    }
