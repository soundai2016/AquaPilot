"""Frequency coverage and example description lengths for a sequence of pilot sets."""
from __future__ import annotations

from math import comb
from typing import Any, Iterable

import numpy as np


_ADDITIVE = ('num_records', 'num_mask_frames', 'num_transitions', 'changed_transitions',
             'replaced_pilots', 'distance_sum_bins', 'worst_distance_sum_bins')


def _derived(row: dict[str, Any]) -> dict[str, Any]:
    n, b = row['n_subcarriers'], row['pilot_budget']
    frames, transitions = row['num_mask_frames'], row['num_transitions']
    changed, records = row['changed_transitions'], row['num_records']
    # Integer ceil(log2): exact also for a singleton or power-of-two alphabet.
    width = (comb(n, b) - 1).bit_length()
    # First set is explicit. Each later snapshot has one change flag and,
    # only for a change, a new full set. At B=N the schedule needs no bits.
    flagged = (records + changed) * width + transitions if width else 0
    return {**row,
            'mean_nearest_pilot_distance_bins': row['distance_sum_bins'] / (frames * n),
            'mean_frame_worst_distance_bins': row['worst_distance_sum_bins'] / frames,
            'changed_transition_fraction': changed / transitions if transitions else None,
            'mean_replaced_pilots_per_transition': row['replaced_pilots'] / transitions if transitions else None,
            'subset_bits_per_explicit_set': width,
            'index_list_bits_per_explicit_set': b * (n - 1).bit_length(),
            'flag_and_set_total_bits': flagged,
            'flag_and_set_bits_per_snapshot': flagged / frames}


def summarize_masks(masks: np.ndarray) -> dict[str, Any]:
    """Summarize one recording. The frequency axis is a line, including band edges."""
    masks = np.asarray(masks)
    if masks.dtype != np.bool_ or masks.ndim != 2 or not masks.size:
        raise ValueError('Expected a nonempty Boolean pilot mask [snapshots, frequency].')
    frames, n = masks.shape
    budgets = masks.sum(axis=1)
    b = int(budgets[0])
    if b == 0 or not np.all(budgets == b):
        raise ValueError('Mask diagnostics require a fixed nonzero pilot count at every snapshot.')
    idx = np.arange(n)
    distance_sum = worst_sum = worst_max = changed = replaced = 0
    for start in range(0, frames, 512):
        block = masks[start:start + 512]
        left = np.maximum.accumulate(np.where(block, idx, -n), axis=1)
        right = np.minimum.accumulate(np.where(block, idx, 2*n)[:, ::-1], axis=1)[:, ::-1]
        distances = np.minimum(idx-left, right-idx)
        worst = distances.max(axis=1)
        distance_sum += int(distances.sum())
        worst_sum += int(worst.sum())
        worst_max = max(worst_max, int(worst.max()))
        first = max(start, 1)
        stop = start + len(block)
        additions = (masks[first:stop] & ~masks[first-1:stop-1]).sum(axis=1)
        changed += int(np.count_nonzero(additions))
        replaced += int(additions.sum())
    return _derived(dict(n_subcarriers=n, pilot_budget=b, num_records=1, num_mask_frames=frames,
                         num_transitions=frames-1, changed_transitions=changed, replaced_pilots=replaced,
                         distance_sum_bins=distance_sum, worst_distance_sum_bins=worst_sum,
                         max_frame_worst_distance_bins=worst_max))


def combine_mask_summaries(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Pool integer counts before computing means; never create inter-record transitions."""
    rows = list(rows)
    if not rows or len({(r['n_subcarriers'], r['pilot_budget']) for r in rows}) != 1:
        raise ValueError('Mask summaries require one frequency grid and pilot budget.')
    row = {k: sum(r[k] for r in rows) for k in _ADDITIVE}
    row.update(n_subcarriers=rows[0]['n_subcarriers'], pilot_budget=rows[0]['pilot_budget'],
               max_frame_worst_distance_bins=max(r['max_frame_worst_distance_bins'] for r in rows))
    if not (row['num_transitions'] == row['num_mask_frames'] - row['num_records']
            and 0 <= row['changed_transitions'] <= row['num_transitions']):
        raise ValueError('Inconsistent mask-transition counts.')
    return _derived(row)
