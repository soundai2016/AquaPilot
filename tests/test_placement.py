from math import comb

import numpy as np
import pytest

from aquapilot.placement import combine_mask_summaries, summarize_masks


def test_band_edges_and_pattern_encoding_restart_at_record_boundaries():
    sequence = np.array([[1,0,1,0,0,0], [1,0,1,0,0,0], [0,1,0,0,1,0]], dtype=bool)
    report = summarize_masks(sequence)
    # Distances on a line: [0,1,0,1,2,3] twice, then [1,0,1,1,0,1].
    assert report['distance_sum_bins'] == 18
    assert report['max_frame_worst_distance_bins'] == 3
    assert report['worst_distance_sum_bins'] == 7
    assert report['subset_bits_per_explicit_set'] == 4
    assert report['flag_and_set_total_bits'] == 10  # explicit 4, repeat 1, changed 1+4
    assert report['mean_replaced_pilots_per_transition'] == 1
    one = summarize_masks(sequence[1:2]); assert one['changed_transition_fraction'] is None
    combined = combine_mask_summaries([report, one]); assert combined['flag_and_set_total_bits'] == 14
    assert combined['num_transitions'] == 2 and combined['mean_nearest_pilot_distance_bins'] == 25/24


def test_accounting_matches_literal_sets_across_chunks():
    rng = np.random.default_rng(17)
    pilots = [set(rng.choice(13, 4, replace=False).tolist()) for _ in range(533)]
    pilots[512] = pilots[511]
    masks = np.array([[k in p for k in range(13)] for p in pilots])
    report = summarize_masks(masks)
    ds = [[min(abs(k-pk) for pk in p) for k in range(13)] for p in pilots]
    assert report['distance_sum_bins'] == sum(map(sum, ds))
    assert report['changed_transitions'] == sum(a != b for a,b in zip(pilots,pilots[1:]))
    assert report['replaced_pilots'] == sum(len(b-a) for a,b in zip(pilots,pilots[1:]))


@pytest.mark.parametrize(('b','bits'), [(8,41), (16,67), (32,101), (128,0)])
def test_watermark_description_lengths(b, bits):
    mask = np.arange(128)[None,:] < b
    out = summarize_masks(np.repeat(mask,3,axis=0))
    assert out['subset_bits_per_explicit_set'] == bits and (1 << bits) >= comb(128,b)
    if b == 128:
        assert out['flag_and_set_total_bits'] == 0 and out['max_frame_worst_distance_bins'] == 0
