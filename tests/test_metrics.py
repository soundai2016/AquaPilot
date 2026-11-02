import json
import numpy as np
import pytest
import torch

from aquapilot.metrics import (
    NMSE_LOG_FLOOR, framewise_diagnostics, summarize_diagnostics,
    summarize_estimates, summarize_prior_ranking, summarize_tensors,
)


def ri(real):
    real = np.asarray(real, dtype=float)
    return np.stack([real, np.zeros_like(real)], axis=-1)


def test_zero_estimate_is_zero_db_and_no_fabricated_coverage():
    target = ri([[1, 2, 3], [2, 4, 6]])
    result = summarize_estimates(np.zeros_like(target), target)
    assert result['nmse_db_mean'] == pytest.approx(0)
    assert result['channel_nrmse_percent_mean'] == pytest.approx(100)
    assert result['pooled_nmse_linear'] == pytest.approx(1)
    assert 'coverage_90' not in result
    assert 'evm_percent_mean' not in result
    json.dumps(result, allow_nan=False)


def test_zero_power_and_nonfinite_frames_are_counted_not_zero_filled():
    target = ri([[0, 0], [1, 1], [1, 1]])
    mean = ri([[2, 2], [0, 0], [np.nan, 0]])
    result = summarize_estimates(mean, target)
    assert result['num_zero_power_frames'] == 1
    assert result['num_nonfinite_frames'] == 1
    assert result['num_valid_error_frames'] == 1
    assert result['nmse_db_mean'] == pytest.approx(0)
    # The undefined first frame still contributes real error energy to pooling.
    assert result['pooled_nmse_linear'] == pytest.approx(5)
    json.dumps(result, allow_nan=False)


def test_framewise_aggregation_differs_from_pooling_and_db_of_mean():
    target = ri([[1], [10]])
    mean = ri([[0], [8]])
    result = summarize_estimates(mean, target)
    assert result['nmse_linear_mean'] == pytest.approx((1 + .04)/2)
    assert result['nmse_db_mean'] == pytest.approx((0 + 10*np.log10(.04))/2)
    assert result['pooled_nmse_linear'] == pytest.approx(5/101)
    assert result['channel_nrmse_percent_mean'] == pytest.approx(60)


def test_physical_scales_only_change_pooled_energy_weighting():
    target = ri([[1], [1]])
    mean = ri([[0], [.8]])
    normalized = summarize_estimates(mean, target)
    physical = summarize_estimates(mean, target, normalization_scale=np.array([1, 10]))
    assert physical['nmse_db_mean'] == pytest.approx(normalized['nmse_db_mean'])
    assert physical['pooled_nmse_linear'] == pytest.approx(5/101)
    assert normalized['pooled_nmse_linear'] == pytest.approx(.52)
    assert physical['energy_coordinates'] == 'physical_via_provided_scale'


def test_nonpilot_mask_and_empty_nonpilot_frames():
    target = ri([[1, 1, 1], [1, 1, 1]])
    mean = ri([[1, 0, 0], [0, 0, 0]])
    pilot = np.array([[1, 0, 0], [1, 1, 1]], dtype=bool)
    result = summarize_estimates(mean, target, observed_mask=pilot)
    assert result['nonpilot']['nmse_db_mean'] == pytest.approx(0)
    assert result['nonpilot']['num_empty_mask_frames'] == 1
    assert result['nonpilot']['num_valid_error_frames'] == 1


def test_rank_is_computed_within_each_frame_and_constant_is_undefined():
    # Frame offsets induce a misleading pooled relationship. Each variable
    # frame has exactly inverse within-frame order; the third score is constant.
    target = ri([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    mean = target + ri([[3, 2, 1], [103, 102, 101], [3, 2, 1]])
    scales = ri([[1, 2, 3], [101, 102, 103], [2, 2, 2]])
    result = summarize_prior_ranking(mean, scales, target)
    assert result['mean_frame_spearman'] == pytest.approx(-1)
    assert result['num_valid_frames'] == 2
    assert result['num_undefined_frames'] == 1
    assert result['num_constant_score_frames'] == 1


def test_no_defined_correlations_returns_null_not_zero():
    target = ri([[1, 1, 1]])
    result = summarize_prior_ranking(target, np.ones_like(target), target)
    assert result['mean_frame_spearman'] is None
    assert result['num_constant_error_frames'] == 1
    assert result['num_constant_score_frames'] == 1
    json.dumps(result, allow_nan=False)


def test_compact_diagnostics_aggregate_identically_and_log_floor_is_disclosed():
    target = torch.from_numpy(ri([[1, 2, 3], [2, 3, 4]]))
    mean = target.clone()
    sigma = torch.ones_like(target)
    diagnostics = framewise_diagnostics(mean, sigma, target, compute_ranking=False)
    joined = {k: np.concatenate([v[:1], v[1:]]) for k, v in diagnostics.items()}
    assert summarize_diagnostics(joined) == summarize_tensors(mean, sigma, target)
    result = summarize_diagnostics(joined)
    assert result['nmse_db_mean'] == pytest.approx(10*np.log10(NMSE_LOG_FLOOR))
    assert result['num_zero_error_frames'] == 2
    assert result['coverage_90'] == 1
    assert result['metric_definitions']['log_floor_linear'] == NMSE_LOG_FLOOR


def test_streaming_accumulator_matches_all_frame_statistics():
    from aquapilot.metrics import DiagnosticsAccumulator
    rng = np.random.default_rng(42)
    target = rng.normal(size=(19, 8, 2))
    mean = target + rng.normal(size=target.shape)
    sigma = np.abs(rng.normal(size=target.shape))
    mask = rng.random(size=target.shape[:2]) < .25
    scale = np.exp(rng.normal(size=len(target)))
    diagnostics = framewise_diagnostics(mean, sigma, target, observed_mask=mask, normalization_scale=scale)
    accumulator = DiagnosticsAccumulator()
    for part in (slice(0, 3), slice(3, 11), slice(11, 19)):
        accumulator.update({key: value[part] for key, value in diagnostics.items()})
    expected, actual = summarize_diagnostics(diagnostics), accumulator.summary()
    def check(a, b):
        assert a.keys() == b.keys()
        for key in a:
            if isinstance(a[key], dict):
                check(a[key], b[key])
            elif isinstance(a[key], float):
                assert a[key] == pytest.approx(b[key], rel=1e-12, abs=1e-12)
            else:
                assert a[key] == b[key]
    check(expected, actual)
