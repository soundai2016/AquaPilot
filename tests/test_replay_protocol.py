import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from aquapilot.engine import _closed_loop_sequence, evaluate_closed_loop, _classical_closed_loop_sequence
from aquapilot.ofdm import simulate_qpsk_link
from aquapilot.pilots import posterior_kernel_update
from aquapilot.replay import check_checkpoint_protocol, matched_information_baselines, update_component_sigma


class HistoryPredictor(nn.Module):
    def forward(self, x):
        mean = 0.8 * x[:, -1, :, :2]
        sigma = torch.ones_like(mean) * torch.linspace(0.1, 0.3, mean.shape[1])[None, :, None]
        return {"mean": mean, "sigma": sigma}


def replay_config():
    return {"seed": 17, "data": {"history": 3, "horizon": 1},
            "pilot": {"ratio": 0.25, "observation_sigma": 0.03, "update_noise_convention": "component", "min_spacing": 2, "update_length_scale": 2},
            "eval": {"prediction_storage": "compact", "run_link_proxy": False}}


def sequence(name="record_A", frames=9):
    rng = np.random.default_rng(4)
    csi = (rng.standard_normal((frames, 8)) + 1j * rng.standard_normal((frames, 8))).astype(np.complex64)
    return {"id": name, "environment": "synthetic_fixture", "source": name, "csi": csi}


def test_component_sigma_and_gaussian_update_have_correct_factor():
    sigma = update_component_sigma(0.2, 2.0, "component")
    comparison = update_component_sigma(0.2, 2.0, "complex_rms_as_component")
    assert sigma**2 == pytest.approx(0.005)
    assert comparison**2 == pytest.approx(0.01)
    prior = torch.zeros((1, 1, 2))
    scales = torch.ones_like(prior)
    observed = torch.ones_like(prior)
    posterior, _ = posterior_kernel_update(prior, scales, observed, torch.ones((1, 1), dtype=torch.bool), observation_sigma=sigma)
    jitter = max(sigma**2 * 1e-4, 1e-8)
    assert posterior[0, 0, 0].item() == pytest.approx(1 / (1+sigma**2+jitter), rel=1e-6)


def test_matched_baselines_use_only_observed_pilots_and_preserve_current_pilots():
    history = np.array([[1, 0, 2, 0, 3], [2, 0, 3, 0, 4]], dtype=np.complex64)
    masks = np.tile(np.array([True, False, True, False, True]), (2, 1))
    current_mask = np.array([False, True, False, True, False])
    observed = np.array([99, 4+2j, -999, 6-1j, 88], dtype=np.complex64)
    estimates = matched_information_baselines(history, masks, observed, current_mask, 2.0)
    changed = observed.copy()
    changed[~current_mask] *= 1000
    comparison = matched_information_baselines(history, masks, changed, current_mask, 2.0)
    for name in estimates:
        np.testing.assert_array_equal(estimates[name], comparison[name])
    for name in ("current_interpolation", "persistence_innovation"):
        complex_estimate = estimates[name][:, 0] + 1j * estimates[name][:, 1]
        np.testing.assert_allclose(complex_estimate[current_mask] * 2, observed[current_mask], atol=1e-6)
    np.testing.assert_array_equal(estimates["zero"], 0)


def test_noise_convention_replays_same_history_without_retraining():
    cfg = replay_config()
    model = HistoryPredictor()
    component = _closed_loop_sequence(model, sequence(), cfg, torch.device("cpu"), "comb", 17)
    comparison_cfg = copy.deepcopy(cfg)
    comparison_cfg["pilot"]["update_noise_convention"] = "complex_rms_as_component"
    comparison = _closed_loop_sequence(model, sequence(), comparison_cfg, torch.device("cpu"), "comb", 17)
    np.testing.assert_array_equal(component["target"], comparison["target"])
    np.testing.assert_array_equal(component["prior_mean"], comparison["prior_mean"])
    np.testing.assert_array_equal(component["mask"], comparison["mask"])
    np.testing.assert_allclose(component["update_component_sigma"] * np.sqrt(2), comparison["update_component_sigma"])
    assert not np.allclose(component["posterior_mean"], comparison["posterior_mean"])
    assert component["frame_identity_sha256"] == comparison["frame_identity_sha256"]


def test_same_mask_classical_interpolation_matches_policy_reference():
    cfg = replay_config()
    learned = _closed_loop_sequence(HistoryPredictor(), sequence(), cfg, torch.device("cpu"), "comb", 17)
    classical = _classical_closed_loop_sequence(sequence(), cfg, "ls_interpolation", None, "comb", 17)
    np.testing.assert_allclose(learned["baseline_current_interpolation_mean"], classical["estimate"], atol=1e-6)
    np.testing.assert_array_equal(learned["target"], classical["target"])


def test_compact_record_shards_and_equal_frame_aggregation(tmp_path: Path):
    cfg = replay_config()
    first, second = sequence("record_A", 9), sequence("record_B", 6)
    result, preview = evaluate_closed_loop(HistoryPredictor(), [first, second], cfg, torch.device("cpu"), "comb", output_dir=tmp_path)
    assert result["posterior"]["num_frames"] == 9
    assert result["posterior"]["nonpilot"]["num_frames"] == 9
    assert "link_proxy" not in result
    assert result["same_information_baselines"]["zero"]["nmse_db_mean"] == pytest.approx(0.0)
    assert len(preview["target"]) <= 80
    means = [record["posterior"]["nmse_db_mean"] for record in result["records"]]
    assert result["posterior"]["nmse_db_mean"] == pytest.approx((6*means[0]+3*means[1])/9)
    assert [(r["frame_start"], r["frame_stop"]) for r in result["records"]] == [(0, 6), (6, 9)]
    for record in result["records"]:
        with np.load(tmp_path / record["arrays_file"], allow_pickle=False) as arrays:
            assert "prior_rank_spearman" in arrays
            assert "prior_mean" not in arrays
            assert arrays["record_id"].item() == record["record_id"]
            assert len(arrays["frame_index"]) == record["num_frames"]
            assert arrays["frame_index"][0] == 3


def test_checkpoint_protocol_rejects_silent_history_horizon_changes():
    cfg = replay_config()
    bundle = {"n_subcarriers": 8, "experiment_config": {"data": {"history": 3, "horizon": 1}}}
    check_checkpoint_protocol(bundle, cfg, 8)
    with pytest.raises(ValueError, match="subcarrier"):
        check_checkpoint_protocol(bundle, cfg, 16)
    wrong = copy.deepcopy(cfg)
    wrong["data"]["history"] = 4
    with pytest.raises(ValueError, match="history"):
        check_checkpoint_protocol(bundle, wrong, 8)
    wrong = copy.deepcopy(cfg)
    wrong["data"]["horizon"] = 2
    with pytest.raises(ValueError, match="horizon"):
        check_checkpoint_protocol(bundle, wrong, 8)


@pytest.mark.parametrize("policy", ["comb", "random", "uncertainty", "active"])
def test_unmeasured_current_responses_cannot_change_future_trajectory(policy):
    cfg, seq, device = replay_config(), sequence(), torch.device("cpu")
    model = HistoryPredictor()
    original = _closed_loop_sequence(model, seq, cfg, device, policy, 17)
    alternate = copy.deepcopy(seq)
    # Change phase on precisely the unseen tones. Magnitudes, hence the
    # simulator's record-level injected noise RMS, stay fixed.
    alternate["csi"][cfg["data"]["history"]:][~original["mask"]] *= -1
    comparison = _closed_loop_sequence(model, alternate, cfg, device, policy, 17)
    assert not np.array_equal(original["target"], comparison["target"])
    for key in ["mask", "history_scale", "prior_mean", "prior_sigma", "posterior_mean",
                "posterior_sigma", "baseline_persistence_prior_mean",
                "baseline_current_interpolation_mean", "baseline_persistence_innovation_mean"]:
        np.testing.assert_array_equal(original[key], comparison[key])


def test_current_pilot_perturbation_changes_update_and_next_history_but_not_current_selection():
    cfg, seq, device = replay_config(), sequence(), torch.device("cpu")
    first = _closed_loop_sequence(HistoryPredictor(), seq, cfg, device, "uncertainty", 17)
    perturbed = copy.deepcopy(seq)
    perturbed["csi"][cfg["data"]["history"], first["mask"][0]] *= -1
    second = _closed_loop_sequence(HistoryPredictor(), perturbed, cfg, device, "uncertainty", 17)
    np.testing.assert_array_equal(first["prior_mean"][0], second["prior_mean"][0])
    np.testing.assert_array_equal(first["mask"][0], second["mask"][0])
    assert not np.allclose(first["posterior_mean"][0], second["posterior_mean"][0])
    assert not np.allclose(first["prior_mean"][1], second["prior_mean"][1])


def test_qpsk_proxy_partitions_the_same_physical_symbols_and_noise():
    csi = sequence()["csi"] * np.linspace(0.1, 3.0, 8)
    est = 0.6*csi + 0.3j
    mask = np.broadcast_to(np.arange(8) < 3, csi.shape)
    def evaluate(p):
        return simulate_qpsk_link(csi, est, p, snr_db=0, seed=29)
    all_bins = evaluate(np.zeros_like(mask))
    left, right = evaluate(mask), evaluate(~mask)
    assert left["record_reference_power"] == right["record_reference_power"]
    for name in ["bit_errors", "total_bits"]:
        assert left[name] + right[name] == all_bins[name]
    empty = evaluate(np.ones_like(mask))
    assert empty["ber"] is None and empty["bler"] is None
    assert empty["normalized_goodput"] == 0


def test_qpsk_proxy_counts_actual_bits_in_partial_blocks_and_preserves_phase_at_low_gain():
    h = np.full((1, 3), 1e-8j)
    est = h.copy()
    est[:, -1] *= -1
    proxy = simulate_qpsk_link(h, est, np.zeros(h.shape, bool), snr_db=100,
                               trials=1, block_subcarriers=2, code_rate=0.5)
    assert proxy["bit_errors"] == 2 and proxy["block_errors"] == 1
    assert proxy["error_free_block_bits"] == 4 and proxy["grid_slots"] == 3
    assert proxy["normalized_goodput"] == pytest.approx(2/3)
