import numpy as np
import torch

from aquapilot.pilots import (
    comb_mask,
    posterior_kernel_update,
    uncertainty_mask,
    uncertainty_top_mask,
)


def test_comb_exact_budget_across_nondivisible_sizes():
    for n_freq in (17, 32, 63, 128):
        for budget in (1, 2, min(7, n_freq), n_freq):
            mask = comb_mask(n_freq, budget, offset=5)
            assert mask.dtype == bool
            assert mask.sum() == budget


def test_uncertainty_selectors_exact_budget():
    mean = np.zeros((32, 2), dtype=np.float32)
    sigma = np.ones((32, 2), dtype=np.float32)
    sigma[10:14] *= 3
    for selector in (
        lambda: uncertainty_mask(mean, sigma, budget=6, min_spacing=2),
        lambda: uncertainty_top_mask(sigma, budget=6, min_spacing=2),
    ):
        mask = selector()
        assert mask.sum() == 6


def test_posterior_moves_toward_pilot_observations_and_reduces_variance():
    prior = torch.zeros(1, 16, 2)
    sigma = torch.ones_like(prior)
    observation = torch.randn_like(prior)
    mask = torch.zeros(1, 16, dtype=torch.bool)
    mask[:, [1, 7, 12]] = True
    posterior, posterior_sigma = posterior_kernel_update(prior, sigma, observation, mask)
    pilot_idx = [1, 7, 12]
    prior_error = (prior[:, pilot_idx] - observation[:, pilot_idx]).abs().mean()
    posterior_error = (posterior[:, pilot_idx] - observation[:, pilot_idx]).abs().mean()
    assert posterior_error < prior_error
    assert torch.all(posterior_sigma[:, pilot_idx] < sigma[:, pilot_idx])
