import numpy as np

from aquapilot.models.classical import (
    fit_frequency_prior,
    interpolate_observation,
    lmmse_estimate,
    omp_delay_estimate,
)


def test_classical_estimators_are_shape_safe_and_finite():
    n_freq = 32
    frequency = np.arange(n_freq)
    frames = np.stack(
        [
            np.exp(-2j * np.pi * frequency * delay / n_freq)
            + 0.2 * np.exp(-2j * np.pi * frequency * (delay + 3) / n_freq)
            for delay in range(1, 9)
        ]
    ).astype(np.complex64)
    prior = fit_frequency_prior([{"csi": frames}], shrinkage=0.1)
    mask = np.zeros(n_freq, dtype=bool)
    mask[::4] = True
    observation = frames[-1]
    estimates = [
        interpolate_observation(observation, mask),
        lmmse_estimate(observation, mask, prior, observation_variance=1e-4),
        omp_delay_estimate(observation, mask, delay_bins=16, sparsity=4),
    ]
    for estimate in estimates:
        assert estimate.shape == (n_freq,)
        assert np.isfinite(estimate.real).all()
        assert np.isfinite(estimate.imag).all()
