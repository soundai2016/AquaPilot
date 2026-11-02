import numpy as np
import pytest

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


def test_complex_covariance_preserves_quadrature_phase():
    # H_1 = i H_0. The conjugated covariance would predict -i H_0 instead.
    latent = np.array([1, -1, 1j, -1j], dtype=np.complex128)
    csi = np.stack((latent, 1j * latent), axis=1)
    prior = fit_frequency_prior([{"csi": csi}], shrinkage=0.0)
    assert prior.covariance[1, 0].imag > 0
    observation = np.array([1.0 + 0j, 0j])
    estimate = lmmse_estimate(observation, np.array([True, False]), prior, observation_variance=1e-8)
    assert abs(estimate[1] - 1j) < 1e-6


def test_capped_frequency_prior_covers_records_and_time_instead_of_file_prefix():
    rng = np.random.default_rng(71)
    records = [
        {"id": f"channel_{n}", "environment": env,
         "csi": rng.normal(size=(n, 4)) + 1j * rng.normal(size=(n, 4))}
        for n, env in [(24, "NCS"), (13, "NOF"), (5, "BCH")]
    ]
    fit = fit_frequency_prior(records, max_frames=10, shrinkage=0)
    reversed_fit = fit_frequency_prior(records[::-1], max_frames=10, shrinkage=0)
    np.testing.assert_array_equal(fit.covariance, reversed_fit.covariance)
    assert sum(r["used_frames"] for r in fit.fit_records) == 10
    assert {r["environment"] for r in fit.fit_records} == {"NCS", "NOF", "BCH"}
    by_id = {r["id"]: r["csi"] for r in records}
    used = []
    for row, idx in zip(fit.fit_records, fit.fit_indices):
        csi = by_id[row["record_id"]]
        edges = np.arange(len(idx) + 1) * len(csi) // len(idx)
        assert len(idx) >= 1 and np.all((edges[:-1] <= idx) & (idx < edges[1:]))
        used.extend(csi[idx] / np.sqrt(np.mean(np.abs(csi)**2)))
    np.testing.assert_allclose(fit.mean, np.mean(used, axis=0))
    np.testing.assert_allclose(fit.covariance, np.cov(np.asarray(used), rowvar=False))
    with pytest.raises(ValueError, match="one frame from each"):
        fit_frequency_prior(records, max_frames=2)
