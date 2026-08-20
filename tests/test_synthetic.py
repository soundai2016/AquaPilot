import numpy as np

from aquapilot.data.synthetic import DEFAULT_ENVIRONMENTS, generate_episode


def test_generate_episode_shape_and_finite():
    episode = generate_episode(DEFAULT_ENVIRONMENTS["calm_fjord"], 20, 32, seed=3)
    assert episode.shape == (20, 32)
    assert np.iscomplexobj(episode)
    assert np.isfinite(episode.real).all()
    assert np.isfinite(episode.imag).all()
    assert 0.8 < np.mean(np.abs(episode) ** 2) < 1.2
