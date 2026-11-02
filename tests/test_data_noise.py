import numpy as np

from aquapilot.data.prepared import ChannelWindowDataset, sequence_observation_noise
from aquapilot.engine import _build_datasets


def test_sequence_noise_complex_rms_and_component_variance():
    csi = np.full((2000, 64), 2.0 + 0j, dtype=np.complex64)
    noise, _ = sequence_observation_noise(csi, 0.2, 17)
    assert np.isclose(np.mean(np.abs(noise)**2), 0.16, rtol=0.015)
    assert np.isclose(np.var(noise.real), 0.08, rtol=0.02)
    assert np.isclose(np.var(noise.imag), 0.08, rtol=0.02)


def test_overlapping_history_windows_share_physical_noise():
    csi = np.ones((10, 8), dtype=np.complex64)
    dataset = ChannelWindowDataset([{"id": "fixture", "csi": csi}], history=3, horizon=1,
                                   pilot_ratio=1.0, mask_mode="comb", observation_sigma=0.1, noise_seed=19)
    first, second = dataset[0], dataset[1]
    a = first["x"][..., :2].numpy() * first["scale"].item()
    b = second["x"][..., :2].numpy() * second["scale"].item()
    np.testing.assert_allclose(a[1:], b[:-1], atol=1e-7)
    assert not np.allclose(a[..., 0], 1)
    np.testing.assert_allclose(first["target"].numpy()[:, 0] * first["scale"].item(), 1, atol=1e-7)
    np.testing.assert_array_equal(dataset[0]["x"].numpy(), first["x"].numpy())


def test_validation_and_test_do_not_inherit_training_mask_augmentation():
    sequence = {"id": "fixture", "csi": np.ones((10, 16), dtype=np.complex64)}
    cfg = {"seed": 17, "data": {"history": 3, "horizon": 1},
           "pilot": {"ratio": 0.25, "training_mask_mode": "random", "extra_block_mask": 1.0, "observation_sigma": 0.08}}
    datasets = _build_datasets(cfg, {name: [sequence] for name in ("train", "val", "test")})
    assert datasets["train"].mask_mode == "random"
    for name in ("val", "test"):
        assert datasets[name].mask_mode == "comb"
        assert datasets[name].extra_block_mask == 0
        assert datasets[name].observation_sigma == 0.08
        np.testing.assert_array_equal(datasets[name][0]["mask"].sum(axis=1).numpy(), [4, 4, 4])
