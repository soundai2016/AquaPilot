import torch

from aquapilot.models.phymask import AquaPilotNet


def test_model_forward_shapes():
    model = AquaPilotNet(n_subcarriers=32, hidden_dim=16, rank=2)
    x = torch.randn(4, 6, 32, 3)
    x[..., 2] = (torch.rand(4, 6, 32) > 0.75).float()
    out = model(x)
    assert out["mean"].shape == (4, 32, 2)
    assert out["sigma"].shape == (4, 32, 2)
    assert torch.all(out["sigma"] > 0)
    assert out["low_rank_component"].shape == (4, 32, 2)


def test_homoscedastic_ablation_uses_fixed_sigma():
    model = AquaPilotNet(n_subcarriers=16, hidden_dim=8, rank=1, fixed_sigma=0.25)
    x = torch.randn(2, 4, 16, 3)
    output = model(x)
    assert torch.allclose(output["sigma"], torch.full_like(output["sigma"], 0.25))
