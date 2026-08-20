import torch

from aquapilot.models import build_forecast_model


def test_unstructured_baseline_matches_output_contract():
    model = build_forecast_model(
        24,
        {"kind": "unstructured", "hidden_dim": 12, "conv_kernel": 3, "dropout": 0.0},
    )
    x = torch.randn(3, 5, 24, 3)
    x[..., 2] = (torch.rand(3, 5, 24) > 0.7).float()
    output = model(x)
    assert output["mean"].shape == (3, 24, 2)
    assert output["sigma"].shape == (3, 24, 2)
    assert output["coefficients"].shape == (3, 2, 0)
    assert torch.all(output["sigma"] > 0)
