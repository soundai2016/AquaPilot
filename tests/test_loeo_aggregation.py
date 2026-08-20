from pathlib import Path

from scripts.run_watermark_loeo import flatten_run


def test_flatten_run_keeps_posterior_coverage() -> None:
    result = {
        "model_kind": "aquapilot",
        "model_parameters": 123,
        "model_test": {
            "nmse_db_mean": -1.0,
            "evm_percent_mean": 10.0,
            "latency_ms_per_frame": 0.5,
        },
        "baselines": {},
        "closed_loop": {
            "aoptimal": {
                "pilot_ratio": 0.125,
                "posterior": {
                    "nmse_db_mean": -2.0,
                    "evm_percent_mean": 8.0,
                    "coverage_90": 0.89,
                },
                "link": {},
            }
        },
        "classical_closed_loop": {},
    }

    row = flatten_run(result, "fold_test_KAU", 0.125, 17, Path("out"))
    assert row["closed_aoptimal_coverage90"] == 0.89
