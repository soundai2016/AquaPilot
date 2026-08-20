from aquapilot.engine import one_standard_error_select


def test_one_se_prefers_smallest_eligible_rank():
    rows = [
        {"rank": 0.0, "mean_nmse_db": -5.0, "se_nmse_db": 0.1},
        {"rank": 1.0, "mean_nmse_db": -5.7, "se_nmse_db": 0.2},
        {"rank": 2.0, "mean_nmse_db": -5.8, "se_nmse_db": 0.4},
        {"rank": 4.0, "mean_nmse_db": -5.9, "se_nmse_db": 0.5},
    ]
    selected = one_standard_error_select(rows)
    assert selected["rank"] == 1.0
