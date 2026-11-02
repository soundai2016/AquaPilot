"""One-case plots must not fabricate variance or hide missing run outcomes."""
import sys
import numpy as np
import pandas as pd
import pytest

from aquapilot.provenance import PROTOCOL_ID
from scripts.summarize_watermark import main, mean_std


def test_single_case_summary_plots_and_withholds_undefined_policy(tmp_path, monkeypatch):
    row = dict(fold='fold_test_NOF', pilot_ratio_requested=.25, seed=17,
               protocol_id=PROTOCOL_ID, protocol_fingerprint='synthetic-test-protocol',
               noise_convention='component', run_mode='fresh_training')
    for policy in ('comb', 'random', 'uncertainty', 'active'):
        row[f'closed_{policy}_nmse_db'] = np.nan if policy == 'uncertainty' else 1.
        row[f'closed_{policy}_channel_nrmse_percent'] = np.nan if policy == 'uncertainty' else 112.
    source, output = tmp_path/'aggregate_runs.csv', tmp_path/'summary'
    pd.DataFrame([row]).to_csv(source, index=False)
    output.mkdir()
    (output/'environment_differences.csv').write_text('obsolete unverified paired table')
    monkeypatch.setattr(sys, 'argv', ['summarize_watermark.py', '--input', str(source), '--output', str(output), '--allow-partial'])
    main()
    table = pd.read_csv(output/'aggregate.csv').set_index('policy')
    assert set(table['sd_ddof']) == {0}
    assert table.loc['Comb', 'nmse_db_sd'] == 0
    assert table.loc['Comb', 'nmse_db_valid_runs'] == 1
    assert table.loc['Uncertainty top-B', 'nmse_db_undefined_runs'] == 1
    assert np.isnan(table.loc['Uncertainty top-B', 'nmse_db_mean'])
    assert (output/'nmse_vs_pilot.pdf').stat().st_size > 1000
    assert not (output/'environment_differences.csv').exists()
    assert 'original frame matching' in (output/'summary_notes.txt').read_text()
    mean, sd = mean_std(pd.Series([1., np.nan]))
    assert np.isnan(mean) and np.isnan(sd)
    assert mean_std(pd.Series([1., 3.])) == pytest.approx((2., 1.))
