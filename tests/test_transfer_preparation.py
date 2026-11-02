import json
import sys

import numpy as np
import pytest

from scripts.prepare_acommset import convert_array, main


def test_distinct_source_paths_preserve_both_channels(tmp_path, monkeypatch):
    raw, out = tmp_path/'raw', tmp_path/'prepared'
    for i, name in enumerate(('A', 'B'), 1):
        src = raw/name; src.mkdir(parents=True)
        np.save(src/'channel.npy', np.full((18,6), i+1j, np.complex64))
    monkeypatch.setattr(sys, 'argv', ['prepare', '--input', str(raw), '--output', str(out), '--time-axis','0','--frequency-axis','1','--n-subcarriers','4'])
    main()
    rows = json.loads((out/'index.json').read_text())['records']
    assert len({r['prepared'] for r in rows}) == len(rows) == 2
    assert all(r['source_frequency_bins'] == 6 for r in rows)
    for i, row in enumerate(rows,1):
        with np.load(out/row['prepared']) as data: np.testing.assert_array_equal(data['csi_real'],np.full((18,4),i))
    with pytest.raises(SystemExit): main()


def test_bad_source_never_publishes_partial_transfer_split(tmp_path, monkeypatch):
    raw, out = tmp_path/'raw', tmp_path/'prepared'; raw.mkdir()
    np.save(raw/'a.npy', np.ones((18,4),np.complex64)); np.save(raw/'b.npy',np.ones((18,4)))
    monkeypatch.setattr(sys, 'argv', ['prepare','--input',str(raw),'--output',str(out),'--time-axis','0','--frequency-axis','1'])
    with pytest.raises(ValueError,match='b.npy'): main()
    assert not out.exists()


def test_axis_selection_uses_loaded_dimensions_without_squeezing():
    a = np.ones((1,18,6,2))
    result = convert_array(a,1,2)
    assert len(result)==1 and result[0].shape==(18,6)
    with pytest.raises(ValueError,match='outside'): convert_array(a,5,2)
