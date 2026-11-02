"""Filesystem behavior of the one-command handoff (Linux training host)."""
import hashlib
import zipfile

import pytest

from scripts.analysis_handoff import FORMAT, choose_source, json_bytes, publish


def handoff(payload):
    return {'scores.csv':payload, 'export_manifest.json':json_bytes({
        'format': FORMAT, 'members': {'scores.csv': {'size_bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}}})}


def test_repeated_exports_preserve_changes_and_refuse_to_replace_a_training_archive(tmp_path):
    path = tmp_path/'results/AquaPilot_analysis.zip'
    publish(path, handoff(b'previous'))
    before = path.read_bytes()
    stamp = path.stat().st_mtime_ns
    publish(path, handoff(b'previous'))
    assert path.stat().st_mtime_ns == stamp
    publish(path, handoff(b'changed'))
    previous, = (path.parent/'export_history').iterdir()
    assert previous.read_bytes() == before
    with zipfile.ZipFile(path) as z:
        assert z.read('scores.csv') == b'changed'
    unknown = path.parent/'results.zip'
    with zipfile.ZipFile(unknown, 'w') as z:
        z.writestr('bundle_manifest.json','{}')
    with pytest.raises(FileExistsError):
        publish(unknown, handoff(b'refuse'))
    with zipfile.ZipFile(unknown) as z:
        assert z.namelist() == ['bundle_manifest.json']


def test_no_silent_fallback_from_partial_new_training_to_earlier_run(tmp_path):
    old = tmp_path/'results/watermark/results.zip'
    old.parent.mkdir(parents=True)
    old.touch()
    assert choose_source(tmp_path, None) == old
    model = tmp_path/'results/watermark_inband/aquapilot'
    (model/'fold_test_BCH').mkdir(parents=True)
    assert choose_source(tmp_path, None) == model
