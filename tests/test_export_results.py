"""Constructed source runs exercise the offline evidence exporter."""
import csv
import hashlib
import io
import json
import sys
import zipfile

import pytest

from aquapilot.provenance import file_fingerprint
from aquapilot.analysis import analyze_results
from scripts.export_results import export_bundle
from test_analysis import completed_fixture


@pytest.fixture
def source(tmp_path):
    root = tmp_path/'private_server_path'/'aquapilot'
    completed_fixture(root/'seed17', seed=17)
    analyze_results([root], root/'analysis')
    return root


def test_bundle_keeps_counts_undefined_intervals_and_hashes_but_no_private_paths(source, tmp_path):
    output = tmp_path/'diagnostic.zip'
    with pytest.raises(ValueError, match='diagnostic'):
        export_bundle(source, output)
    assert not output.exists()
    manifest = export_bundle(source, output, diagnostic=True)
    assert manifest['classification'] == 'diagnostic only'
    assert manifest['input_classification'] == 'contains synthetic data'
    assert manifest['missing_cells'] is None
    assert manifest['members']['baseline_comparisons.csv']['size_bytes'] > 0
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None
        assert not any(n.endswith(('.npz', '.pt', '.tex')) or 'by_record' in n for n in z.namelist())
        assert all(b'private_server_path' not in z.read(n) for n in z.namelist())
        for n, expected in manifest['members'].items():
            assert hashlib.sha256(z.read(n)).hexdigest() == expected['sha256']
        interval = json.loads(z.read('paired_summary.json'))[0]['conditional_seed_interval']
        assert interval['num_seed_blocks'] == 1 and interval['t_lower'] is None
        pairs = list(csv.DictReader(io.StringIO(z.read('paired_top_b_025.csv').decode())))
        assert pairs[0]['metrics_sha256'] == file_fingerprint(source/'seed17'/'metrics.json')['sha256']
        assert 'source_metrics' not in pairs[0]
    with pytest.raises(FileExistsError):
        export_bundle(source, output, diagnostic=True)


def test_explicit_provisional_matrix_preserves_each_missing_cell(source, tmp_path):
    analyze_results([source], source/'analysis', expect_watermark_matrix=True, allow_incomplete=True)
    report = export_bundle(source, tmp_path/'provisional.zip', diagnostic=True)
    assert report['classification'] == 'diagnostic only'
    assert len(report['missing_cells']) == 35
    assert ['NOF', 17, .25] not in report['missing_cells']
    assert 'rank_placement.pdf' not in report['members']


@pytest.mark.parametrize('case', ['edited_csv', 'edited_csv_and_manifest', 'new_seed', 'changed_run'])
def test_stale_analysis_or_modified_completed_run_is_never_packaged(source, tmp_path, case):
    analysis = source/'analysis'
    path = analysis/'baseline_comparisons.csv'
    if case.startswith('edited_csv'):
        path.write_text(path.read_text() + '\n')
        if case == 'edited_csv_and_manifest':
            manifest_path = analysis/'analysis_summary.json'
            manifest = json.loads(manifest_path.read_text())
            manifest['analysis_artifacts'][path.name] = file_fingerprint(path)
            manifest_path.write_text(json.dumps(manifest))
    elif case == 'new_seed':
        completed_fixture(source/'seed37', seed=37)
    else:
        metrics = source/'seed17'/'metrics.json'
        metrics.write_text(metrics.read_text() + '\n')
    output = tmp_path/'not_created.zip'
    with pytest.raises(ValueError, match='changed|modified|does not reproduce'):
        export_bundle(source, output, diagnostic=True)
    assert not output.exists()


def test_fold_manifest_rejects_training_changes_even_if_test_set_is_identical(tmp_path):
    from aquapilot.data.audit import _inventory, EXPECTED_COUNTS, expected_split_groups
    from aquapilot.provenance import fingerprint
    from scripts.export_results import _formal_inputs

    inventory = _inventory(EXPECTED_COUNTS)
    splits = {split: [{'path': f'{split}/{rid}.npz', 'size_bytes': 99, 'sha256': fingerprint(rid)}
                      for rid, (env, _) in inventory.items() if env in members]
              for split, members in expected_split_groups('BCH').items()}
    main = {'splits': splits, 'split_fingerprints': {s: fingerprint(v) for s,v in splits.items()}}
    manifest = {'resolved_config': {'data': {'root': '/unneeded/fold_test_BCH'}}, 'data': {'main': main}}
    sources = {}
    for seed in (17, 37):
        folder = tmp_path/str(seed)
        folder.mkdir()
        (folder/'run_manifest.json').write_text(json.dumps(manifest))
        sources[str(folder/'metrics.json')] = str(seed)
    cells = {('BCH', seed, .25, str(seed)) for seed in (17, 37)}
    assert _formal_inputs(sources, cells)['BCH']['train']['num_files'] == 120
    # Source manifests are individually consistent. Across seeds, one training array changed.
    splits['train'][0]['sha256'] = fingerprint('changed training recording')
    main['split_fingerprints']['train'] = fingerprint(splits['train'])
    (tmp_path/'37'/'run_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='inputs differ across seeds'):
        _formal_inputs(sources, cells)


def test_default_paths_point_to_current_watermark_matrix(tmp_path, monkeypatch):
    from scripts import export_results

    root = tmp_path/'results/watermark_inband/aquapilot'
    completed_fixture(root/'seed17', seed=17)
    analyze_results([root], root/'analysis')
    monkeypatch.setattr(export_results, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['export_results.py'])
    with pytest.raises(ValueError, match='complete measured Watermark'):
        export_results.main()
    monkeypatch.setattr(sys, 'argv', ['export_results.py', '--diagnostic'])
    export_results.main()
    with zipfile.ZipFile(tmp_path/'results/AquaPilot_analysis.zip') as handoff:
        assert json.loads(handoff.read('export_manifest.json'))['classification'] == 'partial_or_test_diagnostic'
        evidence = handoff.read('results.zip')
    with zipfile.ZipFile(io.BytesIO(evidence)) as archive:
        manifest = json.loads(archive.read('bundle_manifest.json'))
        assert manifest['includes_result_tables'] is False
        assert 'summary.txt' in manifest['members'] and 'error_flags.csv' in manifest['members']
