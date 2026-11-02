"""Test transport and relocation using fabricated run files, never measurements."""
import csv
import hashlib
import io
import itertools
import json
import zipfile

import pytest
from aquapilot import post_legacy as old
from aquapilot.data.audit import _inventory, EXPECTED_COUNTS, expected_split_groups
from aquapilot.provenance import fingerprint, finish_run
from test_analysis import completed_fixture


def format_fixture(root):
    # Exercise the old manifest schema entirely offline. The declaration field is part
    # of that schema; every actual record below carries the 'synthetic-fixture' label.
    for env in sorted(old._ENVS):
        records = _inventory(EXPECTED_COUNTS)
        splits = {s: [dict(path=f'{s}/{rid}.npz', size_bytes=99, sha256=fingerprint(rid))
                      for rid,(g,_) in records.items() if g in groups]
                  for s,groups in expected_split_groups(env).items()}
        for seed,frac in itertools.product((17,37,57),(.0625,.125,.25)):
            folder = root/env/str(seed)/str(frac)
            result = completed_fixture(folder,seed=seed,environment=env,fraction=frac)
            manifest = json.loads((folder/'run_manifest.json').read_text())
            cfg = manifest['resolved_config']
            cfg['data'] = {'kind':'prepared','root':f'/test-only/fold_test_{env}'}
            manifest.update(protocol_id=old.LEGACY_PROTOCOL,config_fingerprint=fingerprint(cfg),
                            data={'main':{'splits':splits,'split_fingerprints':{s:fingerprint(v) for s,v in splits.items()}}})
            result['artifact_status'] = 'measured-channel experiment'  # historical field syntax in this test only
            result['fixture_notice'] = 'Synthetic compatibility fixture. No Watermark recordings.'
            (folder/'resolved_config.json').write_text(json.dumps(cfg))
            (folder/'run_manifest.json').write_text(json.dumps(manifest))
            finish_run({'output_dir':folder, 'manifest':manifest},result)


def old_zip(path, tables, report, *, unexpected=False, intervals_changed=False):
    members = {k.removeprefix('legacy/'):v for k,v in tables.items() if k.removeprefix('legacy/') not in old._RECORD_TABLES}
    if unexpected:
        members['../unlisted.csv'] = b'outside'
    if intervals_changed:
        members['paired_summary.json'] = b'[]'
    manifest = {**report, 'run_mode':'fresh_training', 'num_environment_seed_budget_cells':36,
                'input_classification':'declared measured channels','members': {
                    k:dict(size_bytes=len(v),sha256=hashlib.sha256(v).hexdigest()) for k,v in members.items()}}
    with zipfile.ZipFile(path,'w') as z:
        for k,v in members.items(): z.writestr(k,v)
        z.writestr('bundle_manifest.json',json.dumps(manifest))


def test_relocated_tree_and_archive_have_consistent_receipts(tmp_path):
    tree = tmp_path/'synthetic-format-fixture'
    format_fixture(tree)
    payload, info = old.legacy_results(tree)
    archive = tmp_path/'numeric-fixture.zip'
    old_zip(archive,payload,info)
    archived, archived_info = old.legacy_results(archive)
    assert archived_info['record_tables_missing'] == list(old._RECORD_TABLES)
    assert not archived_info['paper_eligible'] and not archived_info['record_tables_available']
    moved = tree.rename(tmp_path/'relocated')
    restored, provenance = old.legacy_results(moved, prior_zip=archive)
    assert restored == payload and provenance['matched_prior_archive_sha256'] == archived_info['source_archive_sha256']
    for name,n in zip(old._RECORD_TABLES,(108,144)):
        data = restored['legacy/'+name]
        rows = list(csv.DictReader(io.StringIO(data.decode())))
        assert len(rows) == n and all(len(r['record_key'])==64 for r in rows)
        assert b'actual-test-record' not in data and str(tmp_path).encode() not in data
    # Neither an extra member nor a self-consistently hashed table contradicting
    # the available run tree may pass verification.
    old_zip(archive,payload,info,unexpected=True)
    with pytest.raises(ValueError, match='members'): old.legacy_results(archive)
    old_zip(archive,payload,info,intervals_changed=True)
    with pytest.raises(ValueError, match='differ'): old.legacy_results(moved,prior_zip=archive)
    next(moved.rglob('metrics.json')).unlink()
    with pytest.raises(ValueError, match='Incomplete Watermark matrix'): old.legacy_results(moved,prior_zip=archive)
