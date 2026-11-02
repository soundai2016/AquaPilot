"""Synthetic fixtures test statistical bookkeeping, not channel performance."""
import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from aquapilot.metrics import summarize_estimates, summarize_prior_ranking, summarize_tensors
from aquapilot.provenance import PROTOCOL_ID, file_fingerprint, finish_run, fingerprint
from aquapilot.analysis import Run, analyze_results, conditional_seed_interval, paired_record, pool_pairs


def ri(values):
    values = np.asarray(values, dtype=float)
    return np.stack([values, np.zeros_like(values)], axis=-1)


def completed_fixture(root: Path, *, seed=17, environment='NOF', fraction=.25,
                      protocol='same-test-protocol', bad_frame_hash=False):
    root.mkdir(parents=True)
    cfg = {'seed': seed, 'pilot': {'ratio': fraction, 'update_noise_convention': 'component'},
           'data': {'kind': 'synthetic'}, 'output_dir': str(root)}
    manifest = dict(protocol_id=PROTOCOL_ID, protocol_fingerprint=protocol,
                    run_mode='fresh_training', config_fingerprint=fingerprint(cfg),
                    evaluator_source_fingerprint='test-source', runtime_fingerprint='test-runtime',
                    data_fingerprint='fixed-synthetic-test-data', test_sequence_fingerprint='fixed-test-frames',
                    checkpoint_fingerprint='none', resolved_config=cfg, source={'git_commit': None})
    (root/'run_manifest.json').write_text(json.dumps(manifest))
    (root/'resolved_config.json').write_text(json.dumps(cfg))
    result = {'artifact_status': 'synthetic unit-test fixture, not a benchmark result', 'closed_loop': {}}
    physical_target = ri([[1, 2, 3], [3, 2, 1]])
    for i, policy in enumerate(('comb', 'random', 'uncertainty', 'active')):
        # Different history normalizations retain identical physical targets.
        scale = np.array([1., 2.])*(i+1)
        target = physical_target/scale[:, None, None]
        mean = target + (i+1)*.1/scale[:, None, None]
        sigma = np.abs(target)+.2
        mask = np.array([[1, 0, 0], [0, 1, 0]], bool)
        posterior = summarize_tensors(mean, sigma, target, observed_mask=mask, normalization_scale=scale)
        prior = summarize_tensors(mean+.4, sigma, target, observed_mask=mask, normalization_scale=scale)
        ranking = summarize_prior_ranking(mean+.4, sigma, target)
        baseline = summarize_estimates(target+.25, target, observed_mask=mask, normalization_scale=scale)
        record = dict(record_id='actual-test-record', environment=environment, source='synthetic-fixture',
                      noise_seed=seed, num_frames=2, first_frame_index=16, frame_start=0, frame_stop=2,
                      frame_identity_sha256=fingerprint(['actual-test-record', [16, 17]]),
                      prior=prior, posterior=posterior, prior_ranking=ranking,
                      same_information_baselines={name:baseline for name in ('zero','persistence_prior','current_interpolation','persistence_innovation')})
        if bad_frame_hash and policy == 'uncertainty':
            record['frame_identity_sha256'] = 'f'*64
        result['closed_loop'][policy] = dict(prior=prior, posterior=posterior,
            prior_ranking=ranking, same_information_baselines=record['same_information_baselines'], records=[record])
    return finish_run({'output_dir':root, 'manifest':manifest, 'cached_result':None}, result)


def test_end_to_end_uses_record_identity_not_policy_normalization(tmp_path):
    completed_fixture(tmp_path/'runs'/'seed17', seed=17)
    completed_fixture(tmp_path/'runs'/'seed37', seed=37)
    result = analyze_results([tmp_path/'runs'], tmp_path/'analysis')
    assert result['num_runs'] == 2
    assert result['num_environment_seed_budget_cells'] == 2
    paired = json.loads((tmp_path/'analysis'/'paired_summary.json').read_text())
    assert len(paired) == 3
    assert all(row['conditional_seed_interval']['degrees_of_freedom'] == 1 for row in paired)
    assert (tmp_path/'analysis'/'baseline_comparisons.csv').is_file()
    assert 'mean_nmse_above_zero_estimate' not in (tmp_path/'analysis'/'error_flags.csv').read_text()


def test_seed_blocks_not_twelve_environment_replicates():
    result = conditional_seed_interval({17:-.3, 37:.1, 57:-.4})
    assert result['num_seed_blocks'] == 3
    assert result['degrees_of_freedom'] == 2
    assert result['mean_delta_db'] == pytest.approx(-.2)
    assert 'No environment-population inference' in result['interval_scope']
    assert conditional_seed_interval({17:1})['t_lower'] is None


def test_old_and_new_protocols_are_not_silently_mixed(tmp_path):
    completed_fixture(tmp_path/'new')
    (tmp_path/'old').mkdir()
    (tmp_path/'old'/'metrics.json').write_text('{"closed_loop": {}}')
    with pytest.raises(ValueError, match='incompatible result schema or protocol'):
        analyze_results([tmp_path], tmp_path/'analysis')


def test_distinct_new_protocols_and_modified_summaries_fail(tmp_path):
    completed_fixture(tmp_path/'runs'/'a', protocol='A')
    completed_fixture(tmp_path/'runs'/'b', protocol='B')
    with pytest.raises(ValueError, match='Mixed protocol_fingerprint'):
        analyze_results([tmp_path/'runs'], tmp_path/'analysis')
    path = tmp_path/'runs'/'a'/'metrics.json'
    with path.open('a') as stream:
        stream.write(' ')
    with pytest.raises(ValueError, match='modified completed artifact'):
        analyze_results([tmp_path/'runs'/'a'], tmp_path/'analysis')


def test_paired_frame_identity_mismatch_is_an_error(tmp_path):
    completed_fixture(tmp_path/'run', bad_frame_hash=True)
    with pytest.raises(ValueError, match='unmatched frame_identity'):
        analyze_results([tmp_path/'run'], tmp_path/'analysis')


def test_incomplete_watermark_matrix_requires_explicit_provisional_mode(tmp_path):
    completed_fixture(tmp_path/'run')
    with pytest.raises(ValueError, match='Incomplete Watermark matrix'):
        analyze_results([tmp_path/'run'], tmp_path/'analysis', expect_watermark_matrix=True)
    result = analyze_results([tmp_path/'run'], tmp_path/'analysis', expect_watermark_matrix=True, allow_incomplete=True)
    assert result['matrix_complete'] is False
    assert len(result['missing_cells']) == 35


def test_partial_validity_pairs_only_identical_finite_frames(tmp_path):
    artifacts, records = {}, []
    for name, values in (('comb', [1., np.nan, 3.]), ('top', [.5, 2., np.nan])):
        path = tmp_path/(name+'.npz')
        np.savez_compressed(path, frame_index=np.array([16,17,18]), posterior_nmse_db=np.array(values))
        artifacts[path.name] = file_fingerprint(path)
        records.append(dict(record_id='r', num_frames=3, noise_seed=17, frame_identity_sha256='a'*64,
                            posterior={'num_valid_error_frames':2}, arrays_file=path.name))
    run = Run(tmp_path/'metrics.json', {}, {}, {'artifacts':artifacts}, 17, .25)
    result = paired_record(run, *records)
    assert result['num_matched_valid_frames'] == 1
    assert result['num_excluded_frames'] == 2
    assert result['delta_db'] == pytest.approx(-.5)


def test_no_saved_frames_does_not_guess_a_pair_from_different_valid_subsets(tmp_path):
    record = dict(record_id='r', num_frames=3, noise_seed=17, frame_identity_sha256='a'*64,
                  posterior={'num_valid_error_frames':2,'nmse_db_mean':1.})
    run = Run(tmp_path/'metrics.json', {}, {}, {}, 17, .25)
    result = paired_record(run, record, copy.deepcopy(record))
    assert result['delta_db'] is None
    assert 'unavailable' in result['pairing_status']


def test_compact_and_full_prediction_shards_give_identical_pairs(tmp_path):
    from aquapilot.metrics import framewise_diagnostics
    target = ri([[0, 0], [1, 2], [2, 1]])
    scale = np.array([1., 3., 2.])
    estimates = {'comb':target+ri([[1, 1], [.1, .2], [.2, .1]]),
                 'top':target+ri([[2, 2], [.2, .1], [.1, .2]])}
    results = []
    for storage in ('compact', 'full'):
        folder = tmp_path/storage
        folder.mkdir()
        artifacts, records = {}, []
        for policy, mean in estimates.items():
            path = folder/(policy+'.npz')
            diagnostics = framewise_diagnostics(mean, None, target, normalization_scale=scale)
            arrays = {'frame_index':np.array([16,17,18]), 'history_scale':scale}
            if storage == 'compact':
                arrays['posterior_nmse_db'] = diagnostics['nmse_db']
            else:
                arrays.update(posterior_mean=mean, target=target)
            np.savez_compressed(path, **arrays)
            artifacts[path.name] = file_fingerprint(path)
            records.append(dict(record_id='r',num_frames=3,noise_seed=17,
                frame_identity_sha256='a'*64, posterior={'num_valid_error_frames':2},arrays_file=path.name))
        run = Run(folder/'metrics.json', {}, {}, {'artifacts':artifacts},17,.25)
        results.append(paired_record(run,*records))
    assert results[0] == results[1]
    assert results[0]['num_excluded_frames'] == 1


def test_same_record_names_do_not_hide_changed_test_data(tmp_path):
    completed_fixture(tmp_path/'runs'/'a',seed=17)
    completed_fixture(tmp_path/'runs'/'b',seed=37)
    # Re-finalize a self-consistent provenance with a different actual input
    # fingerprint. Matching display record names do not make these datasets equal.
    root = tmp_path/'runs'/'b'
    manifest = json.loads((root/'run_manifest.json').read_text())
    manifest['test_sequence_fingerprint'] = 'changed-test-data'
    result = json.loads((root/'metrics.json').read_text())
    (root/'run_manifest.json').write_text(json.dumps(manifest))
    finish_run({'output_dir':root,'manifest':manifest,'cached_result':None},result)
    with pytest.raises(ValueError,match='test record/frame identities differ'):
        analyze_results([tmp_path/'runs'],tmp_path/'analysis')


def test_stage_matched_baselines_and_joint_density_report(tmp_path):
    root = tmp_path/'runs'
    completed_fixture(root/'one')
    output = tmp_path/'analysis'
    analyze_results([root], output)
    raw = json.loads((root/'one/metrics.json').read_text())['closed_loop']['uncertainty']
    with (output/'baseline_comparisons.csv').open() as stream:
        rows = {r['baseline']: r for r in csv.DictReader(stream) if r['policy'] == 'uncertainty'}
    prior, post = raw['prior']['nmse_db_mean'], raw['posterior']['nmse_db_mean']
    assert prior != pytest.approx(post)
    for name, row in rows.items():
        stage = 'prior' if name == 'persistence_prior' else 'posterior'
        assert row['receiver_stage'] == stage
        assert float(row['receiver_minus_baseline_db']) == pytest.approx(
            raw[stage]['nmse_db_mean'] - raw['same_information_baselines'][name]['nmse_db_mean'])
    with (output/'ranking_placement_by_environment.csv').open() as stream:
        joined, = csv.DictReader(stream)
    assert (joined['environment'], float(joined['pilot_fraction'])) == ('NOF', .25)
    assert float(joined['equal_seed_mean_frame_spearman']) == pytest.approx(raw['prior_ranking']['mean_frame_spearman'])
    with (output/'paired_top_b_025.csv').open() as stream:
        pair, = csv.DictReader(stream)
    assert float(joined['mean_delta_db']) == pytest.approx(float(pair['delta_db']))


def test_a_verified_empty_record_does_not_erase_other_matched_frames(tmp_path):
    run = Run(tmp_path/'metrics.json', {}, {}, {}, 17, .25)
    def record(name, n, valid, mean):
        return dict(record_id=name, num_frames=n, noise_seed=17, frame_identity_sha256='a'*64,
                    posterior={'num_valid_error_frames':valid, 'nmse_db_mean':mean})
    a = paired_record(run, record('valid',2,2,3.), record('valid',2,2,2.))
    b = paired_record(run, record('empty',3,0,None), record('empty',3,2,4.))
    pooled = pool_pairs([a, b])
    assert pooled['support_known'] and pooled['num_excluded_frames'] == 3
    assert pooled['num_matched_valid_frames'] == 2 and pooled['delta_db'] == pytest.approx(-1.)
    unknown = paired_record(run, record('unavailable',3,2,1.), record('unavailable',3,2,2.))
    assert pool_pairs([a, unknown])['delta_db'] is None


def test_explicit_file_must_be_the_hashed_metric_file(tmp_path):
    completed_fixture(tmp_path/'run')
    other = tmp_path/'run'/'unverified.json'
    other.write_bytes((other.parent/'metrics.json').read_bytes())
    with pytest.raises(ValueError, match='expected the completed metrics.json'):
        analyze_results([other], tmp_path/'analysis')
