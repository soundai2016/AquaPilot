"""Test the corrected handoff contract with generated numbers in a temporary directory.

The fixture is a schema/execution control: it contains no measured response, run
output, or numbers copied from an experiment. Nothing it generates is published.
"""
from collections import defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from scripts.post_corrected import add_corrected_files
from aquapilot.verification import (FORMAL_MEMBERS, FRAMES, RECORDS, SPLIT_SIZES,
                                       ENVS, SEEDS, BUDGETS, POLICIES, BASELINES, PROTOCOL)


def digest(x):
    return hashlib.sha256(x).hexdigest()


def dump(rows):
    io_ = io.StringIO()
    writer=csv.DictWriter(io_,fieldnames=list(rows[0]),lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    return io_.getvalue().encode()


def controls(folder, *, old=False, tamper=False):
    data={k:b'generated test control\n' for k in FORMAL_MEMBERS}; tables=defaultdict(list)
    source=[]; summaries=[]; deltas={'comb':0,'random':1,'uncertainty':-2,'active':-.5}
    for b in BUDGETS:
        f=b/128
        for policy in POLICIES:
            delta=deltas[policy]; nmse=-3+delta
            tables['aggregate_metrics'].append(dict(pilot_fraction=f,policy=policy,
                equal_environment_seed_nmse_db=nmse,descriptive_sd_nmse_db=0,
                equal_environment_seed_channel_nrmse_percent=50,frame_weighted_nmse_db=nmse,
                num_environment_seed_cells=12,num_valid_frames=3*sum(FRAMES.values())))
            if policy!='comb':
                summaries.append(dict(pilot_fraction=f,policy=policy,complete_environment_seed_grid=True,
                    improved_pair_count=12 if delta<0 else 0,num_paired_outcomes=12,equal_environment_equal_seed_delta_db=delta,
                    conditional_seed_interval=dict(degrees_of_freedom=2,num_seed_blocks=3,
                        seed_block_means_db={s:delta for s in SEEDS},t_lower=delta,t_upper=delta)))
            for env in ENVS:
                n=FRAMES[env]
                common=dict(environment=env,pilot_fraction=f,policy=policy)
                ranks=dict(num_valid_frames=3*n,num_undefined_frames=0,num_seeds=3,num_seeds_with_defined_ranks=3,equal_seed_mean_frame_spearman=.25)
                tables['prior_ranking_by_environment'].append({**common,**ranks})
                if policy!='comb':
                    tables['paired_by_environment'].append({**common,'num_seeds':3,'improved_seed_count':3 if delta<0 else 0,'mean_delta_db':delta})
                if policy=='uncertainty':
                    tables['ranking_placement_by_environment'].append({**common,**ranks,'num_frames':3*n,'mean_delta_db':delta,'undefined_delta_seed_count':0})
                for s in SEEDS:
                    h=digest(f'FAKE-TEST-{env}-{b}-{s}'.encode())
                    ident=dict(environment=env,seed=s,pilot_fraction=f,metrics_sha256=h)
                    if policy=='comb':source.append(ident)
                    run={**ident,'run_mode':'fresh_training','policy':policy,'num_frames':n}
                    tables['configuration_metrics'].append({**run,'num_valid_error_frames':n,'num_valid_coverage_frames':n,'nmse_db_mean':nmse,'channel_nrmse_percent_mean':50,'coverage_90':.8})
                    for stage in ('prior','posterior'):
                        tables['calibration'].append({**run,'stage':stage,'num_valid_coverage_frames':n,'nominal_coverage':.9,'coverage_90':.8})
                    for baseline in BASELINES:
                        prior=baseline=='persistence_prior'
                        value=nmse+1 if prior else nmse
                        tables['baseline_comparisons'].append({**run,'baseline':baseline,'receiver_stage':'prior' if prior else 'posterior','support_known':True,'num_matched_valid_frames':n,'num_excluded_frames':0,'receiver_num_valid_frames':n,'baseline_num_valid_frames':n,'receiver_nmse_db':value,'baseline_nmse_db':0,'receiver_minus_baseline_db':value})
                    tables['placement_diagnostics'].append({**run,'num_mask_frames':n,'num_records':RECORDS[env],'pilot_budget':b,'n_subcarriers':128,'distance_sum_bins':128*n,'mean_nearest_pilot_distance_bins':1})
                    tables['prior_ranking_by_run'].append({**run,'num_valid_frames':n,'num_undefined_frames':0,'num_records':RECORDS[env],'mean_frame_spearman':.25})
                    if policy!='comb':
                        pair={**run,'reference_policy':'comb','support_known':True,'num_matched_valid_frames':n,'num_excluded_frames':0,'comb_nmse_db':-3,'policy_nmse_db':nmse,'delta_db':delta}
                        tables['paired_differences'].append(pair)
                        if policy=='uncertainty' and b==32:tables['paired_top_b_025'].append(pair)
    for k,rows in tables.items():data[k+'.csv']=dump(rows)
    data['paired_summary.json']=json.dumps(summaries).encode()
    if old:
        from aquapilot.verification import OLDER_EXTRAS
        data.update({name: b"archived auxiliary file" for name in OLDER_EXTRAS})
    if tamper:
        records = list(csv.DictReader(io.StringIO(data['paired_top_b_025.csv'].decode())))
        records[0]['delta_db'] = '0'
        data['paired_top_b_025.csv'] = dump(records)
    folds={g:{s:{'num_files':n,'ordered_input_fingerprint':digest(f'FAKE-{g}{s}'.encode())}
                for s,n in zip(('train','val','test'),SPLIT_SIZES[g])} for g in ENVS}
    manifest=dict(bundle_schema_version=1 if old else 2,protocol_id=PROTOCOL,classification='complete measured Watermark fresh-training analysis',
        input_classification='declared measured channels',run_mode='fresh_training',num_runs=36,includes_result_tables=True,
        num_environment_seed_budget_cells=36,missing_cells=[],protocol_fingerprint=digest(b'test'),source_analysis_summary_sha256=digest(b'test summary'),
        source_cells=source,prepared_input_manifest_by_fold=folds,
        members={k:{'sha256':digest(v),'size_bytes':len(v)} for k,v in data.items()})
    source_zip=folder/'test_only.zip'; axes=folder/'test_only.json'
    with ZipFile(source_zip,'x') as z:
        for k,v in data.items():z.writestr(k,v)
        z.writestr('bundle_manifest.json',json.dumps(manifest))
    receipt=dict(status='ok',method='independent_delay_axis_fft',frequency_grid='uniform_inner_probe_band',
        band_fraction=.8,n_subcarriers=128,history=16,horizon=1,num_unique_records=156,total_prepared_files=624,
        ordered_split_fingerprints={g:{s:v['ordered_input_fingerprint'] for s,v in splits.items()} for g,splits in folds.items()},
        environments=[dict(environment=g,scored=FRAMES[g],records=RECORDS[g],snapshots=FRAMES[g]+16*RECORDS[g],max_error_over_peak=0) for g in ENVS])
    axes.write_text(json.dumps(receipt))
    return source_zip,axes


class CorrectedHandoff(unittest.TestCase):
    def test_bind_all_input_splits_and_omit_auxiliary_files(self):
        for old in (False, True):
            with self.subTest(old=old), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                archive, axes = controls(root, old=old)
                out = add_corrected_files(archive, axes, root/'stage')
                self.assertEqual(set(out), {'axes.json', 'results.zip'})
                self.assertEqual(out['axes.json'].read_bytes(), axes.read_bytes())
                with ZipFile(out['results.zip']) as z:
                    self.assertEqual(set(z.namelist()), FORMAL_MEMBERS | {'bundle_manifest.json'})
                    manifest = json.loads(z.read('bundle_manifest.json'))
                    self.assertEqual(manifest['bundle_schema_version'], 2)
                    self.assertEqual(set(manifest['members']), FORMAL_MEMBERS)
                wrong = json.loads(axes.read_text())
                wrong['ordered_split_fingerprints']['BCH']['val'] = 64*'0'
                axes.write_text(json.dumps(wrong))
                with self.assertRaisesRegex(ValueError, 'trained inputs differ'):
                    add_corrected_files(archive, axes, root/'test_bad')
                self.assertFalse((root/'test_bad').exists())

    def test_rehashed_but_inconsistent_pairs_fail(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            archive, axes = controls(root, tamper=True)
            with self.assertRaisesRegex(ValueError, 'paired difference'):
                add_corrected_files(archive, axes, root/'stage')
            self.assertFalse((root/'stage').exists())
