"""Verify the Watermark experiment matrix, numerical contrasts and raw-to-CFR receipt."""
from __future__ import annotations

import csv
import hashlib
import io
import itertools
import json
import math
import re
import stat
import statistics
import zipfile

ENVS = ('NOF', 'NCS', 'BCH', 'KAU')
SEEDS = (17, 37, 57)
BUDGETS = (8, 16, 32)
POLICIES = ('comb', 'random', 'uncertainty', 'active')
CANDIDATES = POLICIES[1:]
BASELINES = ('persistence_prior', 'current_interpolation', 'persistence_innovation', 'zero')
RECORDS = dict(zip(ENVS, (60, 60, 4, 32)))
FRAMES = dict(zip(ENVS, (14520, 60480, 2264, 7744)))
SPLIT_SIZES = dict(zip(ENVS, ((36,60,60),(92,4,60),(120,32,4),(64,60,32))))
CELLS = tuple(itertools.product(ENVS, SEEDS, BUDGETS))
from aquapilot.provenance import PROTOCOL_ID as PROTOCOL
FRACTIONS = {b/128:b for b in BUDGETS}
SHA256 = re.compile(r'[0-9a-f]{64}\Z')
mean = statistics.mean
FORMAL_MEMBERS = {
    'aggregate_metrics.csv', 'configuration_metrics.csv', 'paired_top_b_025.csv',
    'paired_differences.csv', 'paired_by_environment.csv', 'paired_summary.json',
    'prior_ranking_by_environment.csv', 'prior_ranking_by_run.csv', 'baseline_comparisons.csv',
    'calibration.csv', 'error_flags.csv', 'summary.txt', 'ranking_placement_by_environment.csv',
    'placement_diagnostics.csv',
}


# Pre-1.1 corrected bundles included derived documents. Verify and omit them on export.
OLDER_EXTRAS = {
    'rank_placement.pdf', 'tables/main_results.tex', 'tables/paired_top_b_025.tex',
    'tables/ranking_by_environment.tex', 'tables/table_notes.txt',
}

def require(condition, message):
    if not condition:
        raise ValueError(message)


def numeric(value):
    if value is None or isinstance(value, bool):
        raise ValueError(f'Expected a finite number, found {value!r}')
    x = float(value)
    require(math.isfinite(x), 'A nonfinite number was supplied')
    return x


def integer(value):
    number = numeric(value)
    require(number.is_integer(), f'Expected integer: {value!r}')
    return int(number)


def same(x, y, what):
    require(math.isclose(numeric(x), numeric(y), rel_tol=5e-10, abs_tol=1e-9),
            f'Inconsistent {what}: {x} versus {y}')


def nullable_spearman(x):
    if x is None or x == '':
        return None
    x = numeric(x)
    require(abs(x) <= 1+1e-12, 'Correlation outside [-1, 1]')
    return x


def objects(items):
    d = {}
    for k,v in items:
        require(k not in d, f'Duplicate JSON key: {k}')
        d[k] = v
    return d


def read_json(data):
    return json.loads(data, object_pairs_hook=objects,
                      parse_constant=lambda s: (_ for _ in ()).throw(ValueError('Invalid JSON number: '+s)))


def index(rows, key, expected, label):
    out = {}
    for r in rows:
        k = key(r)
        require(k not in out, f'{label} contains a duplicate: {k}')
        out[k] = r
    require(set(out) == set(expected), f'Incomplete or unexpected cells in {label}')
    return out


def budget(row):
    try:
        return FRACTIONS[numeric(row['pilot_fraction'])]
    except KeyError as exc:
        raise ValueError('Expected a pilot fraction of 1/16, 1/8 or 1/4') from exc


def cell(r):
    return r['environment'], integer(r['seed']), budget(r)


def key(r):
    return (*cell(r), r['policy'])


def digest(data):
    return hashlib.sha256(data).hexdigest()


def bundle(path):
    with zipfile.ZipFile(path) as z:
        entries = z.infolist()
        require(len(set(z.namelist()))==len(entries) and all(not it.is_dir() for it in entries),
                'Duplicate or directory entries in the results ZIP')
        require(sum(it.file_size for it in entries) <= 50_000_000 and
                all(it.file_size <= 16_000_000 and not it.flag_bits & 1 and
                    not stat.S_ISLNK(it.external_attr >> 16) for it in entries),
                'Expected bounded, ordinary aggregate results files')
        require('bundle_manifest.json' in z.namelist(), 'Missing bundle manifest')
        m = read_json(z.read('bundle_manifest.json'))
        require(m.get('protocol_id') == PROTOCOL,
                'This archive does not use the corrected delay-axis / in-band Watermark protocol.')
        require(m['classification']=='complete measured Watermark fresh-training analysis'
                and m['input_classification']=='declared measured channels'
                and m['run_mode']=='fresh_training' and m['includes_result_tables'] is True
                and m['num_runs']==m['num_environment_seed_budget_cells']==36 and m['missing_cells']==[],
                'Expected a complete, measured, fresh-training matrix of 36 fits')
        require(SHA256.fullmatch(m['protocol_fingerprint']) and SHA256.fullmatch(m['source_analysis_summary_sha256']),
                'Invalid provenance digest')
        schema = m.get('bundle_schema_version')
        members = set(m['members'])
        allowed = FORMAL_MEMBERS | OLDER_EXTRAS if schema == 1 else FORMAL_MEMBERS
        require(schema in (1, 2) and members == set(z.namelist()) - {'bundle_manifest.json'}
                and FORMAL_MEMBERS <= members <= allowed,
                'Unexpected or missing aggregate results ZIP member')
        data = {}
        for name, spec in m['members'].items():
            value = z.read(name)
            require(len(value)==spec['size_bytes'] and digest(value)==spec['sha256'], 'Failed archive check: '+name)
            if name in FORMAL_MEMBERS:
                data[name] = value
        return m, data


def verify_receipt(axes):
    """Check the complete independent raw-to-CFR audit and twelve ordered input fingerprints."""
    require(axes.get('status')=='ok' and axes.get('method')=='independent_delay_axis_fft'
            and axes.get('frequency_grid')=='uniform_inner_probe_band', 'Incorrect source-to-CFR audit method or status')
    require(axes['band_fraction']==.8 and axes['n_subcarriers']==128 and axes['history']==16
            and axes['horizon']==1 and axes['num_unique_records']==156 and axes['total_prepared_files']==624,
            'Source-to-CFR audit has unexpected data dimensions or counts')
    envrows = index(axes['environments'], lambda r:r['environment'], ENVS, 'raw-to-CFR audit')
    require(set(axes['ordered_split_fingerprints'])==set(ENVS),
            'Missing or unexpected fold fingerprints')
    for g in ENVS:
        row = envrows[g]
        require(row['records']==RECORDS[g] and row['scored']==FRAMES[g] and
                row['snapshots']==FRAMES[g]+16*RECORDS[g] and 0<=numeric(row['max_error_over_peak'])<=4*2**-23,
                f'{g}: raw time-snapshot count or independent transform check failed')
        raw = axes['ordered_split_fingerprints'][g]
        require(set(raw) == {'train', 'val', 'test'} and
                all(isinstance(h, str) and SHA256.fullmatch(h) for h in raw.values()),
                f'{g}: invalid split fingerprints')


def verify_axes(m, axes):
    """Bind each seed's training fold inventory to the source-data verification."""
    verify_receipt(axes)
    fitted = m['prepared_input_manifest_by_fold']
    require(set(fitted) == set(ENVS), 'Missing or unexpected fitted folds')
    for g in ENVS:
        manifest = fitted[g]
        raw = axes['ordered_split_fingerprints'][g]
        require(set(manifest) == set(raw), f'{g}: missing fitted split fingerprints')
        for part, n in zip(('train', 'val', 'test'), SPLIT_SIZES[g]):
            require(raw[part] == manifest[part]['ordered_input_fingerprint'] and manifest[part]['num_files'] == n,
                    f'{g}/{part}: trained inputs differ from the raw-verified prepared split')


def csvrows(data, stem):
    reader = csv.DictReader(io.StringIO(data[stem+'.csv'].decode('utf-8-sig'), newline=''))
    require(reader.fieldnames and len(set(reader.fieldnames))==len(reader.fieldnames), f'{stem}: CSV header')
    rows = list(reader)
    require(all(None not in r and None not in r.values() for r in rows), f'{stem}: malformed row')
    return rows


def t_interval(blocks):
    # For df=2, F(t)=1/2+t/(2*sqrt(2+t^2)).
    require(len(blocks)==3, 'The paired interval requires three complete joint-seed blocks')
    d = mean(blocks)
    half = math.sqrt(2*.95**2/(1-.95**2))*statistics.stdev(blocks)/math.sqrt(3)
    return d-half,d+half


def verify_numeric_tables(m, data):
    """Recompute aggregate metrics and paired statistics from their configuration rows."""
    source = index(m['source_cells'], cell, CELLS, 'source fits')
    hashes={k:v['metrics_sha256'] for k,v in source.items()}
    require(len(set(hashes.values()))==36 and all(SHA256.fullmatch(h) for h in hashes.values()), 'Invalid or duplicate fit digests')

    def per_run(stem, keyfn, expected):
        rows=csvrows(data, stem)
        for row in rows:
            require(row['metrics_sha256']==hashes.get(cell(row)) and row['run_mode']=='fresh_training',
                    stem+': row does not belong to its measured training run')
            require(integer(row['num_frames'])==FRAMES[row['environment']],stem+': incorrect scored snapshot count')
        return index(rows,keyfn,expected,stem)

    all4=tuple((*c,p) for c in CELLS for p in POLICIES)
    cfg=per_run('configuration_metrics',key,all4)
    rec=per_run('baseline_comparisons',lambda r:(*key(r),r['baseline']),[(*k,a) for k in all4 for a in BASELINES])
    cal=per_run('calibration',lambda r:(*key(r),r['stage']),[(*k,s) for k in all4 for s in ('prior','posterior')])
    ranks=per_run('prior_ranking_by_run',key,all4)
    geo=index(csvrows(data,'placement_diagnostics'),key,all4,'pilot geometry')
    agg=index(csvrows(data,'aggregate_metrics'),lambda r:(budget(r),r['policy']),itertools.product(BUDGETS,POLICIES),'aggregate metrics')
    fullpair=per_run('paired_differences',key,[(*c,p) for c in CELLS for p in CANDIDATES])
    quarter=per_run('paired_top_b_025',key,[(e,s,32,'uncertainty') for e in ENVS for s in SEEDS])
    reports=index(read_json(data['paired_summary.json']),lambda r:(budget(r),r['policy']),itertools.product(BUDGETS,CANDIDATES),'paired intervals')
    envpairs=index(csvrows(data,'paired_by_environment'),lambda r:(r['environment'],budget(r),r['policy']),itertools.product(ENVS,BUDGETS,CANDIDATES),'environment differences')
    renv=index(csvrows(data,'prior_ranking_by_environment'),lambda r:(r['environment'],budget(r),r['policy']),itertools.product(ENVS,BUDGETS,POLICIES),'environment ranks')
    joint=index(csvrows(data,'ranking_placement_by_environment'),lambda r:(r['environment'],budget(r)),itertools.product(ENVS,BUDGETS),'rank and placement')

    post={}; rho={}; validrank={}; u={}
    for k in all4:
        e,s,b,p=k; n=FRAMES[e]; row=cfg[k]
        require(integer(row['num_valid_error_frames'])==integer(row['num_valid_coverage_frames'])==n,
                'Verified numerical tables require full, common error and coverage support')
        post[k]=numeric(row['nmse_db_mean'])
        require(numeric(row['channel_nrmse_percent_mean'])>=0,'Negative channel NRMSE')
        for stage in ('prior','posterior'):
            cr=cal[k+(stage,)];require(integer(cr['num_valid_coverage_frames'])==n and numeric(cr['nominal_coverage'])==.9 and 0<=numeric(cr['coverage_90'])<=1,'Invalid coverage support or level')
        same(cal[k+('posterior',)]['coverage_90'],row['coverage_90'],'posterior coverage')
        for method in BASELINES:
            rr=rec[k+(method,)];stage='prior' if method=='persistence_prior' else 'posterior'
            require(rr['receiver_stage']==stage and rr['support_known']=='True' and integer(rr['num_excluded_frames'])==0
                    and all(integer(rr[col])==n for col in ('num_matched_valid_frames','receiver_num_valid_frames','baseline_num_valid_frames')),
                    f'Unmatched reference observations: {k}, {method}')
            predicted=numeric(rr['receiver_nmse_db']);base=numeric(rr['baseline_nmse_db'])
            same(rr['receiver_minus_baseline_db'],predicted-base,'reference contrast')
            if stage=='posterior': same(post[k],predicted,'stage-matched posterior')
            if method=='zero': same(base,0,'zero-channel reference')
        gr=geo[k]
        require(gr['metrics_sha256']==hashes[e,s,b] and gr['run_mode']=='fresh_training' and
                integer(gr['num_mask_frames'])==n and integer(gr['num_records'])==RECORDS[e] and
                integer(gr['pilot_budget'])==b and integer(gr['n_subcarriers'])==128,'Incorrect pilot geometry')
        distance=integer(gr['distance_sum_bins'])/(128*n)
        same(distance,gr['mean_nearest_pilot_distance_bins'],'nearest pilot distance')
        rr=ranks[k];validrank[k]=integer(rr['num_valid_frames']);u[k]=integer(rr['num_undefined_frames'])
        require(integer(rr['num_records'])==RECORDS[e] and 0<=validrank[k]<=n and validrank[k]+u[k]==n,'Inconsistent rank support')
        rho[k]=nullable_spearman(rr['mean_frame_spearman'])
        require((rho[k] is None)==(validrank[k]==0),'Undefined rank/support disagreement')

    deltas={}; site={}; site_rho={}
    for b,p in itertools.product(BUDGETS,POLICIES):
        ks=[(e,s,b,p) for e,s in itertools.product(ENVS,SEEDS)]
        ar=agg[b,p]; vals=[post[k] for k in ks]
        same(ar['equal_environment_seed_nmse_db'],mean(vals),'equal-cell NMSE')
        same(ar['descriptive_sd_nmse_db'],statistics.pstdev(vals),'descriptive SD')
        same(ar['equal_environment_seed_channel_nrmse_percent'],mean(numeric(cfg[k]['channel_nrmse_percent_mean']) for k in ks),'channel NRMSE')
        weight=sum(FRAMES[k[0]] for k in ks)
        same(ar['frame_weighted_nmse_db'],sum(FRAMES[k[0]]*post[k] for k in ks)/weight,'snapshot-weighted NMSE')
        require(integer(ar['num_environment_seed_cells'])==12 and integer(ar['num_valid_frames'])==weight,'Incomplete aggregate support')
        for e in ENVS:
            ek=[(e,s,b,p) for s in SEEDS];rr=renv[e,b,p]; valid=sum(validrank[k] for k in ek)
            require(integer(rr['num_valid_frames'])==valid and integer(rr['num_undefined_frames'])==sum(u[k] for k in ek)
                    and integer(rr['num_seeds'])==3 and integer(rr['num_seeds_with_defined_ranks'])==sum(rho[k] is not None for k in ek),'Invalid site rank counts')
            expected=mean(rho[k] for k in ek) if all(rho[k] is not None for k in ek) else None
            reported=nullable_spearman(rr['equal_seed_mean_frame_spearman'])
            if expected is None:require(reported is None,'Undefined site rank presented as a number')
            else:same(reported,expected,'site rank mean')
            if p=='uncertainty':site_rho[e,b]=expected
        if p=='comb':continue
        for k in ks:
            e,s,_,_=k;d=post[k]-post[e,s,b,'comb'];deltas[k]=d
            for pr in [fullpair[k]]+([quarter[k]] if p=='uncertainty' and b==32 else []):
                require(pr['reference_policy']=='comb' and pr['support_known']=='True' and
                        integer(pr['num_matched_valid_frames'])==FRAMES[e] and integer(pr['num_excluded_frames'])==0,
                        'Incomplete matched-frame support')
                same(pr['comb_nmse_db'],post[e,s,b,'comb'],'paired comb');same(pr['policy_nmse_db'],post[k],'paired policy');same(pr['delta_db'],d,'paired difference')
        for e in ENVS:
            vals=[deltas[e,s,b,p] for s in SEEDS]; ep=envpairs[e,b,p]
            site[e,b,p]=mean(vals);same(ep['mean_delta_db'],mean(vals),'environment contrast')
            require(integer(ep['num_seeds'])==3 and integer(ep['improved_seed_count'])==sum(x<0 for x in vals),'Invalid environment pair counts')
        means={s:mean(deltas[e,s,b,p] for e in ENVS) for s in SEEDS}
        blockmean=mean(means.values()); wins=sum(deltas[k]<0 for k in ks); bounds=t_interval(list(means.values()))
        pr=reports[b,p];ci=pr['conditional_seed_interval']
        require(pr['complete_environment_seed_grid'] is True and integer(pr['improved_pair_count'])==wins
                and integer(pr['num_paired_outcomes'])==12 and integer(ci['degrees_of_freedom'])==2 and integer(ci['num_seed_blocks'])==3,'Incomplete three-seed interval')
        for s in SEEDS:same(means[s],ci['seed_block_means_db'][str(s)],'joint-seed block')
        same(pr['equal_environment_equal_seed_delta_db'],blockmean,'mean paired difference')
        same(ci['t_lower'],bounds[0],'lower t endpoint');same(ci['t_upper'],bounds[1],'upper t endpoint')

    for (e,b),r in joint.items():
        require(r['policy']=='uncertainty' and integer(r['num_frames'])==3*FRAMES[e] and
                integer(r['num_undefined_frames'])==sum(u[e,s,b,'uncertainty'] for s in SEEDS),'Joint rank/placement support mismatch')
        same(r['mean_delta_db'],site[e,b,'uncertainty'],'rank/placement contrast')
        expected=site_rho[e,b];actual=nullable_spearman(r['equal_seed_mean_frame_spearman'])
        if expected is None:require(actual is None,'Joint rank/placement table rank should be undefined')
        else:same(actual,expected,'rank/placement correlation')
