"""Recover published-format tables from the pre-in-band Watermark experiment.

The old converter chose an axis by length and used a different frequency grid.
Internal agreement of its tables does not certify the physical channel axes.
"""
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
import tempfile
import zipfile
from pathlib import Path

from .export_io import _source_table, _formal_inputs

LEGACY_PROTOCOL = 'aquapilot_closed_loop'
_TABLES = {
    'aggregate_metrics.csv', 'configuration_metrics.csv', 'paired_differences.csv',
    'paired_by_environment.csv', 'paired_summary.json', 'paired_top_b_025.csv',
    'prior_ranking_by_run.csv', 'prior_ranking_by_environment.csv', 'calibration.csv',
    'baseline_comparisons.csv', 'error_flags.csv', 'summary.txt',
}
_EXTRA = {'ranking_placement_by_environment.csv', 'placement_diagnostics.csv'}
_OLD_PAPER = {'rank_placement.pdf', 'tables/main_results.tex', 'tables/paired_top_b_025.tex',
              'tables/ranking_by_environment.tex', 'tables/table_notes.txt'}
_RECORD_TABLES = ('paired_by_record.csv', 'prior_ranking_by_record.csv')
_POLS = {'comb', 'random', 'uncertainty', 'active'}
_ENVS = {'BCH', 'KAU', 'NOF', 'NCS'}
_EXPECTED = set(itertools.product(_ENVS, (17, 37, 57), (.0625, .125, .25)))
_SPLIT_COUNTS = {'BCH': (120,32,4), 'KAU': (64,60,32), 'NCS': (92,4,60), 'NOF': (36,60,60)}


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _rows(data: bytes) -> list[dict]:
    reader = csv.DictReader(io.StringIO(data.decode('utf-8')))
    if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
        raise ValueError('Missing or repeated columns in a legacy results table.')
    result = list(reader)
    if any(None in row or None in row.values() for row in result):
        raise ValueError('Malformed row in a legacy results table.')
    return result


def _cell(row: dict) -> tuple:
    return row['environment'], int(row['seed']), float(row['pilot_fraction'])


def _check_intervals(data: bytes, paired: dict) -> int:
    """Verify the reported fixed-environment t intervals from the three seed blocks."""
    summary = json.loads(data)
    keys = {(c[2],p) for c,p in paired}
    if not isinstance(summary,list) or len(summary) != len(keys) or {
            (r['pilot_fraction'],r['policy']) for r in summary} != keys:
        raise ValueError('Legacy paired summary differs from the paired table.')
    # The df=2 t distribution has F(x) = 1/2 + x/(2*sqrt(x*x+2)).
    t975 = math.sqrt(2*.95**2/(1-.95**2))
    def close(a,b):
        return a is None and b is None or (a is not None and b is not None and
                math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-9))
    for row in summary:
        f, pol = row['pilot_fraction'],row['policy']
        blocks = []
        ci = row['conditional_seed_interval']
        if ci['num_seed_blocks'] != 3 or set(ci['seed_block_means_db']) != {'17','37','57'}:
            raise ValueError('Unexpected seed blocks in legacy confidence interval.')
        for seed in (17,37,57):
            vals = [paired[((e,seed,f),pol)]['delta_db'] for e in sorted(_ENVS)]
            block = statistics.fmean(map(float,vals)) if all(v != '' for v in vals) else None
            if not close(block, ci['seed_block_means_db'][str(seed)]):
                raise ValueError('Legacy seed-block mean differs from the paired differences.')
            blocks.append(block)
        defined = None not in blocks
        m = statistics.fmean(blocks) if defined else None
        w = t975*statistics.stdev(blocks)/math.sqrt(3) if defined else None
        if (not close(m,row['equal_environment_equal_seed_delta_db']) or
                not close(m-w if defined else None,ci['t_lower']) or not close(m+w if defined else None,ci['t_upper'])):
            raise ValueError('Legacy confidence interval differs from the three paired seed blocks.')
    return len(summary)


def _matrix(payload: dict, manifest: dict) -> dict:
    if (manifest.get('protocol_id') != LEGACY_PROTOCOL or manifest.get('run_mode') != 'fresh_training'
            or manifest.get('input_classification') != 'declared measured channels'):
        raise ValueError('Expected measured-channel results from the aquapilot_closed_loop fresh-training protocol.')
    cells = {_cell(r): r['metrics_sha256'] for r in manifest['source_cells']}
    if (len(cells) != len(manifest['source_cells']) or cells.keys() != _EXPECTED
            or len(set(cells.values())) != 36 or manifest.get('num_runs') != 36
            or manifest.get('num_environment_seed_budget_cells') != 36 or manifest.get('missing_cells') != []
            or any(not re.fullmatch(r'[0-9a-f]{64}', h) for h in cells.values())):
        raise ValueError(f'Legacy run matrix is incomplete or inconsistent: missing cells {sorted(_EXPECTED - cells.keys())}.')
    folds = manifest['prepared_input_manifest_by_fold']
    if not isinstance(folds, dict) or set(folds) != _ENVS:
        raise ValueError('Missing prepared-input fingerprints for the legacy folds.')
    for env, counts in _SPLIT_COUNTS.items():
        for split, count in zip(('train','val','test'), counts):
            entry = folds[env][split]
            if entry.get('num_files') != count or not re.fullmatch(r'[0-9a-f]{64}', entry.get('ordered_input_fingerprint','')):
                raise ValueError(f'{env}/{split}: invalid legacy prepared-input receipt.')

    config = _rows(payload['configuration_metrics.csv'])
    policies = {(_cell(r),r['policy']): r for r in config}
    if (len(config) != 144 or set(policies) != set(itertools.product(cells, _POLS))
            or any(r['metrics_sha256'] != cells[_cell(r)] or r['run_mode'] != 'fresh_training' for r in config)):
        raise ValueError('Legacy configuration table does not match the declared source runs and four policies.')
    frames = {env: {int(r['num_frames']) for r in config if r['environment'] == env} for env in _ENVS}
    if any(len(ns) != 1 or min(ns) <= 0 for ns in frames.values()):
        raise ValueError('Legacy frame counts disagree across seeds, pilot fractions or policies within an environment.')
    pairs = _rows(payload['paired_differences.csv'])
    paired = {(_cell(r),r['policy']):r for r in pairs}
    if (len(pairs) != 108 or set(paired) != set(itertools.product(cells, _POLS-{'comb'}))
            or any(r['reference_policy'] != 'comb' or r['metrics_sha256'] != cells[_cell(r)] for r in pairs)):
        raise ValueError('Legacy paired table does not match the declared run/policy matrix.')
    for r in pairs:
        cell = _cell(r)
        total, matched, excluded = (int(r[f]) for f in ('num_frames','num_matched_valid_frames','num_excluded_frames'))
        if total != int(policies[(cell,r['policy'])]['num_frames']) or not 0 <= matched <= total or total-matched != excluded:
            raise ValueError('Invalid legacy paired frame counts.')
        if r['support_known'] not in {'True','False'}:
            raise ValueError('Invalid legacy pairing-support flag.')
        if not matched or r['support_known'] == 'False':
            if any(r[k] for k in ('policy_nmse_db','comb_nmse_db','delta_db')):
                raise ValueError('Legacy paired result estimated on empty or unknown joint support.')
            continue
        a, b, delta = (float(r[k]) for k in ('policy_nmse_db','comb_nmse_db','delta_db'))
        if not all(map(math.isfinite,(a,b,delta))) or not math.isclose(a-b,delta,rel_tol=1e-9,abs_tol=1e-9):
            raise ValueError('Legacy paired difference disagrees with the two matched errors.')
        if total == matched:
            for p, value in ((r['policy'],a),('comb',b)):
                if not math.isclose(float(policies[(cell,p)]['nmse_db_mean']),value,rel_tol=1e-9,abs_tol=1e-9):
                    raise ValueError('Paired and configuration NMSE disagree on identical frame support.')
    subset = _rows(payload['paired_top_b_025.csv'])
    if len(subset) != 12 or {(_cell(r),r['policy']) for r in subset} != {
            k for k in paired if k[0][2] == .25 and k[1] == 'uncertainty'} or any(
                paired[(_cell(r),r['policy'])] != r for r in subset):
        raise ValueError('Legacy quarter-budget Top-B table differs from the complete paired table.')
    return {
        'classification': 'legacy_diagnostic', 'protocol_id': LEGACY_PROTOCOL, 'num_runs': len(cells),
        'num_policy_rows': len(policies), 'num_paired_comparisons': len(pairs), 'num_quarter_budget_top_b_pairs': len(subset),
        'num_seed_intervals_checked': _check_intervals(payload['paired_summary.json'], paired),
        'prepared_input_manifest_by_fold': folds, 'source_cells': manifest['source_cells'],
        'scored_frames_per_configuration_by_environment': {e: next(iter(n)) for e,n in sorted(frames.items())},
        'missing_cells': [], 'checkpoint_weights_reopened': False,
    }


def _archive(path: Path) -> tuple[dict, dict]:
    with zipfile.ZipFile(path) as z:
        entries = z.infolist()
        names = {i.filename for i in entries}
        if (len(names) != len(entries) or 'bundle_manifest.json' not in names
                or names - (_TABLES | _EXTRA | _OLD_PAPER | {'bundle_manifest.json'})
                or not _TABLES <= names or sum(i.file_size for i in entries) > 64*1024**2
                or any(i.file_size > 16*1024**2 or stat.S_ISLNK(i.external_attr >> 16) or i.flag_bits & 1 for i in entries)):
            raise ValueError('Unknown, duplicate, missing, encrypted or oversized legacy ZIP members.')
        m = json.loads(z.read('bundle_manifest.json'))
        if not isinstance(m,dict) or not isinstance(m.get('members'),dict) or set(m['members']) != names-{'bundle_manifest.json'}:
            raise ValueError('Legacy ZIP inventory disagrees with its manifest.')
        payload = {}
        for name, receipt in m['members'].items():
            b = z.read(name)
            if receipt != {'size_bytes':len(b),'sha256':_digest(b)}:
                raise ValueError(f'Legacy archive checksum mismatch: {name}')
            if name in _TABLES | _EXTRA:
                payload[name] = b
    info = _matrix(payload, m)
    info.update(input_source='previous_results_archive', source_archive_sha256=_digest(path.read_bytes()),
                archive_members_verified=len(m['members']), record_tables_available=False,
                verification='ZIP member hashes/lengths, run/policy identities, paired NMSE arithmetic, frame counts and seed-block intervals checked. '
                             'The source run files and the training inputs were not reopened by archive verification.')
    return payload, info


def _record_table(path: Path, sources: dict) -> bytes:
    table = _rows(_source_table(path, sources))
    for row in table:
        key = (row['environment']+'\0'+row.pop('record_id')).encode()
        row['record_key'] = _digest(key)
        row.pop('frame_identity_sha256', None)
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(table[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(table)
    return stream.getvalue().encode()


def _tree(path: Path) -> tuple[dict, dict]:
    from .analysis import analyze_results

    with tempfile.TemporaryDirectory(prefix='aquapilot_legacy_') as temp:
        out = Path(temp)
        report = analyze_results([path], out, expect_watermark_matrix=True, expected_protocol=LEGACY_PROTOCOL)
        if (report['input_data_kinds'] != ['prepared']
                or report['declared_source_statuses'] != ['measured-channel experiment']
                or report['run_mode'] != 'fresh_training'):
            raise ValueError('The legacy diagnostic requires the complete measured Watermark training matrix.')
        sources = dict(zip(report['source_metrics'],report['source_summary_sha256'],strict=True))
        payload = {}
        for name in sorted(_TABLES | (_EXTRA & report['analysis_artifacts'].keys())):
            file = out/name
            payload[name] = _source_table(file,sources) if name.endswith('.csv') else file.read_bytes()
        cells = {(*_cell(r),r['metrics_sha256']) for r in _rows(payload['configuration_metrics.csv'])}
        manifest = {**report,'input_classification':'declared measured channels',
                    'source_cells':[dict(environment=e,seed=s,pilot_fraction=f,metrics_sha256=h) for e,s,f,h in sorted(cells)],
                    'prepared_input_manifest_by_fold':_formal_inputs(sources,cells)}
        info = _matrix(payload, manifest)
        for name in _RECORD_TABLES:
            payload[name] = _record_table(out/name,sources)
    info.update(input_source='completed_run_tree', record_tables_available=True,
                verification='Tables regenerated from completed metrics, configurations and run manifests; checksums '
                             'verified on these artifacts and on prediction shards if the paired calculation opened them.')
    return payload, info


def legacy_results(source: Path, *, prior_zip: Path | None = None) -> tuple[dict[str,bytes],dict]:
    """Export diagnostic evidence, never a corrected experiment or a paper table."""
    source = Path(source)
    if source.is_dir():
        tables, info = _tree(source)
        if prior_zip is not None:
            old, receipt = _archive(Path(prior_zip))
            shared = (_TABLES | _EXTRA) & (old.keys() | tables.keys())
            # Result content is compared, not the path-dependent source analysis hash.
            changed = sorted(k for k in shared if tables.get(k) != old.get(k))
            if changed or info['source_cells'] != receipt['source_cells'] or info['prepared_input_manifest_by_fold'] != receipt['prepared_input_manifest_by_fold']:
                raise ValueError('Existing archive and completed run tree differ; specify one source explicitly: '+', '.join(changed))
            info.update(archive_members_verified=receipt['archive_members_verified'],
                        matched_prior_archive_sha256=receipt['source_archive_sha256'])
    else:
        if prior_zip is not None:
            raise ValueError('Only a run tree can be compared with a separate archive.')
        tables, info = _archive(source)
    info['record_tables_missing'] = [name for name in _RECORD_TABLES if name not in tables]
    info['paper_eligible'] = False
    return {'legacy/'+name:b for name,b in tables.items()}, info
