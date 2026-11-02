"""Paired channel-estimation errors and pre-acquisition ranks from completed runs.

The default intervals describe repeatability over seeds for training and replay on the fixed
set of evaluated environments. Record/cluster resampling is opt-in. Adjacent
frames and the twelve environment-seed cells are never treated as independent
environmental replicates.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from aquapilot.metrics import framewise_diagnostics
from aquapilot.provenance import PROTOCOL_ID, file_fingerprint, fingerprint

POLICIES = ('comb', 'random', 'uncertainty', 'active')
ENVIRONMENTS = ('NOF', 'NCS', 'BCH', 'KAU')
SEEDS = (17, 37, 57)
FRACTIONS = (.0625, .125, .25)


def read_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f'{path}: nonstandard JSON number {value}')
    value = json.loads(path.read_text(encoding='utf-8'), parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f'{path}: expected a JSON object')
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        if fields:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def _finite(value: Any) -> bool:
    return value is not None and isinstance(value, (int, float)) and math.isfinite(value)


def _mean(values: Any) -> float | None:
    values = list(values)
    return float(np.mean(values)) if values and all(_finite(x) for x in values) else None


def _safe_path(root: Path, value: str) -> Path:
    path = (root/value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Artifact path escapes its run directory: {value}')
    return path


@dataclass
class Run:
    path: Path
    result: dict[str, Any]
    manifest: dict[str, Any]
    completion: dict[str, Any]
    seed: int
    fraction: float


def load_run(path: Path, *, expected_protocol: str = PROTOCOL_ID) -> Run:
    """Verify the completed summary/config identity, without reloading raw data."""
    path = path.resolve()
    if path.name != 'metrics.json':
        raise ValueError(f'{path}: expected the completed metrics.json, not another file in the run directory')
    result = read_json(path)
    if result.get('result_schema_version') != 2 or result.get('protocol_id') != expected_protocol:
        raise ValueError(f'{path}: incompatible result schema or protocol.')
    manifest = read_json(path.parent/'run_manifest.json')
    complete = read_json(path.parent/'complete.json')
    artifacts = complete.get('artifacts', {})
    for name in ('metrics.json', 'run_manifest.json', 'resolved_config.json'):
        if name not in artifacts or file_fingerprint(path.parent/name) != artifacts[name]:
            raise ValueError(f'{path.parent}: missing or modified completed artifact {name}')
    identity_keys = ('protocol_id', 'protocol_fingerprint', 'run_mode', 'config_fingerprint',
                     'evaluator_source_fingerprint', 'runtime_fingerprint', 'data_fingerprint',
                     'test_sequence_fingerprint', 'checkpoint_fingerprint')
    if any(k not in manifest for k in identity_keys):
        raise ValueError(f'{path}: incomplete provenance identity')
    identity = {k: manifest[k] for k in identity_keys}
    if complete.get('identity_fingerprint') != fingerprint(identity):
        raise ValueError(f'{path}: completion identity mismatch')
    for key in identity_keys:
        if key in result and result[key] != manifest[key]:
            raise ValueError(f'{path}: result/manifest mismatch for {key}')
    cfg = manifest['resolved_config']
    if read_json(path.parent/'resolved_config.json') != cfg:
        raise ValueError(f'{path}: resolved configuration mismatch')
    seed, ratio = cfg['seed'], float(cfg['pilot']['ratio'])
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 < ratio <= 1:
        raise ValueError(f'{path}: invalid seed or pilot fraction')
    for policy in POLICIES:
        policy_result = result.get('closed_loop', {}).get(policy)
        if not isinstance(policy_result, dict) or not policy_result.get('records'):
            raise ValueError(f'{path}: policy {policy} has no recording-level outcomes')
    return Run(path, result, manifest, complete, seed, ratio)


def discover_runs(roots: list[Path], *, expected_protocol: str = PROTOCOL_ID) -> list[Run]:
    paths = set()
    for root in roots:
        if root.is_file():
            paths.add(root.resolve())
        elif root.is_dir():
            paths.update(p.resolve() for p in root.rglob('metrics.json'))
        else:
            raise FileNotFoundError(root)
    if not paths:
        raise ValueError('No metrics.json experiment outputs were found.')
    runs = [load_run(path, expected_protocol=expected_protocol) for path in sorted(paths)]
    for field in ('protocol_fingerprint', 'evaluator_source_fingerprint', 'run_mode', 'noise_convention'):
        if len({run.result.get(field) for run in runs}) != 1:
            raise ValueError(f'Mixed {field} values: use a separate analysis for each protocol. No runs were discarded.')
    if runs[0].result.get('noise_convention') not in ('component', 'complex_rms_as_component'):
        raise ValueError('Missing explicit noise convention.')
    return runs


def _environment(value: Any, watermark: bool) -> str:
    value = value.strip() if isinstance(value, str) else ''
    if not value:
        raise ValueError('Missing environment identity.')
    if watermark:
        if value in ('KAU1', 'KAU2'):
            value = 'KAU'
        if value not in ENVIRONMENTS:
            raise ValueError(f'Unexpected Watermark environment: {value}')
    return value


def record_map(run: Run, policy: str, watermark: bool) -> dict[tuple[str, str], dict[str, Any]]:
    out = {}
    for record in run.result['closed_loop'][policy]['records']:
        required = {'record_id', 'environment', 'noise_seed', 'num_frames', 'frame_identity_sha256', 'posterior', 'prior', 'prior_ranking', 'same_information_baselines'}
        if not required.issubset(record):
            raise ValueError(f'{run.path}: incomplete record metadata for {policy}: {sorted(required-set(record))}')
        key = (_environment(record['environment'], watermark), str(record['record_id']))
        if not key[1] or key in out or int(record['num_frames']) < 1:
            raise ValueError(f'{run.path}: duplicate/invalid record identity {key}')
        if len(str(record['frame_identity_sha256'])) != 64:
            raise ValueError(f'{run.path}: invalid frame identity hash for {key}')
        if record['posterior'].get('schema_version') != 2:
            raise ValueError(f'{run.path}: incompatible record metric schema')
        summaries = [record['prior'], record['posterior'], record['prior_ranking'], *record['same_information_baselines'].values()]
        if any(r.get('num_frames') != record['num_frames'] for r in summaries):
            raise ValueError(f'{run.path}: receiver or rank summaries have a different frame count for {key}')
        out[key] = record
    return out


def merge_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Frame-weighted arithmetic aggregation of existing recording summaries."""
    n = sum(int(r['num_frames']) for r in records)
    valid = sum(int(r['num_valid_error_frames']) for r in records)
    def weighted(key: str, count_key: str = 'num_valid_error_frames') -> float | None:
        pairs = [(r.get(key), int(r.get(count_key, 0))) for r in records]
        if any(count and not _finite(value) for value, count in pairs):
            raise ValueError(f'Positive metric count with undefined {key}')
        count = sum(count for _, count in pairs)
        return sum(float(value)*weight for value, weight in pairs if weight)/count if count else None
    power = sum(float(r.get('pooled_target_energy', 0)) for r in records)
    error = sum(float(r.get('pooled_squared_error_energy', 0)) for r in records)
    coordinate_systems = {r.get('energy_coordinates') for r in records}
    if len(coordinate_systems) != 1:
        raise ValueError('Cannot pool recording summaries in different amplitude coordinates.')
    result = dict(num_frames=n, num_valid_error_frames=valid,
                  nmse_db_mean=weighted('nmse_db_mean'), nmse_linear_mean=weighted('nmse_linear_mean'),
                  channel_nrmse_percent_mean=weighted('channel_nrmse_percent_mean'),
                  pooled_nmse_linear=error/power if power else None,
                  pooled_nmse_db=10*math.log10(max(error/power, 1e-12)) if power else None,
                  pooled_target_energy=power, pooled_squared_error_energy=error,
                  energy_coordinates=next(iter(coordinate_systems)),
                  num_zero_power_frames=sum(int(r.get('num_zero_power_frames', 0)) for r in records),
                  num_nonfinite_frames=sum(int(r.get('num_nonfinite_frames', 0)) for r in records),
                  num_empty_mask_frames=sum(int(r.get('num_empty_mask_frames', 0)) for r in records))
    if any('coverage_90' in r for r in records):
        result['coverage_90'] = weighted('coverage_90', 'num_valid_coverage_frames')
        result['num_valid_coverage_frames'] = sum(int(r['num_valid_coverage_frames']) for r in records)
    if all('nonpilot' in r for r in records):
        result['nonpilot'] = merge_metrics([r['nonpilot'] for r in records])
    return result


def merge_ranking(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = sum(int(r['num_valid_frames']) for r in records)
    for row in records:
        n, used, undefined = (int(row[k]) for k in ('num_frames', 'num_valid_frames', 'num_undefined_frames'))
        rho = row.get('mean_frame_spearman')
        if min(n, used, undefined) < 0 or n != used + undefined:
            raise ValueError('Ranking frame counts do not sum to the record length.')
        if used and (not _finite(rho) or not -1 <= rho <= 1):
            raise ValueError('A defined mean Spearman coefficient must lie within [-1, 1].')
        if not used and rho is not None:
            raise ValueError('A record with no defined ranks must have a null mean.')
    return dict(num_frames=sum(int(r['num_frames']) for r in records), num_valid_frames=valid,
                num_undefined_frames=sum(int(r['num_undefined_frames']) for r in records),
                mean_frame_spearman=sum(float(r['mean_frame_spearman'])*int(r['num_valid_frames']) for r in records if r['num_valid_frames'])/valid if valid else None,
                equal_record_mean_spearman=_mean(r.get('mean_frame_spearman') for r in records),
                num_records=len(records),
                num_records_without_defined_ranks=sum(r.get('mean_frame_spearman') is None for r in records))


def _frame_errors(run: Run, record: dict[str, Any], stage: str) -> tuple[np.ndarray, np.ndarray] | None:
    name = record.get('arrays_file')
    if not name:
        return None
    path = _safe_path(run.path.parent, str(name))
    stored = run.completion.get('artifacts', {}).get(str(path.relative_to(run.path.parent)))
    if stored is None or file_fingerprint(path) != stored:
        raise ValueError(f'{path}: prediction shard is missing or changed')
    with np.load(path, allow_pickle=False) as data:
        index = data['frame_index']
        if stage + '_nmse_db' in data:
            errors = data[stage + '_nmse_db']
        elif stage + '_mean' in data and 'target' in data:
            errors = framewise_diagnostics(data[stage + '_mean'], None, data['target'],
                normalization_scale=data['history_scale'] if 'history_scale' in data else None,
                compute_ranking=False)['nmse_db']
        else:
            raise ValueError(f'{path}: missing per-frame diagnostics for {stage}')
    if index.shape != (int(record['num_frames']),) or errors.shape != index.shape:
        raise ValueError(f'{path}: invalid per-frame identity or metric length')
    if index.dtype.kind not in 'iu' or not np.all(index[1:] > index[:-1]):
        raise ValueError(f'{path}: expected distinct, increasing original frame indices')
    return index, errors


def _stage_metric(record: dict[str, Any], stage: str) -> dict[str, Any]:
    if stage.startswith('baseline_'):
        return record['same_information_baselines'][stage.removeprefix('baseline_')]
    return record[stage]


def _paired_errors(run: Run, reference: dict[str, Any], candidate: dict[str, Any],
                   ref_stage: str, cand_stage: str) -> dict[str, Any]:
    for field in ('record_id', 'num_frames', 'frame_identity_sha256', 'noise_seed'):
        if reference[field] != candidate[field]:
            raise ValueError(f'{run.path}: unmatched {field} in paired outcomes')
    n = int(reference['num_frames'])
    ref, cand = _stage_metric(reference, ref_stage), _stage_metric(candidate, cand_stage)
    valid = tuple(int(item['num_valid_error_frames']) for item in (ref, cand))
    if any(v < 0 or v > n for v in valid):
        raise ValueError('Invalid defined-frame count in matched comparison.')
    if valid == (n, n):
        x, y = (item['nmse_db_mean'] for item in (ref, cand))
        if not all(_finite(v) for v in (x, y)):
            raise ValueError('Missing mean error for fully defined frames.')
        used, known, status = n, True, 'all frames valid; recorded frame identity and noise seed matched'
    elif 0 in valid:
        x, y, used, known, status = None, None, 0, True, 'no jointly defined frame in this record'
    else:
        ra, ca = _frame_errors(run, reference, ref_stage), _frame_errors(run, candidate, cand_stage)
        if ra is None or ca is None:
            x, y, used, known, status = None, None, 0, False, 'unavailable: frame intersection requires saved diagnostics'
        else:
            if not np.array_equal(ra[0], ca[0]):
                raise ValueError(f'{run.path}: per-frame index mismatch despite record hash')
            rv, cv = ra[1], ca[1]
            if tuple(int(np.isfinite(v).sum()) for v in (rv, cv)) != valid:
                raise ValueError(f'{run.path}: record valid counts disagree with saved frame diagnostics')
            use = np.isfinite(rv) & np.isfinite(cv)
            used, known = int(use.sum()), True
            x = float(rv[use].mean(dtype=np.float64)) if used else None
            y = float(cv[use].mean(dtype=np.float64)) if used else None
            status = 'paired intersection of defined frame outcomes; exclusions explicitly counted'
    return dict(reference_nmse_db=x, candidate_nmse_db=y, delta_db=y-x if used else None,
                num_frames=n, num_matched_valid_frames=used,
                num_excluded_frames=n-used, support_known=known, pairing_status=status)


def paired_record(run: Run, comb: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    pair = _paired_errors(run, comb, other, 'posterior', 'posterior')
    return {**pair, 'comb_nmse_db': pair['reference_nmse_db'], 'policy_nmse_db': pair['candidate_nmse_db']}


def pool_pairs(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Weight jointly valid frames, including zero-overlap records in the exclusion count."""
    count = sum(r['num_matched_valid_frames'] for r in records)
    n = sum(r['num_frames'] for r in records)
    known = all(r['support_known'] for r in records)
    def weighted(key: str) -> float | None:
        return (sum(r[key]*r['num_matched_valid_frames'] for r in records if r['num_matched_valid_frames']) / count
                if known and count else None)
    return dict(num_frames=n, num_matched_valid_frames=count, num_excluded_frames=n-count,
                support_known=known, reference_nmse_db=weighted('reference_nmse_db'),
                candidate_nmse_db=weighted('candidate_nmse_db'), delta_db=weighted('delta_db'),
                pairing_status=('matched original records and jointly defined frames' if count else 'no jointly defined frames')
                if known else 'unavailable: missing per-frame information; no difference reported')


def conditional_seed_interval(seed_means: dict[int, float | None]) -> dict[str, Any]:
    """Exploratory fixed-environment repeatability interval, one block per seed."""
    result: dict[str, Any] = dict(num_seed_blocks=len(seed_means), seed_block_means_db=seed_means,
        interval_scope='Conditional on the fixed evaluated environments; each seed indexes training and replay randomness, and one block averages the environments. No environment-population inference.',
        t_lower=None, t_upper=None, bootstrap_lower=None, bootstrap_upper=None,
        interval_status='unavailable: fewer than two complete seed blocks')
    if len(seed_means) < 2 or not all(_finite(x) for x in seed_means.values()):
        return result
    values = np.asarray(list(seed_means.values()), dtype=float)
    n = len(values)
    half = float(student_t.ppf(.975, n-1)*values.std(ddof=1)/np.sqrt(n))
    result.update(mean_delta_db=float(values.mean()), t_lower=float(values.mean()-half),
                  t_upper=float(values.mean()+half), degrees_of_freedom=n-1,
                  interval_status='exploratory seed-block intervals; few seeds limit precision; no multiplicity adjustment')
    if n <= 7:
        samples = np.asarray([values[list(index)].mean() for index in itertools.product(range(n), repeat=n)])
        low, high = np.quantile(samples, [.025, .975])
        result.update(bootstrap_lower=float(low), bootstrap_upper=float(high),
                      bootstrap_method='Exact enumeration of nonparametric seed-block resamples; environments kept together')
    else:
        result['bootstrap_method'] = 'not computed for more than seven seed blocks; approximate t interval provided'
    return result


def paired_tables(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_env, by_policy = defaultdict(list), defaultdict(list)
    for row in rows:
        by_env[(row['pilot_fraction'], row['policy'], row['environment'])].append(row)
        by_policy[(row['pilot_fraction'], row['policy'])].append(row)
    environments = []
    for (ratio, policy, env), group in sorted(by_env.items()):
        values = [r['delta_db'] for r in group]
        complete = all(_finite(x) for x in values)
        environments.append(dict(pilot_fraction=ratio, policy=policy, environment=env, num_seeds=len(group),
            mean_delta_db=_mean(values), sample_sd_delta_db=float(np.std(values, ddof=1)) if complete and len(values)>1 else None,
            minimum_delta_db=min(values) if complete else None, maximum_delta_db=max(values) if complete else None,
            improved_seed_count=sum(_finite(v) and v < 0 for v in values),
            undefined_seed_count=sum(not _finite(v) for v in values),
            excluded_frame_count=sum(r['num_excluded_frames'] for r in group)))
    totals = []
    for (ratio, policy), group in sorted(by_policy.items()):
        envs, seeds = sorted({r['environment'] for r in group}), sorted({r['seed'] for r in group})
        cells = {(r['environment'], r['seed']): r for r in group}
        if len(cells) != len(group):
            raise ValueError('Duplicate environment/seed/budget/policy outcomes.')
        complete_grid = set(cells) == set(itertools.product(envs, seeds))
        seed_means = {seed: _mean(cells[(env, seed)]['delta_db'] for env in envs) for seed in seeds} if complete_grid else {}
        interval = conditional_seed_interval(seed_means)
        totals.append(dict(pilot_fraction=ratio, policy=policy, num_environments=len(envs), num_seeds=len(seeds),
            num_paired_outcomes=len(group), complete_environment_seed_grid=complete_grid,
            equal_environment_equal_seed_delta_db=_mean(seed_means.values()),
            improved_pair_count=sum(_finite(r['delta_db']) and r['delta_db'] < 0 for r in group),
            worse_pair_count=sum(_finite(r['delta_db']) and r['delta_db'] > 0 for r in group),
            conditional_seed_interval=interval))
    return environments, totals


def record_cluster_intervals(path: Path, rows: list[dict[str, Any]], replicates: int,
                             seed: int) -> list[dict[str, Any]]:
    """Opt-in resampling of user-declared independent original-record clusters."""
    with path.open(newline='', encoding='utf-8-sig') as stream:
        source = list(csv.DictReader(stream))
    required = {'environment', 'record_id', 'cluster_id'}
    if not source or not required.issubset(source[0]):
        raise ValueError('Cluster CSV requires environment, record_id, cluster_id.')
    mapping = {}
    for row in source:
        key = row['environment'], row['record_id']
        if key in mapping or not row['cluster_id'].strip():
            raise ValueError(f'Duplicate or missing original-record cluster: {key}')
        mapping[key] = row['cluster_id'].strip()
    identities = {(r['environment'], r['record_id']) for r in rows}
    if set(mapping) != identities:
        raise ValueError(f'Cluster map must cover exactly the analyzed records; missing={len(identities-set(mapping))}, extra={len(set(mapping)-identities)}')
    groups = defaultdict(list)
    for row in rows:
        groups[(row['environment'], row['pilot_fraction'], row['policy'])].append(row)
    rng = np.random.default_rng(seed)
    out = []
    for (env, fraction, policy), group in sorted(groups.items()):
        per_record = defaultdict(dict)
        seeds = {r['seed'] for r in group}
        for row in group:
            per_record[row['record_id']][row['seed']] = row['mean_frame_spearman']
        complete = all(set(values) == seeds and all(_finite(v) for v in values.values()) for values in per_record.values())
        base = dict(environment=env, pilot_fraction=fraction, policy=policy, num_seeds=len(seeds),
                    num_records=len(per_record), lower_95=None, upper_95=None,
                    estimand='Equal-cluster mean of equal-record means; seed means averaged within each original record',
                    scope='Conditional on fitted models and these environments; user-supplied clusters must be independent sampling units')
        if not complete:
            out.append({**base, 'interval_status': 'unavailable: incomplete record-seed grid or undefined rank; no favorable subset selected'})
            continue
        clusters = defaultdict(list)
        for record, values in per_record.items():
            clusters[mapping[(env, record)]].append(float(np.mean(list(values.values()))))
        values = np.asarray([np.mean(v) for v in clusters.values()])
        base.update(num_clusters=len(values), equal_cluster_mean=float(values.mean()))
        if len(values) < 2:
            out.append({**base, 'interval_status': 'unavailable: fewer than two declared independent record clusters'})
            continue
        # Resample whole acquisition clusters.
        draws = np.asarray([rng.choice(values, len(values), replace=True).mean() for _ in range(replicates)])
        low, high = np.quantile(draws, [.025, .975])
        out.append({**base, 'lower_95': float(low), 'upper_95': float(high),
                    'interval_status': 'exploratory percentile bootstrap of explicitly declared record clusters'})
    return out


def analyze_results(roots: list[Path], output: Path, *, expect_watermark_matrix: bool = False,
                    allow_incomplete: bool = False, record_clusters: Path | None = None,
                    bootstrap_replicates: int = 5000, bootstrap_seed: int = 17,
                    expected_protocol: str = PROTOCOL_ID) -> dict[str, Any]:
    runs = discover_runs(roots, expected_protocol=expected_protocol)
    if any(output.resolve().is_relative_to(run.path.parent) for run in runs):
        raise ValueError('Analysis output must be outside each completed run directory.')
    previous_summary = output/'analysis_summary.json'
    if previous_summary.is_file():
        previous = read_json(previous_summary)
        if previous.get('protocol_fingerprint') != runs[0].result['protocol_fingerprint']:
            raise ValueError('Analysis output already belongs to a different protocol; choose a new output directory.')
    configuration, rank_runs, rank_records, calibration, baselines, pairs = [], [], [], [], [], []
    record_pair_rows, placement_rows = [], []
    cells, env_test_identities = set(), {}
    for run in runs:
        maps = {p: record_map(run, p, expect_watermark_matrix) for p in POLICIES}
        identities = set(maps['comb'])
        if any(set(value) != identities for value in maps.values()):
            raise ValueError(f'{run.path}: policies evaluated different record sets')
        by_env = defaultdict(list)
        for identity in sorted(identities):
            by_env[identity[0]].append(identity)
        for env, keys in by_env.items():
            run_cell = env, run.seed, run.fraction
            if run_cell in cells:
                raise ValueError(f'Duplicate environment/seed/pilot-fraction run: {run_cell}')
            cells.add(run_cell)
            identity = fingerprint([(key[1], maps['comb'][key]['frame_identity_sha256']) for key in keys])
            test_identity = (identity, run.result['test_sequence_fingerprint'],
                             run.manifest['data_fingerprint'] if expect_watermark_matrix else None)
            if env in env_test_identities and env_test_identities[env] != test_identity:
                raise ValueError(f'{env}: test record/frame identities differ across seeds or budgets; use separate analyses')
            env_test_identities[env] = test_identity
            meta = dict(environment=env, seed=run.seed, pilot_fraction=run.fraction,
                        source_metrics=str(run.path), run_mode=run.result['run_mode'])
            for policy in POLICIES:
                records = [maps[policy][key] for key in keys]
                placement = [record.get('placement_diagnostics') for record in records]
                if any(v is not None for v in placement):
                    if any(v is None for v in placement):
                        raise ValueError(f'{run.path}: placement diagnostics missing for some {policy} records')
                    from aquapilot.placement import combine_mask_summaries
                    geometry = combine_mask_summaries(placement)
                    if (geometry['num_mask_frames'] != sum(record['num_frames'] for record in records)
                            or geometry['num_records'] != len(records)):
                        raise ValueError(f'{run.path}: placement counts differ from evaluated {policy} records')
                    placement_rows.append({**meta, 'policy': policy, **geometry})
                metric = merge_metrics([r['posterior'] for r in records])
                row = {**meta, 'policy': policy, **{k:v for k,v in metric.items() if k != 'nonpilot'}}
                row.update({'nonpilot_'+k: v for k,v in metric.get('nonpilot', {}).items()})
                configuration.append(row)
                ranking = merge_ranking([r['prior_ranking'] for r in records])
                rank_runs.append({**meta, 'policy': policy, **ranking})
                for record in records:
                    ranking_record = record['prior_ranking']
                    rank_records.append({**meta, 'policy': policy, 'record_id': str(record['record_id']),
                        'frame_identity_sha256': record['frame_identity_sha256'],
                        'mean_frame_spearman': ranking_record.get('mean_frame_spearman'),
                        'num_frames': int(record['num_frames']),
                        'num_valid_frames': int(ranking_record['num_valid_frames']),
                        'num_undefined_frames': int(ranking_record['num_undefined_frames'])})
                for stage in ('prior', 'posterior'):
                    stage_metrics = merge_metrics([r[stage] for r in records])
                    calibration.append({**meta, 'policy': policy, 'stage': stage,
                        'nominal_coverage': .9, 'coverage_90': stage_metrics.get('coverage_90'),
                        'num_frames': stage_metrics['num_frames'],
                        'num_valid_coverage_frames': stage_metrics.get('num_valid_coverage_frames'),
                        'definition': 'Model-supplied marginal real/imaginary Gaussian intervals; frame-mean coverage'})
                names = set(records[0]['same_information_baselines'])
                if any(set(r['same_information_baselines']) != names for r in records):
                    raise ValueError('Same-information baseline sets differ between records.')
                required_baselines = {'zero', 'persistence_prior', 'current_interpolation', 'persistence_innovation'}
                if names != required_baselines:
                    raise ValueError(f'{run.path}: unexpected set of same-information baselines: {names}')
                for name in sorted(names):
                    stage = 'prior' if name == 'persistence_prior' else 'posterior'
                    baseline_pairs = [_paired_errors(run, r, r, 'baseline_'+name, stage) for r in records]
                    matched = pool_pairs(baseline_pairs)
                    baselines.append({**meta, 'policy': policy, 'baseline': name, 'receiver_stage': stage,
                        'receiver_nmse_db': matched['candidate_nmse_db'], 'baseline_nmse_db': matched['reference_nmse_db'],
                        'receiver_minus_baseline_db': matched['delta_db'],
                        'receiver_num_valid_frames': sum(r[stage]['num_valid_error_frames'] for r in records),
                        'baseline_num_valid_frames': sum(r['same_information_baselines'][name]['num_valid_error_frames'] for r in records),
                        **{k: matched[k] for k in ('num_frames', 'num_matched_valid_frames', 'num_excluded_frames', 'support_known')},
                        'comparison_status': matched['pairing_status'],
                        'baseline_information': {'persistence_prior': 'previous noisy interpolated frame; no current pilot',
                            'current_interpolation': 'current noisy pilots only',
                            'persistence_innovation': 'previous interpolated frame and current noisy pilots',
                            'zero': 'analytical zero estimate; no observations'}[name]})
                if policy == 'comb':
                    continue
                per_record = [paired_record(run, maps['comb'][key], maps[policy][key]) for key in keys]
                matched = pool_pairs(per_record)
                pair = {**meta, 'policy': policy, 'reference_policy': 'comb', 'num_records': len(keys),
                    'comb_nmse_db': matched['reference_nmse_db'], 'policy_nmse_db': matched['candidate_nmse_db'],
                    **{k: matched[k] for k in ('delta_db', 'num_frames', 'num_matched_valid_frames',
                        'num_excluded_frames', 'support_known', 'pairing_status')},
                    'difference_convention': 'policy minus comb; negative favors policy'}
                pairs.append(pair)
                for key, record_result in zip(keys, per_record):
                    record_pair_rows.append({**meta, 'policy': policy, 'record_id': key[1], **record_result})
    expected = set(itertools.product(ENVIRONMENTS, SEEDS, FRACTIONS)) if expect_watermark_matrix else cells
    missing, unexpected = sorted(expected-cells), sorted(cells-expected)
    if (missing or unexpected) and not allow_incomplete:
        raise ValueError(f'Incomplete Watermark matrix: missing={missing}, unexpected={unexpected}. Use --allow-incomplete only to label a provisional analysis.')
    if unexpected:
        raise ValueError(f'Unexpected Watermark cells: {unexpected}')
    env_pairs, pair_totals = paired_tables(pairs)
    ranked_groups = defaultdict(list)
    for row in rank_runs:
        ranked_groups[(row['environment'], row['pilot_fraction'], row['policy'])].append(row)
    rank_env = []
    for (env, fraction, policy), group in sorted(ranked_groups.items()):
        rank_env.append(dict(num_frames=sum(r['num_frames'] for r in group),
            num_valid_frames=sum(r['num_valid_frames'] for r in group),
            num_seeds_with_defined_ranks=sum(r['num_valid_frames'] > 0 for r in group), environment=env, pilot_fraction=fraction, policy=policy,
            num_seeds=len(group), num_records_per_seed=min(r['num_records'] for r in group),
            equal_seed_mean_frame_spearman=_mean(r['mean_frame_spearman'] for r in group),
            equal_seed_equal_record_mean_spearman=_mean(r['equal_record_mean_spearman'] for r in group),
            num_undefined_frames=sum(r['num_undefined_frames'] for r in group),
            confidence_interval='not computed; no independent-frame or environment-population inference'))
    environment_deltas = {(r['environment'], r['pilot_fraction']): r for r in env_pairs if r['policy'] == 'uncertainty'}
    ranking_placement = []
    for rank in rank_env:
        if rank['policy'] != 'uncertainty':
            continue
        delta = environment_deltas[(rank['environment'], rank['pilot_fraction'])]
        ranking_placement.append({**rank, 'mean_delta_db': delta['mean_delta_db'],
            'improved_seed_count': delta['improved_seed_count'], 'undefined_delta_seed_count': delta['undefined_seed_count'],
            'excluded_pair_frames': delta['excluded_frame_count'],
            'comparison': 'top-B minus comb full-band dB NMSE; rho uses the top-B pre-acquisition trajectory'})
    groups = defaultdict(list)
    for row in configuration:
        groups[(row['pilot_fraction'], row['policy'])].append(row)
    aggregate = []
    for (fraction, policy), group in sorted(groups.items()):
        values = [r['nmse_db_mean'] for r in group]
        frame_count = sum(r['num_valid_error_frames'] for r in group)
        total_power = sum(r['pooled_target_energy'] for r in group)
        total_error = sum(r['pooled_squared_error_energy'] for r in group)
        def frame_weighted(name: str) -> float | None:
            if any(r['num_valid_error_frames'] and not _finite(r[name]) for r in group):
                return None
            return sum(r[name]*r['num_valid_error_frames'] for r in group if r['num_valid_error_frames'])/frame_count if frame_count else None
        aggregate.append(dict(pilot_fraction=fraction, policy=policy, num_environment_seed_cells=len(group),
            equal_environment_seed_nmse_db=_mean(values),
            descriptive_sd_nmse_db=float(np.std(values, ddof=0)) if all(_finite(v) for v in values) else None,
            equal_environment_seed_channel_nrmse_percent=_mean(r['channel_nrmse_percent_mean'] for r in group),
            frame_weighted_nmse_db=frame_weighted('nmse_db_mean'),
            frame_weighted_channel_nrmse_percent=frame_weighted('channel_nrmse_percent_mean'),
            pooled_nmse_linear=total_error/total_power if total_power > 0 else None,
            pooled_nmse_db=10*math.log10(max(total_error/total_power, 1e-12)) if total_power > 0 else None,
            num_valid_frames=frame_count,
            averaging='Equal-cell columns weight environment-seed cells equally; frame-weighted and pooled-energy columns include all evaluated frames and seeds'))
    flags = []
    for row in configuration:
        meta = {k: row[k] for k in ('environment', 'pilot_fraction', 'seed', 'policy')}
        if _finite(row['nmse_db_mean']) and row['nmse_db_mean'] > 0:
            flags.append({**meta, 'issue': 'mean_nmse_above_zero_estimate', 'value': row['nmse_db_mean']})
        if row['num_valid_error_frames'] != row['num_frames']:
            flags.append({**meta, 'issue': 'undefined_frame_error_metrics', 'value': row['num_frames']-row['num_valid_error_frames']})
    for row in baselines:
        if row['baseline'] == 'current_interpolation' and _finite(row['receiver_minus_baseline_db']) and row['receiver_minus_baseline_db'] > 0:
            flags.append({**{k:row[k] for k in ('environment', 'pilot_fraction', 'seed', 'policy')},
                          'issue': 'worse_than_same_information_interpolation', 'value': row['receiver_minus_baseline_db']})
    cluster_rows = None
    if record_clusters is not None:
        if bootstrap_replicates < 100:
            raise ValueError('Use at least 100 explicitly requested record-cluster bootstrap replicates.')
        cluster_rows = record_cluster_intervals(record_clusters, rank_records, bootstrap_replicates, bootstrap_seed)
    output.mkdir(parents=True, exist_ok=True)
    # A failed write must not leave an old success summary beside new tables.
    previous_summary.unlink(missing_ok=True)
    files = {'configuration_metrics.csv': configuration, 'aggregate_metrics.csv': aggregate,
             'paired_differences.csv': pairs, 'paired_top_b_025.csv': [r for r in pairs if r['policy']=='uncertainty' and math.isclose(r['pilot_fraction'], .25)],
             'ranking_placement_by_environment.csv': ranking_placement, 'paired_by_environment.csv': env_pairs, 'paired_by_record.csv': record_pair_rows,
             'prior_ranking_by_run.csv': rank_runs, 'prior_ranking_by_record.csv': rank_records,
             'prior_ranking_by_environment.csv': rank_env, 'calibration.csv': calibration,
             'baseline_comparisons.csv': baselines, 'error_flags.csv': flags}
    if placement_rows:
        files['placement_diagnostics.csv'] = placement_rows
    else:
        (output/'placement_diagnostics.csv').unlink(missing_ok=True)
    for name, rows in files.items():
        write_csv(output/name, rows)
    write_json(output/'paired_summary.json', pair_totals)
    if cluster_rows is not None:
        write_csv(output/'record_cluster_intervals.csv', cluster_rows)
    else:
        (output/'record_cluster_intervals.csv').unlink(missing_ok=True)
    report = dict(analysis_schema_version=2, protocol_id=expected_protocol,
        protocol_fingerprint=runs[0].result['protocol_fingerprint'], run_mode=runs[0].result['run_mode'],
        noise_convention=runs[0].result['noise_convention'], num_runs=len(runs), num_environment_seed_budget_cells=len(cells),
        input_data_kinds=sorted({run.manifest['resolved_config']['data']['kind'] for run in runs}),
        declared_source_statuses=sorted({str(run.result.get('artifact_status', 'unspecified')) for run in runs}),
        data_kind=runs[0].manifest['resolved_config']['data']['kind'],
        artifact_status=runs[0].result.get('artifact_status', 'unspecified'),
        expected_watermark_matrix=expect_watermark_matrix, matrix_complete=(not missing) if expect_watermark_matrix else None,
        missing_cells=missing, source_metrics=[str(run.path) for run in runs],
        source_summary_sha256=[file_fingerprint(run.path)['sha256'] for run in runs],
        pairing='Matched record and frame identities, noise seeds, and jointly defined errors',
        primary_policy_metric='Mean full-band framewise NMSE in dB; nonpilot frequency sets vary by policy',
        inference='95% t interval over joint training/replay seed blocks, conditional on evaluated environments',
        record_cluster_bootstrap_requested=record_clusters is not None,
        calibration='Real/imaginary marginal coverage for estimators with predicted scales',
        artifact_scope='Completed summary, manifest and resolved configuration hashes checked; prediction shards checked when opened',
        num_error_flags=len(flags), selection='All completed runs under the requested roots, with one shared protocol')
    matrix = (f'{len(missing)} Watermark cells missing' if missing else 'complete Watermark matrix') if expect_watermark_matrix else 'Watermark completeness not assessed'
    text = [f'AquaPilot: {len(runs)} runs, {len(cells)} environment/seed/budget cells ({matrix}).',
            f'Mode: {report["run_mode"]}; noise convention: {report["noise_convention"]}.', '',
            'Full-band NMSE difference from comb (dB). A negative difference favors the named policy.',
            'B/F      policy          mean difference    95% seed interval       lower / higher pairs']
    labels = {'random': 'Random', 'uncertainty': 'Top-B', 'active': 'A-optimal'}
    for row in pair_totals:
        value = row['equal_environment_equal_seed_delta_db']
        display = f'{value:+.4f}' if value is not None else 'NA'
        ci = row['conditional_seed_interval']
        interval = f"[{ci['t_lower']:+.4f}, {ci['t_upper']:+.4f}]" if ci['t_lower'] is not None else 'NA'
        text.append(f'{row["pilot_fraction"]:<8g} {labels[row["policy"]]:<15} {display:<18} {interval:<23} '
                    f'{row["improved_pair_count"]} / {row["worse_pair_count"]}')
    zero_fail = sum(r['issue']=='mean_nmse_above_zero_estimate' for r in flags)
    interp_fail = sum(r['issue']=='worse_than_same_information_interpolation' for r in flags)
    text += ['', 'Intervals use joint training/replay seed blocks and condition on the evaluated environments.',
             f'Policy cells above the zero reference (0 dB): {zero_fail}.',
             f'Policy cells above interpolation with the same acquired pilots: {interp_fail}. See error_flags.csv.',
             'ranking_placement_by_environment.csv pairs within-frame, pre-acquisition Spearman correlations with posterior NMSE differences.']
    (output/'summary.txt').write_text('\n'.join(text)+'\n', encoding='utf-8')
    artifact_names = list(files)+['paired_summary.json', 'summary.txt']
    if cluster_rows is not None:
        artifact_names.append('record_cluster_intervals.csv')
    report['analysis_artifacts'] = {name:file_fingerprint(output/name) for name in artifact_names}
    write_json(output/'analysis_summary.json', report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', '--root', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expect-watermark-matrix', action='store_true')
    parser.add_argument('--allow-incomplete', action='store_true', help='Label missing Watermark cells; never skip incompatible or invalid runs')
    parser.add_argument('--record-clusters', type=Path, help='Explicit independent-record cluster map: environment,record_id,cluster_id')
    parser.add_argument('--bootstrap-replicates', type=int, default=5000)
    parser.add_argument('--bootstrap-seed', type=int, default=17)
    args = parser.parse_args()
    analyze_results(args.results, args.output, expect_watermark_matrix=args.expect_watermark_matrix,
                    allow_incomplete=args.allow_incomplete, record_clusters=args.record_clusters,
                    bootstrap_replicates=args.bootstrap_replicates, bootstrap_seed=args.bootstrap_seed)
    print(f'Wrote experiment analysis to {args.output.resolve()}')


if __name__ == '__main__':
    main()
