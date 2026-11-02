#!/usr/bin/env python3
"""Export the completed Watermark experiment for analysis. No training or CUDA is required."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if not __package__:  # Resolve this checkout even if the editable install points elsewhere.
    sys.path[:0] = [str(PROJECT_ROOT / 'src'), str(PROJECT_ROOT)]

from aquapilot.export_io import _source_table, _formal_inputs, _fold_inputs
from aquapilot.provenance import PROTOCOL_ID, file_fingerprint

# Only aggregate outputs: recording IDs, predictions, data paths and weights stay on the host.
BASE_TABLES = (
    'aggregate_metrics.csv', 'configuration_metrics.csv', 'paired_top_b_025.csv',
    'paired_differences.csv', 'paired_by_environment.csv', 'paired_summary.json',
    'prior_ranking_by_environment.csv', 'prior_ranking_by_run.csv',
    'baseline_comparisons.csv', 'calibration.csv', 'error_flags.csv',
    'summary.txt',
)
EXTRA_TABLES = ('ranking_placement_by_environment.csv', 'placement_diagnostics.csv')


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')


def _check_source_files(analysis: Path, report: dict) -> None:
    for name, expected in report['analysis_artifacts'].items():
        if Path(name).name != name:
            raise ValueError('Invalid analysis artifact name')
        path = analysis / name
        if path.is_symlink() or not path.is_file() or file_fingerprint(path) != expected:
            raise ValueError(f'Missing or changed analysis artifact: {name}. Regenerate the analysis.')


def _verify_analysis(results: Path, analysis: Path, stage: Path, diagnostic: bool) -> dict:
    from aquapilot.analysis import analyze_results, read_json

    report = read_json(analysis / 'analysis_summary.json')
    if report.get('protocol_id') != PROTOCOL_ID:
        raise ValueError('Analysis protocol differs from the current analyzer.')
    if report.get('record_cluster_bootstrap_requested'):
        raise ValueError('This bundle uses the default seed-block analysis. Export the optional recording-cluster sensitivity analysis separately.')
    full = report.get('expected_watermark_matrix') is True and report.get('matrix_complete') is True
    if not diagnostic and not (full and report.get('run_mode') == 'fresh_training'
            and report.get('declared_source_statuses') == ['measured-channel experiment']
            and report.get('input_data_kinds') == ['prepared'] and report.get('num_runs') == 36):
        raise ValueError('Expected the complete measured Watermark fresh-training matrix; use --diagnostic for a diagnostic archive.')
    _check_source_files(analysis, report)
    # Use the current directory inventory. A stale summary can be internally hash-consistent
    # and still omit a newly completed seed, or carry a statistic from an earlier analyzer.
    analyze_results([results], stage, expect_watermark_matrix=report['expected_watermark_matrix'],
                    allow_incomplete=diagnostic)
    fresh = read_json(stage / 'analysis_summary.json')
    if fresh != report:
        changed = sorted(k for k in set(fresh) | set(report) if fresh.get(k) != report.get(k))
        raise ValueError('Analysis does not reproduce from the current source runs (' + ', '.join(changed) + '). Regenerate the analysis.')
    return fresh


def export_bundle(results: Path, output: Path, *, diagnostic: bool = False, analysis: Path | None = None) -> dict:
    results, output = Path(results).resolve(), Path(output).absolute()
    analysis = Path(analysis) if analysis is not None else results / 'analysis'
    if not results.is_dir() or not (analysis / 'analysis_summary.json').is_file():
        raise ValueError('Pass the model results directory containing analysis/analysis_summary.json.')
    if output.suffix.lower() != '.zip' or output.resolve().is_relative_to(results) or output.resolve().is_relative_to(PROJECT_ROOT / 'data'):
        raise ValueError('Choose a new ZIP outside the model results and data directories.')
    if output.is_symlink() or output.exists():
        raise FileExistsError(f'Archive already exists: {output}')
    with tempfile.TemporaryDirectory(prefix='aquapilot_evidence_') as scratch:
        stage = Path(scratch) / 'analysis'
        report = _verify_analysis(results, analysis, stage, diagnostic)
        sources = dict(zip(report['source_metrics'], report['source_summary_sha256'], strict=True))
        payload = {}
        for name in (*BASE_TABLES, *(n for n in EXTRA_TABLES if n in report['analysis_artifacts'])):
            path = stage / name
            payload[name] = _source_table(path, sources) if path.suffix == '.csv' else path.read_bytes()
        if not diagnostic:
            from aquapilot.result_figure import rank_placement_pdf
            from scripts.build_result_tables import build_tables

            tables = Path(scratch) / 'tables'
            exported = build_tables(stage, tables)
            for name in exported['outputs']:
                if Path(name).name != name or Path(name).suffix not in {'.tex', '.txt'}:
                    raise ValueError(f'Unexpected table output: {name}')
                payload['tables/' + name] = (tables / name).read_bytes()
            payload['rank_placement.pdf'] = rank_placement_pdf(payload['ranking_placement_by_environment.csv'])
        source_kinds = set(report['input_data_kinds'])
        status = set(report['declared_source_statuses'])
        if 'synthetic' in source_kinds or any('synthetic' in item.lower() for item in status):
            data_status = 'contains synthetic data'
        elif source_kinds == {'prepared'} and status == {'measured-channel experiment'}:
            data_status = 'declared measured channels'
        else:
            data_status = 'other or mixed inputs'
        cells = set()
        input_rows = csv.DictReader(io.StringIO(payload['configuration_metrics.csv'].decode('utf-8')))
        for row in input_rows:
            cells.add((row['environment'], int(row['seed']), float(row['pilot_fraction']), row['metrics_sha256']))
        manifest = {
            'bundle_schema_version': 1, 'protocol_id': PROTOCOL_ID,
            'protocol_fingerprint': report['protocol_fingerprint'], 'run_mode': report['run_mode'],
            'classification': 'diagnostic only' if diagnostic else 'complete measured Watermark fresh-training analysis',
            'input_classification': data_status, 'includes_result_tables': not diagnostic,
            'num_runs': report['num_runs'], 'num_environment_seed_budget_cells': report['num_environment_seed_budget_cells'],
            'missing_cells': report['missing_cells'] if report['expected_watermark_matrix'] else None,
            'source_cells': [dict(environment=e, seed=s, pilot_fraction=f, metrics_sha256=h) for e,s,f,h in sorted(cells)],
            'prepared_input_manifest_by_fold': None if diagnostic else _formal_inputs(sources, cells),
            'source_analysis_summary_sha256': file_fingerprint(analysis/'analysis_summary.json')['sha256'],
            'num_error_flags': report['num_error_flags'],
            'verification': 'Aggregates reproduced from completed run outputs. Formal bundles check ordered prepared split hashes saved by the trainer; raw measurement files are not reopened.',
            'members': {name: {'size_bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest()}
                        for name, value in sorted(payload.items())},
        }
        payload['bundle_manifest.json'] = _json_bytes(manifest)
        output.parent.mkdir(parents=True, exist_ok=True)
        # Publish a complete archive only. Concurrent exports cannot overwrite an existing ZIP.
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix='.zip') as tmp:
            with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
                for name, value in sorted(payload.items()):
                    archive.writestr(zipfile.ZipInfo(name), value, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
            tmp.flush()
            os.fsync(tmp.fileno())
            os.link(tmp.name, output)
    return manifest


def main(argv=None) -> None:
    from scripts.analysis_handoff import handoff

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, help='Model directory or a previous numeric results.zip; default: discover the completed run in this project')
    parser.add_argument('--output', type=Path, default=Path('results/AquaPilot_analysis.zip'), help='Upload ZIP (default: results/AquaPilot_analysis.zip)')
    parser.add_argument('--raw-root', type=Path, default=Path('data/raw/watermark/WatermarkV1/Watermark/input/channels'), help='Original Watermark channels directory for inspecting earlier training inputs')
    parser.add_argument('--prepared-root', type=Path, default=Path('data/prepared/watermark'), help='Earlier training folds for matching to saved input hashes')
    parser.add_argument('--axes', type=Path, help='Corrected-run axes.json (default: beside the model directory or input ZIP)')
    parser.add_argument('--full-audit', action='store_true', help='For legacy results, compare all 156 original records instead of one per channel bank')
    parser.add_argument('--diagnostic', action='store_true', help='For a partial or synthetic current-protocol run: export diagnostics without manuscript sources')
    args = parser.parse_args(argv)
    manifest, output = handoff(PROJECT_ROOT, args, export_bundle)
    print(f"Mode: {manifest['classification']} | fits: {manifest['num_runs']}")
    if manifest['classification'] == 'legacy_diagnostic':
        check = manifest['input_check']
        print(f"Paired comparisons: {manifest['num_paired_comparisons']} | top-B vs comb at 1/4: {manifest['num_quarter_budget_top_b_pairs']}")
        print(f"MAT/NPZ comparison: {check['status']} ({check['compared_copies']} copies); training-input hashes: {check['trained_input_binding']}")
        print("Earlier frequency grid: diagnostics only.")
    print(f"Upload: {output}")


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        sys.exit(f'Export stopped: {exc}')
