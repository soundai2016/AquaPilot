"""Select one experiment and write one read-only analysis handoff."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

from aquapilot import __version__
from aquapilot.provenance import PROTOCOL_ID, file_fingerprint

LEGACY = 'aquapilot_closed_loop'
FORMAT = 'aquapilot_analysis_handoff'
CURRENT = 'results/watermark_inband'
EARLIER = 'results/watermark'


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def choose_source(root, supplied):
    if supplied is not None:
        return (root / supplied).resolve()
    for folder in (root / CURRENT, root / EARLIER):
        model = folder / 'aquapilot'
        aggregate = folder / 'results.zip'
        if model.is_dir() and any(model.iterdir()):
            # A user may have archived the 36 old run folders and kept only analysis/.
            # Only the earlier location may use its own sibling ZIP in this case.
            if (folder == root / EARLIER and aggregate.is_file() and
                    all(p.name in {'analysis', 'tables'} for p in model.iterdir())):
                return aggregate
            return model
        if aggregate.is_file():
            return aggregate
    raise FileNotFoundError(f'No training outputs found under {CURRENT} or {EARLIER}. Pass --results MODEL_DIRECTORY or --results PATH/TO/results.zip.')


def protocol(source):
    if source.is_file():
        if not zipfile.is_zipfile(source):
            raise ValueError('Pass a model directory or a numeric results ZIP.')
        with zipfile.ZipFile(source) as archive:
            if 'export_manifest.json' in archive.namelist():
                raise ValueError('This is already an AquaPilot analysis handoff. Upload it directly.')
            if 'bundle_manifest.json' not in archive.namelist() or archive.getinfo('bundle_manifest.json').file_size > 2_000_000:
                raise ValueError('Expected a numeric results ZIP with a bundle_manifest.json.')
            return json.loads(archive.read('bundle_manifest.json')).get('protocol_id')
    if not source.is_dir():
        raise FileNotFoundError(f'Results not found: {source}')
    report = source / 'analysis/analysis_summary.json'
    if not report.is_file():
        report = next(source.rglob('run_manifest.json'), None)
    if report is None:
        raise ValueError(f'No completed experiment or analysis at {source}. For an archived run use --results PATH/TO/results.zip.')
    return json.loads(report.read_bytes()).get('protocol_id')


def archive_bytes(path, members):
    with zipfile.ZipFile(path, 'w') as z:
        for name, data in sorted(members.items()):
            z.writestr(zipfile.ZipInfo(name), data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)


def check_old_handoff(path):
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if 'export_manifest.json' not in names or z.getinfo('export_manifest.json').file_size > 2_000_000:
            raise FileExistsError(f'{path} already exists and is not an AquaPilot handoff. Choose a different --output.')
        prior = json.loads(z.read('export_manifest.json'))
        members = prior.get('members', {})
        if (prior.get('format') != FORMAT or len(names) != len(set(names)) or
            set(names) != set(members) | {'export_manifest.json'} or
            sum(x.file_size for x in z.infolist()) > 256 * 1024**2 or
            any(z.getinfo(n).file_size != spec['size_bytes'] or
                hashlib.sha256(z.read(n)).hexdigest() != spec['sha256'] for n, spec in members.items())):
            raise FileExistsError(f'Existing output is not a valid earlier handoff. Choose another --output: {path}')


def publish(output, payload):
    # Linux: lock the directory itself. Never overwrite training outputs or an unknown ZIP.
    import fcntl

    output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(output.parent, os.O_RDONLY)
    try:
        with tempfile.TemporaryDirectory(dir=output.parent, prefix='.aquapilot_export_') as temp:
            staged = Path(temp) / 'handoff.zip'
            archive_bytes(staged, payload)
            with staged.open('rb') as fp:
                os.fsync(fp.fileno())
            fcntl.flock(fd, fcntl.LOCK_EX)
            if output.is_symlink():
                raise ValueError(f'Not replacing a symbolic link: {output}')
            if output.exists():
                check_old_handoff(output)
                previous = file_fingerprint(output)
                if previous == file_fingerprint(staged):
                    return
                history = output.parent / 'export_history'
                if history.is_symlink():
                    raise ValueError('export_history must be a regular directory.')
                history.mkdir(exist_ok=True)
                keep = history / f"{output.stem}-{previous['sha256']}.zip"
                if keep.is_symlink() or (keep.exists() and file_fingerprint(keep) != previous):
                    raise ValueError(f'Invalid prior copy: {keep}')
                if not keep.exists():
                    os.link(output, keep)
            os.replace(staged, output)
            os.fsync(fd)
    finally:
        os.close(fd)


def handoff(root, args, export_bundle):
    root = Path(root).resolve()
    source = choose_source(root, args.results)
    out_arg = root / args.output
    output = out_arg.parent.resolve() / out_arg.name
    if (output.suffix.lower() != '.zip' or output.is_symlink() or output == source or
        output.is_relative_to(root / 'data') or (source.is_dir() and output.is_relative_to(source))):
        raise ValueError('Choose a new ZIP outside the dataset and the source experiment.')
    pid = protocol(source)
    if pid not in (PROTOCOL_ID, LEGACY):
        raise ValueError(f'Unrecognized training protocol: {pid!r}. No data were exported.')
    info = {'format': FORMAT, 'software_version': __version__, 'protocol_id': pid}
    payload = {}
    with tempfile.TemporaryDirectory(prefix='aquapilot_post_') as temp:
        scratch = Path(temp)
        if pid == LEGACY:
            if args.axes:
                raise ValueError('A corrected receipt cannot be attached to an earlier training run.')
            from aquapilot.post_legacy import legacy_results
            from scripts.check_watermark_axes import diagnose

            # If both exist on the host, the saved ZIP and regenerated numbers must agree.
            old_zip = source.parent / 'results.zip' if source.is_dir() else None
            payload, legacy = legacy_results(source, prior_zip=old_zip if args.results is None and old_zip and old_zip.is_file() else None)
            probe = diagnose((root / args.raw_root).resolve(), (root / args.prepared_root).resolve(),
                             full=args.full_audit, trained_manifest=legacy.get('prepared_input_manifest_by_fold'))
            payload['input_check.json'] = json_bytes(probe)
            info.update(legacy)
            info['input_check'] = {k: probe[k] for k in ('status','compared_records','compared_copies','scope')}
            info['input_check']['trained_input_binding'] = probe['trained_input_binding']['status']
            info.pop('paper_eligible', None)
            info['classification'] = 'legacy_diagnostic'
            info['manuscript_eligible'] = False
            text = ('Earlier Watermark experiment (diagnostics only).\n'
                    'Start with input_check.json: it describes the MAT/NPZ checks actually possible on this host.\n'
                    'legacy/ contains the earlier numerical evidence. The quarter-budget pairs are top-B minus comb (dB).\n'
                    'These results do not establish performance on the corrected in-band frequency grid.\n')
        else:
            if args.full_audit:
                raise ValueError('--full-audit applies to earlier results. Corrected runs use the full axes.json receipt.')
            if source.is_dir():
                results_zip = scratch / 'results.zip'
                analysis = source / 'analysis'
                if not (analysis / 'analysis_summary.json').is_file():
                    from aquapilot.analysis import analyze_results
                    analysis = scratch / 'rebuilt_analysis'
                    analyze_results([source], analysis, expect_watermark_matrix=True, allow_incomplete=args.diagnostic)
                summary = export_bundle(source, results_zip, diagnostic=args.diagnostic, analysis=analysis)
            else:
                if args.diagnostic:
                    raise ValueError('Use --diagnostic with a run directory, not a pre-exported aggregate.')
                results_zip = source
                with zipfile.ZipFile(results_zip) as z:
                    summary = json.loads(z.read('bundle_manifest.json'))
            if args.diagnostic:
                payload['results.zip'] = results_zip.read_bytes()
                info.update(classification='partial_or_test_diagnostic', manuscript_eligible=False, num_runs=summary['num_runs'])
                text = 'Explicit diagnostic export. This bundle is not a complete, verified measured-channel experiment.\n'
            else:
                from scripts.post_corrected import add_corrected_files
                receipt = (root / args.axes).resolve() if args.axes else source.parent / 'axes.json'
                if not receipt.is_file():
                    raise ValueError(f'Corrected export needs the independent raw-to-prepared receipt: {receipt}. Use --axes PATH/TO/axes.json.')
                files = add_corrected_files(results_zip, receipt, scratch)
                payload = {name: path.read_bytes() for name, path in files.items()}
                info.update(classification='verified_corrected', manuscript_eligible=True, num_runs=summary['num_runs'],
                            input_source='run_tree' if source.is_dir() else 'numeric_archive')
                text = ('Corrected in-band Watermark experiment.\n'
                        'results.zip: numerical tables and paired comparisons. axes.json: raw-to-CFR verification.\n')
        payload['READ_ME.txt'] = text.encode()
        info['members'] = {n: {'size_bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()} for n,b in sorted(payload.items())}
        payload['export_manifest.json'] = json_bytes(info)
        publish(output, payload)
    return info, output
