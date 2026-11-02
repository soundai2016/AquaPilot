"""CSV sanitization and ordered training-file receipts shared by both export protocols."""
from __future__ import annotations

import csv
import io
import re
from pathlib import Path


def _source_table(path: Path, sources: dict[str, str]) -> bytes:
    """Preserve blank statistics and their counts; replace source paths with hashes."""
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if not fields:  # A table with no QA findings or no 1/4 configuration can be empty.
            return b''
        if len(set(fields)) != len(fields):
            raise ValueError(f'{path.name}: duplicate column name')
        buffer = io.StringIO(newline='')
        out_fields = ['metrics_sha256' if key == 'source_metrics' else key for key in fields]
        writer = csv.DictWriter(buffer, fieldnames=out_fields, lineterminator='\n')
        writer.writeheader()
        for row in reader:
            if None in row or None in row.values():
                raise ValueError(f'{path.name}: malformed CSV row')
            if 'source_metrics' in row:
                source = row.pop('source_metrics')
                if source not in sources:
                    raise ValueError(f'{path.name}: unknown source run')
                row['metrics_sha256'] = sources[source]
            if 'environment' in row and not re.fullmatch(r'[A-Za-z0-9_.-]{1,100}', row['environment']):
                raise ValueError('An environment label contains a path or unsupported characters; choose a shareable label.')
            writer.writerow(row)
        return buffer.getvalue().encode('utf-8')


def _fold_inputs(manifest: dict, environment: str) -> dict:
    """Check ordered training/validation/test input identities saved at training time."""
    from aquapilot.data.audit import EXPECTED_COUNTS, SPLITS, _inventory, expected_split_groups
    from aquapilot.provenance import fingerprint

    cfg = manifest['resolved_config']['data']
    if environment not in EXPECTED_COUNTS or Path(cfg['root']).name != 'fold_test_' + environment:
        raise ValueError('Prepared source fold and the evaluated environment do not agree.')
    groups = expected_split_groups(environment)
    inventory = _inventory(EXPECTED_COUNTS)
    main = manifest['data']['main']
    result = {}
    for split in SPLITS:
        rows = main['splits'][split]
        expected = {f'{split}/{rid}.npz' for rid, (group, _) in inventory.items() if group in groups[split]}
        if (len(rows) != len(expected) or {r['path'] for r in rows} != expected
                or any(not re.fullmatch('[0-9a-f]{64}', r['sha256']) or r['size_bytes'] < 1 for r in rows)):
            raise ValueError(f'{environment}: invalid {split} provenance inventory.')
        digest = fingerprint(rows)  # Record order determines the per-sequence noise seed.
        if digest != main['split_fingerprints'][split]:
            raise ValueError(f'{environment}: invalid {split} fingerprint in the run manifest.')
        result[split] = {'num_files': len(rows), 'ordered_input_fingerprint': digest}
    return result


def _formal_inputs(sources: dict[str, str], cells: set[tuple]) -> dict:
    from aquapilot.analysis import read_json

    environments = {digest: env for env, _, _, digest in cells}
    by_fold = {}
    for path, digest in sources.items():
        env = environments[digest]
        summary = _fold_inputs(read_json(Path(path).parent/'run_manifest.json'), env)
        if by_fold.setdefault(env, summary) != summary:
            raise ValueError(f'{env}: prepared split inputs differ across seeds or pilot fractions.')
    return by_fold
