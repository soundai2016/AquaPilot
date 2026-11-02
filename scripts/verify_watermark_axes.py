#!/usr/bin/env python3
"""Verify WatermarkV1 prepared CFR values against the raw MATLAB TVIR matrices."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from scipy.signal import CZT

from aquapilot.data.audit import EXPECTED_COUNTS, audit_watermark, project_path
from aquapilot.provenance import prepared_fingerprint
from aquapilot.watermark_grid import BANKS, GRID_NAME

if __package__:
    from .prepare_watermark import load_mat, scalar
else:
    from prepare_watermark import load_mat, scalar


def verify_delay_transform(source_h: np.ndarray, prepared: np.ndarray, fraction: float, *,
                           probe_band_hz, fc: float, fs_tau: float) -> float:
    """Independent reference: MATLAB rows are slow time; native DFT uses the columns."""
    h = np.asarray(source_h).squeeze()
    if h.ndim != 2 or prepared.ndim != 2 or h.shape[0] != prepared.shape[0]:
        raise ValueError(f'MATLAB h shape {h.shape}, prepared CSI shape {prepared.shape}: prepared rows must be slow-time snapshots.')
    frames, taps = h.shape
    bins = prepared.shape[1]
    if not 0 < fraction <= 1 or not 2 <= bins <= taps:
        raise ValueError('Incompatible retained band or subcarrier count.')
    # Directly evaluate the finite delay response on the specified physical tones.
    # This chirp-z transform does not use the preparation FFT length or index list.
    spacing = fraction*(probe_band_hz[1]-probe_band_hz[0])/bins
    first = (probe_band_hz[0]+probe_band_hz[1])/2 - fc - (bins//2)*spacing
    check = CZT(taps,m=bins,w=np.exp(-2j*np.pi*spacing/fs_tau),a=np.exp(2j*np.pi*first/fs_tau))
    max_err, ref_peak = 0.0, 0.0
    for i in range(0, frames, 32):
        block = np.asarray(h[i:i+32], dtype=np.complex128)
        expected = check(block, axis=1)
        observed = prepared[i:i+len(block)]
        if not np.isfinite(expected).all() or not np.isfinite(observed).all():
            raise ValueError('Raw or prepared CFR contains a nonfinite value.')
        max_err = max(max_err, float(np.abs(observed-expected).max()))
        ref_peak = max(ref_peak, float(np.abs(expected).max()))
    if ref_peak == 0 or max_err > 4*np.finfo(np.float32).eps*ref_peak:
        raise ValueError(f'Prepared bins differ from a delay-column DFT (max abs error {max_err:.6g}, peak reference {ref_peak:.6g}).')
    return max_err/ref_peak


def verify(raw_root: Path, prepared_root: Path) -> dict:
    raw_root, prepared_root = raw_root.resolve(), prepared_root.resolve()
    audit = audit_watermark(prepared_root)
    if audit['band_fraction'] != 0.8 or audit['n_subcarriers'] != 128:
        raise ValueError('The Watermark manuscript protocol requires 128 uniformly spaced physical tones within the central 80% of the probe band.')
    summaries = []
    fingerprints = {}
    for group in EXPECTED_COUNTS:
        name = 'fold_test_'+group; rows = audit['folds'][name]['records']
        held_out = [r for r in rows if r['path'].startswith(name+'/test/')]
        lengths = []; errors = []
        for record in held_out:
            code = record['channel_code']; stem = record['record_id'].removeprefix(code+'_')
            raw = raw_root/code/'mat'/(stem+'.mat')
            if not raw.is_file() or not raw.resolve().is_relative_to(raw_root):
                raise ValueError(f'Missing or out-of-tree original file: {raw}')
            try:
                fields = load_mat(raw); h = np.asarray(fields['h']).squeeze()
                fc, fs_tau, fs_t = (scalar(fields,k) for k in ('fc','fs_tau','fs_t'))
                if tuple(h.shape) != BANKS[code][0] or any(a is None or not np.isclose(record.get(k, np.nan), a, rtol=1e-9)
                        for k,a in [('fc',fc),('fs_tau',fs_tau),('fs_t',fs_t)]):
                    raise ValueError('Original h dimensions or sampling rates disagree with the prepared record.')
                if list(h.shape) != record.get('matlab_h_shape'):
                    raise ValueError(f'Raw h shape {h.shape} differs from prepared provenance {record.get("matlab_h_shape")}')
                with np.load(prepared_root/record['path'], allow_pickle=False) as z:
                    csi = (z['csi_real'].astype(np.float32) + 1j*z['csi_imag'].astype(np.float32)
                           if 'csi_real' in z and 'csi_imag' in z else z['csi'].astype(np.complex64))
                errors.append(verify_delay_transform(h,csi,audit['band_fraction'],probe_band_hz=BANKS[code][1],fc=fc,fs_tau=fs_tau)); lengths.append(len(h))
            except (KeyError, ValueError, OSError) as exc:
                raise ValueError(f'{raw.name}: {exc}') from exc
        snapshot = prepared_fingerprint(prepared_root/name, ['train','val','test'])
        audited_files = {record['path'].split('/', 1)[1]: record['file'] for record in rows}
        for entries in snapshot['splits'].values():
            for entry in entries:
                if audited_files.get(entry['path']) != {key: entry[key] for key in ('size_bytes','sha256')}:
                    raise ValueError(f'Prepared file changed while checking the delay transform: {entry["path"]}')
        fingerprints[group] = snapshot['split_fingerprints']
        summaries.append(dict(environment=group, records=len(held_out), snapshots=sum(lengths),
                              scored=sum(n-audit['history'] for n in lengths), max_error_over_peak=max(errors)))
    return {'status':'ok','method':'independent_delay_axis_fft',
            'num_unique_records':audit['num_unique_records'],'total_prepared_files':audit['total_prepared_files'],
            'history':audit['history'], 'horizon':audit['horizon'], 'n_subcarriers':audit['n_subcarriers'],
            'band_fraction':audit['band_fraction'], 'frequency_grid':GRID_NAME, 'environments':summaries,
            'ordered_split_fingerprints':fingerprints}


def write_receipt(report: dict, output: Path) -> None:
    """Publish a completed verification; never replace an existing receipt."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=output.parent) as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
        os.link(handle.name, output)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root', type=Path, required=True, help='Watermark/input/channels directory')
    p.add_argument('--prepared-root', type=Path, required=True, help='Prepared four-fold root')
    p.add_argument('--output',type=Path,help='New JSON receipt outside raw and prepared data')
    args = p.parse_args(argv)
    raw, prepared = project_path(args.raw_root), project_path(args.prepared_root)
    output = project_path(args.output) if args.output is not None else None
    if output is not None and (output.suffix != '.json' or output.exists() or any(
            output.is_relative_to(protected) for protected in (raw,prepared,project_path('data')))):
        p.error('Use a new .json receipt outside the raw and prepared data.')
    try:
        report = verify(raw, prepared)
        if output is not None:
            write_receipt(report, output)
    except (OSError, KeyError, ValueError) as exc:
        p.exit(2, f'Axis check failed: {exc}\n')
    print('Held-out group  Records  Raw snapshots  Scored (history=16)  Max |difference|/peak')
    for r in report['environments']:
        print(f"{r['environment']:14} {r['records']:7d} {r['snapshots']:14d} {r['scored']:20d} {r['max_error_over_peak']:22.3e}")
    print('All original records matched the DFT along the MATLAB delay columns; all four copies were checked for identical CSI.')


if __name__ == '__main__':
    main()
