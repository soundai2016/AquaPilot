#!/usr/bin/env python3
"""Identify which delay/frequency convention produced a set of Watermark NPZ files."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from scipy.io import loadmat
from scipy.io.matlab import MatReadError
from scipy.signal import CZT

ROOT = Path(__file__).resolve().parents[1]
if not __package__:
    sys.path.insert(0, str(ROOT/'src'))
from aquapilot.provenance import file_fingerprint, prepared_fingerprint
from aquapilot.watermark_grid import BANKS

FOLDS = ('NOF', 'NCS', 'BCH', 'KAU')
SPLITS = ('train', 'val', 'test')
COUNTS = {'NOF1': 60, 'NCS1': 60, 'BCH1': 4, 'KAU1': 16, 'KAU2': 16}
# Allow small FFT roundoff when comparing files created with different NumPy versions.
MAX_RELATIVE_TO_PEAK = 2e-6
RAWDIR = Path('data/raw/watermark/WatermarkV1/Watermark/input/channels')
PREPDIR = Path('data/prepared/watermark')


def _raw_data(path):
    try:
        return loadmat(path, variable_names=['h','fc','fs_t','fs_tau'], simplify_cells=True)
    except NotImplementedError:
        # MATLAB v7.3 reverses the order of HDF5 dimensions on disk.
        if __package__:
            from .prepare_watermark import load_mat
        else:
            from prepare_watermark import load_mat
        return load_mat(path)


def _old_grid(h):
    length = h.shape[1]
    width = min(length, max(128, round(.8*length)))
    first = (length-width)//2
    idx = np.linspace(first, first+width-1, 128, dtype=int)
    return np.fft.fftshift(np.fft.fft(h, axis=1), axes=1)[:,idx]


def _relative_error(got, ref):
    if got.shape != ref.shape:
        return None
    peak = float(np.max(np.abs(ref)))
    if not peak or not np.isfinite(peak):
        return None
    return float(np.max(np.abs(got-ref))/peak)


def _references(fields, bank):
    (shape, band, ft0, fd0) = BANKS[bank]
    h = np.asarray(fields['h']).squeeze()
    ft, fd, fc = (float(np.asarray(fields[k]).item()) for k in ('fs_t','fs_tau','fc'))
    if (h.shape!=shape or not np.isfinite(h).all() or not np.any(h) or
            not np.allclose((ft, fd, fc),(ft0, fd0, sum(band)/2),rtol=1e-6,atol=0)):
        raise ValueError('Original dimensions or sampling rates differ from the Watermark bank.')
    df = .8*(band[1]-band[0])/128
    base = np.arange(-64,64)*df
    length = h.shape[1]
    czt = CZT(length, m=128, a=np.exp(2j*np.pi*base[0]/fd), w=np.exp(-2j*np.pi*df/fd))
    in_band = czt(h, axis=1)
    period = round(fd/df)
    padded = ((length+period-1)//period)*period
    return h, {'in_band':in_band,'old_band':_old_grid(h)}, {
        'subcarrier_hz': fc+base,
        'padded_dft_length':padded,
        'shifted_dft_bin_indices': padded//2 + np.arange(-64,64)*(padded//period)}


def _read_cfr(record):
    if 'csi_real' in record and 'csi_imag' in record:
        re, im = record['csi_real'], record['csi_imag']
        if re.shape != im.shape or re.dtype.kind not in 'fiu' or im.dtype.kind not in 'fiu':
            raise ValueError('Invalid real/imaginary response arrays.')
        result = re+1j*im
    else:
        result = np.asarray(record['csi'])
    if result.ndim != 2 or result.dtype.kind != 'c' or not np.isfinite(result).all():
        raise ValueError('Expected a finite complex frequency-response matrix.')
    return result


def _compare_copy(path, raw_h, refs, metadata):
    with np.load(path, allow_pickle=False) as z:
        observed = _read_cfr(z)
        if observed.shape == (raw_h.shape[1],128) and 'exchanged_old_band' not in refs:
            refs['exchanged_old_band'] = _old_grid(raw_h.T)
        errors = {k:_relative_error(observed,v) for k,v in refs.items()}
        matches = [k for k,e in errors.items() if e is not None and e<=MAX_RELATIVE_TO_PEAK]
        result = {'npz_shape':list(observed.shape), 'max_error_over_reference_peak':errors}
        result['classification'] = matches[0] if len(matches)==1 else ('ambiguous' if matches else 'other')
        if result['classification']=='in_band':
            # Some legacy containers have no frequency metadata: matching the
            # complex samples by itself does not supply a formal axis receipt.
            bad = []
            for key, ref in metadata.items():
                if key not in z:
                    continue
                got = z[key]
                if np.shape(got) != np.shape(ref) or not np.allclose(got,ref,rtol=0,atol=2e-3 if key=='subcarrier_hz' else 0):
                    bad.append(key)
            if bad:
                result.update(classification='metadata_mismatch', differing_fields=bad)
        return result


def _trained_binding(prepared_root, trained):
    if not trained:
        return {'status':'unavailable', 'reason':'No ordered split fingerprints in the input results.'}
    if not prepared_root.is_dir():
        return {'status':'unavailable', 'reason':'The prepared input directory is unavailable.'}
    splits = []
    for fold in FOLDS:
        for name in SPLITS:
            cited = (trained.get(fold) or {}).get(name, {})
            try:
                snapshot = prepared_fingerprint(prepared_root/('fold_test_'+fold), [name])
                digest = snapshot['split_fingerprints'][name]
                n = len(snapshot['splits'][name])
                if not cited.get('ordered_input_fingerprint'):
                    status = 'missing_trained_fingerprint'
                else:
                    status = 'match' if (cited.get('ordered_input_fingerprint')==digest and cited.get('num_files')==n) else 'mismatch'
                splits.append(dict(fold=fold,split=name,status=status,num_files=n,ordered_input_fingerprint=digest))
            except (OSError, KeyError, TypeError, ValueError):
                splits.append(dict(fold=fold,split=name,status='unreadable'))
    statuses = [r['status'] for r in splits]
    return {'status':'match' if set(statuses)=={'match'} else ('mismatch' if 'mismatch' in statuses else 'incomplete'),
            'matched_splits':statuses.count('match'),'expected_splits':12,'splits':splits}


def diagnose(raw_root: Path, prepared_root: Path, *, full=False, trained_manifest=None, record=None):
    """Compare first record in each bank (or all 156) with each of its four prepared copies."""
    raw_root, prepared_root = Path(raw_root).resolve(), Path(prepared_root).resolve()
    if record and full:
        raise ValueError('Select either a single record or the full inventory.')
    all_records = [(bank,f'{bank}_{i:03d}') for bank,n in COUNTS.items() for i in range(1,n+1)]
    chosen = [r for r in all_records if (r[1]==record if record else (full or r[1].endswith('_001')))]
    if not chosen:
        raise ValueError('Unknown WatermarkV1 source record.')
    binding = _trained_binding(prepared_root,trained_manifest)
    info = dict(method='direct_comparison_of_prepared_complex_response',
                scope=('one_named_record' if record else ('all_156_records' if full else 'first_record_per_bank')),
                raw_available=raw_root.is_dir(), prepared_available=prepared_root.is_dir(),
                expected_records_in_protocol=156, requested_records=len(chosen), requested_copies=len(chosen)*len(FOLDS),
                compared_records=0, compared_copies=0, max_relative_to_peak_tolerance=MAX_RELATIVE_TO_PEAK,
                samples_per_record='all slow-time rows and all 128 tones', trained_input_binding=binding,
                interpretation='Diagnostic for the inspected files only. An ordered-split match also binds the prepared inputs to the cited training run. This is not a corrected-run receipt.')
    if not (info['raw_available'] and info['prepared_available']):
        return {**info,'status':'unavailable','counts':{},'records':[]}
    rows, totals = [], Counter()
    for bank, rid in chosen:
        row = {'record':rid,'copies':[]}
        raw = raw_root/bank/'mat'/(rid+'.mat')
        try:
            if not raw.is_file() or not raw.resolve().is_relative_to(raw_root):
                raise FileNotFoundError
            raw_h, refs, metadata = _references(_raw_data(raw),bank)
            row.update(raw_mat_sha256=file_fingerprint(raw)['sha256'],raw_shape=list(raw_h.shape))
        except (OSError, ImportError, IndexError, KeyError, ValueError, TypeError, MatReadError) as exc:
            row.update(raw_issue='missing_or_outside_root' if isinstance(exc,FileNotFoundError) else 'unreadable_or_unexpected_geometry')
            row['copies']=[{'fold':env,'classification':'raw_unavailable'} for env in FOLDS]
            totals['raw_unavailable']+=len(FOLDS)
            rows.append(row)
            continue
        compared_this = 0
        for env in FOLDS:
            fold = prepared_root/('fold_test_'+env)
            options = [fold/split/(bank+'_'+rid+'.npz') for split in SPLITS]
            found = [p for p in options if p.is_file()]
            if len(found)!=1 or not found[0].resolve().is_relative_to(prepared_root):
                result = {'classification':'missing_duplicate_or_outside_root'}
            else:
                try:
                    result = _compare_copy(found[0],raw_h,refs,metadata)
                    result['split']=found[0].parent.name
                    compared_this += 1
                except (KeyError, ValueError, TypeError, OSError, zipfile.BadZipFile):
                    result = {'classification':'unreadable_prepared'}
            row['copies'].append({'fold':env,**result})
            totals[result['classification']] += 1
        rows.append(row)
        info['compared_copies'] += compared_this
        info['compared_records'] += int(compared_this>0)
    return {**info,'status':'compared' if info['compared_copies']==info['requested_copies'] else 'incomplete',
            'counts':dict(sorted(totals.items())),'records':rows}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channels',type=Path,default=ROOT/RAWDIR)
    parser.add_argument('--prepared',type=Path,default=ROOT/PREPDIR)
    which=parser.add_mutually_exclusive_group()
    which.add_argument('--full',action='store_true',help='Compare all 156 original records (624 fold copies).')
    which.add_argument('--record',help='Compare one record, e.g. NCS1_001.')
    args=parser.parse_args()
    try:
        report=diagnose(args.channels,args.prepared,full=args.full,record=args.record)
    except ValueError as exc:
        parser.exit(2,f'Axis diagnosis failed: {exc}\n')
    print(json.dumps(report,allow_nan=False,sort_keys=True,indent=2))
    return 0 if report['status']=='compared' else 2


if __name__=='__main__':
    raise SystemExit(main())
