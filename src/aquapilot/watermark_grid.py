"""WatermarkV1 channel geometry (FFI Watermark manual, §3.1/Table 2.1; original MAT headers)."""
from __future__ import annotations

import numpy as np

# MATLAB h dimensions [slow time, delay], probe band Hz, slow-time Hz, delay sampling Hz.
BANKS = {
    'NOF1': ((258, 2048), (10000, 18000), 7.8125, 16000),
    'NCS1': ((1024, 510), (10000, 18000), 16000/510, 16000),
    'BCH1': ((582, 2044), (32500, 37500), 20000/2044, 20000),
    'KAU1': ((258, 1024), (4000, 8000), 7.8125, 8000),
    'KAU2': ((258, 1024), (4000, 8000), 7.8125, 8000),
}
GRID_NAME = 'uniform_inner_probe_band'


def subcarrier_grid(code, taps, count, fraction, fc, fs_tau):
    _, band, _, native_rate = BANKS[code]
    if fc is None or fs_tau is None or not (np.isclose(fc, sum(band)/2, rtol=1e-6) and np.isclose(fs_tau, native_rate, rtol=1e-6)):
        raise ValueError(f'{code}: expected native fc={sum(band)/2:g} Hz and fs_tau={native_rate} Hz, found {fc}, {fs_tau}.')
    if not 0 < fraction <= 1:
        raise ValueError('The fraction of the published probe band must be in (0,1].')
    if not isinstance(count,(int,np.integer)) or not 2 <= count <= taps:
        raise ValueError(f'{code}: need 2 to {taps} subcarriers on this delay record.')
    spacing = fraction*(band[1]-band[0])/count
    period = int(round(fs_tau/spacing))
    if not np.isclose(fs_tau/spacing,period,rtol=1e-12,atol=1e-12):
        raise ValueError('This fraction and tone count require a noninteger FFT period; choose a commensurate grid.')
    # Zero padding evaluates the finite delay response at the requested tones. It
    # does not improve the delay resolution of the original acoustic sounding.
    padded_length = ((taps+period-1)//period)*period
    offsets = np.arange(count,dtype=np.int64) - count//2
    indices = padded_length//2 + offsets*(padded_length//period)
    frequencies = fc + offsets*spacing
    if indices.min()<0 or indices.max()>=padded_length:
        raise ValueError('Requested tones are outside the sampled delay-FFT grid.')
    return padded_length, indices, frequencies
