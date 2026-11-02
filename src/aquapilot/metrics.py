"""Channel errors, nominal intervals, and within-frame uncertainty rankings.

Arrays are [frame, frequency, 2] real/imaginary pairs. Observation masks are
True on pilots. Undefined frame ratios/correlations remain NaN and are counted.
"""
from __future__ import annotations
from typing import Any
import numpy as np
import torch
from scipy.stats import rankdata

METRIC_SCHEMA_VERSION = 2
NMSE_LOG_FLOOR = 1e-12
NORMAL_90_Z = 1.6448536269514722


def _numpy(value: Any) -> np.ndarray:
    return np.asarray(value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value)


def _pair(mean: Any, target: Any) -> tuple[np.ndarray, np.ndarray]:
    m, y = _numpy(mean), _numpy(target)
    if m.shape != y.shape or m.ndim != 3 or m.shape[-1] != 2:
        raise ValueError('Expected matching [frames, subcarriers, 2] real/imaginary arrays.')
    if m.dtype.kind not in 'fiu' or y.dtype.kind not in 'fiu':
        raise ValueError('Real-valued real/imaginary arrays are required.')
    return m.astype(np.float64, copy=False), y.astype(np.float64, copy=False)


def _mask(value: Any | None, shape: tuple[int, int]) -> np.ndarray:
    if value is None:
        return np.ones(shape, dtype=bool)
    a = _numpy(value)
    if a.shape == (shape[1],):
        a = np.broadcast_to(a, shape)
    if a.shape != shape or not np.isin(a, [0, 1]).all():
        raise ValueError('A mask must be boolean/0/1 and match [frames, subcarriers].')
    return a.astype(bool, copy=False)


def _errors(m: np.ndarray, y: np.ndarray, selected: np.ndarray,
            scale: np.ndarray, rescaled: bool) -> dict[str, np.ndarray]:
    n = selected.sum(axis=1)
    bad = ((~np.isfinite(m).all(axis=-1) | ~np.isfinite(y).all(axis=-1)) & selected).any(axis=1)
    with np.errstate(over='ignore', invalid='ignore'):
        err = np.where(selected[..., None], (m - y) ** 2, 0).sum(axis=(1, 2)) * scale**2
        power = np.where(selected[..., None], y**2, 0).sum(axis=(1, 2)) * scale**2
    bad |= ~np.isfinite(err) | ~np.isfinite(power)
    valid = (~bad) & (n > 0) & (power > 0)
    ratio = np.full(len(m), np.nan)
    np.divide(err, power, out=ratio, where=valid)
    db = np.full(len(m), np.nan)
    db[valid] = 10 * np.log10(np.maximum(ratio[valid], NMSE_LOG_FLOOR))
    return dict(nmse_db=db, nmse_linear=ratio, channel_nrmse_percent=100*np.sqrt(ratio),
                squared_error_energy=np.where((~bad) & (n > 0), err, np.nan),
                target_energy=np.where((~bad) & (n > 0), power, np.nan),
                valid_error_frame=valid, zero_power_frame=(~bad) & (n > 0) & (power == 0),
                nonfinite_frame=bad, empty_mask_frame=n == 0,
                zero_error_frame=valid & (err == 0), below_log_floor_frame=valid & (ratio < NMSE_LOG_FLOOR),
                num_selected_bins=n, energy_rescale_applied=np.full(len(m), rescaled, dtype=bool))


def _ranks(m: np.ndarray, s: np.ndarray, y: np.ndarray, selected: np.ndarray,
           chunk_size: int = 2048) -> dict[str, np.ndarray]:
    count = selected.sum(axis=1)
    bad = ((~np.isfinite(m).all(axis=-1) | ~np.isfinite(y).all(axis=-1)
            | ~np.isfinite(s).all(axis=-1) | (s < 0).any(axis=-1)) & selected).any(axis=1)
    rho = np.full(len(m), np.nan)
    const_u, const_v = np.zeros(len(m), bool), np.zeros(len(m), bool)
    for start in range(0, len(m), chunk_size):
        end = min(start+chunk_size, len(m))
        use = selected[start:end]
        with np.errstate(over='ignore', invalid='ignore'):
            score = np.sum(s[start:end]**2, axis=-1)
            error = np.sum((m[start:end]-y[start:end])**2, axis=-1)
        bad[start:end] |= ((~np.isfinite(score) | ~np.isfinite(error)) & use).any(axis=1)
        u = rankdata(np.where(use, score, np.nan), axis=1, method='average', nan_policy='omit')
        v = rankdata(np.where(use, error, np.nan), axis=1, method='average', nan_policy='omit')
        denom_n = np.maximum(count[start:end], 1)[:, None]
        u = np.where(use, u-np.nansum(u, axis=1, keepdims=True)/denom_n, 0)
        v = np.where(use, v-np.nansum(v, axis=1, keepdims=True)/denom_n, 0)
        u2, v2 = np.sum(u*u, axis=1), np.sum(v*v, axis=1)
        eligible = (~bad[start:end]) & (count[start:end] >= 2)
        const_u[start:end], const_v[start:end] = eligible & (u2 == 0), eligible & (v2 == 0)
        denom = np.sqrt(u2*v2)
        values = np.full(end-start, np.nan)
        np.divide(np.sum(u*v, axis=1), denom, out=values, where=eligible & np.isfinite(denom) & (denom > 0))
        rho[start:end] = np.clip(values, -1, 1)
    return dict(rank_spearman=rho, rank_valid_frame=np.isfinite(rho),
                rank_constant_score_frame=const_u, rank_constant_error_frame=const_v,
                rank_nonfinite_frame=bad, rank_insufficient_bins_frame=count < 2)


def _coverage(m: np.ndarray, s: np.ndarray, y: np.ndarray, selected: np.ndarray,
              z: float = NORMAL_90_Z) -> dict[str, np.ndarray]:
    count = 2*selected.sum(axis=1)
    bad = ((~np.isfinite(m).all(axis=-1) | ~np.isfinite(y).all(axis=-1)
            | ~np.isfinite(s).all(axis=-1) | (s < 0).any(axis=-1)) & selected).any(axis=1)
    valid = (~bad) & (count > 0)
    covered = ((y >= m-z*s) & (y <= m+z*s) & selected[..., None]).sum(axis=(1, 2))
    values = np.full(len(m), np.nan)
    np.divide(covered, count, out=values, where=valid)
    return dict(coverage_90=values, coverage_valid_frame=valid, coverage_invalid_frame=bad,
                covered_component_count=np.where(valid, covered, 0), evaluated_component_count=np.where(valid, count, 0))


def framewise_diagnostics(mean: Any, sigma: Any | None, target: Any, *,
                          observed_mask: Any | None = None, normalization_scale: Any | None = None,
                          compute_ranking: bool = True) -> dict[str, np.ndarray]:
    """Compact exact frame diagnostics; provided scales restore physical energies.

    A common amplitude scale leaves framewise ratios, coverage, and ranks
    unchanged. It changes pooled energy weights and therefore must be supplied
    when normalized frames are pooled on a physical-amplitude basis.
    """
    m, y = _pair(mean, target)
    scale = np.ones(len(m), dtype=float)
    if normalization_scale is not None:
        scale = _numpy(normalization_scale).astype(float)
        if scale.ndim == 0:
            scale = np.full(len(m), float(scale))
        if scale.shape != (len(m),) or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError('normalization_scale must be finite, positive, and scalar or [frames].')
    subsets = {'': np.ones(m.shape[:2], dtype=bool)}
    if observed_mask is not None:
        subsets['nonpilot_'] = ~_mask(observed_mask, m.shape[:2])
    s = None if sigma is None else _numpy(sigma).astype(float)
    if s is not None and s.shape != m.shape:
        raise ValueError('sigma must match the prediction shape.')
    result = {}
    for prefix, selected in subsets.items():
        values = _errors(m, y, selected, scale, normalization_scale is not None)
        if s is not None:
            values.update(_coverage(m, s, y, selected))
            if compute_ranking:
                values.update(_ranks(m, s, y, selected))
        result.update({prefix+k: v for k, v in values.items()})
    return result


def _mean(a: Any) -> float | None:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else None


def _std(a: Any) -> float | None:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.std(ddof=0)) if a.size else None


def _rank_summary(d: dict[str, np.ndarray]) -> dict[str, Any]:
    return dict(mean_frame_spearman=_mean(d['rank_spearman']), std_frame_spearman=_std(d['rank_spearman']),
                num_frames=len(d['rank_spearman']), num_valid_frames=int(np.count_nonzero(d['rank_valid_frame'])),
                num_undefined_frames=int(np.count_nonzero(~d['rank_valid_frame'].astype(bool))),
                num_constant_score_frames=int(np.count_nonzero(d['rank_constant_score_frame'])),
                num_constant_error_frames=int(np.count_nonzero(d['rank_constant_error_frame'])),
                num_nonfinite_or_negative_scale_frames=int(np.count_nonzero(d['rank_nonfinite_frame'])),
                num_insufficient_bins_frames=int(np.count_nonzero(d['rank_insufficient_bins_frame'])),
                definition='Per-frame Spearman across frequency of sum_c sigma_c^2 versus sum_c (mean_c-target_c)^2; finite frame correlations averaged equally.',
                uncertainty_scope='Descriptive frame variability; no independent-frame confidence interval.')


def summarize_diagnostics(diagnostics: dict[str, np.ndarray]) -> dict[str, Any]:
    """Same aggregation for full arrays and concatenated compact record outputs."""
    d = {k: np.asarray(v) for k, v in diagnostics.items() if not k.startswith('nonpilot_')}
    n = len(d['nmse_db'])
    if any(v.shape != (n,) for v in d.values()):
        raise ValueError('Each compact diagnostic must be a one-dimensional per-frame array.')
    energy_valid = np.isfinite(d['target_energy']) & np.isfinite(d['squared_error_energy'])
    power = float(d['target_energy'][energy_valid].sum())
    error = float(d['squared_error_energy'][energy_valid].sum())
    pooled = error/power if power > 0 else None
    rescaled = d['energy_rescale_applied'].astype(bool)
    if rescaled.any() and not rescaled.all():
        raise ValueError('Cannot mix rescaled and unscaled energy diagnostics.')
    result = dict(schema_version=METRIC_SCHEMA_VERSION, num_frames=n,
                  num_valid_error_frames=int(d['valid_error_frame'].sum()),
                  num_zero_power_frames=int(d['zero_power_frame'].sum()),
                  num_nonfinite_frames=int(d['nonfinite_frame'].sum()),
                  num_empty_mask_frames=int(d['empty_mask_frame'].sum()),
                  num_zero_error_frames=int(d['zero_error_frame'].sum()),
                  num_below_log_floor_frames=int(d['below_log_floor_frame'].sum()),
                  nmse_db_mean=_mean(d['nmse_db']), nmse_db_std=_std(d['nmse_db']),
                  nmse_linear_mean=_mean(d['nmse_linear']),
                  channel_nrmse_percent_mean=_mean(d['channel_nrmse_percent']),
                  channel_nrmse_percent_std=_std(d['channel_nrmse_percent']),
                  pooled_nmse_linear=pooled,
                  pooled_nmse_db=10*float(np.log10(max(pooled, NMSE_LOG_FLOOR))) if pooled is not None else None,
                  pooled_target_energy=power, pooled_squared_error_energy=error,
                  num_pooled_energy_frames=int(energy_valid.sum()),
                  energy_coordinates='physical_via_provided_scale' if n and rescaled.all() else 'input_coordinates')
    result['metric_definitions'] = {
        'frame_ratio': 'sum_selected |estimate-target|^2 / sum_selected |target|^2; zero reference power is undefined',
        'nmse_db_mean': 'Arithmetic mean of valid framewise 10 log10(error ratio)',
        'channel_nrmse_percent_mean': 'Arithmetic mean of valid framewise 100 sqrt(error ratio); not detected-symbol EVM',
        'pooled_nmse_linear': 'Sum of finite frame error energies / sum of corresponding reference energies, including zero-power-reference frames',
        'std': 'Descriptive population standard deviation (ddof=0), not a confidence interval',
        'log_floor_linear': NMSE_LOG_FLOOR,
        'undefined': 'Undefined frame values excluded and counted; summaries use null when no finite values exist',
    }
    if 'coverage_90' in d:
        cov = _mean(d['coverage_90'])
        count = int(d['evaluated_component_count'].sum())
        result.update(coverage_90=cov, coverage_error_90=abs(cov-.9) if cov is not None else None,
                      coverage_90_component_weighted=float(d['covered_component_count'].sum())/count if count else None,
                      num_valid_coverage_frames=int(d['coverage_valid_frame'].sum()),
                      num_invalid_coverage_frames=int(d['coverage_invalid_frame'].sum()), num_evaluated_components=count)
    if 'rank_spearman' in d:
        result['rank'] = _rank_summary(d)
    if any(k.startswith('nonpilot_') for k in diagnostics):
        result['nonpilot'] = summarize_diagnostics({k.removeprefix('nonpilot_'): v for k, v in diagnostics.items() if k.startswith('nonpilot_')})
    return result


def summarize_estimates(mean: Any, target: Any, *, observed_mask: Any | None = None,
                        normalization_scale: Any | None = None) -> dict[str, Any]:
    """Errors for deterministic estimates; no test-residual uncertainty fitted."""
    return summarize_diagnostics(framewise_diagnostics(mean, None, target, observed_mask=observed_mask,
                                                       normalization_scale=normalization_scale))


def summarize_tensors(mean: Any, sigma: Any, target: Any, *, observed_mask: Any | None = None,
                      normalization_scale: Any | None = None) -> dict[str, Any]:
    """Errors and nominal interval coverage for model-supplied predictive scales."""
    return summarize_diagnostics(framewise_diagnostics(mean, sigma, target, observed_mask=observed_mask,
                                                       normalization_scale=normalization_scale, compute_ranking=False))


def summarize_prior_ranking(mean: Any, sigma: Any, target: Any, *, observed_mask: Any | None = None) -> dict[str, Any]:
    """Use only predictions made before the current pilots are acquired."""
    m, y = _pair(mean, target)
    s = _numpy(sigma).astype(float)
    if s.shape != m.shape:
        raise ValueError('sigma must match mean and target.')
    result = _rank_summary(_ranks(m, s, y, np.ones(m.shape[:2], dtype=bool)))
    if observed_mask is not None:
        result['nonpilot'] = _rank_summary(_ranks(m, s, y, ~_mask(observed_mask, m.shape[:2])))
    return result


def _tensor_metric(pred: torch.Tensor, target: torch.Tensor, name: str, mask: Any | None = None) -> torch.Tensor:
    m, y = _pair(pred, target)
    values = _errors(m, y, _mask(mask, m.shape[:2]), np.ones(len(m)), False)[name]
    return torch.as_tensor(values, dtype=torch.float64, device=pred.device)


def nmse_db(pred: torch.Tensor, target: torch.Tensor, mask: Any | None = None) -> torch.Tensor:
    return _tensor_metric(pred, target, 'nmse_db', mask)


def channel_nrmse_percent(pred: torch.Tensor, target: torch.Tensor, mask: Any | None = None) -> torch.Tensor:
    return _tensor_metric(pred, target, 'channel_nrmse_percent', mask)


def interval_coverage(mean: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor, z: float = NORMAL_90_Z) -> torch.Tensor:
    m, y = _pair(mean, target)
    s = _numpy(sigma).astype(float)
    if s.shape != m.shape:
        raise ValueError('sigma must match mean and target.')
    values = _coverage(m, s, y, np.ones(m.shape[:2], dtype=bool), z)['coverage_90']
    return torch.as_tensor(values, dtype=torch.float64, device=mean.device)


def uncertainty_error_spearman(mean: Any, sigma: Any, target: Any) -> float | None:
    """Mean within-snapshot Spearman correlation across subcarriers."""
    return summarize_prior_ranking(mean, sigma, target)['mean_frame_spearman']


class DiagnosticsAccumulator:
    """Combine recording-level batches by their frame counts and central moments.
    """

    def __init__(self) -> None:
        self._stats: dict[str, dict[str, float | int]] = {}
        self._keys: set[str] | None = None
        self._num_frames = 0

    def update(self, diagnostics: dict[str, np.ndarray]) -> None:
        d = {key: np.asarray(value) for key, value in diagnostics.items()}
        if 'nmse_db' not in d:
            raise ValueError('Missing per-frame NMSE diagnostics.')
        n = len(d['nmse_db'])
        if any(value.shape != (n,) for value in d.values()):
            raise ValueError('Every diagnostic must be a one-dimensional per-frame array.')
        if self._keys is not None and set(d) != self._keys:
            raise ValueError('Diagnostic schemas differ between recordings.')
        self._keys = set(d)
        self._num_frames += n
        for prefix in ('', 'nonpilot_'):
            if prefix+'target_energy' not in d:
                continue
            valid = np.isfinite(d[prefix+'target_energy']) & np.isfinite(d[prefix+'squared_error_energy'])
            d[prefix+'energy_valid_frame'] = valid
            for name in ('target_energy', 'squared_error_energy'):
                d[prefix+name] = np.where(valid, d[prefix+name], np.nan)
        for key, value in d.items():
            x = value.astype(np.float64, copy=False)
            finite = x[np.isfinite(x)]
            new_n = len(finite)
            state = self._stats.setdefault(key, {'count': 0, 'mean': 0., 'm2': 0., 'sum': 0.})
            if not new_n:
                continue
            new_mean = float(finite.mean())
            new_m2 = float(np.sum((finite-new_mean)**2))
            old_n = int(state['count'])
            combined = old_n+new_n
            delta = new_mean-float(state['mean'])
            state['mean'] = float(state['mean']) + delta*new_n/combined
            state['m2'] = float(state['m2']) + new_m2 + delta**2*old_n*new_n/combined
            state['sum'] = float(state['sum']) + float(finite.sum())
            state['count'] = combined

    def _summary(self, prefix: str = '') -> dict[str, Any]:
        def state(key: str) -> dict[str, float | int]:
            return self._stats[prefix+key]
        def mean(key: str) -> float | None:
            s = state(key)
            return float(s['mean']) if s['count'] else None
        def std(key: str) -> float | None:
            s = state(key)
            return float(np.sqrt(max(0., float(s['m2']))/int(s['count']))) if s['count'] else None
        def total(key: str) -> float:
            return float(state(key)['sum'])
        def count(key: str) -> int:
            return int(round(total(key)))
        n = self._num_frames
        scaled = count('energy_rescale_applied')
        if scaled not in (0, n):
            raise ValueError('Cannot mix rescaled and unscaled energy diagnostics.')
        power, error = total('target_energy'), total('squared_error_energy')
        pooled = error/power if power > 0 else None
        result = dict(schema_version=METRIC_SCHEMA_VERSION, num_frames=n,
                      num_valid_error_frames=count('valid_error_frame'),
                      num_zero_power_frames=count('zero_power_frame'),
                      num_nonfinite_frames=count('nonfinite_frame'),
                      num_empty_mask_frames=count('empty_mask_frame'),
                      num_zero_error_frames=count('zero_error_frame'),
                      num_below_log_floor_frames=count('below_log_floor_frame'),
                      nmse_db_mean=mean('nmse_db'), nmse_db_std=std('nmse_db'),
                      nmse_linear_mean=mean('nmse_linear'),
                      channel_nrmse_percent_mean=mean('channel_nrmse_percent'),
                      channel_nrmse_percent_std=std('channel_nrmse_percent'),
                      pooled_nmse_linear=pooled,
                      pooled_nmse_db=10*float(np.log10(max(pooled, NMSE_LOG_FLOOR))) if pooled is not None else None,
                      pooled_target_energy=power, pooled_squared_error_energy=error,
                      num_pooled_energy_frames=count('energy_valid_frame'),
                      energy_coordinates='physical_via_provided_scale' if n and scaled == n else 'input_coordinates')
        result['metric_definitions'] = {
            'frame_ratio': 'sum_selected |estimate-target|^2 / sum_selected |target|^2; zero reference power is undefined',
            'nmse_db_mean': 'Arithmetic mean of valid framewise 10 log10(error ratio)',
            'channel_nrmse_percent_mean': 'Arithmetic mean of valid framewise 100 sqrt(error ratio); not detected-symbol EVM',
            'pooled_nmse_linear': 'Sum of finite frame error energies / sum of corresponding reference energies, including zero-power-reference frames',
            'std': 'Descriptive population standard deviation (ddof=0), not a confidence interval',
            'log_floor_linear': NMSE_LOG_FLOOR,
            'undefined': 'Undefined frame values excluded and counted; summaries use null when no finite values exist',
        }
        if prefix+'coverage_90' in self._stats:
            cov = mean('coverage_90')
            components = count('evaluated_component_count')
            result.update(coverage_90=cov, coverage_error_90=abs(cov-.9) if cov is not None else None,
                          coverage_90_component_weighted=total('covered_component_count')/components if components else None,
                          num_valid_coverage_frames=count('coverage_valid_frame'),
                          num_invalid_coverage_frames=count('coverage_invalid_frame'), num_evaluated_components=components)
        if prefix+'rank_spearman' in self._stats:
            result['rank'] = dict(mean_frame_spearman=mean('rank_spearman'), std_frame_spearman=std('rank_spearman'),
                                  num_frames=n, num_valid_frames=count('rank_valid_frame'),
                                  num_undefined_frames=n-count('rank_valid_frame'),
                                  num_constant_score_frames=count('rank_constant_score_frame'),
                                  num_constant_error_frames=count('rank_constant_error_frame'),
                                  num_nonfinite_or_negative_scale_frames=count('rank_nonfinite_frame'),
                                  num_insufficient_bins_frames=count('rank_insufficient_bins_frame'),
                                  definition='Per-frame Spearman across frequency of sum_c sigma_c^2 versus sum_c (mean_c-target_c)^2; finite frame correlations averaged equally.',
                                  uncertainty_scope='Descriptive frame variability; no independent-frame confidence interval.')
        return result

    def summary(self) -> dict[str, Any]:
        if self._keys is None:
            raise ValueError('Cannot summarize an accumulator before any diagnostic update.')
        result = self._summary()
        if 'nonpilot_nmse_db' in self._stats:
            result['nonpilot'] = self._summary('nonpilot_')
        return result
