"""One-column vector figure from the verified Watermark comparison table."""
from __future__ import annotations

import csv
import io
import math

from matplotlib import rc_context
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle


_GROUPS = ('NOF', 'NCS', 'BCH', 'KAU')
_RATIOS = (1/16, 1/8, 1/4)


def _value(raw: str) -> float | None:
    if raw == '':
        return None
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError('Expected a finite statistic or an empty field for an undefined statistic.')
    return value


def rank_placement_pdf(csv_bytes: bytes) -> bytes:
    """Render equal-seed environment means; never impute an undefined rank."""
    cells = {}
    for row in csv.DictReader(io.StringIO(csv_bytes.decode('utf-8-sig'))):
        key = row['environment'], float(row['pilot_fraction'])
        if key in cells or row['policy'] != 'uncertainty' or int(row['num_seeds']) != 3:
            raise ValueError('Expected one top-B row per environment and budget, with all three seeds.')
        rho, delta = _value(row['equal_seed_mean_frame_spearman']), _value(row['mean_delta_db'])
        if rho is not None and (abs(rho) > 1.0000000001 or int(row['num_seeds_with_defined_ranks']) != 3):
            raise ValueError('Invalid or incompletely supported three-seed rank mean.')
        if delta is not None and int(row['undefined_delta_seed_count']) != 0:
            raise ValueError('Invalid three-seed paired difference.')
        cells[key] = rho, delta
    if set(cells) != {(g,r) for g in _GROUPS for r in _RATIOS}:
        raise ValueError('The comparison figure requires four environments and all three pilot fractions.')
    with rc_context({'font.family':'DejaVu Sans', 'font.size':7.1, 'pdf.fonttype':42}):
        fig = Figure(figsize=(3.34, 1.72))
        for j, (x0, title) in enumerate(((0.13, '(a) Prior rank, ρ'), (0.58, '(b) ΔNMSE (dB)'))):
            ax = fig.add_axes([x0, 0.27, 0.39, 0.58]); ax.set_title(title, fontsize=7.5, pad=4)
            for i,g in enumerate(_GROUPS):
                for k,r in enumerate(_RATIOS):
                    val = cells[g,r][j]
                    color = '#f4f4f4' if val is None or abs(val)<0.005 else ('#dcebf3' if val<0 else '#f8e8dc')
                    ax.add_patch(Rectangle((k-.5,i-.5),1,1,facecolor=color,edgecolor='white',linewidth=.9))
                    label = '—' if val is None else ('0.00' if abs(val)<0.005 else f'{val:+.2f}')
                    ax.text(k,i,label,ha='center',va='center',fontsize=6.8)
            ax.set(xlim=(-.5,2.5), ylim=(3.5,-.5))
            ax.set_xticks(range(3), ['1/16','1/8','1/4'])
            if j == 0:
                ax.set_yticks(range(4), _GROUPS)
            else:
                ax.set_yticks([])
            ax.tick_params(length=0,pad=2.4)
            for spine in ax.spines.values(): spine.set_visible(False)
        fig.text(.55,.11,'Pilot fraction',ha='center',fontsize=7.1)
        fig.text(.55,.014,'Δ = top-B − comb; negative values favor top-B.  — undefined.',ha='center',fontsize=6)
        out = io.BytesIO()
        fig.savefig(out, format='pdf', metadata={'Title':'Watermark: prior ranking and paired placement error', 'CreationDate':None, 'ModDate':None})
        return out.getvalue()
