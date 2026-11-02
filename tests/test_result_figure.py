import csv
import io

import pytest

from aquapilot.result_figure import rank_placement_pdf


def fixture_table():
    s=io.StringIO(); w=csv.writer(s); w.writerow(['environment','pilot_fraction','policy','num_seeds','num_seeds_with_defined_ranks','equal_seed_mean_frame_spearman','undefined_delta_seed_count','mean_delta_db'])
    for i,g in enumerate(('NOF','NCS','BCH','KAU')):
        for j,r in enumerate((1/16,1/8,1/4)):
            undefined=(i,j)==(3,0)
            w.writerow([g,r,'uncertainty',3,2 if undefined else 3,'' if undefined else (-.15+i*.06+j*.07),0,1.6-i*.2-j*.62])
    return s.getvalue().encode()


def test_figure_renders_all_cells_including_undefined():
    assert rank_placement_pdf(fixture_table()).startswith(b'%PDF-')


def test_incomplete_figure_is_not_exportable():
    with pytest.raises(ValueError,match='four environments'):
        rank_placement_pdf(b'\n'.join(fixture_table().splitlines()[:-1]))
