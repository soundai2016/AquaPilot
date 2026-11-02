"""Reconcile the formal result tables with the independent source-data receipt."""
from pathlib import Path
from zipfile import ZipFile

from writing.render_submission import render


def add_corrected_files(results_zip: Path, axes_json: Path, stage_dir: Path) -> dict[str, Path]:
    """Build the shareable file inventory; never compile or submit the paper."""
    results_zip, axes_json, stage_dir = map(Path, (results_zip, axes_json, stage_dir))
    if not axes_json.is_file():
        raise ValueError('Missing raw-to-prepared audit receipt (axes.json). Run scripts/verify_watermark_axes.py on the prepared folds used for training.')
    dest = stage_dir / 'verified_source'
    # render verifies both manifests, scored counts, 36 fits, all pairings,
    # and the values used by the TeX tables. It requires a new output path.
    render(results_zip, axes_json, dest)
    # Plot the checked CSV. A valid checksum identifies an archived PDF but does
    # not establish that the numbers printed in it agree with the table.
    from aquapilot.result_figure import rank_placement_pdf
    with ZipFile(results_zip) as archive:
        plot = rank_placement_pdf(archive.read('ranking_placement_by_environment.csv'))
    (dest / 'paper/figures/rank_placement.pdf').write_bytes(plot)
    files = {'results.zip': results_zip, 'axes.json': axes_json}
    files.update({p.relative_to(stage_dir).as_posix(): p for p in sorted(dest.rglob('*')) if p.is_file()})
    return files
