"""Validate the complete numerical Watermark archive against the source-data receipt."""
from pathlib import Path

from aquapilot.verification import bundle, read_json, verify_axes, verify_numeric_tables
from scripts.analysis_handoff import archive_bytes, json_bytes


def add_corrected_files(results_zip: Path, axes_json: Path, stage_dir: Path) -> dict[str, Path]:
    """Stage verified numerical members, also when importing a previous archive."""
    axes_bytes = Path(axes_json).read_bytes()
    manifest, tables = bundle(results_zip)
    verify_axes(manifest, read_json(axes_bytes))
    verify_numeric_tables(manifest, tables)
    manifest["members"] = {name: manifest["members"][name] for name in tables}
    manifest.update(bundle_schema_version=2, table_format="numeric")
    stage = Path(stage_dir) / "verified_numeric"
    stage.mkdir(parents=True)
    archive = stage / "results.zip"
    archive_bytes(archive, {**tables, "bundle_manifest.json": json_bytes(manifest)})
    receipt = stage / "axes.json"
    receipt.write_bytes(axes_bytes)
    return {"results.zip": archive, "axes.json": receipt}
