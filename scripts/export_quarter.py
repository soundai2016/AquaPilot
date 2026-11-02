#!/usr/bin/env python3
"""Export twelve completed corrected-grid fits for portable manuscript analysis."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
if not __package__:
    sys.path[:0] = [str(PROJECT / "src"), str(PROJECT)]

from aquapilot.quarter_export import FORMAT, PAIR_FILE, PROTOCOL, load_json, member_info, read_quarter_export, summarize_pairs
from scripts.quarter_budget import default_quarter_receipt, write_quarter_report

DEFAULT_OUTPUT = Path("results/AquaPilot_quarter_results.zip")


def export_quarter(model_root: Path, axes: Path, output: Path) -> Path:
    """Re-verify completed 1.0/1.1 runs in place and publish one numerical ZIP."""
    model_root, axes = Path(model_root).resolve(), Path(axes).resolve()
    output = Path(output).absolute()
    output = output.parent.resolve() / output.name
    if (output.suffix.lower() != ".zip" or output.is_symlink() or output.is_relative_to(model_root)
            or output.is_relative_to(PROJECT / "data")):
        raise ValueError("Choose a ZIP outside the training runs and dataset.")
    receipt = axes.read_bytes()
    if len(receipt) > 2_000_000:
        raise ValueError("Unexpectedly large axis receipt.")
    with tempfile.TemporaryDirectory(prefix="aquapilot-quarter-") as scratch:
        report = write_quarter_report(model_root, axes, output_parent=Path(scratch))
        members = {"axes.json": receipt, PAIR_FILE: (report.parent / PAIR_FILE).read_bytes(), "report.md": report.read_bytes()}
    if axes.read_bytes() != receipt:
        raise ValueError("Axis receipt changed during export.")
    rows, summary = summarize_pairs(members[PAIR_FILE])
    cells, trained_splits = [], {}
    for row in rows:
        env, seed = row["environment"], int(row["seed"])
        run = model_root / f"fold_test_{env}/ratio_0.25/seed_{seed}"
        completed = load_json((run / "complete.json").read_bytes())["artifacts"]
        source = (run / "run_manifest.json").read_bytes()
        run_identity = member_info(source)
        metric_identity = member_info((run / "metrics.json").read_bytes())
        if (metric_identity["sha256"] != row["metrics_sha256"] or completed["metrics.json"] != metric_identity
                or completed["run_manifest.json"] != run_identity):
            raise ValueError(f"Source run changed during quarter-budget export: {env}, seed {seed}.")
        splits = load_json(source)["data"]["main"]["split_fingerprints"]
        if trained_splits.setdefault(env, splits) != splits:
            raise ValueError(f"Seeds within {env} refer to different fitted data.")
        cells.append(dict(environment=env, seed=seed, metrics_sha256=row["metrics_sha256"],
                          run_manifest_sha256=run_identity["sha256"]))
    manifest = dict(format=FORMAT, schema_version=1, export_version="1.1", protocol_id=PROTOCOL,
                    num_fits=12, pilot_count=32, num_subcarriers=128, summary=summary,
                    source_cells=cells, trained_split_fingerprints=trained_splits,
                    members={name: member_info(data) for name, data in members.items()})
    members["manifest.json"] = (json.dumps(manifest, allow_nan=False, sort_keys=True, indent=2) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".zip") as temp:
        with zipfile.ZipFile(temp, "w") as archive:
            for name, data in sorted(members.items()):
                archive.writestr(zipfile.ZipInfo(name), data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
        temp.flush()
        os.fsync(temp.fileno())
        read_quarter_export(temp.name)
        try:
            os.link(temp.name, output)
        except FileExistsError:
            if output.is_symlink() or not output.is_file() or read_quarter_export(output)["manifest"] != manifest:
                raise FileExistsError(f"Different output exists at {output}. Choose a new --output.") from None
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/watermark_quarter/aquapilot"),
                        help="Model directory with the four fold_test_* directories, including a full corrected matrix")
    parser.add_argument("--axes", type=Path, help="Verified axis receipt; default: the run receipt, then the packaged results/axes.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        model = (PROJECT / args.results).resolve()
        axes = (PROJECT / args.axes).resolve() if args.axes is not None else default_quarter_receipt(model, PROJECT)
        destination = export_quarter(model, axes, PROJECT / args.output)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        parser.exit(2, f"Quarter-budget export failed: {exc}\n")
    print(destination)


if __name__ == "__main__":
    main()
