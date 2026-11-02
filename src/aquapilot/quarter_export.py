"""Read a portable AquaPilot quarter-budget export. Uses only the Python standard library."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import zipfile
from pathlib import Path
from statistics import fmean, stdev

ENVIRONMENTS = ("NOF", "NCS", "BCH", "KAU")
SEEDS = (17, 37, 57)
FRAMES = dict(zip(ENVIRONMENTS, (14520, 60480, 2264, 7744)))
RECORDS = dict(zip(ENVIRONMENTS, (60, 60, 4, 32)))
PROTOCOL = "aquapilot_inband_delay_fft"
FORMAT = "aquapilot_verified_quarter"
PAIR_FILE = "paired_top_b_025.csv"
DATA_FILES = ("axes.json", PAIR_FILE, "report.md")
T_975_DF2 = 4.302652729696142
CELLS = {(env, seed) for env in ENVIRONMENTS for seed in SEEDS}
SHA = re.compile(r"[0-9a-f]{64}\Z")


def load_json(data):
    def unique(pairs):
        values = dict(pairs)
        if len(values) != len(pairs):
            raise ValueError("Duplicate JSON key in quarter-budget export.")
        return values

    def invalid_constant(value):
        raise ValueError(f"Non-finite JSON number: {value}.")

    return json.loads(data, object_pairs_hook=unique, parse_constant=invalid_constant)


def member_info(data):
    return {"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def summarize_pairs(data):
    reader = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""))
    fields = reader.fieldnames or []
    if not fields or len(fields) != len(set(fields)):
        raise ValueError("Missing or duplicate paired CSV columns.")
    rows = list(reader)
    cells = {}
    digests = set()
    for r in rows:
        if None in r or None in r.values():
            raise ValueError("Malformed paired CSV row.")
        env, seed = r["environment"], int(r["seed"])
        cell = (env, seed)
        if cell not in CELLS or cell in cells:
            raise ValueError(f"Unexpected or duplicate environment/seed pair: {cell}.")
        if (r["run_mode"] != "fresh_training" or float(r["pilot_fraction"]) != .25 or
                r["reference_policy"] != "comb" or r["policy"] != "uncertainty" or
                r["support_known"] != "True" or int(r["num_records"]) != RECORDS[env] or
                int(r["num_frames"]) != FRAMES[env] or int(r["num_matched_valid_frames"]) != FRAMES[env] or
                int(r["num_excluded_frames"]) != 0 or r["difference_convention"] != "policy minus comb; negative favors policy"):
            raise ValueError(f"Unexpected policies, budget, frame counts or difference convention for {cell}.")
        top, comb, delta = (float(r[key]) for key in ("policy_nmse_db", "comb_nmse_db", "delta_db"))
        if not all(map(math.isfinite, (top, comb, delta))) or not math.isclose(top-comb, delta, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError(f"Inconsistent signed Top-B minus comb difference for {cell}.")
        digest = r["metrics_sha256"]
        if not SHA.fullmatch(digest) or digest in digests:
            raise ValueError(f"Invalid or repeated source metrics identity for {cell}.")
        digests.add(digest)
        cells[cell] = (top, comb, delta)
    if cells.keys() != CELLS:
        raise ValueError("The corrected quarter-budget experiment requires all twelve pairs.")
    seed_means = {str(s): fmean(cells[env, s][2] for env in ENVIRONMENTS) for s in SEEDS}
    delta = fmean(seed_means.values())
    half = T_975_DF2 * stdev(seed_means.values()) / math.sqrt(3)
    summary = dict(top_db=fmean(x[0] for x in cells.values()), comb_db=fmean(x[1] for x in cells.values()),
                   delta_db=delta, ci_low_db=delta-half, ci_high_db=delta+half, degrees_of_freedom=2,
                   wins=sum(x[2] < 0 for x in cells.values()), ties=sum(x[2] == 0 for x in cells.values()),
                   losses=sum(x[2] > 0 for x in cells.values()), seed_means_db=seed_means)
    return rows, summary


def check_receipt(axes, training_splits):
    expected = dict(status="ok", method="independent_delay_axis_fft", frequency_grid="uniform_inner_probe_band",
                    history=16, horizon=1, band_fraction=.8, n_subcarriers=128, num_unique_records=156,
                    total_prepared_files=624)
    if any(axes.get(key) != value for key, value in expected.items()):
        raise ValueError("Incorrect in-band axis verification or record inventory.")
    audited = axes["ordered_split_fingerprints"]
    records = {r["environment"]: r for r in axes["environments"]}
    if (audited != training_splits or set(audited) != set(ENVIRONMENTS) or
            set(records) != set(ENVIRONMENTS) or len(axes["environments"]) != len(records)):
        raise ValueError("Source-to-frequency receipt does not match the fitted inputs.")
    for env, r in records.items():
        splits = audited[env]
        if (set(splits) != {"train", "val", "test"} or not all(SHA.fullmatch(s) for s in splits.values()) or
                r["records"] != RECORDS[env] or r["scored"] != FRAMES[env] or
                r["snapshots"] != FRAMES[env] + 16*RECORDS[env] or not 0 <= r["max_error_over_peak"] <= 4*2**-23):
            raise ValueError(f"Invalid source-to-frequency receipt for {env}.")


def read_quarter_export(path):
    """Verify the server's portable numerical evidence without reopening server data or weights."""
    try:
        return _read_quarter_export(path)
    except (AttributeError, KeyError, TypeError, OverflowError) as exc:
        raise ValueError(f"Invalid quarter-budget export metadata: {exc}.") from exc


def _read_quarter_export(path):
    with zipfile.ZipFile(Path(path)) as archive:
        names = archive.namelist()
        if set(names) != {*DATA_FILES, "manifest.json"} or len(names) != len(set(names)):
            raise ValueError("Unexpected, missing or duplicate members in quarter-budget archive.")
        if any(info.flag_bits & 1 or info.file_size > 2_000_000 for info in archive.infolist()):
            raise ValueError("Invalid member size or encryption in quarter-budget archive.")
        manifest = load_json(archive.read("manifest.json"))
        data = {name: archive.read(name) for name in DATA_FILES}
    if (manifest.get("format") != FORMAT or manifest.get("schema_version") != 1 or
            manifest.get("export_version") != "1.1" or manifest.get("protocol_id") != PROTOCOL or
            manifest.get("num_fits") != 12 or manifest.get("pilot_count") != 32 or
            manifest.get("num_subcarriers") != 128):
        raise ValueError("Expected the verified 32/128 corrected-grid export from AquaPilot 1.1.")
    if manifest["members"] != {name: member_info(value) for name, value in data.items()}:
        raise ValueError("A quarter-budget member differs from its recorded size or SHA-256.")
    rows, summary = summarize_pairs(data[PAIR_FILE])
    axes = load_json(data["axes.json"])
    check_receipt(axes, manifest["trained_split_fingerprints"])
    sources = manifest["source_cells"]
    identities = {(s["environment"], s["seed"]): s["metrics_sha256"] for s in sources}
    if (len(sources) != 12 or identities != {(r["environment"], int(r["seed"])): r["metrics_sha256"] for r in rows}
            or any(not SHA.fullmatch(s["run_manifest_sha256"]) for s in sources)):
        raise ValueError("The paired rows do not match the twelve recorded run identities.")
    # Recompute from the CSV; the manifest carries a convenient copy of the same statistics.
    if not _same_summary(summary, manifest["summary"]):
        raise ValueError("Stored summary does not reproduce from the paired CSV.")
    return dict(rows=rows, summary=summary, axes=axes, manifest=manifest,
                axes_sha256=manifest["members"]["axes.json"]["sha256"], paired_sha256=manifest["members"][PAIR_FILE]["sha256"])


def _same_summary(expected, actual):
    if not isinstance(actual, dict) or actual.keys() != expected.keys():
        return False
    for key, value in expected.items():
        other = actual[key]
        if isinstance(value, dict):
            if not _same_summary(value, other):
                return False
        elif isinstance(value, int):
            if type(other) is not int or other != value:
                return False
        elif type(other) not in (int, float) or not math.isclose(value, other, rel_tol=1e-13, abs_tol=1e-13):
            return False
    return True
