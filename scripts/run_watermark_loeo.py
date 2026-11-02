#!/usr/bin/env python3
"""Train and evaluate the Watermark leave-one-environment-out matrix."""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import importlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aquapilot.config import load_config
from aquapilot.provenance import PROTOCOL_ID, resolve_config_paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FOLDS = ("fold_test_NOF", "fold_test_NCS", "fold_test_BCH", "fold_test_KAU")
DEFAULT_RATIOS = (0.25, 0.125, 0.0625)
DEFAULT_SEEDS = (17, 37, 57)
DEFAULT_RESULTS_ROOT = "results/watermark_inband"
DEFAULT_RAW_ROOT = "data/raw/watermark/WatermarkV1/Watermark/input/channels"


def project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def ratio_tag(ratio: float) -> str:
    return f"ratio_{ratio:.10g}"


def flatten_run(result: dict[str, Any], fold: str, ratio: float, seed: int, output_dir: Path) -> dict[str, Any]:
    if result.get("result_schema_version") != 2 or result.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Only completed results from the current evaluation protocol can enter this aggregate.")
    row: dict[str, Any] = {"fold": fold, "pilot_ratio_requested": ratio, "seed": seed,
                           "output_dir": str(output_dir), "model_kind": result.get("model_kind"),
                           "parameters": result.get("model_parameters")}
    for key in ("protocol_id", "protocol_fingerprint", "config_fingerprint", "data_fingerprint",
                "test_sequence_fingerprint", "evaluator_source_fingerprint", "noise_convention", "run_mode"):
        row[key] = result.get(key)

    def add_metrics(prefix: str, metrics: dict[str, Any]) -> None:
        for source, suffix in (("nmse_db_mean", "nmse_db"), ("nmse_linear_mean", "nmse_linear"),
                               ("channel_nrmse_percent_mean", "channel_nrmse_percent"),
                               ("coverage_90", "coverage90"), ("pooled_nmse_db", "pooled_nmse_db"),
                               ("num_frames", "num_frames"), ("num_valid_error_frames", "num_valid_frames")):
            if source in metrics:
                row[f"{prefix}_{suffix}"] = metrics[source]
        if "nonpilot" in metrics:
            add_metrics(f"{prefix}_nonpilot", metrics["nonpilot"])

    add_metrics("forecast", result.get("model_test", {}))
    row["forecast_latency_ms"] = result.get("model_test", {}).get("latency_ms_per_frame")
    for name, metrics in result.get("baselines", {}).items():
        add_metrics(f"forecast_baseline_{name}", metrics)
    for policy, metrics in result.get("closed_loop", {}).items():
        prefix = f"closed_{policy}"
        row[f"{prefix}_pilot_ratio"] = metrics.get("pilot_ratio")
        add_metrics(prefix, metrics.get("posterior", {}))
        add_metrics(f"{prefix}_prior", metrics.get("prior", {}))
        for key, value in metrics.get("prior_ranking", {}).items():
            if value is None or isinstance(value, (int, float, str, bool)):
                row[f"{prefix}_ranking_{key}"] = value
        for name, baseline in metrics.get("same_information_baselines", {}).items():
            add_metrics(f"{prefix}_matched_{name}", baseline)
    for name, metrics in result.get("classical_closed_loop", {}).items():
        add_metrics(f"classical_{name}", metrics)
    return row


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", newline="", encoding="utf-8", delete=False) as handle:
            temporary = handle.name
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and Path(temporary).exists():
            Path(temporary).unlink()


def add_matrix_arguments(parser: argparse.ArgumentParser, *, results_root: str) -> None:
    parser.add_argument("--config", type=Path, default=Path("configs/watermark.yaml"))
    parser.add_argument("--prepared-root", type=Path, default=Path("data/prepared/watermark_inband"))
    parser.add_argument("--results-root", type=Path, default=Path(results_root))
    parser.add_argument("--folds", nargs="+", default=list(DEFAULT_FOLDS))
    parser.add_argument("--pilot-ratios", nargs="+", type=float, default=list(DEFAULT_RATIOS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--skip-existing", action="store_true", help="Reuse only completed runs with identical verified inputs and evaluator")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without reading datasets or writing outputs")


def build_plan(args: argparse.Namespace, base: dict[str, Any]) -> list[dict[str, Any]]:
    for values, name in ((args.folds, "folds"), (args.pilot_ratios, "pilot ratios"), (args.seeds, "seeds")):
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate {name} would reuse an output path.")
    if any(not 0.0 < value <= 1.0 for value in args.pilot_ratios):
        raise ValueError("Each pilot ratio must be in (0, 1].")
    if any(value < 0 for value in args.seeds):
        raise ValueError("Seeds must be nonnegative integers.")
    if any(Path(fold).name != fold or fold in {".", ".."} for fold in args.folds):
        raise ValueError("Each fold must be a directory name, not a path.")
    if set(args.folds) - set(DEFAULT_FOLDS):
        raise ValueError(f"Watermark folds must be chosen from {', '.join(DEFAULT_FOLDS)}.")
    kind = str(base["model"].get("kind", "aquapilot"))
    root = project_path(args.results_root)
    return [{"fold": fold, "root": str(project_path(args.prepared_root) / fold),
             "pilot_ratio": ratio, "seed": seed,
             "output_dir": str(root / kind / fold / ratio_tag(ratio) / f"seed_{seed}"),
             "config": str(project_path(args.config))}
            for fold in args.folds for ratio in args.pilot_ratios for seed in args.seeds]


def config_for_run(base: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg["seed"] = int(item["seed"])
    cfg["data"]["root"] = item["root"]
    cfg["pilot"]["ratio"] = float(item["pilot_ratio"])
    cfg["output_dir"] = item["output_dir"]
    return resolve_config_paths(cfg, PROJECT_ROOT)


def save_plan(path: Path, plan: list[dict[str, Any]]) -> None:
    from aquapilot.utils import save_json
    payload = {"protocol_id": PROTOCOL_ID, "runs": plan}
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != payload:
        raise ValueError(f"{path} describes a different matrix. Choose a new --results-root instead of mixing runs.")
    save_json(path, payload)


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_matrix_arguments(parser, results_root=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--prepare", action="store_true", help="Prepare and verify the in-band Watermark folds, then exit without training")
    parser.add_argument("--raw-root", type=Path, metavar="DIR", help=f"Raw channel directory for --prepare (default: {DEFAULT_RAW_ROOT})")
    parser.add_argument("--axes", type=Path, help="Raw-to-prepared verification receipt (default: RESULTS_ROOT/axes.json)")
    parser.add_argument("--device", help="Override train.device: auto, cpu, cuda, or cuda:N")
    parser.add_argument("--check-data", action="store_true", help="Check the existing prepared folds and runtime, then exit without writing results")
    parser.add_argument("--background", action="store_true", help="Validate data/runtime first, then start a detached training process")
    parser.add_argument("--export-tables", action="store_true", help="Export verified LaTeX tables after a complete 36-run matrix")
    parser.add_argument("--session-id", help=argparse.SUPPRESS)
    parser.add_argument("--lock-fd", type=int, help=argparse.SUPPRESS)
    return parser


def axes_path(args: argparse.Namespace) -> Path:
    return project_path(args.axes if args.axes is not None else args.results_root / "axes.json")


def prepare_inputs(args: argparse.Namespace) -> None:
    raw = project_path(args.raw_root or DEFAULT_RAW_ROOT)
    prepared, receipt = project_path(args.prepared_root), axes_path(args)
    legacy_prepared, legacy_results = project_path("data/prepared/watermark"), project_path("results/watermark")
    if prepared.is_relative_to(legacy_prepared) or receipt.is_relative_to(legacy_results):
        raise ValueError("Retain data/prepared/watermark and results/watermark for the previous run. Use the watermark_inband defaults or new directories.")
    if not raw.is_dir():
        raise FileNotFoundError(f"Watermark input/channels directory not found: {raw}. Use --raw-root for a different location.")
    if receipt.suffix != ".json" or any(receipt.is_relative_to(p) for p in (raw, prepared, PROJECT_ROOT / "data")):
        raise ValueError("The axis receipt must be a .json file outside the raw and prepared datasets.")
    if __package__:
        from .prepare_watermark_loeo import main as prepare
        from .verify_watermark_axes import verify, write_receipt
    else:
        from prepare_watermark_loeo import main as prepare
        from verify_watermark_axes import verify, write_receipt
    if not prepared.exists() or (prepared.is_dir() and not any(prepared.iterdir())):
        if receipt.exists():
            raise ValueError(f"{receipt} exists without its prepared inputs. Choose a new directory for both.")
        prepare(["--input", str(raw), "--output-root", str(prepared)])
    report = verify(raw, prepared)
    if receipt.exists():
        if json.loads(receipt.read_text(encoding="utf-8")) != report:
            raise ValueError(f"{receipt} refers to different inputs. Keep it with its run and choose another --results-root.")
    else:
        write_receipt(report, receipt)
    print(f"Verified {report['num_unique_records']} recordings and {report['total_prepared_files']} prepared files.\nReceipt: {receipt}")


def match_receipt(args: argparse.Namespace, data: dict[str, Any]) -> Path:
    """Do not spend a new training run on data other than those checked against the raw soundings."""
    from aquapilot.provenance import prepared_fingerprint
    from aquapilot.watermark_grid import GRID_NAME
    receipt = axes_path(args)
    if not receipt.is_file():
        raise FileNotFoundError(f"Missing raw-to-prepared verification: {receipt}. Run bash run.sh --prepare first or pass --axes for these inputs.")
    raw_check = json.loads(receipt.read_text(encoding="utf-8"))
    required = {"status": "ok", "method": "independent_delay_axis_fft", "frequency_grid": GRID_NAME,
                **{key: data[key] for key in ("band_fraction", "num_unique_records", "total_prepared_files",
                                              "n_subcarriers", "history", "horizon")}}
    if not isinstance(raw_check, dict) or any(raw_check.get(k) != v for k, v in required.items()):
        raise ValueError(f"{receipt}: source verification does not match the Watermark training protocol.")
    root = project_path(args.prepared_root)
    checked = raw_check.get("ordered_split_fingerprints")
    for fold in DEFAULT_FOLDS:
        signatures = prepared_fingerprint(root / fold, ("train", "val", "test"))["split_fingerprints"]
        if not isinstance(checked, dict) or checked.get(fold.removeprefix("fold_test_")) != signatures:
            raise ValueError(f"{receipt}: ordered training, validation or test inputs changed in {fold}. Verify the source files before training.")
    return receipt


def load_fresh_config(args: argparse.Namespace) -> dict[str, Any]:
    base = load_config(project_path(args.config))
    if base["data"]["kind"] != "prepared":
        raise ValueError("The Watermark matrix requires data.kind=prepared.")
    if base["train"].get("init_checkpoint"):
        raise ValueError("Fresh training requires train.init_checkpoint to be unset; no previous model is loaded.")
    required = {'comb', 'random', 'uncertainty', 'active'}
    if set(base['eval'].get('mask_strategies', required)) != required:
        raise ValueError('The Watermark comparison needs all four placement policies; use the dedicated ablation entry for a different design.')
    if int(base['data'].get('horizon', 1)) != 1:
        raise ValueError('The Watermark closed-loop comparison requires a one-snapshot prediction horizon.')
    if base['eval'].get('prediction_storage', 'compact') not in {'compact', 'full'}:
        raise ValueError('Use compact or full prediction_storage so comparisons retain frame-matched diagnostics.')
    if args.device is not None:
        if not re.fullmatch(r"auto|cpu|cuda(?::\d+)?", args.device):
            raise ValueError("--device must be auto, cpu, cuda, or cuda:N.")
        base["train"]["device"] = args.device
    return base


def runtime_preflight(cfg: dict[str, Any]) -> dict[str, Any]:
    """Load the training/analysis dependencies and execute a small device operation."""
    for name in ("aquapilot.engine", "pandas", "matplotlib"):
        importlib.import_module(name)
    import torch
    from aquapilot.provenance import runtime_snapshot
    from aquapilot.utils import choose_device, configure_torch_runtime
    configure_torch_runtime(cfg["train"].get("torch_threads"))
    device = choose_device(str(cfg["train"].get("device", "auto")))
    value = torch.ones(2, device=device).sum().item()
    if value != 2:
        raise RuntimeError(f"Device operation failed on {device}.")
    result = runtime_snapshot()
    result.update(executable=sys.executable, selected_device=str(device),
                  requested_device=str(cfg["train"].get("device", "auto")),
                  cuda_available=torch.cuda.is_available())
    if device.type == "cuda":
        index = torch.cuda.current_device() if device.index is None else device.index
        result["cuda_device"] = {"index": index, "name": torch.cuda.get_device_name(index),
                                 "memory_bytes": torch.cuda.get_device_properties(index).total_memory}
        try:
            query = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                                   capture_output=True, text=True, timeout=10, check=False)
            result["nvidia_smi"] = query.stdout.strip() if query.returncode == 0 else None
        except (FileNotFoundError, subprocess.TimeoutExpired):
            result["nvidia_smi"] = None
    return result


def validate_output_paths(args: argparse.Namespace, base: dict[str, Any], plan: list[dict[str, Any]]) -> Path:
    destination = project_path(args.results_root) / str(base["model"].get("kind", "aquapilot"))
    if destination.is_relative_to(project_path("results/watermark")):
        raise ValueError("Retain results/watermark for the previous run. Use results/watermark_inband or a new --results-root.")
    for protected in (project_path(args.prepared_root), PROJECT_ROOT / "data"):
        if destination.is_relative_to(protected) or protected.is_relative_to(destination):
            raise ValueError(f"Training results must be separate from data: {destination}")
    existing_plan = destination / "run_plan.json"
    expected = {"protocol_id": PROTOCOL_ID, "runs": plan}
    if existing_plan.exists() and json.loads(existing_plan.read_text(encoding="utf-8")) != expected:
        raise ValueError(f"{existing_plan} describes a different matrix; choose a new --results-root.")
    for item in plan:
        path = Path(item["output_dir"])
        if path.exists() and any(path.iterdir()):
            if not args.skip_existing:
                raise FileExistsError(f"Run directory is not empty: {path}. Choose a new --results-root for fresh training, or explicitly reuse verified completed runs with --skip-existing.")
            if not (path / "complete.json").is_file():
                raise RuntimeError(f"Interrupted or incomplete run: {path}. Retain it and choose a new --results-root; incomplete runs are not silently resumed.")
    return destination


def preflight(args: argparse.Namespace, base: dict[str, Any], plan: list[dict[str, Any]]) -> dict[str, Any]:
    from aquapilot.data.audit import audit_watermark
    if not args.check_data:
        validate_output_paths(args, base, plan)
    runtime = runtime_preflight(base)
    print(f"Checking prepared Watermark data: {project_path(args.prepared_root)}", flush=True)
    data = audit_watermark(project_path(args.prepared_root), history=int(base["data"].get("history", 16)),
                          horizon=int(base["data"].get("horizon", 1)),
                          n_subcarriers=int(base["data"].get("n_subcarriers", 128)))
    receipt = match_receipt(args, data)
    if args.skip_existing and not args.check_data:
        from aquapilot.provenance import begin_run
        for item in plan:
            path = Path(item["output_dir"])
            if path.exists() and any(path.iterdir()):
                begin_run(config_for_run(base, item), mode="fresh_training", skip_existing=True)
    report = {"mode": "fresh_training", "initialized_from": None, "runtime": runtime, "data": data,
              "axes_receipt": str(receipt), "num_configurations": len(plan)}
    print(f"Data check passed: {data['num_unique_records']} records, {data['total_prepared_files']} prepared files; device={runtime['selected_device']}", flush=True)
    return report


def _session_paths(args: argparse.Namespace) -> dict[str, Path]:
    session = args.session_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{os.getpid()}_{uuid.uuid4().hex[:6]}"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", session):
        raise ValueError("Invalid training session identifier.")
    root = project_path(args.results_root) / "logs"
    return {name: root / f"train_{session}{suffix}" for name, suffix in (
        ("log", ".log"), ("pid", ".pid"), ("status", ".status.json"), ("preflight", ".preflight.json"))} | {"id": Path(session)}


def _lock(args: argparse.Namespace) -> int:
    root = project_path(args.results_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".train.lock"
    if args.lock_fd is not None:
        fd = args.lock_fd
        if os.fstat(fd).st_ino != path.stat().st_ino or os.fstat(fd).st_dev != path.stat().st_dev:
            raise RuntimeError("The inherited training lock does not belong to this result root.")
    else:
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(fd)
        raise RuntimeError(f"Another training process is using {root}.") from exc
    return fd


def _status(paths: dict[str, Path], state: str, **values: Any) -> None:
    from aquapilot.utils import save_json
    save_json(paths["status"], {"session_id": str(paths["id"]), "state": state,
                               "updated_utc": datetime.now(timezone.utc).isoformat(),
                               "pid": os.getpid(), "mode": "fresh_training",
                               "command": shlex.join(sys.argv), **values})


def _launch_background(args: argparse.Namespace, argv: list[str], report: dict[str, Any]) -> None:
    from aquapilot.utils import save_json
    fd = _lock(args)
    paths = _session_paths(args)
    paths["log"].parent.mkdir(parents=True, exist_ok=True)
    save_json(paths["preflight"], report)
    _status(paths, "launching", results_root=str(project_path(args.results_root)))
    command = [sys.executable, "-u", str(Path(__file__).resolve()),
               *[value for value in argv if value != "--background"],
               "--session-id", str(paths["id"]), "--lock-fd", str(fd)]
    try:
        with paths["log"].open("x", encoding="utf-8") as log:
            child = subprocess.Popen(command, cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                     pass_fds=(fd,), env={**os.environ, "PYTHONUNBUFFERED": "1"})
        paths["pid"].write_text(str(child.pid) + "\n", encoding="utf-8")
    except Exception as exc:
        _status(paths, "failed_to_launch", error=str(exc))
        raise
    finally:
        os.close(fd)
    print(f"Started fresh training: PID {child.pid}\nLog: {paths['log']}\nStatus: {paths['status']}\nResults: {project_path(args.results_root)}", flush=True)


def full_matrix(plan: list[dict[str, Any]]) -> bool:
    expected = {(fold, ratio, seed) for fold in DEFAULT_FOLDS for ratio in DEFAULT_RATIOS for seed in DEFAULT_SEEDS}
    return {(item["fold"], item["pilot_ratio"], item["seed"]) for item in plan} == expected


def postprocess(destination: Path, plan: list[dict[str, Any]], export_tables: bool) -> None:
    complete = full_matrix(plan)
    summary = [sys.executable, str(PROJECT_ROOT / "scripts/summarize_watermark.py"), "--input", str(destination / "aggregate_runs.csv")]
    if not complete:
        summary.append("--allow-partial")
    subprocess.run(summary, cwd=PROJECT_ROOT, check=True)
    analysis = [sys.executable, str(PROJECT_ROOT / "scripts/analyze_results.py"), "--results", str(destination),
                "--output", str(destination / "analysis")]
    if complete:
        analysis.append("--expect-watermark-matrix")
    subprocess.run(analysis, cwd=PROJECT_ROOT, check=True)
    if export_tables:
        if not complete:
            raise ValueError("Table export requires the complete 36-run matrix.")
        subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts/build_result_tables.py"),
                        "--analysis", str(destination / "analysis"), "--output", str(destination / "tables")],
                       cwd=PROJECT_ROOT, check=True)


def execute_matrix(args: argparse.Namespace, base: dict[str, Any], plan: list[dict[str, Any]], report: dict[str, Any]) -> None:
    from aquapilot.engine import run_experiment
    from aquapilot.utils import save_json
    fd = _lock(args)
    paths = _session_paths(args)
    destination = project_path(args.results_root) / str(base["model"].get("kind", "aquapilot"))
    start = time.monotonic()
    rows: list[dict[str, Any]] = []
    try:
        save_json(paths["preflight"], report)
        paths["pid"].write_text(str(os.getpid()) + "\n", encoding="utf-8")
        save_plan(destination / "run_plan.json", plan)
        _status(paths, "running", completed_runs=0, total_runs=len(plan), results_root=str(destination))
        for index, item in enumerate(plan, 1):
            reusing = args.skip_existing and (Path(item["output_dir"]) / "complete.json").is_file()
            action = "reuse verified completed run" if reusing else "fresh initialization"
            print(f"[{index}/{len(plan)}] {item['fold']} ratio={item['pilot_ratio']} seed={item['seed']} {action}", flush=True)
            result = run_experiment(config_for_run(base, item), skip_existing=args.skip_existing)
            rows.append(flatten_run(result, item["fold"], item["pilot_ratio"], item["seed"], Path(item["output_dir"])))
            write_rows(destination / "aggregate_runs.csv", rows)
            _status(paths, "running", completed_runs=len(rows), total_runs=len(plan), results_root=str(destination),
                    latest_completed=item, elapsed_seconds=time.monotonic() - start)
        _status(paths, "analyzing", completed_runs=len(rows), total_runs=len(plan), results_root=str(destination))
        postprocess(destination, plan, args.export_tables)
        _status(paths, "completed", completed_runs=len(rows), total_runs=len(plan), results_root=str(destination),
                elapsed_seconds=time.monotonic() - start)
    except BaseException as exc:
        _status(paths, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", completed_runs=len(rows),
                total_runs=len(plan), error=str(exc), results_root=str(destination))
        raise
    finally:
        os.close(fd)
    print(json.dumps({"runs": len(rows), "aggregate": str(destination / "aggregate_runs.csv"), "status": str(paths["status"])}, indent=2))


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    command_parser = parser()
    args = command_parser.parse_args(argv)
    if sum((args.prepare, args.dry_run, args.check_data, args.background)) > 1:
        command_parser.error("Use --prepare, --dry-run, --check-data, or --background separately.")
    if args.raw_root is not None and not args.prepare:
        command_parser.error("--raw-root is used with --prepare.")
    if args.prepare:
        try:
            prepare_inputs(args)
        except (OSError, ValueError) as exc:
            command_parser.exit(2, f"Watermark preparation failed: {exc}\n")
        return
    if args.lock_fd is not None and args.session_id is None:
        command_parser.error("An inherited lock requires a session identifier.")
    base = load_fresh_config(args)
    plan = build_plan(args, base)
    if args.export_tables and not full_matrix(plan):
        command_parser.error("--export-tables requires all four folds, three pilot fractions, and three seeds.")
    if args.dry_run:
        print(json.dumps({"mode": "fresh_training", "initialized_from": None,
                          "device": base["train"].get("device", "auto"), "runs": len(plan), "plan": plan}, indent=2))
        return
    try:
        report = preflight(args, base, plan)
    except BaseException as exc:
        if args.session_id is not None:
            _status(_session_paths(args), "failed_preflight", error=str(exc))
        if args.lock_fd is not None:
            os.close(args.lock_fd)
        raise
    if args.check_data:
        print(json.dumps({"status": "ok", "data": {key: report["data"][key] for key in ("root", "num_unique_records", "total_prepared_files", "data_fingerprint")},
                          "device": report["runtime"]["selected_device"], "executable": sys.executable,
                          "planned_configurations": len(plan)}, indent=2))
        return
    if args.background:
        _launch_background(args, argv, report)
        return
    execute_matrix(args, base, plan, report)


if __name__ == "__main__":
    main()
