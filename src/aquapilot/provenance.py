"""Experiment identity, input hashes, and completed-run validation."""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROTOCOL_ID = "aquapilot_inband_delay_fft"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_fingerprint(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def resolve_config_paths(cfg: dict[str, Any], project_root: str | Path | None = None) -> dict[str, Any]:
    """Resolve experiment paths against the project, independently of the shell cwd."""
    root = Path(project_root or PROJECT_ROOT).resolve()
    result = copy.deepcopy(cfg)
    locations = [(result, "output_dir"), (result.get("data", {}), "root"),
                 (result.get("train", {}), "init_checkpoint"),
                 (result.get("classical", {}), "training_root")]
    for section, key in locations:
        if section.get(key) is not None:
            path = Path(section[key]).expanduser()
            section[key] = str((root / path).resolve() if not path.is_absolute() else path.resolve())
    return result


def _git(root: Path, *args: str) -> bytes | None:
    try:
        return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return None


def source_snapshot(project_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root or PROJECT_ROOT).resolve()
    files = set(root.glob("src/**/*.py")) | set(root.glob("scripts/**/*.py")) | set(root.glob("scripts/*.sh"))
    files.update(root.glob("configs/*.yaml"))
    files.update(path for path in (root / "pyproject.toml", root / "run.sh") if path.is_file())
    records = [{"path": str(path.relative_to(root)), **file_fingerprint(path)} for path in sorted(files)]
    revision = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain", "--untracked-files=all", "--", "src", "scripts", "configs", "pyproject.toml", "run.sh")
    diff = _git(root, "diff", "HEAD", "--binary", "--", "src", "scripts", "configs", "pyproject.toml", "run.sh")
    dirty_payload = {"status": (status or b"").decode("utf-8", errors="replace"),
                     "tracked_diff_sha256": hashlib.sha256(diff or b"").hexdigest(),
                     "working_source_sha256": fingerprint(records)}
    return {
        "git_commit": revision.decode().strip() if revision else None,
        "git_available": revision is not None,
        "dirty": bool(status),
        "dirty_code_fingerprint": fingerprint(dirty_payload),
        "dirty_scope": ["src", "scripts", "configs", "pyproject.toml", "run.sh"],
        "evaluator_source_fingerprint": fingerprint(records),
        "files": records,
    }


def runtime_snapshot() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("numpy", "scipy", "torch", "PyYAML", "pandas", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result: dict[str, Any] = {"python": sys.version, "platform": platform.platform(), "packages": packages}
    torch = sys.modules.get("torch")
    if torch is not None:
        result["torch_cuda_build"] = torch.version.cuda
        result["torch_threads"] = torch.get_num_threads()
    return result


def _split_files(root: Path, split: str) -> list[Path]:
    manifest = root / "splits.json"
    if manifest.is_file():
        mapping = json.loads(manifest.read_text(encoding="utf-8"))
        paths = [(root / str(item)).resolve() for item in mapping.get(split, [])]
    else:
        paths = sorted((root / split).rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No prepared {split} sequences under {root}. Prepare this split before running.")
    if len(paths) != len(set(paths)):
        raise ValueError(f"Duplicate sequence paths in {root}/splits.json for {split}.")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Prepared sequence does not exist: {path}")
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Prepared split refers outside its data root: {path}")
    return paths


def prepared_fingerprint(root: str | Path, splits: Iterable[str]) -> dict[str, Any]:
    root = Path(root).resolve()
    records = {}
    for split in splits:
        # Preserve manifest order: it determines the per-sequence noise seeds.
        records[split] = [{"path": str(path.relative_to(root)), **file_fingerprint(path)}
                          for path in _split_files(root, split)]
    return {"root": str(root), "splits": records,
            "fingerprint": fingerprint(records),
            "split_fingerprints": {name: fingerprint(items) for name, items in records.items()}}


def _data_snapshot(cfg: dict[str, Any], input_splits: Iterable[str]) -> dict[str, Any]:
    if cfg["data"]["kind"] == "synthetic":
        generation = {"data": cfg["data"], "seed": cfg["seed"]}
        digest = fingerprint(generation)
        return {"kind": "synthetic", "generation": generation, "fingerprint": digest,
                "test_sequence_fingerprint": fingerprint({**generation, "split": "test"})}
    main = prepared_fingerprint(cfg["data"]["root"], input_splits)
    result: dict[str, Any] = {"kind": "prepared", "main": main,
                             "test_sequence_fingerprint": main["split_fingerprints"].get("test")}
    classical = cfg.get("classical", {})
    training_root = classical.get("training_root")
    if cfg.get("eval", {}).get("run_classical", False) and training_root:
        result["classical_training"] = prepared_fingerprint(training_root, ["train"])
    result["fingerprint"] = fingerprint(result)
    return result


def _common_protocol(cfg: dict[str, Any]) -> dict[str, Any]:
    common = copy.deepcopy(cfg)
    for key in ("seed", "output_dir"):
        common.pop(key, None)
    common.get("data", {}).pop("root", None)
    common.get("pilot", {}).pop("ratio", None)
    common.get("train", {}).pop("init_checkpoint", None)
    common.get("classical", {}).pop("training_root", None)
    return common


def _save(path: Path, value: dict[str, Any]) -> None:
    from aquapilot.utils import save_json
    save_json(path, value)


def _identity(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: manifest[key] for key in (
        "protocol_id", "protocol_fingerprint", "run_mode", "config_fingerprint",
        "evaluator_source_fingerprint", "runtime_fingerprint", "data_fingerprint",
        "test_sequence_fingerprint", "checkpoint_fingerprint")}


def _validate_completed(output: Path, expected: dict[str, Any]) -> dict[str, Any]:
    marker = output / "complete.json"
    manifest_path = output / "run_manifest.json"
    if not marker.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"{output} is incomplete or predates this protocol. Choose a new --results-root; existing files were not changed.")
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    completion = json.loads(marker.read_text(encoding="utf-8"))
    if stored.get("manifest_schema_version") != 2 or completion.get("completion_schema_version") != 2:
        raise RuntimeError(f"Unsupported provenance schema in {output}; choose a new output directory.")
    mismatches = [key for key, value in _identity(expected).items() if stored.get(key) != value]
    if mismatches:
        raise RuntimeError(f"Cannot reuse {output}: changed {', '.join(mismatches)}. Choose a new --results-root to retain both evaluations.")
    if completion.get("identity_fingerprint") != fingerprint(_identity(stored)):
        raise RuntimeError(f"Completion identity does not match {manifest_path}; use a new output directory.")
    artifacts = completion.get("artifacts", {})
    if not {"metrics.json", "run_manifest.json", "resolved_config.json"}.issubset(artifacts):
        raise RuntimeError(f"Incomplete artifact inventory in {marker}; use a new output directory.")
    actual = {str(path.relative_to(output)) for path in output.rglob("*") if path.is_file() and path != marker}
    if actual != set(artifacts):
        raise RuntimeError(f"The artifact inventory in {output} changed after completion. Use a new output directory.")
    for name, saved in artifacts.items():
        path = output / name
        if path.is_symlink() or not path.is_file() or file_fingerprint(path) != saved:
            raise RuntimeError(f"Completed artifact changed or is missing: {path}. Use a new output directory.")
    return json.loads((output / "metrics.json").read_text(encoding="utf-8"))


def begin_run(cfg: dict[str, Any], *, mode: str, checkpoint: str | Path | None = None,
              skip_existing: bool = False, input_splits: Iterable[str] | None = None,
              project_root: str | Path | None = None) -> dict[str, Any]:
    """Validate run identity before evaluating or creating an output directory."""
    if mode not in {"fresh_training", "checkpoint_reevaluation", "fine_tuning"}:
        raise ValueError(f"Unknown run mode: {mode}")
    cfg = resolve_config_paths(cfg, project_root)
    output = Path(cfg["output_dir"])
    for root in (cfg.get("data", {}).get("root"), cfg.get("classical", {}).get("training_root")):
        if root and (output.is_relative_to(Path(root)) or Path(root).is_relative_to(output)):
            raise ValueError(f"Output {output} must be separate from the data tree {root}.")
    if input_splits is None:
        input_splits = ["test"] if mode == "checkpoint_reevaluation" else ["train", "val", "test"]
    data = _data_snapshot(cfg, input_splits)
    source = source_snapshot(project_root)
    runtime = runtime_snapshot()
    checkpoint_path = checkpoint or cfg.get("train", {}).get("init_checkpoint")
    checkpoint_info = None
    if checkpoint_path:
        path = Path(checkpoint_path).expanduser()
        if not path.is_absolute():
            path = Path(project_root or PROJECT_ROOT) / path
        checkpoint_info = {"path": str(path.resolve()), **file_fingerprint(path)}
    if mode == "checkpoint_reevaluation" and checkpoint_info is None:
        raise ValueError("Checkpoint reevaluation requires an existing checkpoint.")
    protocol = {"protocol_id": PROTOCOL_ID, "run_mode": mode, "config": _common_protocol(cfg),
                "evaluator_source_fingerprint": source["evaluator_source_fingerprint"], "runtime": runtime}
    manifest = {
        "manifest_schema_version": 2, "protocol_id": PROTOCOL_ID, "run_mode": mode,
        "created_utc": datetime.now(timezone.utc).isoformat(), "seed": int(cfg["seed"]),
        "resolved_config": cfg, "config_fingerprint": fingerprint(cfg),
        "protocol_fingerprint": fingerprint(protocol), "protocol_definition": protocol,
        "evaluator_source_fingerprint": source["evaluator_source_fingerprint"],
        "source": source, "runtime": runtime, "runtime_fingerprint": fingerprint(runtime),
        "data": data, "data_fingerprint": data["fingerprint"],
        "test_sequence_fingerprint": data["test_sequence_fingerprint"],
        "checkpoint": checkpoint_info, "checkpoint_fingerprint": fingerprint(checkpoint_info),
    }
    context = {"output_dir": output, "manifest": manifest, "cached_result": None}
    if output.exists() and any(output.iterdir()):
        if not skip_existing:
            raise FileExistsError(f"Run directory is not empty: {output}. Use --skip-existing to validate a completed identical run, or choose a new --results-root.")
        context["cached_result"] = _validate_completed(output, manifest)
        return context
    output.mkdir(parents=True, exist_ok=True)
    _save(output / "resolved_config.json", cfg)
    _save(output / "run_manifest.json", manifest)
    return context


def finish_run(context: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Write metrics and an atomic completion marker after all outputs are closed."""
    if context.get("cached_result") is not None:
        return context["cached_result"]
    output = Path(context["output_dir"])
    manifest = context["manifest"]
    result = copy.deepcopy(result)
    for key in ("protocol_id", "protocol_fingerprint", "config_fingerprint", "data_fingerprint",
                "test_sequence_fingerprint", "evaluator_source_fingerprint", "run_mode"):
        result[key] = manifest[key]
    result.update(result_schema_version=2, code_commit=manifest["source"]["git_commit"],
                  noise_convention=manifest["resolved_config"]["pilot"].get("update_noise_convention", "component"),
                  manifest_file="run_manifest.json")
    _save(output / "metrics.json", result)
    artifacts = {}
    for path in sorted(output.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Run artifacts must be regular local files: {path}")
        if path.is_file() and path.name != "complete.json":
            artifacts[str(path.relative_to(output))] = file_fingerprint(path)
    _save(output / "complete.json", {"completion_schema_version": 2,
          "completed_utc": datetime.now(timezone.utc).isoformat(),
          "identity_fingerprint": fingerprint(_identity(manifest)), "artifacts": artifacts})
    return result
