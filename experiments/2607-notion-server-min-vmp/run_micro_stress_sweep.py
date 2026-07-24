#!/usr/bin/env python3
"""Run the dedicated micro-stress first-stage OFAT sensitivity suite.

The canonical baseline and its launcher remain untouched.  This entry point
requires the derived builder output explicitly, materializes immutable configs,
then delegates scheduling to the shared sweep runner. Every child receives a
disjoint 16-CPU set and Gurobi SoftMemLimit=80 GB; aggregate memory is checked
before any child is launched.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
BASE_RUNNER_PATH = EXPERIMENT_ROOT / "run_sweep.py"
SINGLE_RUN_ENTRYPOINT = EXPERIMENT_ROOT / "run_experiment.py"
VENV_PYTHON = REPOSITORY_ROOT / ".venv" / "bin" / "python"
DEFAULT_PLAN = EXPERIMENT_ROOT / "plans" / "micro_stress_first_stage_symbreak_v2.yaml"
MODEL_SOURCE_ROOT = EXPERIMENT_ROOT / "src" / "notion_server_min_vmp"
REQUIRED_DATA_FILES = (
    "vm_requests.csv",
    "vm_usage_5min_scenarios.csv",
    "servers.csv",
)
OPTIONAL_LINEAGE_FILES = (
    "metadata.json",
    "stress_transform.json",
    "validation_report.json",
    "selection_manifest.csv",
)


def _load_base_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "notion_server_min_shared_sweep_runner", BASE_RUNNER_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load shared sweep runner: {BASE_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SHARED = _load_base_runner()


def _resolve_from_experiment(value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (EXPERIMENT_ROOT / candidate).resolve()


def _resolve_google_dir(value: str | Path) -> Path:
    """Resolve CLI data paths relative to repository root, not CWD."""

    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    google_dir = candidate.resolve()
    if not google_dir.is_dir():
        raise FileNotFoundError(f"Micro-stress --google-dir is not a directory: {google_dir}")
    missing = [name for name in REQUIRED_DATA_FILES if not (google_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Micro-stress dataset is incomplete at {google_dir}; missing {missing}"
        )
    return google_dir


def _resolve_formulation(
    plan: dict[str, Any],
    spec_path_override: str | Path | None = None,
) -> dict[str, Any] | None:
    formulation = plan.get("formulation")
    if formulation is None:
        # Backward compatibility for the already-completed v1 plan. New plans
        # are expected to pin formulation provenance explicitly.
        return None
    if not isinstance(formulation, dict):
        raise ValueError("Micro-stress plan must declare formulation provenance")
    formulation_id = str(formulation.get("id", "")).strip()
    if not formulation_id:
        raise ValueError("Micro-stress plan formulation.id must be nonempty")
    expected_sha256 = str(formulation.get("spec_sha256", "")).strip().lower()
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise ValueError("Micro-stress plan formulation.spec_sha256 must be a SHA-256 hex digest")

    # The authoritative Notion export is private workspace input and is not
    # vendored in this repository. Plans retain its immutable checksum. A user
    # may optionally provide a local copy; when supplied, it is verified and
    # included in the source hash manifest before any candidate is launched.
    configured_spec_path = formulation.get("spec_path")
    selected_spec_path = (
        spec_path_override
        if spec_path_override is not None
        else configured_spec_path
    )
    result: dict[str, Any] = {
        "id": formulation_id,
        "spec_sha256": expected_sha256,
        "spec_verified": False,
    }
    if selected_spec_path is not None and str(selected_spec_path).strip():
        spec_path = _resolve_from_experiment(str(selected_spec_path))
        if not spec_path.is_file():
            raise FileNotFoundError(f"Formulation specification is missing: {spec_path}")
        actual_sha256 = SHARED._sha256(spec_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "Formulation specification SHA-256 mismatch: "
                f"expected={expected_sha256}, actual={actual_sha256}, path={spec_path}"
            )
        result["spec_path"] = str(spec_path)
        result["spec_verified"] = True
    change_summary = formulation.get("change_summary_path")
    if change_summary:
        summary_path = _resolve_from_experiment(str(change_summary))
        if not summary_path.is_file():
            raise FileNotFoundError(f"Formulation change summary is missing: {summary_path}")
        result["change_summary_path"] = str(summary_path)
        result["change_summary_sha256"] = SHARED._sha256(summary_path)
    return result


def _capture_source_hashes(
    *,
    plan_path: Path,
    base_path: Path,
    formulation: dict[str, Any] | None,
) -> dict[str, str]:
    paths = {
        Path(__file__).resolve(),
        BASE_RUNNER_PATH.resolve(),
        SINGLE_RUN_ENTRYPOINT.resolve(),
        plan_path.resolve(),
        base_path.resolve(),
    }
    if formulation is not None:
        spec_path = formulation.get("spec_path")
        if spec_path:
            paths.add(Path(spec_path).resolve())
        if "change_summary_path" in formulation:
            paths.add(Path(formulation["change_summary_path"]).resolve())
    paths.update(path.resolve() for path in MODEL_SOURCE_ROOT.glob("*.py"))
    missing = [str(path) for path in sorted(paths) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot capture formulation source hashes; missing={missing}")
    return {str(path): SHARED._sha256(path) for path in sorted(paths)}


def _capture_dataset_hashes(google_dir: Path) -> dict[str, str]:
    files = sorted(path.resolve() for path in google_dir.iterdir() if path.is_file())
    if not files:
        raise ValueError(f"Cannot capture hashes for empty dataset directory: {google_dir}")
    return {str(path): SHARED._sha256(path) for path in files}


def _effective_base_config(base_config: dict[str, Any], google_dir: Path) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config["data"]["google_dir"] = str(google_dir)
    lineage = list(config["data"].get("additional_source_files", []))
    for name in OPTIONAL_LINEAGE_FILES:
        path = google_dir / name
        if path.is_file():
            lineage.append(str(path))
    # Preserve ordering while preventing duplicate provenance entries.
    config["data"]["additional_source_files"] = list(dict.fromkeys(lineage))
    return config


def _validate_dataset_shape(google_dir: Path) -> dict[str, Any]:
    """Ensure the loader cannot reselect from a pool larger than the builder result."""

    requests_path = google_dir / "vm_requests.csv"
    with requests_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "class" not in reader.fieldnames:
            raise ValueError(f"{requests_path} is missing the class column")
        class_counts: dict[str, int] = {}
        row_count = 0
        for row in reader:
            class_name = str(row["class"])
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            row_count += 1
    expected_counts = {"on_demand": 10, "spot": 10, "batch_candidate": 10}
    if class_counts != expected_counts:
        raise ValueError(
            "Micro-stress builder output must contain exactly the selected 10/10/10 "
            f"rows so the experiment loader cannot reselect them; expected={expected_counts}, "
            f"actual={class_counts}"
        )

    servers_path = google_dir / "servers.csv"
    with servers_path.open(newline="", encoding="utf-8") as stream:
        server_rows = sum(1 for _ in csv.DictReader(stream))
    if server_rows < 6:
        raise ValueError(
            f"Micro-stress dataset requires at least six server rows, found {server_rows}"
        )
    return {
        "vm_request_rows": row_count,
        "class_counts": class_counts,
        "server_rows": server_rows,
        "selection_preservation": (
            "exact builder-selected pool; no larger pool is exposed to loader stable ordering"
        ),
    }


def _validate_contract(config: dict[str, Any], plan: dict[str, Any]) -> None:
    experiment = config["experiment"]
    counts = experiment["class_counts"]
    batch = config["batch"]
    policy = plan["execution_policy"]
    expected = plan["validation"]["expected_dimensions"]
    actual = {
        "scenarios": int(experiment["num_scenarios"]),
        "on_demand": int(counts["on_demand"]),
        "spot": int(counts["spot"]),
        "batch_jobs": int(counts["batch_jobs"]),
        "max_batch_families": int(batch["max_families"]),
        "servers": int(experiment["num_servers"]),
    }
    normalized_expected = {key: int(value) for key, value in expected.items()}
    if actual != normalized_expected:
        raise ValueError(
            f"Micro-stress dimensions drifted: expected={normalized_expected}, actual={actual}"
        )

    workers = int(policy["max_parallel_jobs"])
    threads = int(policy["threads_per_job"])
    if workers != 5 or threads != 16 or workers * threads != 80:
        raise ValueError("Micro-stress execution must use at most five 16-thread jobs")
    if float(policy["time_limit_seconds"]) != 3600.0:
        raise ValueError("Micro-stress solver time limit must be 3600 seconds")
    if float(policy["mip_gap"]) != 0.001:
        raise ValueError("Micro-stress MIP gap must be 0.001")
    if float(policy["soft_mem_limit_gb_per_job"]) != 80.0:
        raise ValueError("Micro-stress Gurobi SoftMemLimit must be 80 GB per job")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 10/10/10, three-family, ten-scenario micro-stress first-stage suite"
    )
    parser.add_argument(
        "--google-dir",
        required=True,
        help="Builder output directory, absolute or relative to repository root",
    )
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument(
        "--spec-path",
        help=(
            "Optional local copy of the non-vendored formulation specification; "
            "its SHA-256 must match formulation.spec_sha256 in the plan"
        ),
    )
    parser.add_argument("--suite-root", help="Fresh output root; defaults to a UTC launch-id path")
    parser.add_argument("--generate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    plan_path = _resolve_from_experiment(args.plan)
    plan = SHARED._load_yaml(plan_path)
    base_path = _resolve_from_experiment(plan["base_config"])
    formulation_audit = _resolve_formulation(plan, args.spec_path)
    source_file_hashes = _capture_source_hashes(
        plan_path=plan_path,
        base_path=base_path,
        formulation=formulation_audit,
    )
    google_dir = _resolve_google_dir(args.google_dir)
    base_config = _effective_base_config(SHARED._load_yaml(base_path), google_dir)
    _validate_contract(base_config, plan)
    dataset_audit = _validate_dataset_shape(google_dir)
    dataset_file_hashes = _capture_dataset_hashes(google_dir)

    stage_name = "first_stage"
    stage = plan["stages"][stage_name]
    if not stage.get("enabled", False):
        raise ValueError("Micro-stress first stage is disabled")
    launch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.suite_root:
        suite_root = _resolve_from_experiment(args.suite_root)
    else:
        suite_root = (EXPERIMENT_ROOT / plan["output_root"] / launch_id).resolve()
    suite_root.mkdir(parents=True, exist_ok=True)
    unexpected = {
        entry.name
        for entry in suite_root.iterdir()
        if entry.name not in {"launcher.log", "suite.pid"}
    }
    if unexpected:
        raise FileExistsError(f"Suite root is not fresh ({sorted(unexpected)}): {suite_root}")
    if not VENV_PYTHON.is_file():
        raise FileNotFoundError(f"Required virtual environment interpreter is missing: {VENV_PYTHON}")
    if not SINGLE_RUN_ENTRYPOINT.is_file():
        raise FileNotFoundError(SINGLE_RUN_ENTRYPOINT)

    specs = SHARED._build_candidate_specs(base_config, stage)
    expected = int(plan["generation"]["exact_unique_candidate_configs"][stage_name])
    if len(specs) != expected:
        raise ValueError(f"Expected {expected} unique candidates, generated {len(specs)}")

    policy = plan["execution_policy"]
    resource_audit = SHARED._resource_preflight(policy, suite_root)
    resource_audit["memory_limit_enforcement"] = policy["memory_limit_enforcement"]
    rows = SHARED._materialize_candidates(
        specs=specs,
        plan=plan,
        plan_path=plan_path,
        base_path=base_path,
        stage_name=stage_name,
        suite_root=suite_root,
    )
    metadata = {
        "schema_version": 2,
        "suite_name": plan["suite_name"],
        "formulation": formulation_audit,
        "stage": stage_name,
        "launch_id": launch_id,
        "launched_at_utc": datetime.now(timezone.utc).isoformat(),
        "launcher_pid": os.getpid(),
        "hostname": socket.gethostname(),
        "plan_path": str(plan_path),
        "plan_sha256": SHARED._sha256(plan_path),
        "base_config_path": str(base_path),
        "base_config_sha256": SHARED._sha256(base_path),
        "google_dir": str(google_dir),
        "dataset_audit": dataset_audit,
        "source_files_sha256": source_file_hashes,
        "dataset_files_sha256": dataset_file_hashes,
        "candidate_count": len(rows),
        "execution_policy": policy,
        "resource_preflight": resource_audit,
        "manifest": str((suite_root / "manifest.csv").resolve()),
    }
    (suite_root / "suite_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (suite_root / "suite.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)
    if args.generate_only:
        print(f"Generated {len(rows)} immutable candidates in {suite_root}", flush=True)
        return 0

    results = SHARED._execute_candidates(
        rows,
        policy=policy,
        suite_root=suite_root,
        required_source_hashes=source_file_hashes,
    )
    failures = sum(row["state"] in {"FAILED", "LAUNCH_ERROR"} for row in results)
    target_misses = sum(row["state"] == "TARGET_GAP_NOT_REACHED" for row in results)
    target_completions = sum(
        row["state"] in {"COMPLETED_TARGET_GAP", "SKIPPED_COMPLETE"} for row in results
    )
    final = {
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_count": len(results),
        "completed_target_gap_or_skipped": target_completions,
        "target_gap_not_reached": target_misses,
        "failures": failures,
    }
    (suite_root / "suite_summary.json").write_text(
        json.dumps(final, indent=2), encoding="utf-8"
    )
    print(json.dumps(final, indent=2), flush=True)
    return 1 if failures or target_misses else 0


if __name__ == "__main__":
    raise SystemExit(main())
