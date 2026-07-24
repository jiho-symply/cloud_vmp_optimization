#!/usr/bin/env python3
"""Queue one 22-run, three-hour paired migration-policy sensitivity suite.

The launcher deliberately lives beside, rather than modifying, the currently
hash-pinned one-hour launcher.  It waits for an optional predecessor suite,
then interleaves the same 11 OFAT candidates under two policies in one global
five-worker pool:

* ``migration_allowed`` uses the primary experiment entry point;
* ``migration_prohibited`` uses the diagnostic entry point that fixes every
  on-demand migration variable to zero.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import os
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml


EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
SHARED_RUNNER_PATH = EXPERIMENT_ROOT / "run_sweep.py"
MICRO_LAUNCHER_PATH = EXPERIMENT_ROOT / "run_micro_stress_sweep.py"
VENV_PYTHON = REPOSITORY_ROOT / ".venv" / "bin" / "python"
DEFAULT_PLAN = (
    EXPERIMENT_ROOT / "plans" / "micro_stress_first_stage_3h_migration_pair.yaml"
)
ALLOWED_INITIAL_FILES = {"queue.log"}


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name!r} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SHARED = _load_module("notion_server_min_paired_shared_runner", SHARED_RUNNER_PATH)
MICRO = _load_module("notion_server_min_paired_micro_helpers", MICRO_LAUNCHER_PATH)


def _resolve_from_experiment(value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (EXPERIMENT_ROOT / candidate).resolve()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _validate_plan_contract(
    plan: dict[str, Any],
    base_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    variants = plan.get("variants")
    if not isinstance(variants, list) or len(variants) != 2:
        raise ValueError("Paired migration plan must declare exactly two variants")
    expected_variants = {
        "migration_allowed": (False, "run_experiment.py"),
        "migration_prohibited": (True, "run_experiment_no_migration.py"),
    }
    observed: dict[str, tuple[bool, str]] = {}
    for variant in variants:
        variant_id = str(variant.get("id", ""))
        fixed_zero = variant.get("migration_fixed_zero")
        if not isinstance(fixed_zero, bool):
            raise ValueError(f"{variant_id}.migration_fixed_zero must be boolean")
        entrypoint_name = str(variant.get("entrypoint", ""))
        entrypoint = _resolve_from_experiment(entrypoint_name)
        if not entrypoint.is_file():
            raise FileNotFoundError(entrypoint)
        observed[variant_id] = (fixed_zero, entrypoint_name)
    if observed != expected_variants:
        raise ValueError(
            f"Migration variants drifted: expected={expected_variants}, actual={observed}"
        )

    policy = plan["execution_policy"]
    required_policy = {
        "max_parallel_jobs": 5,
        "threads_per_job": 16,
        "total_thread_limit": 80,
        "time_limit_seconds": 10800,
        "mip_gap": 0.001,
        "soft_mem_limit_gb_per_job": 80,
    }
    for key, expected in required_policy.items():
        actual = policy.get(key)
        if float(actual) != float(expected):
            raise ValueError(
                f"Paired migration execution policy drifted for {key}: {actual} != {expected}"
            )

    experiment = base_config["experiment"]
    dimensions = {
        "scenarios": int(experiment["num_scenarios"]),
        "on_demand": int(experiment["class_counts"]["on_demand"]),
        "spot": int(experiment["class_counts"]["spot"]),
        "batch_jobs": int(experiment["class_counts"]["batch_jobs"]),
        "max_batch_families": int(base_config["batch"]["max_families"]),
        "servers": int(experiment["num_servers"]),
    }
    expected_dimensions = {
        key: int(value)
        for key, value in plan["validation"]["expected_dimensions"].items()
    }
    if dimensions != expected_dimensions:
        raise ValueError(
            f"Paired migration dimensions drifted: {dimensions} != {expected_dimensions}"
        )

    stage = {"factors": plan["stage"]["factors"]}
    specs = SHARED._build_candidate_specs(base_config, stage)
    expected_per_variant = int(
        plan["generation"]["exact_unique_candidate_configs_per_variant"]
    )
    if len(specs) != expected_per_variant:
        raise ValueError(
            f"Expected {expected_per_variant} candidates per variant, found {len(specs)}"
        )
    expected_total = int(plan["generation"]["exact_solver_launches"])
    if len(specs) * len(variants) != expected_total:
        raise ValueError(
            f"Expected {expected_total} paired launches, found {len(specs) * len(variants)}"
        )
    return variants, specs


def _materialize_paired_candidates(
    *,
    plan: dict[str, Any],
    plan_path: Path,
    base_path: Path,
    variants: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    suite_root: Path,
) -> list[dict[str, Any]]:
    policy = plan["execution_policy"]
    rows: list[dict[str, Any]] = []
    for spec in specs:
        pair_id = str(spec["run_id"])
        run_slug = pair_id.replace("/", "__")
        for variant in variants:
            variant_id = str(variant["id"])
            fixed_zero = bool(variant["migration_fixed_zero"])
            entrypoint = _resolve_from_experiment(str(variant["entrypoint"]))
            config = copy.deepcopy(spec["config"])
            config["workspace_root"] = str(REPOSITORY_ROOT.resolve())
            config["experiment"]["name"] = (
                f"{plan['suite_name']}__{variant_id}__{run_slug}"
            )
            config["migration"]["fixed_zero"] = fixed_zero
            solver = config.setdefault("solver", {})
            solver.update(
                {
                    "threads": int(policy["threads_per_job"]),
                    "time_limit_seconds": float(policy["time_limit_seconds"]),
                    "mip_gap": float(policy["mip_gap"]),
                    "soft_mem_limit_gb": float(
                        policy["soft_mem_limit_gb_per_job"]
                    ),
                    "nodefile_start_gb": float(policy["nodefile_start_gb"]),
                    "nodefile_dir": "nodefiles",
                    "write_lp": bool(policy["safety"]["write_lp"]),
                    "write_mps": bool(policy["safety"]["write_mps"]),
                }
            )

            config_path = suite_root / "configs" / variant_id / f"{run_slug}.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            run_dir = suite_root / "runs" / variant_id / pair_id
            rows.append(
                {
                    "run_id": f"{variant_id}/{pair_id}",
                    "variant_id": variant_id,
                    "comparison_pair_id": pair_id,
                    "factor_id": spec["factor_id"],
                    "level_id": spec["level_id"],
                    "override_path": spec["override_path"],
                    "override_value_json": json.dumps(
                        spec["override_value"], ensure_ascii=False
                    ),
                    "migration_fixed_zero": fixed_zero,
                    "entrypoint": str(entrypoint),
                    "entrypoint_sha256": SHARED._sha256(entrypoint),
                    "config_path": str(config_path.resolve()),
                    "config_sha256": SHARED._sha256(config_path),
                    "run_dir": str(run_dir.resolve()),
                    "threads": int(solver["threads"]),
                    "time_limit_seconds": float(solver["time_limit_seconds"]),
                    "mip_gap": float(solver["mip_gap"]),
                    "soft_mem_limit_gb": float(solver["soft_mem_limit_gb"]),
                    "nodefile_start_gb": float(solver["nodefile_start_gb"]),
                    "plan_path": str(plan_path),
                    "base_config_path": str(base_path),
                }
            )

    manifest = suite_root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _resolve_formulation_sources(formulation: dict[str, Any] | None) -> list[Path]:
    if formulation is None:
        return []
    sources: list[Path] = []
    spec_path = formulation.get("spec_path")
    if spec_path:
        sources.append(Path(spec_path).resolve())
    change_summary = formulation.get("change_summary_path")
    if change_summary:
        sources.append(Path(change_summary).resolve())
    return sources


def _capture_source_hashes(
    *,
    plan: dict[str, Any],
    plan_path: Path,
    base_path: Path,
    formulation: dict[str, Any] | None = None,
) -> dict[str, str]:
    paths = {
        Path(__file__).resolve(),
        plan_path.resolve(),
        base_path.resolve(),
        SHARED_RUNNER_PATH.resolve(),
        MICRO_LAUNCHER_PATH.resolve(),
    }
    if formulation is None:
        formulation = MICRO._resolve_formulation(plan)
    paths.update(_resolve_formulation_sources(formulation))
    for variant in plan["variants"]:
        paths.add(_resolve_from_experiment(str(variant["entrypoint"])))
    paths.update(
        path.resolve()
        for path in (EXPERIMENT_ROOT / "src" / "notion_server_min_vmp").glob("*.py")
    )
    missing = [str(path) for path in sorted(paths) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot capture paired-suite sources: {missing}")
    return {str(path): SHARED._sha256(path) for path in sorted(paths)}


def _process_command(pid: int) -> str | None:
    stat_path = Path(f"/proc/{pid}/stat")
    cmdline = Path(f"/proc/{pid}/cmdline")
    try:
        # A detached launcher may briefly remain as a zombie until its parent
        # reaps it.  It no longer owns workers or resources at that point.
        stat_fields = stat_path.read_text(encoding="utf-8").split()
        if len(stat_fields) >= 3 and stat_fields[2] == "Z":
            return None
        return cmdline.read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        ).strip()
    except FileNotFoundError:
        return None


def _wait_for_predecessor(
    predecessor_root: Path,
    *,
    queue_state_path: Path,
    poll_seconds: float,
) -> dict[str, Any]:
    pid_path = predecessor_root / "suite.pid"
    if not pid_path.is_file():
        raise FileNotFoundError(f"Predecessor suite PID file is missing: {pid_path}")
    pid = int(pid_path.read_text(encoding="utf-8").strip())
    expected_token = str(predecessor_root.resolve())
    waited_from = datetime.now(timezone.utc)
    while True:
        command = _process_command(pid)
        if command is None:
            break
        if expected_token not in command or "run_micro_stress_sweep.py" not in command:
            raise RuntimeError(
                "Predecessor PID is alive but its command no longer matches the suite: "
                f"pid={pid}, command={command!r}"
            )
        _write_json(
            queue_state_path,
            {
                "state": "WAITING_FOR_PREDECESSOR",
                "predecessor_suite_root": str(predecessor_root),
                "predecessor_pid": pid,
                "waited_from_utc": waited_from.isoformat(),
                "last_checked_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        time.sleep(poll_seconds)

    summary_path = predecessor_root / "suite_summary.json"
    summary = None
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            summary = None
    return {
        "suite_root": str(predecessor_root),
        "pid": pid,
        "waited_from_utc": waited_from.isoformat(),
        "released_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary_path": str(summary_path),
        "summary": summary,
    }


def _write_status_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _execute_paired_candidates(
    rows: list[dict[str, Any]],
    *,
    policy: dict[str, Any],
    suite_root: Path,
    required_source_hashes: dict[str, str],
) -> list[dict[str, Any]]:
    workers = int(policy["max_parallel_jobs"])
    threads = int(policy["threads_per_job"])
    cpus = SHARED._available_cpus()[: workers * threads]
    cpu_groups = [
        cpus[index * threads : (index + 1) * threads]
        for index in range(workers)
    ]
    if len(cpu_groups) != workers or any(len(group) != threads for group in cpu_groups):
        raise RuntimeError("Could not construct five disjoint 16-CPU worker groups")
    slots: queue.Queue[tuple[int, list[int]]] = queue.Queue()
    for slot, group in enumerate(cpu_groups):
        slots.put((slot, group))

    status_jsonl = suite_root / "suite_status.jsonl"
    status_lock = threading.Lock()

    def record(payload: dict[str, Any]) -> None:
        event = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **payload}
        with status_lock:
            with status_jsonl.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            print(json.dumps(event, ensure_ascii=False), flush=True)

    def execute(row: dict[str, Any]) -> dict[str, Any]:
        slot, cpu_group = slots.get()
        started = time.monotonic()
        run_dir = Path(row["run_dir"])
        try:
            if (run_dir / "summary.json").is_file():
                result = {
                    "run_id": row["run_id"],
                    "variant_id": row["variant_id"],
                    "comparison_pair_id": row["comparison_pair_id"],
                    "state": "SKIPPED_COMPLETE",
                    "run_dir": str(run_dir),
                    "elapsed_seconds": 0.0,
                }
                record(result)
                return result
            if run_dir.exists() and any(run_dir.iterdir()):
                raise FileExistsError(
                    f"Refusing to mix an incomplete prior attempt in {run_dir}"
                )
            run_dir.mkdir(parents=True, exist_ok=True)
            SHARED._verify_required_source_hashes(required_source_hashes)
            command: list[str] = []
            if shutil.which("taskset"):
                command.extend(["taskset", "-c", ",".join(map(str, cpu_group))])
            command.extend(
                [
                    str(VENV_PYTHON),
                    "-u",
                    row["entrypoint"],
                    "--config",
                    row["config_path"],
                    "--run-dir",
                    str(run_dir),
                ]
            )
            record(
                {
                    "run_id": row["run_id"],
                    "variant_id": row["variant_id"],
                    "comparison_pair_id": row["comparison_pair_id"],
                    "state": "RUNNING",
                    "slot": slot,
                    "cpu_ids": cpu_group,
                    "threads": threads,
                    "entrypoint": row["entrypoint"],
                    "run_dir": str(run_dir),
                }
            )
            console_path = run_dir / "console.log"
            with console_path.open("w", encoding="utf-8") as console:
                process = subprocess.Popen(
                    command,
                    cwd=REPOSITORY_ROOT,
                    env=SHARED._child_environment(),
                    stdin=subprocess.DEVNULL,
                    stdout=console,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                (run_dir / "child.pid").write_text(
                    f"{process.pid}\n", encoding="utf-8"
                )
                returncode = process.wait()

            summary_path = run_dir / "summary.json"
            summary: dict[str, Any] = {}
            if summary_path.is_file():
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    summary = {}
            solver = summary.get("solver", {})
            solver_status = solver.get("status")
            achieved_gap = solver.get("mip_gap")
            state = SHARED._completion_state(
                returncode=returncode,
                summary_exists=summary_path.is_file(),
                solver_status=solver_status,
                achieved_gap=achieved_gap,
                target_gap=float(row["mip_gap"]),
            )
            result = {
                "run_id": row["run_id"],
                "variant_id": row["variant_id"],
                "comparison_pair_id": row["comparison_pair_id"],
                "state": state,
                "returncode": returncode,
                "elapsed_seconds": time.monotonic() - started,
                "run_dir": str(run_dir),
                "solver_status": solver_status,
                "target_mip_gap": float(row["mip_gap"]),
                "achieved_mip_gap": achieved_gap,
            }
            record(result)
            return result
        except Exception as exc:
            result = {
                "run_id": row["run_id"],
                "variant_id": row["variant_id"],
                "comparison_pair_id": row["comparison_pair_id"],
                "state": "LAUNCH_ERROR",
                "error": repr(exc),
                "elapsed_seconds": time.monotonic() - started,
                "run_dir": str(run_dir),
            }
            record(result)
            return result
        finally:
            slots.put((slot, cpu_group))

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(execute, row) for row in rows]
        for future in as_completed(futures):
            results.append(future.result())
            _write_status_csv(suite_root / "suite_status.csv", results)
    return results


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired 3-hour migration-allowed/prohibited micro-stress OFAT suites"
    )
    parser.add_argument("--google-dir", required=True)
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument(
        "--spec-path",
        help=(
            "Optional local copy of the non-vendored formulation specification; "
            "its SHA-256 must match formulation.spec_sha256 in the plan"
        ),
    )
    parser.add_argument("--suite-root", required=True)
    parser.add_argument("--predecessor-suite-root")
    parser.add_argument("--predecessor-poll-seconds", type=float, default=30.0)
    parser.add_argument("--generate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not (1.0 <= float(args.predecessor_poll_seconds) <= 60.0):
        raise ValueError("--predecessor-poll-seconds must be between 1 and 60")
    plan_path = _resolve_from_experiment(args.plan)
    plan = SHARED._load_yaml(plan_path)
    base_path = _resolve_from_experiment(str(plan["base_config"]))
    google_dir = MICRO._resolve_google_dir(args.google_dir)
    base_config = MICRO._effective_base_config(
        SHARED._load_yaml(base_path), google_dir
    )
    variants, specs = _validate_plan_contract(plan, base_config)
    dataset_audit = MICRO._validate_dataset_shape(google_dir)
    dataset_hashes = MICRO._capture_dataset_hashes(google_dir)
    formulation_audit = MICRO._resolve_formulation(plan, args.spec_path)
    source_hashes = _capture_source_hashes(
        plan=plan,
        plan_path=plan_path,
        base_path=base_path,
        formulation=formulation_audit,
    )

    suite_root = _resolve_from_experiment(args.suite_root)
    suite_root.mkdir(parents=True, exist_ok=True)
    unexpected = {
        entry.name for entry in suite_root.iterdir() if entry.name not in ALLOWED_INITIAL_FILES
    }
    if unexpected:
        raise FileExistsError(
            f"Paired suite root is not fresh ({sorted(unexpected)}): {suite_root}"
        )
    if not VENV_PYTHON.is_file():
        raise FileNotFoundError(VENV_PYTHON)

    rows = _materialize_paired_candidates(
        plan=plan,
        plan_path=plan_path,
        base_path=base_path,
        variants=variants,
        specs=specs,
        suite_root=suite_root,
    )
    queue_state_path = suite_root / "queue_state.json"
    metadata = {
        "schema_version": 1,
        "state": "MATERIALIZED",
        "suite_name": plan["suite_name"],
        "launcher_pid": os.getpid(),
        "hostname": socket.gethostname(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_path": str(plan_path),
        "plan_sha256": SHARED._sha256(plan_path),
        "base_config_path": str(base_path),
        "base_config_sha256": SHARED._sha256(base_path),
        "formulation": formulation_audit,
        "google_dir": str(google_dir),
        "candidate_count": len(rows),
        "comparison_pair_count": len(specs),
        "variant_ids": [str(variant["id"]) for variant in variants],
        "dataset_audit": dataset_audit,
        "source_files_sha256": source_hashes,
        "dataset_files_sha256": dataset_hashes,
        "execution_policy": plan["execution_policy"],
        "manifest": str((suite_root / "manifest.csv").resolve()),
    }
    _write_json(suite_root / "suite_metadata.json", metadata)
    (suite_root / "paired_suite.pid").write_text(
        f"{os.getpid()}\n", encoding="utf-8"
    )
    if args.generate_only:
        _write_json(queue_state_path, {"state": "GENERATED_ONLY"})
        print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)
        return 0

    predecessor_audit = None
    if args.predecessor_suite_root:
        predecessor_root = _resolve_from_experiment(args.predecessor_suite_root)
        predecessor_audit = _wait_for_predecessor(
            predecessor_root,
            queue_state_path=queue_state_path,
            poll_seconds=float(args.predecessor_poll_seconds),
        )

    resource_audit = SHARED._resource_preflight(
        plan["execution_policy"], suite_root
    )
    resource_audit["memory_limit_enforcement"] = plan["execution_policy"][
        "memory_limit_enforcement"
    ]
    metadata["state"] = "RUNNING"
    metadata["started_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["predecessor"] = predecessor_audit
    metadata["resource_preflight"] = resource_audit
    _write_json(suite_root / "suite_metadata.json", metadata)
    _write_json(
        queue_state_path,
        {
            "state": "RUNNING",
            "started_at_utc": metadata["started_at_utc"],
            "candidate_count": len(rows),
        },
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)

    results = _execute_paired_candidates(
        rows,
        policy=plan["execution_policy"],
        suite_root=suite_root,
        required_source_hashes=source_hashes,
    )
    variants_summary: dict[str, dict[str, int]] = {}
    for variant in variants:
        variant_id = str(variant["id"])
        selected = [row for row in results if row["variant_id"] == variant_id]
        variants_summary[variant_id] = {
            "candidate_count": len(selected),
            "target_gap_complete_or_skipped": sum(
                row["state"] in {"COMPLETED_TARGET_GAP", "SKIPPED_COMPLETE"}
                for row in selected
            ),
            "target_gap_not_reached": sum(
                row["state"] == "TARGET_GAP_NOT_REACHED" for row in selected
            ),
            "failures": sum(
                row["state"] in {"FAILED", "LAUNCH_ERROR"} for row in selected
            ),
        }
    failures = sum(
        row["state"] in {"FAILED", "LAUNCH_ERROR"} for row in results
    )
    misses = sum(row["state"] == "TARGET_GAP_NOT_REACHED" for row in results)
    final = {
        "state": "FINISHED",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_count": len(results),
        "variants": variants_summary,
        "failures": failures,
        "target_gap_not_reached": misses,
    }
    _write_json(suite_root / "suite_summary.json", final)
    _write_json(queue_state_path, final)
    print(json.dumps(final, indent=2), flush=True)
    return 1 if failures or misses else 0


if __name__ == "__main__":
    raise SystemExit(main())
