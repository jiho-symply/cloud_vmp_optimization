#!/usr/bin/env python3
"""Generate and execute one deduplicated OFAT sensitivity stage.

The single-run entry point remains the authority for data preparation, model
construction, optimization, and reporting.  This launcher only materializes
immutable derived configs and schedules them with disjoint CPU affinity.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
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
from typing import Any

import yaml


EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
SINGLE_RUN_ENTRYPOINT = EXPERIMENT_ROOT / "run_experiment.py"
VENV_PYTHON = REPOSITORY_ROOT / ".venv" / "bin" / "python"


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    # PyYAML materializes unquoted ISO timestamps as datetime objects.  Their
    # string form is stable and keeps scientific-config deduplication lossless.
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def _set_config_path(config: dict[str, Any], dotted_path: str, value: Any) -> None:
    keys = dotted_path.split(".")
    target: dict[str, Any] = config
    for key in keys[:-1]:
        child = target.get(key)
        if not isinstance(child, dict):
            raise KeyError(f"Invalid config path {dotted_path!r}: {key!r} is not a mapping")
        target = child
    if keys[-1] not in target:
        raise KeyError(f"Invalid config path {dotted_path!r}: missing {keys[-1]!r}")
    target[keys[-1]] = copy.deepcopy(value)


def _safe_component(value: str) -> str:
    if not value or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in value):
        raise ValueError(f"Unsafe run identifier component: {value!r}")
    return value


def _build_candidate_specs(
    base_config: dict[str, Any],
    stage: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return the baseline anchor plus globally deduplicated OFAT overrides."""

    specs: list[dict[str, Any]] = [
        {
            "run_id": "baseline",
            "factor_id": "baseline",
            "level_id": "baseline",
            "config": copy.deepcopy(base_config),
            "override_path": None,
            "override_value": None,
        }
    ]
    seen = {_canonical(base_config)}
    for factor in stage["factors"]:
        factor_id = _safe_component(str(factor["id"]))
        config_path = str(factor["config_path"])
        for level in factor["levels"]:
            level_id = _safe_component(str(level["id"]))
            candidate = copy.deepcopy(base_config)
            _set_config_path(candidate, config_path, level["value"])
            signature = _canonical(candidate)
            if signature in seen:
                continue
            seen.add(signature)
            specs.append(
                {
                    "run_id": f"{factor_id}/{level_id}",
                    "factor_id": factor_id,
                    "level_id": level_id,
                    "config": candidate,
                    "override_path": config_path,
                    "override_value": copy.deepcopy(level["value"]),
                }
            )
    return specs


def _available_cpus() -> list[int]:
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return list(range(os.cpu_count() or 1))


def _memory_available_gib() -> float:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        values[key] = int(raw.strip().split()[0])
    return values["MemAvailable"] / (1024.0**2)


def _resource_preflight(policy: dict[str, Any], suite_root: Path) -> dict[str, Any]:
    workers = int(policy["max_parallel_jobs"])
    threads = int(policy["threads_per_job"])
    total_thread_limit = int(policy.get("total_thread_limit", workers * threads))
    required_threads = workers * threads
    available_cpus = _available_cpus()
    if required_threads > total_thread_limit:
        raise ValueError(
            f"Requested {required_threads} solver threads, exceeding total_thread_limit={total_thread_limit}"
        )
    if required_threads > len(available_cpus):
        raise ValueError(
            f"Requested {required_threads} solver threads, but affinity exposes {len(available_cpus)} CPUs"
        )

    safety = policy["safety"]
    soft_mem = float(policy["soft_mem_limit_gb_per_job"])
    memory_reserve = float(safety["memory_reserve_gb"])
    available_memory = _memory_available_gib()
    required_memory = workers * soft_mem + memory_reserve
    if available_memory < required_memory:
        raise RuntimeError(
            f"Insufficient available memory: {available_memory:.1f} GiB < "
            f"{workers}*{soft_mem:.1f}+{memory_reserve:.1f}={required_memory:.1f} GiB"
        )

    disk = shutil.disk_usage(suite_root.parent)
    available_disk = disk.free / (1024.0**3)
    disk_reserve = float(safety["disk_reserve_gb"])
    if available_disk < disk_reserve:
        raise RuntimeError(
            f"Insufficient disk: {available_disk:.1f} GiB available < {disk_reserve:.1f} GiB reserve"
        )
    return {
        "available_cpu_ids": available_cpus,
        "required_solver_threads": required_threads,
        "total_thread_limit": total_thread_limit,
        "available_memory_gib": available_memory,
        "required_memory_including_reserve_gib": required_memory,
        "available_disk_gib": available_disk,
        "disk_reserve_gib": disk_reserve,
    }


def _materialize_candidates(
    *,
    specs: list[dict[str, Any]],
    plan: dict[str, Any],
    plan_path: Path,
    base_path: Path,
    stage_name: str,
    suite_root: Path,
) -> list[dict[str, Any]]:
    policy = plan["execution_policy"]
    config_root = suite_root / "configs" / stage_name
    config_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        config = copy.deepcopy(spec["config"])
        run_slug = spec["run_id"].replace("/", "__")
        config["workspace_root"] = str(REPOSITORY_ROOT.resolve())
        config["experiment"]["name"] = (
            f"{plan['suite_name']}__{stage_name}__{run_slug}"
        )
        solver = config.setdefault("solver", {})
        configured_time_limit = policy.get("time_limit_seconds")
        solver.update(
            {
                "threads": int(policy["threads_per_job"]),
                "time_limit_seconds": (
                    None
                    if configured_time_limit is None
                    else float(configured_time_limit)
                ),
                "mip_gap": float(policy["mip_gap"]),
                "soft_mem_limit_gb": float(policy["soft_mem_limit_gb_per_job"]),
                "nodefile_start_gb": float(policy["nodefile_start_gb"]),
                "nodefile_dir": "nodefiles",
                "write_lp": bool(policy["safety"]["write_lp"]),
                "write_mps": bool(policy["safety"]["write_mps"]),
            }
        )

        config_path = config_root / f"{run_slug}.yaml"
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        run_dir = suite_root / stage_name / spec["run_id"]
        rows.append(
            {
                "run_id": spec["run_id"],
                "factor_id": spec["factor_id"],
                "level_id": spec["level_id"],
                "override_path": spec["override_path"],
                "override_value_json": json.dumps(spec["override_value"], ensure_ascii=False),
                "config_path": str(config_path.resolve()),
                "config_sha256": _sha256(config_path),
                "run_dir": str(run_dir.resolve()),
                "threads": int(solver["threads"]),
                "time_limit_seconds": solver["time_limit_seconds"],
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


def _child_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    return environment


def _write_status_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _completion_state(
    *,
    returncode: int,
    summary_exists: bool,
    solver_status: str | None,
    achieved_gap: float | None,
    target_gap: float,
) -> str:
    if returncode != 0 or not summary_exists:
        return "FAILED"
    if solver_status == "OPTIMAL":
        return "COMPLETED_TARGET_GAP"
    if achieved_gap is not None and float(achieved_gap) <= target_gap + 1e-12:
        return "COMPLETED_TARGET_GAP"
    return "TARGET_GAP_NOT_REACHED"


def _verify_required_source_hashes(required_source_hashes: dict[str, str]) -> None:
    """Fail before launch if any captured formulation source has drifted."""

    for raw_path, expected_sha256 in sorted(required_source_hashes.items()):
        path = Path(raw_path)
        if not path.is_file():
            raise RuntimeError(f"Captured source file disappeared before child launch: {path}")
        actual_sha256 = _sha256(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                "Captured source hash changed before child launch: "
                f"{path} (expected {expected_sha256}, found {actual_sha256})"
            )


def _execute_candidates(
    rows: list[dict[str, Any]],
    *,
    policy: dict[str, Any],
    suite_root: Path,
    required_source_hashes: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    workers = int(policy["max_parallel_jobs"])
    threads = int(policy["threads_per_job"])
    cpus = _available_cpus()[: workers * threads]
    groups = [cpus[index * threads : (index + 1) * threads] for index in range(workers)]
    slots: queue.Queue[tuple[int, list[int]]] = queue.Queue()
    for slot, group in enumerate(groups):
        slots.put((slot, group))

    status_path = suite_root / "suite_status.jsonl"
    status_lock = threading.Lock()

    def record(payload: dict[str, Any]) -> None:
        payload = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **payload}
        with status_lock:
            with status_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
            print(json.dumps(payload, ensure_ascii=False), flush=True)

    def execute(row: dict[str, Any]) -> dict[str, Any]:
        slot, cpu_group = slots.get()
        started = time.monotonic()
        run_dir = Path(row["run_dir"])
        try:
            if (run_dir / "summary.json").is_file():
                result = {
                    "run_id": row["run_id"],
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
            command: list[str] = []
            if shutil.which("taskset"):
                command.extend(["taskset", "-c", ",".join(map(str, cpu_group))])
            command.extend(
                [
                    str(VENV_PYTHON),
                    "-u",
                    str(SINGLE_RUN_ENTRYPOINT),
                    "--config",
                    row["config_path"],
                    "--run-dir",
                    str(run_dir),
                ]
            )
            console_path = run_dir / "console.log"
            if required_source_hashes is not None:
                # Recompute immediately before each Popen, including jobs that
                # wait for a later worker wave. This prevents one suite from
                # silently mixing formulations when live source files change.
                _verify_required_source_hashes(required_source_hashes)
            record(
                {
                    "run_id": row["run_id"],
                    "state": "RUNNING",
                    "slot": slot,
                    "cpu_ids": cpu_group,
                    "threads": threads,
                    "run_dir": str(run_dir),
                }
            )
            with console_path.open("w", encoding="utf-8") as console:
                process = subprocess.Popen(
                    command,
                    cwd=REPOSITORY_ROOT,
                    env=_child_environment(),
                    stdout=console,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                (run_dir / "child.pid").write_text(f"{process.pid}\n", encoding="utf-8")
                returncode = process.wait()

            summary_path = run_dir / "summary.json"
            summary: dict[str, Any] = {}
            if summary_path.is_file():
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    summary = {}
            solver_summary = summary.get("solver", {})
            solver_status = solver_summary.get("status")
            achieved_gap = solver_summary.get("mip_gap")
            target_gap = float(row["mip_gap"])
            state = _completion_state(
                returncode=returncode,
                summary_exists=summary_path.is_file(),
                solver_status=solver_status,
                achieved_gap=achieved_gap,
                target_gap=target_gap,
            )
            result = {
                "run_id": row["run_id"],
                "state": state,
                "returncode": returncode,
                "elapsed_seconds": time.monotonic() - started,
                "run_dir": str(run_dir),
                "solver_status": solver_status,
                "target_mip_gap": target_gap,
                "achieved_mip_gap": achieved_gap,
            }
            record(result)
            return result
        except Exception as exc:
            result = {
                "run_id": row["run_id"],
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
    parser = argparse.ArgumentParser(description="Run one deduplicated sensitivity-analysis stage")
    parser.add_argument("--plan", default="plans/sweep_plan.yaml")
    parser.add_argument("--stage", default="first_stage", choices=("first_stage", "second_stage"))
    parser.add_argument("--suite-root", help="Fresh output root; defaults to a UTC launch-id path")
    parser.add_argument("--generate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    plan_path = Path(args.plan).expanduser()
    if not plan_path.is_absolute():
        plan_path = EXPERIMENT_ROOT / plan_path
    plan_path = plan_path.resolve()
    plan = _load_yaml(plan_path)
    base_path = Path(plan["base_config"]).expanduser()
    if not base_path.is_absolute():
        base_path = EXPERIMENT_ROOT / base_path
    base_path = base_path.resolve()
    base_config = _load_yaml(base_path)
    stage = plan["stages"][args.stage]
    if not stage.get("enabled", False):
        raise ValueError(f"Sweep stage is disabled: {args.stage}")

    launch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.suite_root:
        suite_root = Path(args.suite_root).expanduser()
        if not suite_root.is_absolute():
            suite_root = EXPERIMENT_ROOT / suite_root
    else:
        suite_root = EXPERIMENT_ROOT / plan["output_root"] / launch_id
    suite_root = suite_root.resolve()
    suite_root.mkdir(parents=True, exist_ok=True)
    unexpected = {
        entry.name for entry in suite_root.iterdir()
        if entry.name not in {"launcher.log", "suite.pid"}
    }
    if unexpected:
        raise FileExistsError(
            f"Suite root is not fresh ({sorted(unexpected)}): {suite_root}"
        )

    if not VENV_PYTHON.is_file():
        raise FileNotFoundError(f"Required virtual environment interpreter is missing: {VENV_PYTHON}")
    if not SINGLE_RUN_ENTRYPOINT.is_file():
        raise FileNotFoundError(SINGLE_RUN_ENTRYPOINT)

    specs = _build_candidate_specs(base_config, stage)
    expected = int(plan["generation"]["exact_unique_candidate_configs"][args.stage])
    if len(specs) != expected:
        raise ValueError(f"Expected {expected} unique {args.stage} candidates, generated {len(specs)}")

    resource_audit = _resource_preflight(plan["execution_policy"], suite_root)
    rows = _materialize_candidates(
        specs=specs,
        plan=plan,
        plan_path=plan_path,
        base_path=base_path,
        stage_name=args.stage,
        suite_root=suite_root,
    )
    metadata = {
        "schema_version": 1,
        "suite_name": plan["suite_name"],
        "stage": args.stage,
        "launch_id": launch_id,
        "launched_at_utc": datetime.now(timezone.utc).isoformat(),
        "launcher_pid": os.getpid(),
        "hostname": socket.gethostname(),
        "plan_path": str(plan_path),
        "plan_sha256": _sha256(plan_path),
        "base_config_path": str(base_path),
        "base_config_sha256": _sha256(base_path),
        "candidate_count": len(rows),
        "execution_policy": plan["execution_policy"],
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

    results = _execute_candidates(
        rows,
        policy=plan["execution_policy"],
        suite_root=suite_root,
    )
    failures = sum(row["state"] in {"FAILED", "LAUNCH_ERROR"} for row in results)
    target_misses = sum(row["state"] == "TARGET_GAP_NOT_REACHED" for row in results)
    target_completions = sum(
        row["state"] in {"COMPLETED_TARGET_GAP", "SKIPPED_COMPLETE"}
        for row in results
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
