#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _cpu_groups(workers: int, threads: int) -> list[list[int]]:
    available = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
    required = workers * threads
    if required > len(available):
        raise ValueError(f"Requested {workers} workers x {threads} threads = {required}, but affinity exposes {len(available)} CPUs")
    return [available[n * threads : (n + 1) * threads] for n in range(workers)]


def _child_environment(base: dict[str, str]) -> dict[str, str]:
    env = base.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    return env


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--threads-per-run", type=int, default=8)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    repo_root = Path.cwd().resolve()
    entrypoint = repo_root / "scripts/run_notion_energy_experiment.py"
    if not entrypoint.is_file():
        raise FileNotFoundError(f"Run from repository root; missing {entrypoint}")
    rows = _read_manifest(manifest_path)
    groups = _cpu_groups(args.max_workers, args.threads_per_run)
    slots: queue.Queue[tuple[int, list[int]]] = queue.Queue()
    for slot, cpus in enumerate(groups):
        slots.put((slot, cpus))

    status_path = manifest_path.parent / "suite_status.jsonl"
    status_lock = threading.Lock()
    if status_path.exists() and not args.rerun:
        status_path.unlink()

    def record(payload: dict[str, Any]) -> None:
        with status_lock:
            with status_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            print(json.dumps(payload, ensure_ascii=False), flush=True)

    def execute(row: dict[str, str]) -> dict[str, Any]:
        run_dir = Path(row["run_dir"])
        summary = run_dir / "summary.json"
        if summary.is_file() and not args.rerun:
            payload = {"run_id": row["run_id"], "phase": row["phase"], "state": "SKIPPED_COMPLETE", "run_dir": str(run_dir)}
            record(payload)
            return payload
        slot, cpus = slots.get()
        started = time.time()
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            command = []
            if shutil.which("taskset"):
                command += ["taskset", "-c", ",".join(map(str, cpus))]
            command += [
                str(repo_root / ".venv/bin/python"),
                "-u",
                str(entrypoint),
                "--config",
                row["config_path"],
                "--run-dir",
                str(run_dir),
                "--threads",
                str(args.threads_per_run),
            ]
            if args.prepare_only:
                command.append("--prepare-only")
            if (run_dir / "prepared_data").exists():
                command.append("--overwrite-prepared")
            env = _child_environment(os.environ)
            console_path = run_dir / "console.log"
            record(
                {
                    "run_id": row["run_id"],
                    "phase": row["phase"],
                    "state": "RUNNING",
                    "slot": slot,
                    "cpus": cpus,
                    "run_dir": str(run_dir),
                }
            )
            with console_path.open("w", encoding="utf-8") as console:
                completed = subprocess.run(
                    command,
                    cwd=repo_root,
                    env=env,
                    stdout=console,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            payload = {
                "run_id": row["run_id"],
                "phase": row["phase"],
                "state": "COMPLETED" if completed.returncode == 0 else "FAILED",
                "returncode": completed.returncode,
                "elapsed_seconds": time.time() - started,
                "run_dir": str(run_dir),
            }
            record(payload)
            return payload
        except Exception as exc:
            payload = {
                "run_id": row["run_id"],
                "phase": row["phase"],
                "state": "LAUNCH_ERROR",
                "error": repr(exc),
                "elapsed_seconds": time.time() - started,
                "run_dir": str(run_dir),
            }
            record(payload)
            return payload
        finally:
            slots.put((slot, cpus))

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(execute, row) for row in rows]
        for future in as_completed(futures):
            results.append(future.result())

    status_csv = manifest_path.parent / "suite_status.csv"
    keys = sorted({key for row in results for key in row})
    with status_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    failed = sum(row["state"] in {"FAILED", "LAUNCH_ERROR"} for row in results)
    print(f"Suite finished: {len(results)} runs, {failed} failures", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
