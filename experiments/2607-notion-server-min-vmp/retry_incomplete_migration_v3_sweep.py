#!/usr/bin/env python3
"""Retry only incomplete cases from an immutable migration-v3 sweep suite.

The original suite is treated as read-only.  Configs and provenance hashes are
verified against its manifest/metadata, while every retried solve writes into a
fresh sibling suite root.  Scheduling is delegated to the established sweep
runner so CPU affinity, parallelism, solver limits, and status reporting remain
identical to the original run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import run_micro_stress_migration_v3_sweep as migration_v3


EXPERIMENT_ROOT = Path(__file__).resolve().parent
SHARED = migration_v3.BASE.SHARED


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = EXPERIMENT_ROOT / path
    return path.resolve()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _verify_hashes(hashes: dict[str, str], *, label: str) -> None:
    for raw_path, expected in sorted(hashes.items()):
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"{label} file is missing: {path}")
        actual = SHARED._sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"{label} hash drift for {path}: expected={expected}, actual={actual}"
            )


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Source manifest is empty: {path}")
    return rows


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retry incomplete cases from a migration-v3 first-stage suite"
    )
    parser.add_argument("--source-suite", required=True)
    parser.add_argument("--suite-root", required=True)
    parser.add_argument("--expected-incomplete", type=int, default=6)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    source_suite = _resolve(args.source_suite)
    suite_root = _resolve(args.suite_root)
    if source_suite == suite_root:
        raise ValueError("Retry suite root must differ from the source suite")

    source_metadata_path = source_suite / "suite_metadata.json"
    source_manifest_path = source_suite / "manifest.csv"
    if not source_metadata_path.is_file() or not source_manifest_path.is_file():
        raise FileNotFoundError(
            f"Source suite lacks suite_metadata.json or manifest.csv: {source_suite}"
        )
    source_metadata = _load_json(source_metadata_path)
    source_rows = _read_manifest(source_manifest_path)

    source_hashes = dict(source_metadata["source_files_sha256"])
    dataset_hashes = dict(source_metadata["dataset_files_sha256"])
    _verify_hashes(source_hashes, label="source")
    _verify_hashes(dataset_hashes, label="dataset")
    for row in source_rows:
        config_path = Path(str(row["config_path"]))
        actual = SHARED._sha256(config_path)
        if actual != row["config_sha256"]:
            raise RuntimeError(
                f"Immutable config hash drift for {config_path}: "
                f"expected={row['config_sha256']}, actual={actual}"
            )

    incomplete_rows = [
        row
        for row in source_rows
        if not (Path(str(row["run_dir"])) / "summary.json").is_file()
    ]
    if len(incomplete_rows) != args.expected_incomplete:
        raise RuntimeError(
            "Unexpected incomplete-case count: "
            f"expected={args.expected_incomplete}, actual={len(incomplete_rows)}, "
            f"run_ids={[row['run_id'] for row in incomplete_rows]}"
        )

    suite_root.mkdir(parents=True, exist_ok=True)
    unexpected = {
        entry.name
        for entry in suite_root.iterdir()
        if entry.name not in {"launcher.log", "suite.pid"}
    }
    if unexpected:
        raise FileExistsError(f"Retry suite root is not fresh: {sorted(unexpected)}")

    plan_path = Path(str(source_metadata["plan_path"]))
    plan = SHARED._load_yaml(plan_path)
    policy = plan["execution_policy"]
    if int(policy["max_parallel_jobs"]) != 5:
        raise ValueError("Retry requires max_parallel_jobs=5")
    if int(policy["threads_per_job"]) != 16:
        raise ValueError("Retry requires threads_per_job=16")
    if float(policy["time_limit_seconds"]) != 10800.0:
        raise ValueError("Retry requires a 10,800-second time limit")
    if float(policy["mip_gap"]) != 0.001:
        raise ValueError("Retry requires MIPGap=0.001")
    if float(policy["soft_mem_limit_gb_per_job"]) != 80.0:
        raise ValueError("Retry requires SoftMemLimit=80 GB per job")

    resource_audit = SHARED._resource_preflight(policy, suite_root)
    retry_rows: list[dict[str, Any]] = []
    for source_row in incomplete_rows:
        row = dict(source_row)
        row["run_dir"] = str(
            (suite_root / "first_stage" / str(row["run_id"])).resolve()
        )
        retry_rows.append(row)

    retry_manifest_path = suite_root / "manifest.csv"
    _write_manifest(retry_manifest_path, retry_rows)
    launcher_path = Path(__file__).resolve()
    required_source_hashes = {
        **source_hashes,
        str(launcher_path): SHARED._sha256(launcher_path),
    }
    metadata = {
        "schema_version": 1,
        "suite_name": f"{source_metadata['suite_name']}__retry_incomplete",
        "retry_of": str(source_suite),
        "source_suite_metadata_sha256": SHARED._sha256(source_metadata_path),
        "source_manifest_sha256": SHARED._sha256(source_manifest_path),
        "stage": "first_stage",
        "launched_at_utc": datetime.now(timezone.utc).isoformat(),
        "launcher_pid": os.getpid(),
        "hostname": socket.gethostname(),
        "plan_path": str(plan_path),
        "plan_sha256": SHARED._sha256(plan_path),
        "candidate_count": len(retry_rows),
        "run_ids": [row["run_id"] for row in retry_rows],
        "execution_policy": policy,
        "resource_preflight": resource_audit,
        "source_files_sha256": required_source_hashes,
        "dataset_files_sha256": dataset_hashes,
        "manifest": str(retry_manifest_path.resolve()),
    }
    (suite_root / "suite_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (suite_root / "suite.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)

    results = SHARED._execute_candidates(
        retry_rows,
        policy=policy,
        suite_root=suite_root,
        required_source_hashes=required_source_hashes,
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
