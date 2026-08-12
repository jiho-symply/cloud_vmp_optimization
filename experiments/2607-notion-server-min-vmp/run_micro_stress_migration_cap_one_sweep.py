#!/usr/bin/env python3
"""Run the migration-cap-one first-stage OFAT suite for three hours per case.

This versioned adapter preserves the shared micro-stress data, provenance,
materialization, resource-preflight, and scheduling code. It selects the
migration-cap entry point, enforces the six-by-sixteen execution contract, and
pins every formulation-specific source before any candidate is launched.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parent
BASE_LAUNCHER_PATH = EXPERIMENT_ROOT / "run_micro_stress_sweep.py"
SINGLE_RUN_ENTRYPOINT = EXPERIMENT_ROOT / "run_experiment_migration_cap_one.py"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "micro_stress_migration_cap_one.yaml"
DEFAULT_PLAN = (
    EXPERIMENT_ROOT
    / "plans"
    / "micro_stress_first_stage_migration_cap_one_3h.yaml"
)


def _load_base_launcher() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "notion_server_min_migration_cap_one_base_launcher", BASE_LAUNCHER_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load base micro-stress launcher: {BASE_LAUNCHER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base_launcher()
_ORIGINAL_VALIDATE_CONTRACT = BASE._validate_contract
_ORIGINAL_CAPTURE_SOURCE_HASHES = BASE._capture_source_hashes
_ORIGINAL_EXECUTE_CANDIDATES = BASE.SHARED._execute_candidates


def _patch_single_run_entrypoint() -> None:
    """Keep the base launcher check and shared child scheduler in agreement."""

    BASE.SINGLE_RUN_ENTRYPOINT = SINGLE_RUN_ENTRYPOINT
    BASE.SHARED.SINGLE_RUN_ENTRYPOINT = SINGLE_RUN_ENTRYPOINT


_patch_single_run_entrypoint()


def _validate_contract(config: dict[str, Any], plan: dict[str, Any]) -> None:
    """Validate the cap-one formulation and its six-worker resource contract."""

    migration = config.get("migration")
    if not isinstance(migration, dict):
        raise ValueError("Migration-cap config must declare a migration mapping")
    cap = migration.get("max_count_per_vm_per_scenario")
    if type(cap) is not int or cap != 1:
        raise ValueError(
            "migration.max_count_per_vm_per_scenario must be the integer 1"
        )

    solver = config.get("solver")
    if not isinstance(solver, dict):
        raise ValueError("Migration-cap config must declare a solver mapping")
    expected_solver = {
        "threads": 16,
        "time_limit_seconds": 10800.0,
        "mip_gap": 0.001,
        "soft_mem_limit_gb": 64.0,
        "nodefile_start_gb": 0.5,
    }
    actual_solver = {
        "threads": int(solver["threads"]),
        "time_limit_seconds": float(solver["time_limit_seconds"]),
        "mip_gap": float(solver["mip_gap"]),
        "soft_mem_limit_gb": float(solver["soft_mem_limit_gb"]),
        "nodefile_start_gb": float(solver["nodefile_start_gb"]),
    }
    if actual_solver != expected_solver:
        raise ValueError(
            "Migration-cap base solver contract drifted: "
            f"expected={expected_solver}, actual={actual_solver}"
        )

    policy = plan["execution_policy"]
    expected_policy = {
        "max_parallel_jobs": 6,
        "threads_per_job": 16,
        "total_thread_limit": 96,
        "time_limit_seconds": 10800.0,
        "mip_gap": 0.001,
        "soft_mem_limit_gb_per_job": 64.0,
        "nodefile_start_gb": 0.5,
    }
    actual_policy = {
        "max_parallel_jobs": int(policy["max_parallel_jobs"]),
        "threads_per_job": int(policy["threads_per_job"]),
        "total_thread_limit": int(policy["total_thread_limit"]),
        "time_limit_seconds": float(policy["time_limit_seconds"]),
        "mip_gap": float(policy["mip_gap"]),
        "soft_mem_limit_gb_per_job": float(policy["soft_mem_limit_gb_per_job"]),
        "nodefile_start_gb": float(policy["nodefile_start_gb"]),
    }
    if actual_policy != expected_policy:
        raise ValueError(
            "Migration-cap execution policy drifted: "
            f"expected={expected_policy}, actual={actual_policy}"
        )
    if (
        actual_policy["max_parallel_jobs"]
        * actual_policy["threads_per_job"]
        != actual_policy["total_thread_limit"]
    ):
        raise ValueError("Migration-cap worker threads must total exactly 96")

    if policy.get("memory_limit_enforcement") != (
        "gurobi_softmemlimit_plus_aggregate_preflight"
    ):
        raise ValueError("Migration-cap memory-limit enforcement policy drifted")
    safety = policy.get("safety")
    if not isinstance(safety, dict):
        raise ValueError("Migration-cap execution policy must declare safety limits")
    expected_safety = {
        "memory_reserve_gb": 64.0,
        "disk_reserve_gb": 60.0,
        "write_lp": False,
        "write_mps": False,
        "require_disjoint_cpu_affinity": True,
    }
    actual_safety = {
        "memory_reserve_gb": float(safety["memory_reserve_gb"]),
        "disk_reserve_gb": float(safety["disk_reserve_gb"]),
        "write_lp": safety["write_lp"],
        "write_mps": safety["write_mps"],
        "require_disjoint_cpu_affinity": safety[
            "require_disjoint_cpu_affinity"
        ],
    }
    if actual_safety != expected_safety:
        raise ValueError(
            "Migration-cap safety policy drifted: "
            f"expected={expected_safety}, actual={actual_safety}"
        )

    validation = plan.get("validation", {})
    if validation.get("migration_count_config_path") != (
        "migration.max_count_per_vm_per_scenario"
    ):
        raise ValueError("Migration-cap validation config path drifted")
    if validation.get("max_migration_count_per_vm_per_scenario") != 1:
        raise ValueError("Migration-cap validation limit must equal one")

    # Reuse the established dimension validation without weakening or mutating
    # the real six-worker plan consumed by preflight/materialization/execution.
    compatibility_plan = copy.deepcopy(plan)
    compatibility_plan["execution_policy"].update(
        {
            "max_parallel_jobs": 5,
            "threads_per_job": 16,
            "total_thread_limit": 80,
            "time_limit_seconds": 3600,
            "soft_mem_limit_gb_per_job": 80,
        }
    )
    _ORIGINAL_VALIDATE_CONTRACT(config, compatibility_plan)


def _capture_source_hashes(
    *,
    plan_path: Path,
    base_path: Path,
    formulation: dict[str, Any] | None,
) -> dict[str, str]:
    """Pin the shared sources and every migration-cap-specific source file."""

    _patch_single_run_entrypoint()
    hashes = _ORIGINAL_CAPTURE_SOURCE_HASHES(
        plan_path=plan_path,
        base_path=base_path,
        formulation=formulation,
    )
    price_path = _resolve_nyiso_price_path(base_path)
    required_paths = {
        Path(__file__).resolve(),
        SINGLE_RUN_ENTRYPOINT.resolve(),
        DEFAULT_CONFIG.resolve(),
        DEFAULT_PLAN.resolve(),
        price_path,
    }
    missing = [str(path) for path in sorted(required_paths) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Cannot capture migration-cap source hashes; missing={missing}"
        )
    for path in required_paths:
        hashes[str(path)] = BASE.SHARED._sha256(path)
    return dict(sorted(hashes.items()))


def _resolve_nyiso_price_path(base_path: Path) -> Path:
    """Resolve the base config's actual price input exactly as the data layer does."""

    resolved_base_path = base_path.expanduser().resolve()
    config = BASE.SHARED._load_yaml(resolved_base_path)
    base_dir = resolved_base_path.parent
    configured_root = Path(config.get("workspace_root", base_dir)).expanduser()
    workspace_root = (
        configured_root.resolve()
        if configured_root.is_absolute()
        else (base_dir / configured_root).resolve()
    )
    data = config.get("data")
    if not isinstance(data, dict) or not data.get("nyiso_prices"):
        raise ValueError("Migration-cap base config must declare data.nyiso_prices")
    configured_price = Path(str(data["nyiso_prices"])).expanduser()
    price_path = (
        configured_price.resolve()
        if configured_price.is_absolute()
        else (workspace_root / configured_price).resolve()
    )
    if not price_path.is_file():
        raise FileNotFoundError(f"NYISO price input is missing: {price_path}")
    return price_path


def _merge_required_hashes(
    *hash_maps: tuple[str, dict[str, str]],
) -> dict[str, str]:
    """Normalize and merge launch hashes without accepting conflicting digests."""

    merged: dict[str, str] = {}
    for label, values in hash_maps:
        if not isinstance(values, dict):
            raise TypeError(f"{label} hashes must be a mapping")
        for raw_path, raw_digest in values.items():
            path = str(Path(raw_path).expanduser().resolve())
            digest = str(raw_digest).lower()
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"Invalid {label} SHA-256 for {path}: {raw_digest}")
            previous = merged.get(path)
            if previous is not None and previous != digest:
                raise ValueError(
                    f"Conflicting required launch hashes for {path}: "
                    f"{previous} != {digest}"
                )
            merged[path] = digest
    return dict(sorted(merged.items()))


def _execute_candidates_with_runtime_hashes(
    rows: list[dict[str, Any]],
    *,
    policy: dict[str, Any],
    suite_root: Path,
    required_source_hashes: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Recheck source, price, dataset, and derived configs before every child."""

    metadata_path = suite_root / "suite_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Migration-cap suite metadata is missing before execution: {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Suite metadata root must be an object: {metadata_path}")
    dataset_hashes = metadata.get("dataset_files_sha256")
    if not isinstance(dataset_hashes, dict) or not dataset_hashes:
        raise ValueError("Suite metadata must contain nonempty dataset_files_sha256")
    if int(metadata.get("candidate_count", -1)) != len(rows):
        raise ValueError(
            "Suite metadata candidate count does not match execution rows: "
            f"metadata={metadata.get('candidate_count')}, rows={len(rows)}"
        )

    config_hashes: dict[str, str] = {}
    for row in rows:
        config_path = str(Path(str(row["config_path"])).expanduser().resolve())
        digest = str(row["config_sha256"])
        previous = config_hashes.get(config_path)
        if previous is not None and previous != digest:
            raise ValueError(f"Conflicting config hashes for {config_path}")
        config_hashes[config_path] = digest
    if len(config_hashes) != len(rows):
        raise ValueError("Every migration-cap candidate must have a unique config path")

    combined_hashes = _merge_required_hashes(
        ("source/runtime", required_source_hashes or {}),
        ("dataset", dataset_hashes),
        ("derived config", config_hashes),
    )
    metadata["required_launch_files_sha256"] = combined_hashes
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return _ORIGINAL_EXECUTE_CANDIDATES(
        rows,
        policy=policy,
        suite_root=suite_root,
        required_source_hashes=combined_hashes,
    )


def _patch_shared_executor() -> None:
    BASE.SHARED._execute_candidates = _execute_candidates_with_runtime_hashes


_patch_shared_executor()


def main(argv: list[str] | None = None) -> int:
    _patch_single_run_entrypoint()
    _patch_shared_executor()
    BASE.DEFAULT_PLAN = DEFAULT_PLAN
    BASE._validate_contract = _validate_contract
    BASE._capture_source_hashes = _capture_source_hashes
    return BASE.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
