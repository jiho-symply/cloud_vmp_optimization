#!/usr/bin/env python3
"""Run the exact six-case cumulative fixed-zero micro-stress suite.

The adapter keeps the shared micro-stress data, materialization, preflight, and
scheduling implementation while selecting the v6 single-run entry point.  It
requires migration, OD excess-branch, and Spot-preemption decisions all to be
fixed to zero, enforces six concurrent 16-thread solves, and rechecks every
captured source, declared data input, dataset file, and derived config before
each candidate subprocess is created.
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
SINGLE_RUN_ENTRYPOINT = (
    EXPERIMENT_ROOT
    / "run_experiment_fixed_zero_migration_excess_preemption.py"
)
V5_FIXED_ZERO_ENTRYPOINT = (
    EXPERIMENT_ROOT / "run_experiment_fixed_zero_migration_excess.py"
)
DEFAULT_CONFIG = (
    EXPERIMENT_ROOT
    / "configs"
    / "micro_stress_fixed_zero_migration_excess_preemption.yaml"
)
DEFAULT_PLAN = (
    EXPERIMENT_ROOT
    / "plans"
    / "micro_stress_first_six_fixed_zero_migration_excess_preemption_3h.yaml"
)
CHANGE_SUMMARY = (
    EXPERIMENT_ROOT
    / "FORMULATION_CHANGES_FIXED_ZERO_MIGRATION_EXCESS_PREEMPTION_V6.md"
)
FORMULATION_ID = (
    "notion_server_min_fixed_zero_migration_excess_preemption_v6_20260725"
)
SPEC_SHA256 = "bf2b4dc77d9fd3cf26b6d7f3314113454103bd50767e9f72b9f4dc3d3c2e26d7"
EXPECTED_RUN_IDS = (
    "baseline",
    "spot_discount_ratio/gamma_0",
    "spot_discount_ratio/gamma_0_1",
    "spot_discount_ratio/gamma_0_5",
    "migration_coefficient/c_mig_0",
    "migration_coefficient/c_mig_0_1",
)
EXPECTED_FACTORS = [
    {
        "id": "spot_discount_ratio",
        "config_path": "economics.spot_discount_ratio",
        "levels": [
            {"id": "gamma_0", "value": 0.0},
            {"id": "gamma_0_1", "value": 0.1},
            {"id": "gamma_0_3", "value": 0.3},
            {"id": "gamma_0_5", "value": 0.5},
        ],
    },
    {
        "id": "migration_coefficient",
        "config_path": "migration.coefficient",
        "levels": [
            {"id": "c_mig_0", "value": 0.0},
            {"id": "c_mig_0_05", "value": 0.05},
            {"id": "c_mig_0_1", "value": 0.1},
        ],
    },
]
FIXED_ZERO_GUARDS = (
    ("migration", "migration.fixed_zero"),
    ("excess_load_indicator", "excess_load_indicator.fixed_zero"),
    ("spot_preemption", "spot_preemption.fixed_zero"),
)


def _load_base_launcher() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "notion_server_min_cumulative_fixed_zero_base_launcher",
        BASE_LAUNCHER_PATH,
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
    """Keep the base launcher check and child scheduler in agreement."""

    BASE.SINGLE_RUN_ENTRYPOINT = SINGLE_RUN_ENTRYPOINT
    BASE.SHARED.SINGLE_RUN_ENTRYPOINT = SINGLE_RUN_ENTRYPOINT


_patch_single_run_entrypoint()


def _require_literal_true(
    mapping: object,
    *,
    key: str,
    dotted_path: str,
) -> None:
    if not isinstance(mapping, dict) or mapping.get(key) is not True:
        raise ValueError(f"{dotted_path} must be the literal YAML boolean true")


def _validate_fixed_zero_guards(config: dict[str, Any]) -> None:
    for section, dotted_path in FIXED_ZERO_GUARDS:
        _require_literal_true(
            config.get(section),
            key="fixed_zero",
            dotted_path=dotted_path,
        )


def _validate_contract(config: dict[str, Any], plan: dict[str, Any]) -> None:
    """Validate all three guards and the immutable six-case launch contract."""

    _validate_fixed_zero_guards(config)
    migration = config.get("migration")
    coefficient = migration.get("coefficient") if isinstance(migration, dict) else None
    if (
        isinstance(coefficient, bool)
        or not isinstance(coefficient, (int, float))
        or float(coefficient) != 0.05
    ):
        raise ValueError(
            "Cumulative fixed-zero base config migration.coefficient must equal 0.05"
        )

    solver = config.get("solver")
    if not isinstance(solver, dict):
        raise ValueError("Cumulative fixed-zero config must declare a solver mapping")
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
            "Cumulative fixed-zero base solver contract drifted: "
            f"expected={expected_solver}, actual={actual_solver}"
        )

    if plan.get("schema_version") != 6:
        raise ValueError("Cumulative fixed-zero plan schema_version must equal 6")
    if plan.get("status") != "immutable_launch_contract":
        raise ValueError(
            "Cumulative fixed-zero plan status must be immutable_launch_contract"
        )
    if plan.get("suite_name") != (
        "notion_server_min_micro_stress_first_six_"
        "fixed_zero_migration_excess_preemption_3h"
    ):
        raise ValueError("Cumulative fixed-zero suite_name drifted")
    if plan.get("base_config") != (
        "configs/micro_stress_fixed_zero_migration_excess_preemption.yaml"
    ):
        raise ValueError("Cumulative fixed-zero plan base_config drifted")
    if plan.get("output_root") != (
        "runs/notion_server_min_micro_stress_first_six_"
        "fixed_zero_migration_excess_preemption_3h"
    ):
        raise ValueError("Cumulative fixed-zero plan output_root drifted")

    formulation = plan.get("formulation")
    expected_formulation = {
        "id": FORMULATION_ID,
        "spec_path": None,
        "spec_sha256": SPEC_SHA256,
        "change_summary_path": CHANGE_SUMMARY.name,
    }
    if formulation != expected_formulation:
        raise ValueError(
            "Cumulative fixed-zero formulation contract drifted: "
            f"expected={expected_formulation}, actual={formulation}"
        )

    generation = plan.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("Cumulative fixed-zero plan must declare generation settings")
    expected_generation = {
        "design": "one_factor_at_a_time",
        "semantics": "copy_baseline_then_override_exactly_one_factor",
        "forbid_full_cartesian_product": True,
        "combine_factors_within_stage": False,
        "deduplicate_identical_configs_globally": True,
        "exact_unique_candidate_configs": {"first_stage": 6},
        "immutable_configs": True,
        "require_common_random_numbers_within_comparison_group": True,
    }
    if generation != expected_generation:
        raise ValueError("Cumulative fixed-zero generation contract drifted")

    stages = plan.get("stages")
    stage = stages.get("first_stage") if isinstance(stages, dict) else None
    if not isinstance(stage, dict) or stage.get("enabled") is not True:
        raise ValueError("Cumulative fixed-zero first stage must be enabled")
    if stage.get("design") != "one_factor_at_a_time":
        raise ValueError("Cumulative fixed-zero first-stage design drifted")
    if stage.get("anchor") != "baseline":
        raise ValueError("Cumulative fixed-zero first-stage anchor drifted")
    if stage.get("factors") != EXPECTED_FACTORS:
        raise ValueError("Cumulative fixed-zero six-case factor contract drifted")

    candidates = BASE.SHARED._build_candidate_specs(config, stage)
    actual_run_ids = tuple(candidate["run_id"] for candidate in candidates)
    if actual_run_ids != EXPECTED_RUN_IDS:
        raise ValueError(
            "Cumulative fixed-zero candidate contract drifted: "
            f"expected={EXPECTED_RUN_IDS}, actual={actual_run_ids}"
        )
    for candidate in candidates:
        _validate_fixed_zero_guards(candidate["config"])

    policy = plan.get("execution_policy")
    if not isinstance(policy, dict):
        raise ValueError("Cumulative fixed-zero plan must declare execution_policy")
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
            "Cumulative fixed-zero execution policy drifted: "
            f"expected={expected_policy}, actual={actual_policy}"
        )
    if policy.get("memory_limit_enforcement") != (
        "gurobi_softmemlimit_plus_aggregate_preflight"
    ):
        raise ValueError(
            "Cumulative fixed-zero memory-limit enforcement policy drifted"
        )

    safety = policy.get("safety")
    if not isinstance(safety, dict):
        raise ValueError(
            "Cumulative fixed-zero execution policy must declare safety limits"
        )
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
            "Cumulative fixed-zero safety policy drifted: "
            f"expected={expected_safety}, actual={actual_safety}"
        )
    if safety["write_lp"] is not False or safety["write_mps"] is not False:
        raise ValueError("Cumulative fixed-zero LP/MPS switches must be literal false")
    if safety["require_disjoint_cpu_affinity"] is not True:
        raise ValueError(
            "Cumulative fixed-zero CPU affinity guard must be literal true"
        )

    validation = plan.get("validation")
    if not isinstance(validation, dict):
        raise ValueError("Cumulative fixed-zero plan must declare validation settings")
    expected_validation = {
        "migration_fixed_zero_config_path": "migration.fixed_zero",
        "excess_load_indicator_fixed_zero_config_path": (
            "excess_load_indicator.fixed_zero"
        ),
        "spot_preemption_fixed_zero_config_path": "spot_preemption.fixed_zero",
        "require_literal_boolean_true": True,
        "exact_candidate_run_ids": list(EXPECTED_RUN_IDS),
    }
    actual_validation = {
        key: validation.get(key) for key in expected_validation
    }
    if actual_validation != expected_validation:
        raise ValueError(
            "Cumulative fixed-zero validation policy drifted: "
            f"expected={expected_validation}, actual={actual_validation}"
        )
    if validation["require_literal_boolean_true"] is not True:
        raise ValueError(
            "Cumulative fixed-zero literal-boolean validation guard drifted"
        )

    # Reuse the established dimension validation without mutating the real
    # six-worker plan consumed by materialization, preflight, and execution.
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


def _resolved_workspace_and_data(base_path: Path) -> tuple[Path, dict[str, Any]]:
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
    if not isinstance(data, dict):
        raise ValueError("Cumulative fixed-zero base config must declare data")
    return workspace_root, data


def _resolve_config_data_path(workspace_root: Path, value: object) -> Path:
    configured = Path(str(value)).expanduser()
    return (
        configured.resolve()
        if configured.is_absolute()
        else (workspace_root / configured).resolve()
    )


def _resolve_declared_data_inputs(base_path: Path) -> set[Path]:
    """Resolve the NYISO input and every declared lineage source."""

    workspace_root, data = _resolved_workspace_and_data(base_path)
    if not data.get("nyiso_prices"):
        raise ValueError("Cumulative fixed-zero config must declare data.nyiso_prices")
    raw_values = [data["nyiso_prices"], *data.get("additional_source_files", [])]
    paths = {
        _resolve_config_data_path(workspace_root, value) for value in raw_values
    }
    missing = [str(path) for path in sorted(paths) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot capture cumulative fixed-zero data-input hashes; "
            f"missing={missing}"
        )
    return paths


def _resolve_nyiso_price_path(base_path: Path) -> Path:
    workspace_root, data = _resolved_workspace_and_data(base_path)
    if not data.get("nyiso_prices"):
        raise ValueError("Cumulative fixed-zero config must declare data.nyiso_prices")
    price_path = _resolve_config_data_path(workspace_root, data["nyiso_prices"])
    if not price_path.is_file():
        raise FileNotFoundError(f"NYISO price input is missing: {price_path}")
    return price_path


def _capture_source_hashes(
    *,
    plan_path: Path,
    base_path: Path,
    formulation: dict[str, Any] | None,
) -> dict[str, str]:
    """Pin shared/v6 sources, plan/config, and all declared static inputs."""

    _patch_single_run_entrypoint()
    hashes = _ORIGINAL_CAPTURE_SOURCE_HASHES(
        plan_path=plan_path,
        base_path=base_path,
        formulation=formulation,
    )
    required_paths = {
        Path(__file__).resolve(),
        SINGLE_RUN_ENTRYPOINT.resolve(),
        V5_FIXED_ZERO_ENTRYPOINT.resolve(),
        DEFAULT_CONFIG.resolve(),
        DEFAULT_PLAN.resolve(),
        CHANGE_SUMMARY.resolve(),
        *_resolve_declared_data_inputs(base_path),
    }
    missing = [str(path) for path in sorted(required_paths) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot capture cumulative fixed-zero source hashes; "
            f"missing={missing}"
        )
    for path in required_paths:
        hashes[str(path)] = BASE.SHARED._sha256(path)
    return dict(sorted(hashes.items()))


def _merge_required_hashes(
    *hash_maps: tuple[str, dict[str, str]],
) -> dict[str, str]:
    """Normalize and merge launch hashes without accepting conflicts."""

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


def _validate_runtime_rows(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Verify all six derived configs retain the three fixed-zero guards."""

    actual_run_ids = tuple(str(row.get("run_id")) for row in rows)
    if actual_run_ids != EXPECTED_RUN_IDS:
        raise ValueError(
            "Cumulative fixed-zero runtime candidate order drifted: "
            f"expected={EXPECTED_RUN_IDS}, actual={actual_run_ids}"
        )

    config_hashes: dict[str, str] = {}
    for row in rows:
        config_path = Path(str(row["config_path"])).expanduser().resolve()
        digest = str(row["config_sha256"])
        path_string = str(config_path)
        if path_string in config_hashes:
            raise ValueError(
                "Every cumulative fixed-zero candidate must have a unique "
                f"config path: {config_path}"
            )
        candidate_config = BASE.SHARED._load_yaml(config_path)
        _validate_fixed_zero_guards(candidate_config)
        solver = candidate_config.get("solver")
        if not isinstance(solver, dict):
            raise ValueError(f"Derived config has no solver mapping: {config_path}")
        actual_solver = (
            int(solver["threads"]),
            float(solver["time_limit_seconds"]),
            float(solver["mip_gap"]),
            float(solver["soft_mem_limit_gb"]),
            float(solver["nodefile_start_gb"]),
        )
        if actual_solver != (16, 10800.0, 0.001, 64.0, 0.5):
            raise ValueError(
                "Derived cumulative fixed-zero solver contract drifted at "
                f"{config_path}: {actual_solver}"
            )
        config_hashes[path_string] = digest
    return config_hashes


def _execute_candidates_with_runtime_hashes(
    rows: list[dict[str, Any]],
    *,
    policy: dict[str, Any],
    suite_root: Path,
    required_source_hashes: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Recheck source, static input, dataset, and six configs per child."""

    metadata_path = suite_root / "suite_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            "Cumulative fixed-zero suite metadata is missing before execution: "
            f"{metadata_path}"
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

    config_hashes = _validate_runtime_rows(rows)
    combined_hashes = _merge_required_hashes(
        ("source/static input", required_source_hashes or {}),
        ("dataset", dataset_hashes),
        ("derived config", config_hashes),
    )
    metadata["required_launch_files_sha256"] = combined_hashes
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # The shared executor recomputes this complete map immediately before each
    # subprocess.Popen, including candidates that ever wait for a worker slot.
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
