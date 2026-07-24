#!/usr/bin/env python3
"""Run the migration-v3 first-stage OFAT suite with a three-hour limit.

The existing one-hour launcher is retained as historical provenance.  This
small versioned adapter reuses its data, hashing, materialization, resource
preflight, and scheduling logic while enforcing the new run's 10,800-second
contract and adding itself to the immutable source manifest.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parent
BASE_LAUNCHER_PATH = EXPERIMENT_ROOT / "run_micro_stress_sweep.py"
DEFAULT_PLAN = (
    EXPERIMENT_ROOT / "plans" / "micro_stress_first_stage_migration_v3_3h.yaml"
)


def _load_base_launcher() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "notion_server_min_migration_v3_base_launcher", BASE_LAUNCHER_PATH
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


def _validate_contract(config: dict[str, Any], plan: dict[str, Any]) -> None:
    """Apply the established micro contract with exactly a three-hour limit."""

    actual_time_limit = float(plan["execution_policy"]["time_limit_seconds"])
    if actual_time_limit != 10800.0:
        raise ValueError(
            "Migration-v3 micro-stress solver time limit must be 10800 seconds"
        )
    compatibility_plan = copy.deepcopy(plan)
    compatibility_plan["execution_policy"]["time_limit_seconds"] = 3600
    _ORIGINAL_VALIDATE_CONTRACT(config, compatibility_plan)


def _capture_source_hashes(
    *,
    plan_path: Path,
    base_path: Path,
    formulation: dict[str, str] | None,
) -> dict[str, str]:
    """Retain the base manifest and pin this versioned adapter as well."""

    hashes = _ORIGINAL_CAPTURE_SOURCE_HASHES(
        plan_path=plan_path,
        base_path=base_path,
        formulation=formulation,
    )
    launcher_path = Path(__file__).resolve()
    hashes[str(launcher_path)] = BASE.SHARED._sha256(launcher_path)
    return dict(sorted(hashes.items()))


def main(argv: list[str] | None = None) -> int:
    BASE.DEFAULT_PLAN = DEFAULT_PLAN
    BASE._validate_contract = _validate_contract
    BASE._capture_source_hashes = _capture_source_hashes
    return BASE.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
