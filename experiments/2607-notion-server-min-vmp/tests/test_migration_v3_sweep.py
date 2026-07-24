from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = EXPERIMENT_ROOT / "run_micro_stress_migration_v3_sweep.py"
PLAN_PATH = EXPERIMENT_ROOT / "plans/micro_stress_first_stage_migration_v3_3h.yaml"


def _load_launcher():
    spec = importlib.util.spec_from_file_location(
        "notion_server_min_migration_v3_sweep_test", LAUNCHER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_migration_v3_plan_pins_spec_and_exact_three_hour_policy() -> None:
    launcher = _load_launcher()
    plan = yaml.safe_load(PLAN_PATH.read_text(encoding="utf-8"))
    base_path = EXPERIMENT_ROOT / plan["base_config"]
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    formulation = launcher.BASE._resolve_formulation(plan)
    candidates = launcher.BASE.SHARED._build_candidate_specs(
        base, plan["stages"]["first_stage"]
    )

    launcher._validate_contract(base, plan)
    hashes = launcher._capture_source_hashes(
        plan_path=PLAN_PATH,
        base_path=base_path,
        formulation=formulation,
    )

    assert len(candidates) == 11
    assert formulation is not None
    assert formulation["spec_verified"] is False
    assert plan["formulation"]["spec_path"] is None
    assert formulation["spec_sha256"] == (
        "bf2b4dc77d9fd3cf26b6d7f3314113454103bd50767e9f72b9f4dc3d3c2e26d7"
    )
    assert plan["status"] == "executed_target_gap_not_reached"
    assert plan["execution_result"] == {
        "completed_launches": 11,
        "original_launches": 5,
        "retry_launches": 6,
        "solver_status": "TIME_LIMIT",
        "target_gap_reached": 0,
    }
    assert plan["execution_policy"] == {
        "max_parallel_jobs": 5,
        "threads_per_job": 16,
        "total_thread_limit": 80,
        "time_limit_seconds": 10800,
        "mip_gap": 0.001,
        "soft_mem_limit_gb_per_job": 80,
        "memory_limit_enforcement": "gurobi_softmemlimit_plus_aggregate_preflight",
        "nodefile_start_gb": 0.5,
        "safety": {
            "memory_reserve_gb": 64,
            "disk_reserve_gb": 60,
            "write_lp": False,
            "write_mps": False,
            "require_disjoint_cpu_affinity": True,
        },
    }
    assert str(LAUNCHER_PATH.resolve()) in hashes
    assert str((EXPERIMENT_ROOT / "run_micro_stress_sweep.py").resolve()) in hashes
    assert formulation["change_summary_path"] in hashes


def test_migration_v3_launcher_rejects_non_three_hour_limit() -> None:
    launcher = _load_launcher()
    plan = yaml.safe_load(PLAN_PATH.read_text(encoding="utf-8"))
    base = yaml.safe_load(
        (EXPERIMENT_ROOT / plan["base_config"]).read_text(encoding="utf-8")
    )
    plan["execution_policy"]["time_limit_seconds"] = 3600

    with pytest.raises(ValueError, match="10800"):
        launcher._validate_contract(base, plan)
