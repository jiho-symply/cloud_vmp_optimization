from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml

from conftest import EXPERIMENT_ROOT


def _load_runner():
    path = EXPERIMENT_ROOT / "run_sweep.py"
    spec = importlib.util.spec_from_file_location("notion_server_min_sweep_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_first_stage_runner_generates_exact_deduplicated_ofat_candidates() -> None:
    runner = _load_runner()
    base = yaml.safe_load((EXPERIMENT_ROOT / "configs/baseline.yaml").read_text(encoding="utf-8"))
    plan = yaml.safe_load((EXPERIMENT_ROOT / "plans/sweep_plan.yaml").read_text(encoding="utf-8"))

    candidates = runner._build_candidate_specs(base, plan["stages"]["first_stage"])

    assert [candidate["run_id"] for candidate in candidates] == [
        "baseline",
        "spot_discount_ratio/gamma_0",
        "spot_discount_ratio/gamma_0_1",
        "spot_discount_ratio/gamma_0_5",
        "migration_coefficient/c_mig_0",
        "migration_coefficient/c_mig_0_1",
        "migration_coefficient/c_mig_1",
        "batch_startup_cpu/startup_cpu_0",
        "batch_startup_cpu/startup_cpu_0_1",
        "batch_base_mem/base_mem_0",
        "batch_base_mem/base_mem_0_1",
    ]


def test_first_stage_execution_policy_honors_requested_gap_and_thread_cap() -> None:
    plan = yaml.safe_load((EXPERIMENT_ROOT / "plans/sweep_plan.yaml").read_text(encoding="utf-8"))
    policy = plan["execution_policy"]

    assert policy["mip_gap"] == 0.001
    assert policy["max_parallel_jobs"] == 3
    assert policy["threads_per_job"] == 29
    assert policy["max_parallel_jobs"] * policy["threads_per_job"] == 87
    assert policy["total_thread_limit"] == 88


def test_time_limited_incumbent_is_not_silently_called_target_gap_complete() -> None:
    runner = _load_runner()

    assert runner._completion_state(
        returncode=0,
        summary_exists=True,
        solver_status="TIME_LIMIT",
        achieved_gap=0.02,
        target_gap=0.001,
    ) == "TARGET_GAP_NOT_REACHED"
    assert runner._completion_state(
        returncode=0,
        summary_exists=True,
        solver_status="OPTIMAL",
        achieved_gap=0.0009,
        target_gap=0.001,
    ) == "COMPLETED_TARGET_GAP"


def test_execute_candidates_rechecks_captured_source_before_child_launch(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    captured_source = tmp_path / "model.py"
    captured_source.write_text("before\n", encoding="utf-8")
    expected_hashes = {str(captured_source): runner._sha256(captured_source)}
    captured_source.write_text("after\n", encoding="utf-8")
    suite_root = tmp_path / "suite"
    suite_root.mkdir()

    results = runner._execute_candidates(
        [
            {
                "run_id": "baseline",
                "run_dir": str(suite_root / "first_stage" / "baseline"),
                "config_path": str(tmp_path / "unused.yaml"),
                "mip_gap": 0.001,
            }
        ],
        policy={"max_parallel_jobs": 1, "threads_per_job": 1},
        suite_root=suite_root,
        required_source_hashes=expected_hashes,
    )

    assert results[0]["state"] == "LAUNCH_ERROR"
    assert "Captured source hash changed before child launch" in results[0]["error"]
    assert not (suite_root / "first_stage" / "baseline" / "child.pid").exists()
