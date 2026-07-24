from __future__ import annotations

import copy
import json
from typing import Any

import yaml

from conftest import EXPERIMENT_ROOT


def _read_yaml(relative_path: str) -> dict:
    return yaml.safe_load((EXPERIMENT_ROOT / relative_path).read_text(encoding="utf-8"))


def _factor_values(stage: dict, factor_id: str) -> list[Any]:
    factor = next(item for item in stage["factors"] if item["id"] == factor_id)
    return [level["value"] for level in factor["levels"]]


def _set_config_path(config: dict, config_path: str, value: Any) -> None:
    keys = config_path.split(".")
    target = config
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = copy.deepcopy(value)


def _stage_configs(baseline: dict, stage: dict) -> set[str]:
    configs: set[str] = set()
    for factor in stage["factors"]:
        for level in factor["levels"]:
            config = copy.deepcopy(baseline)
            _set_config_path(config, factor["config_path"], level["value"])
            configs.add(json.dumps(config, sort_keys=True))
    return configs


def test_latest_page_baseline_is_materialized() -> None:
    config = _read_yaml("configs/baseline.yaml")

    assert config["experiment"]["num_scenarios"] == 20
    assert config["experiment"]["class_counts"] == {
        "on_demand": 50,
        "spot": 50,
        "batch_jobs": 50,
    }
    assert config["experiment"]["num_servers"] == 6
    assert config["batch"]["max_families"] == 10
    assert config["migration"] == {"coefficient": 0.05}
    assert "minimum_5min_coverage_ratio" not in config["data"]


def test_sweep_uses_requested_two_stage_ofat_ranges() -> None:
    plan = _read_yaml("plans/sweep_plan.yaml")
    first_stage = plan["stages"]["first_stage"]
    second_stage = plan["stages"]["second_stage"]

    assert plan["baseline"]["experiment"]["num_scenarios"] == 20
    assert plan["baseline"]["experiment"]["class_counts"] == {
        "on_demand": 50,
        "spot": 50,
        "batch_jobs": 50,
    }
    assert plan["baseline"]["experiment"]["num_servers"] == 6
    assert plan["baseline"]["batch"]["max_families"] == 10
    assert (
        plan["baseline"]["batch"]["family_interpretation"]
        == "deterministic_configured_pair_rank_bins"
    )
    assert plan["validation"]["require_deterministic_batch_family_binning"] is True
    assert "require_exact_batch_families" not in plan["validation"]
    assert plan["generation"]["forbid_full_cartesian_product"] is True
    assert plan["generation"]["combine_factors_within_stage"] is False
    assert plan["generation"]["combine_factors_across_stages"] is False
    assert first_stage["design"] == "one_factor_at_a_time"
    assert second_stage["design"] == "one_factor_at_a_time"

    assert _factor_values(first_stage, "spot_discount_ratio") == [0.0, 0.1, 0.3, 0.5]
    assert _factor_values(first_stage, "migration_coefficient") == [0.0, 0.05, 0.1, 1.0]
    assert _factor_values(first_stage, "batch_startup_cpu") == [0.0, 0.05, 0.1]
    assert _factor_values(first_stage, "batch_base_mem") == [0.0, 0.05, 0.1]

    assert _factor_values(second_stage, "num_servers") == [3, 4, 5, 6, 7, 8]
    assert _factor_values(second_stage, "workload") == [
        {"on_demand": 150, "spot": 0, "batch_jobs": 0},
        {"on_demand": 75, "spot": 75, "batch_jobs": 0},
        {"on_demand": 50, "spot": 50, "batch_jobs": 50},
    ]
    assert _factor_values(second_stage, "risk_alpha") == [0.5, 0.8, 0.9, 0.95]
    assert _factor_values(second_stage, "risk_epsilon") == [0.5, 0.1, 0.05, 0.0]


def test_sweep_unique_candidate_count_is_exact_after_global_deduplication() -> None:
    plan = _read_yaml("plans/sweep_plan.yaml")
    baseline = plan["baseline"]

    first_configs = _stage_configs(baseline, plan["stages"]["first_stage"])
    second_configs = _stage_configs(baseline, plan["stages"]["second_stage"])
    counts = plan["generation"]["exact_unique_candidate_configs"]

    # Each stage shares one anchor config across all of its OFAT factors, and
    # both stages share the same global baseline anchor.
    assert len(first_configs) == counts["first_stage"] == 11
    assert len(second_configs) == counts["second_stage"] == 14
    assert len(first_configs | second_configs) == counts["total"] == 24

    assert (
        plan["generation"]["solver_launch_count"]
        == "determined_by_structural_feasibility_precheck"
    )
    assert plan["generation"]["require_structural_feasibility_precheck"] is True
    assert (
        plan["execution_policy"]["skip_structurally_infeasible_candidates_before_solver_launch"]
        is True
    )


def test_removed_sweep_policies_do_not_reappear() -> None:
    plan = _read_yaml("plans/sweep_plan.yaml")
    serialized = yaml.safe_dump(plan, sort_keys=True)

    assert "minimum_5min_coverage_ratio" not in serialized
    assert "coverage_audit" not in serialized
    assert "batch_discount_sensitivity" not in serialized
    assert "paired_seed_confirmation" not in serialized
    assert "conditional_escalation" not in serialized
    assert "adaptive" not in serialized
