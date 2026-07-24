from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_aggregator_module():
    path = Path(__file__).resolve().parents[1] / "scripts/aggregate_experiment_results.py"
    spec = importlib.util.spec_from_file_location("aggregate_experiment_results", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_latest_suite_states_keeps_last_event(tmp_path: Path) -> None:
    module = _load_aggregator_module()
    status = tmp_path / "suite_status.jsonl"
    status.write_text(
        '\n'.join(
            [
                '{"run_id": "run-a", "state": "RUNNING"}',
                '{"run_id": "run-b", "state": "RUNNING"}',
                '{"run_id": "run-a", "state": "COMPLETED"}',
            ]
        )
        + '\n',
        encoding="utf-8",
    )

    latest = module._latest_suite_states(status)

    assert latest["run-a"]["state"] == "COMPLETED"
    assert latest["run-b"]["state"] == "RUNNING"


def test_missing_summary_separates_incomplete_from_failure() -> None:
    module = _load_aggregator_module()

    assert module._missing_summary_bucket("PENDING") == "incomplete"
    assert module._missing_summary_bucket("RUNNING") == "incomplete"
    assert module._missing_summary_bucket("FAILED") == "failure"
    assert module._missing_summary_bucket("LAUNCH_ERROR") == "failure"
    assert module._missing_summary_bucket("COMPLETED") == "failure"


def test_lhs_main_effects_recover_standardized_linear_direction() -> None:
    module = _load_aggregator_module()
    rng = np.random.default_rng(7)
    size = 96
    values = {column: rng.normal(size=size) for column in module.LHS_FEATURES.values()}
    objective = 3.0 * values["config.solar_capacity_multiplier"] - 2.0 * values["config.kappa_sla"]
    frame = pd.DataFrame(
        {
            **values,
            "result.solver.objective": objective,
            "result.service.expected_od_cpu_excess": rng.normal(size=size),
            "result.service.max_time_empirical_cvar": rng.normal(size=size),
            "result.operations.expected_actual_migrations": rng.normal(size=size),
            "analysis.expected_grid_kwh": rng.normal(size=size),
            "result.energy.renewable_coverage_ratio": rng.normal(size=size),
        }
    )

    _, main_effects, interactions = module._lhs_effect_tables(frame)
    objective_effects = main_effects.loc[main_effects["target"] == "objective"].set_index("feature")

    assert objective_effects.loc["solar", "standardized_coefficient"] > 0.7
    assert objective_effects.loc["kappa_sla", "standardized_coefficient"] < -0.4
    assert objective_effects["model_r2"].iloc[0] > 0.99
    assert len(interactions.loc[interactions["target"] == "objective"]) == 28
