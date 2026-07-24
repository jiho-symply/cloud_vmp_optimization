from __future__ import annotations

import json

import pandas as pd
import pytest

from notion_server_min_vmp.reporting import write_solution_reports

from conftest import build_tiny_model, make_instance, optimize_tiny


def test_reporting_reconstructs_objective_and_audits_server_time_risk(
    solved_full_artifacts, tmp_path
) -> None:
    summary = write_solution_reports(solved_full_artifacts, tmp_path)

    components = summary["objective_components"]
    assert components["constant_on_demand_revenue"] == pytest.approx(
        solved_full_artifacts.objective_constants["on_demand_revenue"]
    )
    assert components["constant_batch_revenue"] == pytest.approx(
        solved_full_artifacts.objective_constants["batch_revenue"]
    )
    assert components["objective_reconstruction_error"] == pytest.approx(0.0, abs=1e-8)
    constant_offset = sum(solved_full_artifacts.objective_constants.values())
    assert components["reconstructed_objective"] == pytest.approx(
        solved_full_artifacts.model.ObjVal
    )
    assert components["reconstructed_total_profit"] == pytest.approx(
        constant_offset + solved_full_artifacts.model.ObjVal
    )
    assert summary["solver"]["objective"] == pytest.approx(
        solved_full_artifacts.model.ObjVal
    )
    assert summary["solver"]["best_bound"] == pytest.approx(
        solved_full_artifacts.model.ObjBound
    )
    assert summary["solver"]["mip_gap"] == pytest.approx(
        solved_full_artifacts.model.MIPGap
    )
    assert summary["solver"]["objective_excludes_constant_revenue"] is True
    assert summary["solver"]["constant_revenue_offset"] == pytest.approx(
        constant_offset
    )
    assert summary["solver"]["total_profit_including_constant_revenue"] == pytest.approx(
        constant_offset + solved_full_artifacts.model.ObjVal
    )
    assert summary["solver"]["best_bound_including_constant_revenue"] == pytest.approx(
        constant_offset + solved_full_artifacts.model.ObjBound
    )
    assert summary["model_validation_diagnostics"][
        "maximum_exact_positive_part_residual"
    ] == pytest.approx(0.0, abs=1e-8)

    risk = pd.read_csv(tmp_path / "cvar_by_server_time.csv")
    data = solved_full_artifacts.data
    assert len(risk) == len(data.S) * len(data.T)
    assert set(zip(risk["server_id"], risk["t"])) == {
        (server, t) for server in data.S for t in data.T
    }
    assert (risk["empirical_cvar"] <= data.epsilon + 1e-8).all()

    loads = pd.read_csv(tmp_path / "server_scenario_load.csv")
    assert loads["positive_part_residual"].abs().max() <= 1e-8
    energy = pd.read_csv(tmp_path / "energy_rt_cost.csv")
    assert energy["energy_definition_residual"].abs().max() <= 1e-8
    assert set(energy["migration_coefficient"]) == {0.05}
    assert set(energy["migration_coefficient_unit"]) == {
        "kWh per normalized memory unit migrated"
    }
    completion = pd.read_csv(tmp_path / "batch_completion_audit.csv")
    assert completion["completion_residual"].abs().max() <= 1e-8
    scenarios = pd.read_csv(tmp_path / "scenario_metrics.csv")
    assert {
        "profit_usd",
        "rt_energy_cost_usd",
        "energy_kwh",
    }.issubset(scenarios.columns)
    assert {
        "migration_mode",
        "migration_unit_warning",
        "energy_quantity_label",
        "modeled_profit_objective_value",
        "energy_quantity_as_modeled",
    }.isdisjoint(
        scenarios.columns
    )
    probability_weighted_profit = float(
        (scenarios["probability"] * scenarios["profit_usd"]).sum()
    )
    assert probability_weighted_profit == pytest.approx(
        summary["solver"]["total_profit_including_constant_revenue"]
    )
    probability_weighted_operating_margin = float(
        (
            scenarios["probability"]
            * (scenarios["spot_revenue_usd"] - scenarios["rt_energy_cost_usd"])
        ).sum()
    )
    assert probability_weighted_operating_margin == pytest.approx(
        summary["solver"]["objective"]
    )
    assert summary["units_and_fidelity"] == {
        "migration_coefficient": 0.05,
        "migration_coefficient_unit": "kWh per normalized memory unit migrated",
        "energy_quantity_unit": "kWh",
    }
    assert summary["operations"]["available_servers"] == len(data.S)
    assert "actual_server_change_count" in summary["operations"]
    assert "logical_batch_start_count" in summary["operations"]
    assert summary["solver"]["effective_parameters"]["threads"] == 1
    on_disk = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert on_disk["objective_components"]["reconstructed_objective"] == pytest.approx(
        solved_full_artifacts.model.ObjVal
    )


def test_reporting_supports_empty_spot_and_batch_sets(
    positive_part_artifacts, tmp_path
) -> None:
    summary = write_solution_reports(positive_part_artifacts, tmp_path)

    assert summary["service"]["spot_candidate_count"] == 0
    assert pd.read_csv(tmp_path / "spot_schedule.csv").empty
    assert pd.read_csv(tmp_path / "batch_schedule.csv").empty
    assert pd.read_csv(tmp_path / "batch_completion_audit.csv").empty


def test_reporting_aggregates_destination_migration_once_per_transition(
    tmp_path,
) -> None:
    pytest.importorskip("gurobipy")
    artifacts = build_tiny_model(
        make_instance(
            spot_ids=(),
            batch_ids=(),
            periods=(0, 1),
            scenarios=(0,),
        )
    )
    x = artifacts.variables["x"]
    x["od0", "server0", 0, 0].LB = 1.0
    x["od0", "server0", 0, 0].UB = 1.0
    x["od0", "server1", 1, 0].LB = 1.0
    x["od0", "server1", 1, 0].UB = 1.0
    optimize_tiny(artifacts)

    summary = write_solution_reports(artifacts, tmp_path)
    migrations = pd.read_csv(tmp_path / "migrations.csv")
    energy = pd.read_csv(tmp_path / "energy_rt_cost.csv")
    expected_migration_energy = (
        artifacts.data.c_mig
        * artifacts.data.load_od["od0", "MEM", 0, 0]
    )

    assert len(migrations) == 1
    row = migrations.iloc[0]
    assert row["from_server"] == "server0"
    assert row["to_server"] == "server1"
    assert row["destination_server"] == "server1"
    assert row["indicator"] == 1
    assert row["relaxed_value"] == pytest.approx(1.0)
    assert row["actual_change"] == 1
    assert row["definition_residual"] == pytest.approx(0.0)
    assert row["migration_energy_kwh"] == pytest.approx(
        expected_migration_energy
    )
    assert row["rt_price_usd_per_kwh"] == pytest.approx(
        artifacts.data.p_rt[0, 0]
    )
    assert row["migration_rt_cost_usd"] == pytest.approx(
        expected_migration_energy * artifacts.data.p_rt[0, 0]
    )
    diagnostics = summary["model_validation_diagnostics"]
    assert diagnostics["migration_indicator_count"] == 1
    assert diagnostics["actual_server_change_count"] == 1
    assert diagnostics["false_migration_indicator_count"] == 0
    assert diagnostics[
        "maximum_exact_migration_definition_residual"
    ] == pytest.approx(0.0)
    assert energy.loc[energy["t"] == 0, "migration_energy_kwh"].iloc[0] == pytest.approx(
        expected_migration_energy
    )
