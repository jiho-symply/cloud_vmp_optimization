from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import build_tiny_model, make_instance, optimize_tiny


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = (
    EXPERIMENT_ROOT
    / "run_experiment_fixed_zero_migration_excess_preemption.py"
)


def _load_entrypoint():
    spec = importlib.util.spec_from_file_location(
        "fixed_zero_migration_excess_preemption_entrypoint_test",
        ENTRYPOINT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _valid_config() -> dict:
    return {
        "migration": {"fixed_zero": True},
        "excess_load_indicator": {"fixed_zero": True},
        "spot_preemption": {"fixed_zero": True},
    }


def _configure(artifacts) -> None:
    artifacts.model.Params.OutputFlag = 0
    artifacts.model.Params.LogToConsole = 0
    artifacts.model.Params.Threads = 1
    artifacts.model.Params.TimeLimit = 5
    artifacts.model.Params.Seed = 1


def _force_spot_acceptance(artifacts) -> None:
    spot_id = artifacts.data.J[0]
    selected = artifacts.data.S[0]
    for server_id in artifacts.data.S:
        value = float(server_id == selected)
        variable = artifacts.variables["y_init"][spot_id, server_id]
        variable.LB = value
        variable.UB = value


@pytest.mark.parametrize(
    ("section", "invalid"),
    [
        ("migration", False),
        ("migration", 1),
        ("excess_load_indicator", "true"),
        ("excess_load_indicator", None),
        ("spot_preemption", False),
        ("spot_preemption", 1),
        ("spot_preemption", "true"),
        ("spot_preemption", None),
    ],
)
def test_configuration_requires_all_literal_boolean_true(
    section: str,
    invalid: object,
) -> None:
    entrypoint = _load_entrypoint()
    config = _valid_config()
    config[section]["fixed_zero"] = invalid

    with pytest.raises(ValueError, match=rf"{section}\.fixed_zero"):
        entrypoint.require_fixed_zero_migration_excess_preemption_configuration(
            config
        )


@pytest.mark.parametrize(
    "missing_section",
    ["migration", "excess_load_indicator", "spot_preemption"],
)
def test_configuration_rejects_a_missing_section(missing_section: str) -> None:
    entrypoint = _load_entrypoint()
    config = _valid_config()
    del config[missing_section]

    with pytest.raises(ValueError, match=rf"{missing_section}\.fixed_zero"):
        entrypoint.require_fixed_zero_migration_excess_preemption_configuration(
            config
        )


def test_build_model_fixes_all_three_variable_groups() -> None:
    pytest.importorskip("gurobipy")
    entrypoint = _load_entrypoint()
    artifacts = entrypoint.build_model_fixed_zero_migration_excess_preemption(
        make_instance(
            on_demand_ids=("od0",),
            spot_ids=("spot0",),
            batch_ids=(),
            periods=(0, 1, 2),
            scenarios=(0, 1),
            active_od={"od0": (0, 1, 2)},
            active_spot={"spot0": (0, 1, 2)},
        )
    )

    for group in ("migration", "excess_branch", "h"):
        assert artifacts.variables[group]
        assert all(
            variable.LB == pytest.approx(0.0)
            and variable.UB == pytest.approx(0.0)
            for variable in artifacts.variables[group].values()
        )
    audit = artifacts.variables["fixed_zero_bound_audit"]
    assert audit["all_required_bounds_fixed_zero"] is True
    assert audit["semantic_support_verified"] is True
    assert audit["observed_support_counts"] == audit["expected_support_counts"]
    assert audit["spot_preemption"]["fixed_variable_count"] == 6


def test_preemption_possible_in_primary_model_but_infeasible_when_h_is_zero(
    tmp_path: Path,
) -> None:
    gp = pytest.importorskip("gurobipy")
    data = make_instance(
        on_demand_ids=(),
        spot_ids=("spot0",),
        batch_ids=(),
        servers=("server0",),
        periods=(0, 1),
        scenarios=(0,),
        active_spot={"spot0": (0, 1)},
    )

    ordinary = build_tiny_model(data)
    _force_spot_acceptance(ordinary)
    ordinary.model.addConstr(
        ordinary.variables["h"]["spot0", 1, 0] == 1,
        name="force_preemption",
    )
    optimize_tiny(ordinary)
    assert ordinary.model.Status == gp.GRB.OPTIMAL

    entrypoint = _load_entrypoint()
    fixed = entrypoint.build_model_fixed_zero_migration_excess_preemption(data)
    _configure(fixed)
    _force_spot_acceptance(fixed)
    fixed.model.addConstr(
        fixed.variables["h"]["spot0", 1, 0] == 1,
        name="force_forbidden_preemption",
    )
    optimize_tiny(fixed)
    assert fixed.model.Status == gp.GRB.INFEASIBLE

    summary = (
        entrypoint.write_no_solution_summary_fixed_zero_migration_excess_preemption(
            fixed.model,
            tmp_path,
        )
    )
    policy = summary["fixed_zero_policy"]
    assert policy["bound_audit"]["all_required_bounds_fixed_zero"] is True
    assert policy["bound_audit"]["semantic_support_verified"] is True
    assert policy["bound_audit"]["spot_preemption"]["fixed_variable_count"] == 2
    assert policy["solution_audit"]["incumbent_available"] is False
    assert json.loads((tmp_path / "summary.json").read_text()) == summary


def test_accepted_spot_is_served_on_initial_server_in_every_active_slot() -> None:
    gp = pytest.importorskip("gurobipy")
    entrypoint = _load_entrypoint()
    artifacts = entrypoint.build_model_fixed_zero_migration_excess_preemption(
        make_instance(
            on_demand_ids=(),
            spot_ids=("spot0",),
            batch_ids=(),
            servers=("server0", "server1"),
            periods=(0, 1, 2),
            scenarios=(0, 1),
            active_spot={"spot0": (0, 1, 2)},
        )
    )
    _configure(artifacts)
    _force_spot_acceptance(artifacts)
    optimize_tiny(artifacts)
    assert artifacts.model.Status == gp.GRB.OPTIMAL

    for scenario_id in artifacts.data.Xi:
        for t in artifacts.data.active_spot["spot0"]:
            assert artifacts.variables["h"]["spot0", t, scenario_id].X == (
                pytest.approx(0.0)
            )
            assert artifacts.variables["y"][
                "spot0", "server0", t, scenario_id
            ].X == pytest.approx(1.0)
            assert artifacts.variables["y"][
                "spot0", "server1", t, scenario_id
            ].X == pytest.approx(0.0)

    audit = entrypoint._solution_audit(artifacts)
    assert audit["preempted_state_count"] == 0
    assert audit["preemption_event_count"] == 0
    assert audit["arrival_preemption_violation_count"] == 0
    assert audit["preemption_monotonicity_violation_count"] == 0
    assert audit["accepted_deactivation_count"] == 0
    assert audit["rejected_activation_count"] == 0
    assert audit["direct_expected_spot_service_rate"] == pytest.approx(1.0)
    assert audit["maximum_spot_placement_fractionality"] == pytest.approx(0.0)
    assert audit["maximum_spot_admission_fractionality"] == pytest.approx(0.0)
    assert audit["maximum_admitted_sum_fractionality"] == pytest.approx(0.0)
    assert audit["maximum_accepted_spot_service_deficit"] == pytest.approx(0.0)


def test_acceptance_cross_check_does_not_accumulate_per_vm_numeric_noise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entrypoint = _load_entrypoint()
    spot_ids = tuple(f"spot{index}" for index in range(10))
    value = 1.0 - 9e-6
    variable = lambda x: SimpleNamespace(X=x)
    data = SimpleNamespace(
        I=[],
        J=list(spot_ids),
        S=["server0"],
        T=[0],
        Xi=[0],
        p={0: 1.0},
        active_od={},
        active_spot={spot_id: [0] for spot_id in spot_ids},
    )
    artifacts = SimpleNamespace(
        data=data,
        variables={
            "h": {(spot_id, 0, 0): variable(0.0) for spot_id in spot_ids},
            "y": {
                (spot_id, "server0", 0, 0): variable(value)
                for spot_id in spot_ids
            },
            "y_init": {
                (spot_id, "server0"): variable(value)
                for spot_id in spot_ids
            },
            "u": {("server0", 0): variable(1.0)},
        },
    )
    monkeypatch.setattr(
        entrypoint.V5,
        "_solution_audit",
        lambda *_args, **_kwargs: {
            "incumbent_available": True,
            "tolerance": 1e-5,
        },
    )

    audit = entrypoint._solution_audit(artifacts, tolerance=1e-5)

    assert audit["accepted_spot_count"] == pytest.approx(10.0)
    assert audit["raw_spot_admission_sum"] == pytest.approx(10.0 * value)
    assert audit["maximum_spot_admission_fractionality"] == pytest.approx(9e-6)

    monkeypatch.setattr(
        entrypoint.V5,
        "_audit_primary_report",
        lambda *_args, **_kwargs: None,
    )
    summary = {
        "service": {
            "spot_acceptance_count": 10,
            "expected_spot_service_rate": value,
        },
        "model_validation_diagnostics": {
            "spot_arrival_violation_count": 0,
            "spot_reactivation_violation_count": 0,
            "maximum_spot_relaxation_fractionality": 9e-6,
        },
    }
    entrypoint._audit_primary_report(summary, audit, tolerance=1e-5)


def test_solution_writer_publishes_synchronized_cumulative_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gp = pytest.importorskip("gurobipy")
    entrypoint = _load_entrypoint()
    artifacts = entrypoint.build_model_fixed_zero_migration_excess_preemption(
        make_instance(
            on_demand_ids=("od0",),
            spot_ids=("spot0",),
            batch_ids=(),
            servers=("server0",),
            periods=(0, 1),
            scenarios=(0,),
            active_od={"od0": (0, 1)},
            active_spot={"spot0": (0, 1)},
        )
    )
    _configure(artifacts)
    _force_spot_acceptance(artifacts)
    optimize_tiny(artifacts)
    assert artifacts.model.Status == gp.GRB.OPTIMAL

    original_audit = entrypoint._audit_primary_report

    def audit_before_completion_marker(*args: object, **kwargs: object) -> None:
        assert not (tmp_path / "summary.json").exists()
        original_audit(*args, **kwargs)

    monkeypatch.setattr(
        entrypoint,
        "_audit_primary_report",
        audit_before_completion_marker,
    )
    summary = (
        entrypoint.write_solution_reports_fixed_zero_migration_excess_preemption(
            artifacts,
            tmp_path,
        )
    )

    disk_summary = json.loads((tmp_path / "summary.json").read_text())
    disk_policy = json.loads((tmp_path / "fixed_zero_policy.json").read_text())
    disk_diagnostics = json.loads(
        (tmp_path / "model_validation_diagnostics.json").read_text()
    )
    policy = summary["fixed_zero_policy"]
    assert disk_summary == summary
    assert disk_policy == policy
    assert disk_diagnostics == summary["model_validation_diagnostics"]
    assert policy["policy_id"] == (
        "notion_server_min_fixed_zero_migration_excess_preemption_v6_20260725"
    )
    assert policy["bound_audit"]["all_required_bounds_fixed_zero"] is True
    assert policy["bound_audit"]["semantic_support_verified"] is True
    assert policy["solution_audit"]["spot_preemption_value_sum"] == (
        pytest.approx(0.0)
    )
    assert policy["solution_audit"]["reported_expected_spot_service_rate"] == (
        pytest.approx(1.0)
    )
    assert policy["solution_audit"][
        "reported_maximum_spot_relaxation_fractionality"
    ] == pytest.approx(0.0)
    assert summary["operations"]["spot_preemption_allowed"] is False


def test_no_solution_audit_requires_construction_support_certificate() -> None:
    pytest.importorskip("gurobipy")
    entrypoint = _load_entrypoint()
    ordinary = build_tiny_model(
        make_instance(
            on_demand_ids=(),
            spot_ids=("spot0",),
            batch_ids=(),
            servers=("server0",),
            periods=(0, 1),
            scenarios=(0,),
            active_spot={"spot0": (0, 1)},
        )
    )

    with pytest.raises(AssertionError, match="construction-time"):
        entrypoint._bound_audit_from_model(ordinary.model)


def test_patched_cli_restores_all_replacements() -> None:
    entrypoint = _load_entrypoint()
    replacements = {
        "build_instance": (
            entrypoint.build_instance_fixed_zero_migration_excess_preemption
        ),
        "build_model": entrypoint.build_model_fixed_zero_migration_excess_preemption,
        "write_solution_reports": (
            entrypoint.write_solution_reports_fixed_zero_migration_excess_preemption
        ),
        "write_no_solution_summary": (
            entrypoint.write_no_solution_summary_fixed_zero_migration_excess_preemption
        ),
    }
    originals = {name: getattr(entrypoint._cli, name) for name in replacements}

    with entrypoint._patched_cli():
        assert all(
            getattr(entrypoint._cli, name) is replacement
            for name, replacement in replacements.items()
        )
    assert all(
        getattr(entrypoint._cli, name) is original
        for name, original in originals.items()
    )
