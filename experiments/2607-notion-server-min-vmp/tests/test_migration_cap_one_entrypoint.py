from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import build_tiny_model, make_instance, optimize_tiny


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = EXPERIMENT_ROOT / "run_experiment_migration_cap_one.py"
CONFIG_KEY = "max_count_per_vm_per_scenario"


def _load_entrypoint():
    spec = importlib.util.spec_from_file_location(
        "notion_server_min_migration_cap_one_entrypoint_test", ENTRYPOINT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configure_tiny_solver(artifacts) -> None:
    artifacts.model.Params.OutputFlag = 0
    artifacts.model.Params.LogToConsole = 0
    artifacts.model.Params.Threads = 1
    artifacts.model.Params.TimeLimit = 5
    artifacts.model.Params.Seed = 1


def _fix_od_path(
    artifacts,
    *,
    vm_id: str,
    scenario_id: int,
    server_path: tuple[str, ...],
) -> None:
    periods = artifacts.data.active_od[vm_id]
    assert len(server_path) == len(periods)
    for t, selected_server in zip(periods, server_path, strict=True):
        for server_id in artifacts.data.S:
            value = float(server_id == selected_server)
            variable = artifacts.variables["x"][vm_id, server_id, t, scenario_id]
            variable.LB = value
            variable.UB = value


def test_configuration_requires_literal_integer_one() -> None:
    entrypoint = _load_entrypoint()

    entrypoint.require_migration_cap_one_configuration(
        {"migration": {CONFIG_KEY: 1}}
    )
    for invalid in (True, False, 0, 2, 1.0, "1", None):
        with pytest.raises(
            ValueError,
            match=r"migration\.max_count_per_vm_per_scenario.*integer 1",
        ):
            entrypoint.require_migration_cap_one_configuration(
                {"migration": {CONFIG_KEY: invalid}}
            )
    with pytest.raises(
        ValueError,
        match=r"migration\.max_count_per_vm_per_scenario.*integer 1",
    ):
        entrypoint.require_migration_cap_one_configuration({})


def test_cap_has_exact_vm_scenario_destination_transition_support() -> None:
    pytest.importorskip("gurobipy")
    entrypoint = _load_entrypoint()
    artifacts = entrypoint.build_model_migration_cap_one(
        make_instance(
            on_demand_ids=("od0", "od1"),
            spot_ids=(),
            batch_ids=(),
            periods=(0, 1, 2, 3),
            scenarios=(0, 1),
            active_od={"od0": (1, 2, 3), "od1": (0, 1)},
        )
    )
    model = artifacts.model
    migration = artifacts.variables["migration"]
    caps = artifacts.variables["migration_count_cap"]

    assert set(caps) == {
        (vm_id, scenario_id)
        for vm_id in artifacts.data.I
        for scenario_id in artifacts.data.Xi
    }
    constraint = caps["od0", 1]
    assert constraint.ConstrName == "migration_count_cap[od0,1]"
    assert constraint.Sense == "<"
    assert constraint.RHS == pytest.approx(1.0)

    observed_support = {
        key
        for key, variable in migration.items()
        if model.getCoeff(constraint, variable) != 0.0
    }
    expected_support = {
        ("od0", server_id, t, 1)
        for server_id in artifacts.data.S
        for t in (1, 2)
    }
    assert observed_support == expected_support
    assert all(
        model.getCoeff(constraint, migration[key]) == pytest.approx(1.0)
        for key in expected_support
    )
    assert artifacts.variables["migration_count_cap_audit"][
        "constraint_count"
    ] == len(artifacts.data.I) * len(artifacts.data.Xi)


def test_two_moves_in_one_scenario_are_infeasible() -> None:
    gp = pytest.importorskip("gurobipy")
    entrypoint = _load_entrypoint()
    artifacts = entrypoint.build_model_migration_cap_one(
        make_instance(
            spot_ids=(),
            batch_ids=(),
            periods=(0, 1, 2),
            scenarios=(0,),
            active_od={"od0": (0, 1, 2)},
        )
    )
    _configure_tiny_solver(artifacts)
    artifacts.model.Params.DualReductions = 0
    _fix_od_path(
        artifacts,
        vm_id="od0",
        scenario_id=0,
        server_path=("server0", "server1", "server0"),
    )

    artifacts.model.optimize()

    assert artifacts.model.Status == gp.GRB.INFEASIBLE


def test_one_move_in_one_scenario_is_feasible_and_counts_once() -> None:
    gp = pytest.importorskip("gurobipy")
    entrypoint = _load_entrypoint()
    artifacts = entrypoint.build_model_migration_cap_one(
        make_instance(
            spot_ids=(),
            batch_ids=(),
            periods=(0, 1, 2),
            scenarios=(0,),
            active_od={"od0": (0, 1, 2)},
        )
    )
    _configure_tiny_solver(artifacts)
    _fix_od_path(
        artifacts,
        vm_id="od0",
        scenario_id=0,
        server_path=("server0", "server1", "server1"),
    )

    optimize_tiny(artifacts)

    assert artifacts.model.Status == gp.GRB.OPTIMAL
    migration_total = sum(
        variable.X
        for (vm_id, _server_id, _t, scenario_id), variable in artifacts.variables[
            "migration"
        ].items()
        if vm_id == "od0" and scenario_id == 0
    )
    assert migration_total == pytest.approx(1.0)


def test_cap_is_independent_per_scenario_not_summed_across_scenarios() -> None:
    gp = pytest.importorskip("gurobipy")
    entrypoint = _load_entrypoint()
    artifacts = entrypoint.build_model_migration_cap_one(
        make_instance(
            spot_ids=(),
            batch_ids=(),
            periods=(0, 1, 2),
            scenarios=(0, 1),
            active_od={"od0": (0, 1, 2)},
        )
    )
    _configure_tiny_solver(artifacts)
    for scenario_id in artifacts.data.Xi:
        _fix_od_path(
            artifacts,
            vm_id="od0",
            scenario_id=scenario_id,
            server_path=("server0", "server1", "server1"),
        )

    optimize_tiny(artifacts)

    assert artifacts.model.Status == gp.GRB.OPTIMAL
    scenario_totals = {
        scenario_id: sum(
            variable.X
            for (
                vm_id,
                _server_id,
                _t,
                xi,
            ), variable in artifacts.variables["migration"].items()
            if vm_id == "od0" and xi == scenario_id
        )
        for scenario_id in artifacts.data.Xi
    }
    assert scenario_totals == {0: pytest.approx(1.0), 1: pytest.approx(1.0)}
    assert sum(scenario_totals.values()) == pytest.approx(2.0)


def test_single_active_slot_gets_one_empty_support_cap_per_scenario() -> None:
    pytest.importorskip("gurobipy")
    entrypoint = _load_entrypoint()
    artifacts = entrypoint.build_model_migration_cap_one(
        make_instance(
            spot_ids=(),
            batch_ids=(),
            periods=(0,),
            scenarios=(0, 1),
            active_od={"od0": (0,)},
        )
    )
    model = artifacts.model
    caps = artifacts.variables["migration_count_cap"]

    assert not artifacts.variables["migration"]
    assert set(caps) == {("od0", 0), ("od0", 1)}
    for constraint in caps.values():
        assert constraint.Sense == "<"
        assert constraint.RHS == pytest.approx(1.0)
        assert all(
            model.getCoeff(constraint, variable) == pytest.approx(0.0)
            for variable in model.getVars()
        )


def test_primary_migration_v3_model_remains_uncapped() -> None:
    gp = pytest.importorskip("gurobipy")
    artifacts = build_tiny_model(
        make_instance(
            spot_ids=(),
            batch_ids=(),
            periods=(0, 1, 2),
            scenarios=(0,),
            active_od={"od0": (0, 1, 2)},
        )
    )

    assert "migration_count_cap" not in artifacts.variables
    assert artifacts.model.getConstrByName("migration_count_cap[od0,0]") is None
    _fix_od_path(
        artifacts,
        vm_id="od0",
        scenario_id=0,
        server_path=("server0", "server1", "server0"),
    )

    optimize_tiny(artifacts)

    assert artifacts.model.Status == gp.GRB.OPTIMAL
    migration_total = sum(
        variable.X for variable in artifacts.variables["migration"].values()
    )
    assert migration_total == pytest.approx(2.0)


def test_solution_writer_persists_synchronized_policy_and_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    gp = pytest.importorskip("gurobipy")
    entrypoint = _load_entrypoint()
    artifacts = entrypoint.build_model_migration_cap_one(
        make_instance(
            spot_ids=(),
            batch_ids=(),
            periods=(0, 1),
            scenarios=(0,),
            active_od={"od0": (0, 1)},
        )
    )
    _configure_tiny_solver(artifacts)
    _fix_od_path(
        artifacts,
        vm_id="od0",
        scenario_id=0,
        server_path=("server0", "server1"),
    )
    optimize_tiny(artifacts)
    assert artifacts.model.Status == gp.GRB.OPTIMAL

    finalization_order: list[str] = []
    original_atomic_write = entrypoint._atomic_write_json

    def tracking_atomic_write(path: Path, payload: object) -> None:
        finalization_order.append(path.name)
        original_atomic_write(path, payload)

    monkeypatch.setattr(entrypoint, "_atomic_write_json", tracking_atomic_write)

    summary = entrypoint.write_solution_reports_migration_cap_one(
        artifacts, tmp_path
    )

    on_disk_summary = json.loads(
        (tmp_path / "summary.json").read_text(encoding="utf-8")
    )
    on_disk_policy = json.loads(
        (tmp_path / "migration_policy.json").read_text(encoding="utf-8")
    )
    on_disk_diagnostics = json.loads(
        (tmp_path / "model_validation_diagnostics.json").read_text(
            encoding="utf-8"
        )
    )
    policy = summary["migration_policy"]
    solution_audit = policy["solution_audit"]

    assert on_disk_summary == summary
    assert on_disk_policy == policy
    assert on_disk_diagnostics == summary["model_validation_diagnostics"]
    assert finalization_order[-3:] == [
        "model_validation_diagnostics.json",
        "migration_policy.json",
        "summary.json",
    ]
    assert policy["policy_id"] == (
        "notion_server_min_migration_cap_one_v4_20260724"
    )
    assert policy["base_formulation_id"] == (
        "notion_server_min_exact_destination_migration_v3_20260723"
    )
    assert policy["extension_revision"] == "migration_cap_one_v4_20260724"
    assert policy["solution_audit_tolerance"] == pytest.approx(1e-5)
    assert solution_audit["tolerance"] == pytest.approx(1e-5)
    assert solution_audit["total_modeled_migration_count"] == pytest.approx(1.0)
    assert solution_audit["total_actual_server_change_count"] == 1
    assert solution_audit["maximum_cap_violation"] == pytest.approx(0.0)
    assert summary["operations"][
        "max_migration_count_per_vm_per_scenario"
    ] == 1


def test_infeasible_no_solution_writer_persists_synchronized_policy(
    tmp_path: Path,
) -> None:
    gp = pytest.importorskip("gurobipy")
    entrypoint = _load_entrypoint()
    artifacts = entrypoint.build_model_migration_cap_one(
        make_instance(
            spot_ids=(),
            batch_ids=(),
            periods=(0, 1, 2),
            scenarios=(0,),
            active_od={"od0": (0, 1, 2)},
        )
    )
    _configure_tiny_solver(artifacts)
    artifacts.model.Params.DualReductions = 0
    _fix_od_path(
        artifacts,
        vm_id="od0",
        scenario_id=0,
        server_path=("server0", "server1", "server0"),
    )
    artifacts.model.optimize()
    assert artifacts.model.Status == gp.GRB.INFEASIBLE

    summary = entrypoint.write_no_solution_summary_migration_cap_one(
        artifacts.model, tmp_path
    )

    on_disk_summary = json.loads(
        (tmp_path / "summary.json").read_text(encoding="utf-8")
    )
    on_disk_policy = json.loads(
        (tmp_path / "migration_policy.json").read_text(encoding="utf-8")
    )
    on_disk_diagnostics = json.loads(
        (tmp_path / "model_validation_diagnostics.json").read_text(
            encoding="utf-8"
        )
    )
    solution_audit = summary["migration_policy"]["solution_audit"]

    assert on_disk_summary == summary
    assert on_disk_policy == summary["migration_policy"]
    assert on_disk_diagnostics == summary["model_validation_diagnostics"]
    assert summary["solver"]["status"] == "INFEASIBLE"
    assert solution_audit == {
        "incumbent_available": False,
        "configured_cap": 1,
        "tolerance": pytest.approx(1e-5),
        "audit_status": "not_applicable_without_incumbent",
    }


def test_solution_audit_failure_precedes_primary_report_and_clears_markers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    entrypoint = _load_entrypoint()
    artifacts = SimpleNamespace(
        variables={"migration_count_cap_audit": {"constraint_count": 1}}
    )
    calls: list[str] = []
    for name in (
        "summary.json",
        "model_validation_diagnostics.json",
        "migration_policy.json",
    ):
        (tmp_path / name).write_text("stale", encoding="utf-8")

    def fail_cap_audit(_artifacts: object, *, tolerance: float) -> dict:
        calls.append(f"cap:{tolerance}")
        raise AssertionError("synthetic cap audit failure")

    def unexpected_primary_report(*_args: object, **_kwargs: object) -> dict:
        calls.append("primary")
        return {}

    monkeypatch.setattr(entrypoint, "_solution_audit", fail_cap_audit)
    monkeypatch.setattr(
        entrypoint,
        "_ORIGINAL_WRITE_SOLUTION_REPORTS",
        unexpected_primary_report,
    )

    with pytest.raises(AssertionError, match="synthetic cap audit failure"):
        entrypoint.write_solution_reports_migration_cap_one(artifacts, tmp_path)

    assert calls == ["cap:1e-05"]
    assert all(
        not (tmp_path / name).exists()
        for name in (
            "summary.json",
            "model_validation_diagnostics.json",
            "migration_policy.json",
        )
    )


def test_patched_cli_restores_replacements_after_normal_and_exceptional_exit() -> None:
    entrypoint = _load_entrypoint()
    replacements = {
        "build_instance": entrypoint.build_instance_migration_cap_one,
        "build_model": entrypoint.build_model_migration_cap_one,
        "write_solution_reports": (
            entrypoint.write_solution_reports_migration_cap_one
        ),
        "write_no_solution_summary": (
            entrypoint.write_no_solution_summary_migration_cap_one
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

    with pytest.raises(RuntimeError, match="synthetic patched CLI failure"):
        with entrypoint._patched_cli():
            assert all(
                getattr(entrypoint._cli, name) is replacement
                for name, replacement in replacements.items()
            )
            raise RuntimeError("synthetic patched CLI failure")
    assert all(
        getattr(entrypoint._cli, name) is original
        for name, original in originals.items()
    )
