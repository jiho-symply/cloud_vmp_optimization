from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from conftest import build_tiny_model, make_instance, optimize_tiny


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = EXPERIMENT_ROOT / "run_experiment_fixed_zero_migration_excess.py"


def _load_entrypoint():
    spec = importlib.util.spec_from_file_location(
        "notion_server_min_fixed_zero_migration_excess_entrypoint_test",
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
    }


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


@pytest.mark.parametrize(
    ("section", "invalid"),
    [
        ("migration", False),
        ("migration", 1),
        ("migration", "true"),
        ("migration", None),
        ("excess_load_indicator", False),
        ("excess_load_indicator", 1),
        ("excess_load_indicator", "true"),
        ("excess_load_indicator", None),
    ],
)
def test_configuration_requires_both_literal_boolean_true(
    section: str,
    invalid: object,
) -> None:
    entrypoint = _load_entrypoint()
    config = _valid_config()
    config[section]["fixed_zero"] = invalid

    with pytest.raises(ValueError, match=rf"{section}\.fixed_zero"):
        entrypoint.require_fixed_zero_migration_excess_configuration(config)


@pytest.mark.parametrize("missing_section", ["migration", "excess_load_indicator"])
def test_configuration_rejects_a_missing_fixed_zero_section(
    missing_section: str,
) -> None:
    entrypoint = _load_entrypoint()
    config = _valid_config()
    del config[missing_section]

    with pytest.raises(ValueError, match=rf"{missing_section}\.fixed_zero"):
        entrypoint.require_fixed_zero_migration_excess_configuration(config)


def test_configuration_accepts_only_the_explicit_combined_opt_in() -> None:
    entrypoint = _load_entrypoint()

    entrypoint.require_fixed_zero_migration_excess_configuration(_valid_config())


def test_build_model_fixes_every_migration_and_excess_branch_variable() -> None:
    pytest.importorskip("gurobipy")
    entrypoint = _load_entrypoint()
    artifacts = entrypoint.build_model_fixed_zero_migration_excess(
        make_instance(
            on_demand_ids=("od0", "od1"),
            spot_ids=(),
            batch_ids=(),
            periods=(0, 1, 2),
            scenarios=(0, 1),
            active_od={"od0": (0, 1, 2), "od1": (1, 2)},
        )
    )

    migration = artifacts.variables["migration"]
    excess_branch = artifacts.variables["excess_branch"]
    assert migration
    assert excess_branch
    assert set(migration) == {
        (vm_id, server_id, t, scenario_id)
        for vm_id in artifacts.data.I
        for server_id in artifacts.data.S
        for t in artifacts.data.active_od[vm_id][:-1]
        for scenario_id in artifacts.data.Xi
    }
    assert set(excess_branch) == {
        (server_id, t, scenario_id)
        for server_id in artifacts.data.S
        for t in artifacts.data.T
        for scenario_id in artifacts.data.Xi
    }
    assert all(
        variable.LB == pytest.approx(0.0)
        and variable.UB == pytest.approx(0.0)
        for variable in migration.values()
    )
    assert all(
        variable.LB == pytest.approx(0.0)
        and variable.UB == pytest.approx(0.0)
        for variable in excess_branch.values()
    )


def test_fixed_zero_migration_prohibits_an_adjacent_server_change() -> None:
    gp = pytest.importorskip("gurobipy")
    entrypoint = _load_entrypoint()
    artifacts = entrypoint.build_model_fixed_zero_migration_excess(
        make_instance(
            spot_ids=(),
            batch_ids=(),
            periods=(0, 1),
            scenarios=(0,),
            active_od={"od0": (0, 1)},
        )
    )
    _configure_tiny_solver(artifacts)
    artifacts.model.Params.DualReductions = 0
    _fix_od_path(
        artifacts,
        vm_id="od0",
        scenario_id=0,
        server_path=("server0", "server1"),
    )

    artifacts.model.optimize()

    assert artifacts.model.Status == gp.GRB.INFEASIBLE


def test_fixed_zero_excess_branch_prohibits_positive_excess_load() -> None:
    gp = pytest.importorskip("gurobipy")
    entrypoint = _load_entrypoint()
    data = make_instance(
        spot_ids=(),
        batch_ids=(),
        servers=("server0",),
        periods=(0,),
        scenarios=(0,),
        active_od={"od0": (0,)},
        od_cpu_by_scenario={0: 1.0},
        C_cpu=0.8,
        C_mem=1.0,
        # The ordinary model must permit the 0.2 CPU excess so this test
        # isolates the new branch=0 restriction rather than the CVaR bound.
        epsilon=0.3,
    )

    ordinary = build_tiny_model(data)
    optimize_tiny(ordinary)
    assert ordinary.model.Status == gp.GRB.OPTIMAL
    assert ordinary.variables["excess_branch"]["server0", 0, 0].X == pytest.approx(
        1.0
    )
    assert ordinary.variables["excess"]["server0", 0, 0].X == pytest.approx(0.2)

    fixed = entrypoint.build_model_fixed_zero_migration_excess(data)
    _configure_tiny_solver(fixed)
    fixed.model.Params.DualReductions = 0
    fixed.model.optimize()

    assert fixed.model.Status == gp.GRB.INFEASIBLE


def test_feasible_fixed_zero_solution_has_zero_branch_and_zero_excess() -> None:
    gp = pytest.importorskip("gurobipy")
    entrypoint = _load_entrypoint()
    artifacts = entrypoint.build_model_fixed_zero_migration_excess(
        make_instance(
            spot_ids=(),
            batch_ids=(),
            servers=("server0",),
            periods=(0,),
            scenarios=(0,),
            active_od={"od0": (0,)},
            od_cpu_by_scenario={0: 0.6},
            C_cpu=0.8,
            C_mem=1.0,
            epsilon=0.2,
        )
    )
    _configure_tiny_solver(artifacts)

    optimize_tiny(artifacts)

    assert artifacts.model.Status == gp.GRB.OPTIMAL
    assert all(
        variable.X == pytest.approx(0.0)
        for variable in artifacts.variables["excess_branch"].values()
    )
    assert all(
        variable.X == pytest.approx(0.0)
        for variable in artifacts.variables["excess"].values()
    )


def test_primary_migration_v3_model_remains_unfixed() -> None:
    gp = pytest.importorskip("gurobipy")
    artifacts = build_tiny_model(
        make_instance(
            spot_ids=(),
            batch_ids=(),
            periods=(0, 1),
            scenarios=(0,),
            active_od={"od0": (0, 1)},
        )
    )

    assert all(
        variable.LB == pytest.approx(0.0)
        and variable.UB == pytest.approx(1.0)
        for variable in artifacts.variables["migration"].values()
    )
    assert all(
        variable.LB == pytest.approx(0.0)
        and variable.UB == pytest.approx(1.0)
        for variable in artifacts.variables["excess_branch"].values()
    )
    _fix_od_path(
        artifacts,
        vm_id="od0",
        scenario_id=0,
        server_path=("server0", "server1"),
    )

    optimize_tiny(artifacts)

    assert artifacts.model.Status == gp.GRB.OPTIMAL
    assert sum(
        variable.X for variable in artifacts.variables["migration"].values()
    ) == pytest.approx(1.0)


def test_instance_wrapper_records_both_fixed_zero_scopes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entrypoint = _load_entrypoint()
    instance = make_instance()
    monkeypatch.setattr(
        entrypoint,
        "_ORIGINAL_BUILD_INSTANCE",
        lambda *_args, **_kwargs: instance,
    )

    returned = entrypoint.build_instance_fixed_zero_migration_excess(_valid_config())

    assert returned is instance
    policy = returned.metadata["fixed_zero_policy"]
    assert policy["migration_fixed_zero"] is True
    assert policy["excess_load_indicator_fixed_zero"] is True
    assert entrypoint.FIDELITY_WARNING_KEY in returned.metadata["fidelity_warnings"]


def test_solution_writer_persists_synchronized_policy_and_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gp = pytest.importorskip("gurobipy")
    entrypoint = _load_entrypoint()
    artifacts = entrypoint.build_model_fixed_zero_migration_excess(
        make_instance(
            spot_ids=(),
            batch_ids=(),
            servers=("server0",),
            periods=(0, 1),
            scenarios=(0,),
            active_od={"od0": (0, 1)},
            od_cpu_by_scenario={0: 0.4},
        )
    )
    _configure_tiny_solver(artifacts)
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

    summary = entrypoint.write_solution_reports_fixed_zero_migration_excess(
        artifacts,
        tmp_path,
    )

    on_disk_summary = json.loads(
        (tmp_path / "summary.json").read_text(encoding="utf-8")
    )
    on_disk_policy = json.loads(
        (tmp_path / "fixed_zero_policy.json").read_text(encoding="utf-8")
    )
    on_disk_diagnostics = json.loads(
        (tmp_path / "model_validation_diagnostics.json").read_text(
            encoding="utf-8"
        )
    )
    policy = summary["fixed_zero_policy"]
    diagnostics = summary["model_validation_diagnostics"]

    assert on_disk_summary == summary
    assert on_disk_policy == policy
    assert on_disk_diagnostics == diagnostics
    assert policy["policy_id"] == (
        "notion_server_min_fixed_zero_migration_excess_v5_20260724"
    )
    assert policy["migration_fixed_zero"] is True
    assert policy["excess_load_indicator_fixed_zero"] is True
    assert diagnostics["fixed_zero_migration_excess_bounds"] == policy[
        "bound_audit"
    ]
    assert diagnostics["fixed_zero_migration_excess_solution"] == policy[
        "solution_audit"
    ]
    assert policy["bound_audit"]["all_required_bounds_fixed_zero"] is True
    assert policy["bound_audit"]["migration"]["all_bounds_fixed_zero"] is True
    assert policy["bound_audit"]["excess_branch"]["all_bounds_fixed_zero"] is True
    assert policy["solution_audit"]["incumbent_available"] is True
    assert policy["solution_audit"]["migration_value_sum"] == pytest.approx(0.0)
    assert policy["solution_audit"]["actual_server_change_count"] == 0
    assert policy["solution_audit"]["excess_branch_value_sum"] == pytest.approx(
        0.0
    )
    assert policy["solution_audit"]["excess_value_sum"] == pytest.approx(0.0)
    assert summary["operations"]["migration_allowed"] is False
    assert summary["operations"]["excess_load_allowed"] is False


def test_no_solution_writer_persists_synchronized_policy_and_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gp = pytest.importorskip("gurobipy")
    entrypoint = _load_entrypoint()
    artifacts = entrypoint.build_model_fixed_zero_migration_excess(
        make_instance(
            spot_ids=(),
            batch_ids=(),
            periods=(0, 1),
            scenarios=(0,),
            active_od={"od0": (0, 1)},
        )
    )
    _configure_tiny_solver(artifacts)
    artifacts.model.Params.DualReductions = 0
    _fix_od_path(
        artifacts,
        vm_id="od0",
        scenario_id=0,
        server_path=("server0", "server1"),
    )
    artifacts.model.optimize()
    assert artifacts.model.Status == gp.GRB.INFEASIBLE

    original_finalize = entrypoint._finalize_policy_report

    def finalize_before_completion_marker(*args: object, **kwargs: object) -> dict:
        assert not (tmp_path / "summary.json").exists()
        return original_finalize(*args, **kwargs)

    monkeypatch.setattr(
        entrypoint,
        "_finalize_policy_report",
        finalize_before_completion_marker,
    )

    summary = entrypoint.write_no_solution_summary_fixed_zero_migration_excess(
        artifacts.model,
        tmp_path,
    )

    on_disk_summary = json.loads(
        (tmp_path / "summary.json").read_text(encoding="utf-8")
    )
    on_disk_policy = json.loads(
        (tmp_path / "fixed_zero_policy.json").read_text(encoding="utf-8")
    )
    on_disk_diagnostics = json.loads(
        (tmp_path / "model_validation_diagnostics.json").read_text(
            encoding="utf-8"
        )
    )
    policy = summary["fixed_zero_policy"]
    diagnostics = summary["model_validation_diagnostics"]

    assert on_disk_summary == summary
    assert on_disk_policy == policy
    assert on_disk_diagnostics == diagnostics
    assert summary["solver"]["status"] == "INFEASIBLE"
    assert diagnostics["fixed_zero_migration_excess_bounds"] == policy[
        "bound_audit"
    ]
    assert diagnostics["fixed_zero_migration_excess_solution"] == policy[
        "solution_audit"
    ]
    assert policy["bound_audit"]["all_required_bounds_fixed_zero"] is True
    assert policy["bound_audit"]["migration"]["all_bounds_fixed_zero"] is True
    assert policy["bound_audit"]["excess_branch"]["all_bounds_fixed_zero"] is True
    assert policy["solution_audit"]["incumbent_available"] is False
    assert policy["solution_audit"]["audit_status"] == (
        "not_applicable_without_incumbent"
    )


def test_patched_cli_restores_replacements_after_normal_and_exceptional_exit() -> None:
    entrypoint = _load_entrypoint()
    replacements = {
        "build_instance": entrypoint.build_instance_fixed_zero_migration_excess,
        "build_model": entrypoint.build_model_fixed_zero_migration_excess,
        "write_solution_reports": (
            entrypoint.write_solution_reports_fixed_zero_migration_excess
        ),
        "write_no_solution_summary": (
            entrypoint.write_no_solution_summary_fixed_zero_migration_excess
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
