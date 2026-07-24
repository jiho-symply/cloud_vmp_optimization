from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from conftest import build_tiny_model, make_instance, optimize_tiny


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = EXPERIMENT_ROOT / "run_experiment_no_migration.py"


def _load_entrypoint():
    spec = importlib.util.spec_from_file_location(
        "notion_server_min_no_migration_entrypoint", ENTRYPOINT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_configuration_requires_literal_true() -> None:
    entrypoint = _load_entrypoint()

    entrypoint.require_fixed_zero_configuration({"migration": {"fixed_zero": True}})
    for invalid in (False, 1, "true", None):
        with pytest.raises(ValueError, match="migration.fixed_zero: true"):
            entrypoint.require_fixed_zero_configuration(
                {"migration": {"fixed_zero": invalid}}
            )
    with pytest.raises(ValueError, match="migration.fixed_zero: true"):
        entrypoint.require_fixed_zero_configuration({})


def test_build_model_fixes_every_migration_variable() -> None:
    pytest.importorskip("gurobipy")
    entrypoint = _load_entrypoint()

    artifacts = entrypoint.build_model_no_migration(
        make_instance(spot_ids=(), batch_ids=(), scenarios=(0,))
    )
    migration_variables = list(artifacts.variables["migration"].values())

    assert migration_variables
    assert set(artifacts.variables["migration"]) == {
        (vm_id, server_id, t, scenario_id)
        for vm_id in artifacts.data.I
        for server_id in artifacts.data.S
        for t in artifacts.data.active_od[vm_id][:-1]
        for scenario_id in artifacts.data.Xi
    }
    assert all(variable.LB == 0.0 and variable.UB == 0.0 for variable in migration_variables)


def test_fixed_zero_migration_makes_different_adjacent_placements_infeasible() -> None:
    gurobipy = pytest.importorskip("gurobipy")
    entrypoint = _load_entrypoint()
    artifacts = entrypoint.build_model_no_migration(
        make_instance(
            spot_ids=(),
            batch_ids=(),
            periods=(0, 1),
            scenarios=(0,),
        )
    )
    model = artifacts.model
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    variables = artifacts.variables
    variables["x"]["od0", "server0", 0, 0].LB = 1.0
    variables["x"]["od0", "server0", 0, 0].UB = 1.0
    variables["x"]["od0", "server1", 1, 0].LB = 1.0
    variables["x"]["od0", "server1", 1, 0].UB = 1.0

    model.optimize()

    assert model.Status == gurobipy.GRB.INFEASIBLE


def test_solution_report_persists_policy_and_zero_audit(tmp_path: Path) -> None:
    pytest.importorskip("gurobipy")
    entrypoint = _load_entrypoint()
    artifacts = entrypoint.build_model_no_migration(
        make_instance(spot_ids=(), batch_ids=(), scenarios=(0,))
    )
    artifacts.model.Params.OutputFlag = 0
    artifacts.model.Params.Threads = 1
    artifacts.model.Params.TimeLimit = 5
    optimize_tiny(artifacts)

    summary = entrypoint.write_solution_reports_no_migration(artifacts, tmp_path)

    policy = summary["migration_policy"]
    audit = policy["audit"]
    assert policy["fixed_zero"] is True
    assert policy["migration_allowed"] is False
    assert audit["fixed_variable_count"] == len(
        artifacts.variables["migration"]
    )
    assert audit["all_migration_bounds_fixed_zero"] is True
    assert audit["migration_relaxed_value_sum"] == pytest.approx(0.0)
    assert audit["actual_server_change_count"] == 0
    assert audit["migration_energy_kwh_sum"] == pytest.approx(0.0)
    assert summary["operations"]["migration_allowed"] is False

    on_disk = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    policy_on_disk = json.loads(
        (tmp_path / "migration_policy.json").read_text(encoding="utf-8")
    )
    assert on_disk["migration_policy"] == policy_on_disk


def test_instance_wrapper_records_fidelity_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    entrypoint = _load_entrypoint()
    instance = make_instance()
    monkeypatch.setattr(entrypoint, "_ORIGINAL_BUILD_INSTANCE", lambda *_args, **_kwargs: instance)

    returned = entrypoint.build_instance_no_migration(
        {"migration": {"fixed_zero": True}}
    )

    assert returned is instance
    assert returned.metadata["migration_policy"]["migration_allowed"] is False
    assert (
        entrypoint.FIDELITY_WARNING_KEY
        in returned.metadata["fidelity_warnings"]
    )
