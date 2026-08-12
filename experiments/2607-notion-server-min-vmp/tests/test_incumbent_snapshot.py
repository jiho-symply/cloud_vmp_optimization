from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import build_tiny_model, make_instance
from notion_server_min_vmp.cli import _apply_overrides, parse_args
from notion_server_min_vmp.incumbent import (
    IncumbentSnapshotWriter,
    _atomic_write_text,
    snapshot_destination,
)


def test_snapshot_destination_is_opt_in_and_stays_in_run_directory(tmp_path) -> None:
    assert snapshot_destination({}, tmp_path) is None
    assert (
        snapshot_destination(
            {"incumbent_snapshot": {"enabled": True}},
            tmp_path,
        )
        == (tmp_path / "incumbent_latest.sol").resolve()
    )
    assert (
        snapshot_destination(
            {
                "incumbent_snapshot": {
                    "enabled": True,
                    "filename": "best_so_far.sol",
                }
            },
            tmp_path,
        )
        == (tmp_path / "best_so_far.sol").resolve()
    )


def test_cli_flag_enables_snapshot_in_resolved_solver_config() -> None:
    args = parse_args(
        [
            "--config",
            "unused.yaml",
            "--save-incumbent-snapshot",
        ]
    )
    resolved = _apply_overrides({"experiment": {}, "solver": {}}, args)
    assert resolved["solver"]["incumbent_snapshot"] == {
        "enabled": True,
        "filename": "incumbent_latest.sol",
    }


@pytest.mark.parametrize(
    "filename",
    ("../outside.sol", "/tmp/outside.sol", "nested/latest.sol", "latest.json", ""),
)
def test_snapshot_destination_rejects_unsafe_or_non_solution_name(
    tmp_path, filename: str
) -> None:
    with pytest.raises(ValueError):
        snapshot_destination(
            {
                "incumbent_snapshot": {
                    "enabled": True,
                    "filename": filename,
                }
            },
            tmp_path,
        )


def test_snapshot_destination_requires_boolean_enabled(tmp_path) -> None:
    with pytest.raises(TypeError):
        snapshot_destination(
            {"incumbent_snapshot": {"enabled": "true"}},
            tmp_path,
        )


def test_atomic_writer_replaces_one_complete_file(tmp_path) -> None:
    destination = tmp_path / "incumbent_latest.sol"
    _atomic_write_text(destination, "old\n")
    _atomic_write_text(destination, "new\ncomplete\n")
    assert destination.read_text(encoding="utf-8") == "new\ncomplete\n"
    assert not list(tmp_path.glob(".incumbent_latest.sol.*.tmp"))


def test_callback_keeps_only_strictly_improving_incumbents(tmp_path) -> None:
    gurobipy = pytest.importorskip("gurobipy")
    callback = gurobipy.GRB.Callback
    variable = SimpleNamespace(VarName="decision")

    class FakeCallbackModel:
        ModelName = "fake_maximize"
        ModelSense = -1

        def __init__(self) -> None:
            self.objective = 0.0
            self.solution_reads = 0

        @staticmethod
        def getVars():
            return [variable]

        def cbGet(self, code):
            values = {
                callback.MIPSOL_OBJ: self.objective,
                callback.MIPSOL_OBJBND: 10.0,
                callback.MIPSOL_SOLCNT: self.solution_reads,
                callback.RUNTIME: float(self.solution_reads),
                callback.MIPSOL_NODCNT: 0.0,
                callback.WORK: 0.0,
            }
            return values[code]

        def cbGetSolution(self, _variables):
            self.solution_reads += 1
            return [self.objective]

    model = FakeCallbackModel()
    destination = tmp_path / "incumbent_latest.sol"
    writer = IncumbentSnapshotWriter(model, destination)

    model.objective = 5.0
    writer(model, callback.MIPSOL)
    first = destination.read_text(encoding="utf-8")
    model.objective = 4.0
    writer(model, callback.MIPSOL)
    assert destination.read_text(encoding="utf-8") == first
    model.objective = 6.0
    writer(model, callback.MIPSOL)

    latest = destination.read_text(encoding="utf-8")
    assert "# Incumbent sequence = 2" in latest
    assert "# Objective value = 6" in latest
    assert latest.endswith("decision 6\n")
    assert model.solution_reads == 2


def test_writer_removes_a_stale_snapshot_before_new_solve(tmp_path) -> None:
    pytest.importorskip("gurobipy")
    artifacts = build_tiny_model(make_instance())
    destination = tmp_path / "incumbent_latest.sol"
    destination.write_text("stale\n", encoding="utf-8")

    IncumbentSnapshotWriter(artifacts.model, destination)

    assert not destination.exists()


def test_tiny_solve_writes_callback_and_terminal_incumbent(tmp_path) -> None:
    pytest.importorskip("gurobipy")
    artifacts = build_tiny_model(make_instance())
    model = artifacts.model
    destination = tmp_path / "incumbent_latest.sol"
    writer = IncumbentSnapshotWriter(
        model,
        destination,
        constant_revenue_offset=sum(artifacts.objective_constants.values()),
    )

    model.optimize(writer)
    assert model.SolCount > 0
    assert writer.sequence >= 1
    assert writer.finalize(model)

    text = destination.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0] == f"# Solution for model {model.ModelName}"
    assert lines[1].startswith("# Objective value = ")
    assert "# Snapshot source = terminal incumbent" in lines
    assert "# Total profit including constant revenue = " in text
    assert "# Relative MIP gap for solver objective = " in text
    variable_lines = [line for line in lines if line and not line.startswith("#")]
    assert len(variable_lines) == model.NumVars
    assert any(line.startswith("x_init[") for line in variable_lines)
    assert not list(tmp_path.glob(".incumbent_latest.sol.*.tmp"))
