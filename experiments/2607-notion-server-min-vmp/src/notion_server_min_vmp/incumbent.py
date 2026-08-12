from __future__ import annotations

import math
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _relative_gap(objective: float | None, bound: float | None) -> float | None:
    if objective is None or bound is None:
        return None
    if abs(objective) <= 1e-10:
        return 0.0 if abs(bound - objective) <= 1e-10 else None
    return abs(bound - objective) / abs(objective)


def _format_number(value: Any) -> str:
    number = _finite_or_none(value)
    return "unavailable" if number is None else format(number, ".17g")


def _atomic_write_text(path: Path, text: str) -> None:
    """Publish a complete snapshot at one stable path.

    A reader therefore sees either the previous incumbent or the new incumbent,
    never a partially written multi-megabyte solution file.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class SnapshotMetrics:
    source: str
    saved_at_utc: str
    sequence: int
    solution_count: int | None
    runtime_seconds: float | None
    work_units: float | None
    node_count: float | None
    objective: float | None
    best_bound: float | None
    relative_gap: float | None
    status_code: int | None


class IncumbentSnapshotWriter:
    """Atomically overwrite one ``.sol`` file for each improving MIPSOL."""

    def __init__(
        self,
        model: Any,
        destination: str | Path,
        *,
        constant_revenue_offset: float = 0.0,
    ) -> None:
        self.destination = Path(destination).resolve()
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.destination.unlink(missing_ok=True)
        for stale_temporary in self.destination.parent.glob(
            f".{self.destination.name}.*.tmp"
        ):
            stale_temporary.unlink(missing_ok=True)
        self.variables: tuple[Any, ...] = tuple(model.getVars())
        self.variable_names: tuple[str, ...] = tuple(
            str(variable.VarName) for variable in self.variables
        )
        self.model_name = str(model.ModelName)
        self.objective_sense = int(model.ModelSense)
        self.constant_revenue_offset = float(constant_revenue_offset)
        self.sequence = 0
        self.best_objective: float | None = None
        self.disabled = False
        self.last_error: str | None = None
        self.last_metrics: SnapshotMetrics | None = None

    def __call__(self, model: Any, where: int) -> None:
        if self.disabled:
            return
        try:
            from gurobipy import GRB

            if where != GRB.Callback.MIPSOL:
                return
            objective = _finite_or_none(model.cbGet(GRB.Callback.MIPSOL_OBJ))
            if objective is None or not self._is_improving(objective):
                return
            values = tuple(float(value) for value in model.cbGetSolution(self.variables))
            bound = _finite_or_none(model.cbGet(GRB.Callback.MIPSOL_OBJBND))
            next_sequence = self.sequence + 1
            metrics = SnapshotMetrics(
                source="MIPSOL callback",
                saved_at_utc=datetime.now(timezone.utc).isoformat(),
                sequence=next_sequence,
                solution_count=int(model.cbGet(GRB.Callback.MIPSOL_SOLCNT)) + 1,
                runtime_seconds=_finite_or_none(model.cbGet(GRB.Callback.RUNTIME)),
                work_units=self._callback_value(
                    model, getattr(GRB.Callback, "WORK", None)
                ),
                node_count=_finite_or_none(
                    model.cbGet(GRB.Callback.MIPSOL_NODCNT)
                ),
                objective=objective,
                best_bound=bound,
                relative_gap=_relative_gap(objective, bound),
                status_code=None,
            )
            self._publish(values, metrics)
            self.sequence = next_sequence
            self.best_objective = objective
        except Exception as exc:  # callback I/O must never abort a long solve
            self._disable_after_error(exc)

    @staticmethod
    def _callback_value(model: Any, code: Any) -> float | None:
        if code is None:
            return None
        try:
            return _finite_or_none(model.cbGet(code))
        except Exception:
            return None

    def _is_improving(self, objective: float) -> bool:
        if self.best_objective is None:
            return True
        tolerance = 1e-10 * max(
            1.0, abs(objective), abs(self.best_objective)
        )
        if self.objective_sense == -1:
            return objective > self.best_objective + tolerance
        return objective < self.best_objective - tolerance

    def finalize(self, model: Any) -> bool:
        """Refresh the same path with terminal bound/gap metadata after optimize."""

        if self.disabled or int(model.SolCount) <= 0:
            return False
        try:
            values = tuple(
                float(value) for value in model.getAttr("X", list(self.variables))
            )
            objective = _finite_or_none(model.ObjVal)
            bound = _finite_or_none(model.ObjBound)
            reported_gap = _finite_or_none(model.MIPGap)
            metrics = SnapshotMetrics(
                source="terminal incumbent",
                saved_at_utc=datetime.now(timezone.utc).isoformat(),
                sequence=max(1, self.sequence),
                solution_count=int(model.SolCount),
                runtime_seconds=_finite_or_none(model.Runtime),
                work_units=_finite_or_none(getattr(model, "Work", None)),
                node_count=_finite_or_none(model.NodeCount),
                objective=objective,
                best_bound=bound,
                relative_gap=(
                    reported_gap
                    if reported_gap is not None
                    else _relative_gap(objective, bound)
                ),
                status_code=int(model.Status),
            )
            self._publish(values, metrics)
            return True
        except Exception as exc:
            self._disable_after_error(exc)
            return False

    def _publish(
        self, values: Sequence[float], metrics: SnapshotMetrics
    ) -> None:
        if len(values) != len(self.variable_names):
            raise ValueError(
                "incumbent vector length does not match the model variable list: "
                f"{len(values)} != {len(self.variable_names)}"
            )
        objective = metrics.objective
        bound = metrics.best_bound
        total_profit = (
            objective + self.constant_revenue_offset
            if objective is not None
            else None
        )
        total_profit_bound = (
            bound + self.constant_revenue_offset if bound is not None else None
        )
        sense = "MAXIMIZE" if self.objective_sense == -1 else "MINIMIZE"
        solution_count_label = (
            "Callback solution ordinal (MIPSOL_SOLCNT + 1)"
            if metrics.source == "MIPSOL callback"
            else "Solver solution count"
        )
        lines = [
            f"# Solution for model {self.model_name}",
            f"# Objective value = {_format_number(objective)}",
            "# Snapshot format = complete Gurobi solution vector",
            f"# Snapshot source = {metrics.source}",
            f"# Saved at UTC = {metrics.saved_at_utc}",
            f"# Incumbent sequence = {metrics.sequence}",
            f"# {solution_count_label} = {_format_number(metrics.solution_count)}",
            f"# Runtime seconds = {_format_number(metrics.runtime_seconds)}",
            f"# Work units = {_format_number(metrics.work_units)}",
            f"# Explored node count = {_format_number(metrics.node_count)}",
            f"# Objective sense = {sense}",
            f"# Best bound = {_format_number(bound)}",
            (
                "# Relative MIP gap for solver objective = "
                f"{_format_number(metrics.relative_gap)}"
            ),
            (
                "# Constant On-demand + Batch revenue offset = "
                f"{_format_number(self.constant_revenue_offset)}"
            ),
            (
                "# Total profit including constant revenue = "
                f"{_format_number(total_profit)}"
            ),
            (
                "# Best bound including constant revenue = "
                f"{_format_number(total_profit_bound)}"
            ),
            f"# Terminal status code = {_format_number(metrics.status_code)}",
            f"# Variable count = {len(self.variable_names)}",
        ]
        lines.extend(
            f"{name} {format(float(value), '.17g')}"
            for name, value in zip(self.variable_names, values)
        )
        _atomic_write_text(self.destination, "\n".join(lines) + "\n")
        self.last_metrics = metrics
        gap = (
            "unavailable"
            if metrics.relative_gap is None
            else f"{100.0 * metrics.relative_gap:.4f}%"
        )
        profit = (
            "unavailable" if total_profit is None else f"{total_profit:.9g}"
        )
        print(
            "[incumbent snapshot] "
            f"#{metrics.sequence} runtime={_format_number(metrics.runtime_seconds)}s "
            f"total_profit={profit} gap={gap} -> {self.destination}",
            flush=True,
        )

    def _disable_after_error(self, error: Exception) -> None:
        self.disabled = True
        self.last_error = f"{type(error).__name__}: {error}"
        print(
            "[incumbent snapshot] disabled after write failure; "
            f"optimization will continue: {self.last_error}",
            file=sys.stderr,
            flush=True,
        )


def snapshot_destination(
    solver_config: dict[str, Any], run_dir: str | Path
) -> Path | None:
    raw = solver_config.get("incumbent_snapshot", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError("solver.incumbent_snapshot must be a mapping")
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError("solver.incumbent_snapshot.enabled must be a boolean")
    if not enabled:
        return None
    filename = str(raw.get("filename", "incumbent_latest.sol")).strip()
    relative = Path(filename)
    if (
        not filename
        or relative.is_absolute()
        or len(relative.parts) != 1
        or relative.suffix.lower() != ".sol"
    ):
        raise ValueError(
            "solver.incumbent_snapshot.filename must be one relative .sol filename"
        )
    return (Path(run_dir) / relative).resolve()
