#!/usr/bin/env python3
"""Run the experiment with all on-demand migration explicitly prohibited.

This is a deliberately separate diagnostic entry point.  It reuses the current
data/model/reporting implementation, but requires an explicit opt-in in the
derived configuration and fixes every migration variable to zero after model
construction.  Keeping the override here prevents the primary formulation from
silently changing.
"""

from __future__ import annotations

import copy
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


EXPERIMENT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from notion_server_min_vmp import cli as _cli  # noqa: E402
from notion_server_min_vmp.data import MEM  # noqa: E402


POLICY_ID = "on_demand_migration_fixed_zero_v1"
FIDELITY_WARNING_KEY = "migration_fixed_zero_experimental_override"
FIDELITY_WARNING = (
    "Intentional diagnostic override: all on-demand migration variables are fixed "
    "to zero. This run does not implement the source formulation's migration option."
)

_ORIGINAL_BUILD_INSTANCE = _cli.build_instance
_ORIGINAL_BUILD_MODEL = _cli.build_model
_ORIGINAL_WRITE_SOLUTION_REPORTS = _cli.write_solution_reports
_ORIGINAL_WRITE_NO_SOLUTION_SUMMARY = _cli.write_no_solution_summary


def _policy_payload() -> dict[str, Any]:
    return {
        "policy_id": POLICY_ID,
        "configuration_guard": "migration.fixed_zero is literal true",
        "fixed_zero": True,
        "migration_allowed": False,
        "scope": (
            "all destination-indexed on-demand migration decision variables in "
            "every transition and scenario"
        ),
        "implementation": "set every migration variable LB=UB=0 after primary model construction",
        "source_formulation_fidelity": "intentional diagnostic override",
        "warning": FIDELITY_WARNING,
    }


def require_fixed_zero_configuration(config: dict[str, Any]) -> None:
    """Reject accidental use unless the derived config literally opts in."""

    migration = config.get("migration")
    configured_value = migration.get("fixed_zero") if isinstance(migration, dict) else None
    if configured_value is not True:
        raise ValueError(
            "run_experiment_no_migration.py requires the derived config to set "
            "migration.fixed_zero: true (the value must be the YAML boolean true)"
        )


def build_instance_no_migration(
    config: dict[str, Any], *args: Any, **kwargs: Any
) -> Any:
    """Build the ordinary instance and annotate the intentional fidelity change."""

    require_fixed_zero_configuration(config)
    instance = _ORIGINAL_BUILD_INSTANCE(config, *args, **kwargs)
    metadata = instance.metadata
    metadata["migration_policy"] = copy.deepcopy(_policy_payload())
    warnings = metadata.setdefault("fidelity_warnings", {})
    if not isinstance(warnings, dict):
        raise TypeError("instance metadata fidelity_warnings must be a mapping")
    warnings[FIDELITY_WARNING_KEY] = FIDELITY_WARNING
    return instance


def _migration_variables(artifacts: Any) -> list[Any]:
    variables = artifacts.variables.get("migration")
    if variables is None:
        raise KeyError("model artifacts do not expose the migration variable mapping")
    return list(variables.values())


def build_model_no_migration(data: Any, *args: Any, **kwargs: Any) -> Any:
    """Build the primary model, then prohibit every modeled migration."""

    artifacts = _ORIGINAL_BUILD_MODEL(data, *args, **kwargs)
    migration_variables = _migration_variables(artifacts)
    for variable in migration_variables:
        variable.LB = 0.0
        variable.UB = 0.0
    artifacts.model.update()
    return artifacts


def _bound_audit_from_variables(variables: list[Any], *, tolerance: float) -> dict[str, Any]:
    maximum_bound_deviation = max(
        (
            max(abs(float(variable.LB)), abs(float(variable.UB)))
            for variable in variables
        ),
        default=0.0,
    )
    audit = {
        "fixed_variable_count": len(variables),
        "all_migration_bounds_fixed_zero": maximum_bound_deviation <= tolerance,
        "maximum_migration_bound_deviation_from_zero": maximum_bound_deviation,
    }
    if not audit["all_migration_bounds_fixed_zero"]:
        raise AssertionError(
            "no-migration entry point found a migration variable whose bounds were not "
            f"fixed to zero (max deviation={maximum_bound_deviation})"
        )
    return audit


def _solution_audit(artifacts: Any, *, tolerance: float) -> dict[str, Any]:
    data = artifacts.data
    variables = artifacts.variables
    migration_mapping = variables["migration"]
    migration_variables = list(migration_mapping.values())
    audit = _bound_audit_from_variables(migration_variables, tolerance=tolerance)

    values = [float(variable.X) for variable in migration_variables]
    if any(not math.isfinite(value) for value in values):
        raise AssertionError("a migration variable has a non-finite incumbent value")

    actual_server_change_count = 0
    maximum_adjacent_assignment_change = 0.0
    migration_energy_kwh_sum = 0.0
    transition_keys = sorted(
        {
            (vm_id, t, scenario_id)
            for vm_id, _server_id, t, scenario_id in migration_mapping
        }
    )
    for vm_id, t, scenario_id in transition_keys:
        transition_change = max(
            abs(
                float(variables["x"][vm_id, server_id, t + 1, scenario_id].X)
                - float(variables["x"][vm_id, server_id, t, scenario_id].X)
            )
            for server_id in data.S
        )
        maximum_adjacent_assignment_change = max(
            maximum_adjacent_assignment_change, transition_change
        )
        if transition_change > tolerance:
            actual_server_change_count += 1
        transition_migration_value = sum(
            float(migration_mapping[vm_id, server_id, t, scenario_id].X)
            for server_id in data.S
        )
        migration_energy_kwh_sum += (
            data.c_mig
            * data.load_od[vm_id, MEM, t, scenario_id]
            * transition_migration_value
        )

    relaxed_value_sum = sum(values)
    maximum_absolute_value = max((abs(value) for value in values), default=0.0)
    audit.update(
        {
            "incumbent_available": True,
            "migration_relaxed_value_sum": relaxed_value_sum,
            "maximum_absolute_migration_value": maximum_absolute_value,
            "actual_server_change_count": actual_server_change_count,
            "maximum_adjacent_assignment_change": maximum_adjacent_assignment_change,
            "migration_energy_kwh_sum": migration_energy_kwh_sum,
        }
    )
    violations = {
        "migration value": maximum_absolute_value,
        "adjacent assignment change": maximum_adjacent_assignment_change,
        "migration energy": abs(migration_energy_kwh_sum),
    }
    failed = {name: value for name, value in violations.items() if value > tolerance}
    if actual_server_change_count or failed:
        raise AssertionError(
            "no-migration solution audit failed: "
            f"actual changes={actual_server_change_count}, nonzero metrics={failed}"
        )
    return audit


def _write_policy_report(
    run_dir: str | Path,
    summary: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    output = Path(run_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = _policy_payload()
    report["audit"] = audit
    summary["migration_policy"] = report
    diagnostics = summary.setdefault("model_validation_diagnostics", {})
    if not isinstance(diagnostics, dict):
        raise TypeError("solution summary model_validation_diagnostics must be a mapping")
    diagnostics["no_migration_policy"] = audit
    operations = summary.get("operations")
    if isinstance(operations, dict):
        operations["migration_allowed"] = False
    serialized = json.dumps(summary, indent=2, ensure_ascii=False)
    (output / "summary.json").write_text(serialized, encoding="utf-8")
    (output / "migration_policy.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def write_solution_reports_no_migration(
    artifacts: Any,
    run_dir: str | Path,
    *,
    tolerance: float = 1e-7,
) -> dict[str, Any]:
    """Write ordinary reports, then fail unless migration is exactly absent."""

    summary = _ORIGINAL_WRITE_SOLUTION_REPORTS(
        artifacts, run_dir, tolerance=tolerance
    )
    audit = _solution_audit(artifacts, tolerance=tolerance)

    diagnostics = summary.get("model_validation_diagnostics", {})
    reported_checks = {
        "migration_relaxed_value_sum": float(
            diagnostics.get("migration_relaxed_value_sum", math.inf)
        ),
        "actual_server_change_count": float(
            diagnostics.get("actual_server_change_count", math.inf)
        ),
    }
    nonzero_reported = {
        name: value
        for name, value in reported_checks.items()
        if not math.isfinite(value) or abs(value) > tolerance
    }
    if nonzero_reported:
        raise AssertionError(
            "primary solution report disagrees with the no-migration policy: "
            f"{nonzero_reported}"
        )
    return _write_policy_report(run_dir, summary, audit)


def write_no_solution_summary_no_migration(
    model: Any,
    run_dir: str | Path,
    *,
    tolerance: float = 1e-7,
) -> dict[str, Any]:
    """Persist the fixed-zero policy even when Gurobi finds no incumbent."""

    summary = _ORIGINAL_WRITE_NO_SOLUTION_SUMMARY(model, run_dir)
    migration_variables = [
        variable
        for variable in model.getVars()
        if str(variable.VarName).startswith("migration[")
    ]
    audit = _bound_audit_from_variables(migration_variables, tolerance=tolerance)
    audit["incumbent_available"] = False
    return _write_policy_report(run_dir, summary, audit)


@contextmanager
def _patched_cli() -> Iterator[None]:
    """Install process-local CLI overrides and restore them after invocation."""

    replacements = {
        "build_instance": build_instance_no_migration,
        "build_model": build_model_no_migration,
        "write_solution_reports": write_solution_reports_no_migration,
        "write_no_solution_summary": write_no_solution_summary_no_migration,
    }
    previous = {name: getattr(_cli, name) for name in replacements}
    try:
        for name, replacement in replacements.items():
            setattr(_cli, name, replacement)
        yield
    finally:
        for name, original in previous.items():
            setattr(_cli, name, original)


def main(argv: list[str] | None = None) -> int:
    with _patched_cli():
        return _cli.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
