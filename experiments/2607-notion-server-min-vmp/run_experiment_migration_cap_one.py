#!/usr/bin/env python3
"""Run the experiment with at most one OD migration per VM and scenario.

This is a deliberately separate experimental entry point.  It reuses the
current migration-v3 data, model, and reporting implementation, but requires an
explicit derived-configuration opt-in and adds one linear migration-count cap
for every on-demand VM/scenario pair after the primary model is built.  Keeping
the extension here leaves the primary migration-v3 formulation unchanged.
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


EXPERIMENT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from notion_server_min_vmp import cli as _cli  # noqa: E402


POLICY_ID = "notion_server_min_migration_cap_one_v4_20260724"
BASE_FORMULATION_ID = "notion_server_min_exact_destination_migration_v3_20260723"
EXTENSION_REVISION = "migration_cap_one_v4_20260724"
CONFIG_KEY = "migration.max_count_per_vm_per_scenario"
MAX_COUNT_PER_VM_PER_SCENARIO = 1
CONSTRAINT_PREFIX = "migration_count_cap"
PRIMARY_HARD_AUDIT_TOLERANCE = 1e-5
CONSTRAINT_STRUCTURE_AUDIT_TOLERANCE = 1e-12
FIDELITY_WARNING_KEY = "migration_cap_one_experimental_extension"
FIDELITY_WARNING = (
    "Intentional experimental extension: each on-demand VM may migrate at most "
    "once in each scenario. This cap is not part of the primary migration-v3 "
    "formulation."
)

_ORIGINAL_BUILD_INSTANCE = _cli.build_instance
_ORIGINAL_BUILD_MODEL = _cli.build_model
_ORIGINAL_WRITE_SOLUTION_REPORTS = _cli.write_solution_reports
_ORIGINAL_WRITE_NO_SOLUTION_SUMMARY = _cli.write_no_solution_summary


def _policy_payload(
    *,
    solution_audit_tolerance: float = PRIMARY_HARD_AUDIT_TOLERANCE,
) -> dict[str, Any]:
    return {
        "policy_id": POLICY_ID,
        "base_formulation_id": BASE_FORMULATION_ID,
        "extension_revision": EXTENSION_REVISION,
        "configuration_guard": f"{CONFIG_KEY} is the literal integer 1",
        "max_count_per_vm_per_scenario": MAX_COUNT_PER_VM_PER_SCENARIO,
        "solution_audit_tolerance": float(solution_audit_tolerance),
        "constraint_structure_audit_tolerance": (
            CONSTRAINT_STRUCTURE_AUDIT_TOLERANCE
        ),
        "migration_allowed": True,
        "scope": (
            "each on-demand VM independently in each workload scenario over all "
            "modeled active-slot transitions and destination servers"
        ),
        "mathematical_constraint": (
            "sum_{s in S} sum_{t: t,t+1 in A_i^O} m[i,s,t,xi] <= 1 "
            "for every on-demand VM i and scenario xi"
        ),
        "time_index_semantics": (
            "m[i,s,t,xi] denotes destination-server entry on transition t -> t+1; "
            "the last active slot has no outgoing migration variable"
        ),
        "implementation": (
            "add one linear cap constraint per (on-demand VM, scenario) after "
            "constructing the unchanged primary migration-v3 model"
        ),
        "source_formulation_fidelity": "intentional experimental extension",
        "warning": FIDELITY_WARNING,
    }


def require_migration_cap_one_configuration(config: dict[str, Any]) -> None:
    """Reject accidental use unless the derived config opts into exactly one."""

    migration = config.get("migration")
    configured_value = (
        migration.get("max_count_per_vm_per_scenario")
        if isinstance(migration, dict)
        else None
    )
    if type(configured_value) is not int or configured_value != 1:
        raise ValueError(
            "run_experiment_migration_cap_one.py requires the derived config to "
            f"set {CONFIG_KEY}: 1 (the value must be the literal integer 1)"
        )


def build_instance_migration_cap_one(
    config: dict[str, Any], *args: Any, **kwargs: Any
) -> Any:
    """Build the ordinary instance and record the experimental cap policy."""

    require_migration_cap_one_configuration(config)
    instance = _ORIGINAL_BUILD_INSTANCE(config, *args, **kwargs)
    metadata = instance.metadata
    metadata["migration_policy"] = copy.deepcopy(_policy_payload())
    warnings = metadata.setdefault("fidelity_warnings", {})
    if not isinstance(warnings, dict):
        raise TypeError("instance metadata fidelity_warnings must be a mapping")
    warnings[FIDELITY_WARNING_KEY] = FIDELITY_WARNING
    return instance


def _active_transition_times(periods: list[int] | tuple[int, ...]) -> list[int]:
    """Return the left endpoint of every consecutive active-slot transition."""

    transitions: list[int] = []
    for left, right in zip(periods[:-1], periods[1:]):
        if right != left + 1:
            raise ValueError(
                f"active periods must be consecutive; found {left} -> {right}"
            )
        transitions.append(left)
    return transitions


def _expected_migration_keys(
    artifacts: Any, vm_id: str, scenario_id: int
) -> list[tuple[str, str, int, int]]:
    data = artifacts.data
    return [
        (vm_id, server_id, t, scenario_id)
        for t in _active_transition_times(data.active_od[vm_id])
        for server_id in data.S
    ]


def _add_migration_cap_constraints(artifacts: Any) -> dict[tuple[str, int], Any]:
    """Add and return one exact count-cap constraint for every ``(i, xi)``."""

    try:
        import gurobipy as gp
    except ImportError as exc:  # pragma: no cover - primary builder fails first
        raise RuntimeError("gurobipy is required to build this experiment") from exc

    data = artifacts.data
    model = artifacts.model
    migration = artifacts.variables.get("migration")
    if migration is None:
        raise KeyError("model artifacts do not expose the migration variable mapping")

    constraints: dict[tuple[str, int], Any] = {}
    for vm_id in data.I:
        for scenario_id in data.Xi:
            keys = _expected_migration_keys(artifacts, vm_id, scenario_id)
            missing = [key for key in keys if key not in migration]
            if missing:
                raise KeyError(
                    "primary model is missing migration variables required by the "
                    f"count cap for ({vm_id}, {scenario_id}): {missing[:5]}"
                )
            constraints[vm_id, scenario_id] = model.addConstr(
                gp.quicksum(migration[key] for key in keys)
                <= MAX_COUNT_PER_VM_PER_SCENARIO,
                name=f"{CONSTRAINT_PREFIX}[{vm_id},{scenario_id}]",
            )
    model.update()
    return constraints


def _constraint_audit(
    artifacts: Any,
    constraints: dict[tuple[str, int], Any],
    *,
    tolerance: float = CONSTRAINT_STRUCTURE_AUDIT_TOLERANCE,
) -> dict[str, Any]:
    """Verify the cap rows have exactly the intended variables and coefficients."""

    data = artifacts.data
    model = artifacts.model
    migration = artifacts.variables["migration"]
    pair_rows: list[dict[str, Any]] = []
    invalid_rows: list[str] = []

    for vm_id in data.I:
        for scenario_id in data.Xi:
            constraint = constraints[vm_id, scenario_id]
            expected_keys = _expected_migration_keys(
                artifacts, vm_id, scenario_id
            )
            row = model.getRow(constraint)
            expected_variable_names = {
                str(migration[key].VarName) for key in expected_keys
            }
            observed_coefficients = {
                str(row.getVar(position).VarName): float(row.getCoeff(position))
                for position in range(row.size())
            }
            support_matches = set(observed_coefficients) == expected_variable_names
            coefficients_are_one = all(
                abs(coefficient - 1.0) <= tolerance
                for coefficient in observed_coefficients.values()
            )
            sense_is_leq = str(constraint.Sense) == "<"
            rhs_is_one = (
                abs(float(constraint.RHS) - MAX_COUNT_PER_VM_PER_SCENARIO)
                <= tolerance
            )
            valid = (
                support_matches
                and coefficients_are_one
                and sense_is_leq
                and rhs_is_one
            )
            if not valid:
                invalid_rows.append(str(constraint.ConstrName))
            pair_rows.append(
                {
                    "vm_id": vm_id,
                    "scenario_id": scenario_id,
                    "constraint_name": str(constraint.ConstrName),
                    "active_transition_count": len(
                        _active_transition_times(data.active_od[vm_id])
                    ),
                    "expected_variable_count": len(expected_keys),
                    "observed_variable_count": row.size(),
                    "support_matches": support_matches,
                    "all_coefficients_equal_one": coefficients_are_one,
                    "sense": str(constraint.Sense),
                    "rhs": float(constraint.RHS),
                    "valid": valid,
                }
            )

    audit = {
        "configured_cap": MAX_COUNT_PER_VM_PER_SCENARIO,
        "coefficient_and_rhs_tolerance": float(tolerance),
        "expected_constraint_count": len(data.I) * len(data.Xi),
        "constraint_count": len(constraints),
        "all_constraints_valid": not invalid_rows,
        "invalid_constraint_names": invalid_rows,
        "per_vm_scenario": pair_rows,
    }
    if (
        audit["constraint_count"] != audit["expected_constraint_count"]
        or invalid_rows
    ):
        raise AssertionError(f"migration-count cap constraint audit failed: {audit}")
    return audit


def build_model_migration_cap_one(
    data: Any, *args: Any, **kwargs: Any
) -> Any:
    """Build migration-v3, then cap each VM/scenario migration count at one."""

    artifacts = _ORIGINAL_BUILD_MODEL(data, *args, **kwargs)
    constraints = _add_migration_cap_constraints(artifacts)
    audit = _constraint_audit(artifacts, constraints)
    artifacts.variables["migration_count_cap"] = constraints
    artifacts.variables["migration_count_cap_audit"] = audit
    artifacts.model._migration_count_cap_constraint_audit = copy.deepcopy(audit)
    artifacts.model.update()
    return artifacts


def _solution_audit(
    artifacts: Any, *, tolerance: float
) -> dict[str, Any]:
    """Audit modeled and realized migration counts for every VM/scenario pair."""

    data = artifacts.data
    variables = artifacts.variables
    migration = variables["migration"]
    cap_constraints = variables["migration_count_cap"]
    pair_rows: list[dict[str, Any]] = []
    total_modeled_count = 0.0
    total_actual_count = 0
    maximum_count_definition_residual = 0.0
    maximum_cap_violation = 0.0
    maximum_constraint_slack_residual = 0.0
    violating_pair_count = 0
    pairs_at_cap_count = 0

    for vm_id in data.I:
        for scenario_id in data.Xi:
            transitions = _active_transition_times(data.active_od[vm_id])
            keys = _expected_migration_keys(artifacts, vm_id, scenario_id)
            migration_values = [float(migration[key].X) for key in keys]
            if any(not math.isfinite(value) for value in migration_values):
                raise AssertionError(
                    "a migration variable has a non-finite incumbent value for "
                    f"({vm_id}, {scenario_id})"
                )
            modeled_count = float(sum(migration_values))
            actual_count = 0
            maximum_transition_definition_residual = 0.0
            for t in transitions:
                old_server = next(
                    (
                        server_id
                        for server_id in data.S
                        if float(variables["x"][vm_id, server_id, t, scenario_id].X)
                        > 0.5
                    ),
                    None,
                )
                new_server = next(
                    (
                        server_id
                        for server_id in data.S
                        if float(
                            variables["x"][
                                vm_id, server_id, t + 1, scenario_id
                            ].X
                        )
                        > 0.5
                    ),
                    None,
                )
                actual_change = int(
                    old_server is not None
                    and new_server is not None
                    and old_server != new_server
                )
                transition_value = sum(
                    float(migration[vm_id, server_id, t, scenario_id].X)
                    for server_id in data.S
                )
                actual_count += actual_change
                maximum_transition_definition_residual = max(
                    maximum_transition_definition_residual,
                    abs(transition_value - actual_change),
                )

            count_definition_residual = abs(modeled_count - actual_count)
            cap_violation = max(
                0.0, modeled_count - MAX_COUNT_PER_VM_PER_SCENARIO
            )
            constraint_slack = float(cap_constraints[vm_id, scenario_id].Slack)
            expected_slack = MAX_COUNT_PER_VM_PER_SCENARIO - modeled_count
            constraint_slack_residual = abs(constraint_slack - expected_slack)
            if cap_violation > tolerance:
                violating_pair_count += 1
            if abs(modeled_count - MAX_COUNT_PER_VM_PER_SCENARIO) <= tolerance:
                pairs_at_cap_count += 1
            total_modeled_count += modeled_count
            total_actual_count += actual_count
            maximum_count_definition_residual = max(
                maximum_count_definition_residual,
                count_definition_residual,
                maximum_transition_definition_residual,
            )
            maximum_cap_violation = max(maximum_cap_violation, cap_violation)
            maximum_constraint_slack_residual = max(
                maximum_constraint_slack_residual,
                constraint_slack_residual,
            )
            pair_rows.append(
                {
                    "vm_id": vm_id,
                    "scenario_id": scenario_id,
                    "active_transition_count": len(transitions),
                    "migration_variable_count": len(keys),
                    "modeled_migration_count": modeled_count,
                    "actual_server_change_count": actual_count,
                    "count_definition_residual": count_definition_residual,
                    "maximum_transition_definition_residual": (
                        maximum_transition_definition_residual
                    ),
                    "constraint_slack": constraint_slack,
                    "constraint_slack_residual": constraint_slack_residual,
                    "cap_violation": cap_violation,
                    "within_cap": cap_violation <= tolerance,
                }
            )

    audit = {
        "incumbent_available": True,
        "configured_cap": MAX_COUNT_PER_VM_PER_SCENARIO,
        "tolerance": float(tolerance),
        "vm_scenario_pair_count": len(pair_rows),
        "total_modeled_migration_count": total_modeled_count,
        "total_actual_server_change_count": total_actual_count,
        "pairs_at_cap_count": pairs_at_cap_count,
        "violating_pair_count": violating_pair_count,
        "maximum_count_definition_residual": maximum_count_definition_residual,
        "maximum_constraint_slack_residual": maximum_constraint_slack_residual,
        "maximum_cap_violation": maximum_cap_violation,
        "per_vm_scenario": pair_rows,
    }
    failures = {
        "count definition residual": maximum_count_definition_residual,
        "constraint slack residual": maximum_constraint_slack_residual,
        "cap violation": maximum_cap_violation,
    }
    failed = {name: value for name, value in failures.items() if value > tolerance}
    if violating_pair_count or failed:
        raise AssertionError(
            "migration-count cap solution audit failed: "
            f"violating pairs={violating_pair_count}, metrics={failed}"
        )
    return audit


def _constraint_audit_from_model(model: Any) -> dict[str, Any]:
    audit = getattr(model, "_migration_count_cap_constraint_audit", None)
    if not isinstance(audit, dict):
        raise AssertionError(
            "migration-cap model does not expose its constraint-construction audit"
        )
    return copy.deepcopy(audit)


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically replace one JSON artifact in its destination directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _final_output_paths(output: Path) -> tuple[Path, Path, Path]:
    return (
        output / "summary.json",
        output / "model_validation_diagnostics.json",
        output / "migration_policy.json",
    )


def _clear_final_outputs(output: Path) -> None:
    """Remove stale or partially finalized policy-report artifacts."""

    for path in _final_output_paths(output):
        path.unlink(missing_ok=True)


def _finalize_policy_report(
    run_dir: str | Path,
    summary: dict[str, Any],
    *,
    constraint_audit: dict[str, Any],
    solution_audit: dict[str, Any],
) -> dict[str, Any]:
    """Write diagnostics and policy first, then publish final summary last."""

    output = Path(run_dir)
    output.mkdir(parents=True, exist_ok=True)
    audit_tolerance = float(
        solution_audit.get("tolerance", PRIMARY_HARD_AUDIT_TOLERANCE)
    )
    report = _policy_payload(solution_audit_tolerance=audit_tolerance)
    report["constraint_audit"] = constraint_audit
    report["solution_audit"] = solution_audit
    summary["migration_policy"] = report

    diagnostics = summary.setdefault("model_validation_diagnostics", {})
    if not isinstance(diagnostics, dict):
        raise TypeError(
            "solution summary model_validation_diagnostics must be a mapping"
        )
    diagnostics["migration_count_cap_constraint"] = constraint_audit
    diagnostics["migration_count_cap_solution"] = solution_audit

    operations = summary.get("operations")
    if isinstance(operations, dict):
        operations["migration_allowed"] = True
        operations["max_migration_count_per_vm_per_scenario"] = (
            MAX_COUNT_PER_VM_PER_SCENARIO
        )

    summary_path, diagnostics_path, policy_path = _final_output_paths(output)
    _atomic_write_json(diagnostics_path, diagnostics)
    _atomic_write_json(policy_path, report)
    # A summary is the completion marker used by the sweep runner. Publish it
    # only after both standalone cap artifacts have been finalized.
    _atomic_write_json(summary_path, summary)
    return summary


def write_solution_reports_migration_cap_one(
    artifacts: Any,
    run_dir: str | Path,
    *,
    tolerance: float = PRIMARY_HARD_AUDIT_TOLERANCE,
) -> dict[str, Any]:
    """Audit the cap, write primary reports, and safely finalize cap reports."""

    output = Path(run_dir)
    _clear_final_outputs(output)
    constraint_audit = copy.deepcopy(
        artifacts.variables["migration_count_cap_audit"]
    )
    try:
        # Validate the formulation-specific incumbent before the primary
        # reporter writes any completion marker.
        solution_audit = _solution_audit(artifacts, tolerance=tolerance)
        summary = _ORIGINAL_WRITE_SOLUTION_REPORTS(
            artifacts, run_dir, tolerance=tolerance
        )
        # The primary reporter writes its own summary. Remove that intermediate
        # marker before cap-specific finalization can fail.
        (output / "summary.json").unlink(missing_ok=True)
        return _finalize_policy_report(
            run_dir,
            summary,
            constraint_audit=constraint_audit,
            solution_audit=solution_audit,
        )
    except Exception:
        _clear_final_outputs(output)
        raise


def write_no_solution_summary_migration_cap_one(
    model: Any,
    run_dir: str | Path,
) -> dict[str, Any]:
    """Persist the cap policy and constraint audit when there is no incumbent."""

    output = Path(run_dir)
    _clear_final_outputs(output)
    try:
        constraint_audit = _constraint_audit_from_model(model)
        solution_audit = {
            "incumbent_available": False,
            "configured_cap": MAX_COUNT_PER_VM_PER_SCENARIO,
            "tolerance": PRIMARY_HARD_AUDIT_TOLERANCE,
            "audit_status": "not_applicable_without_incumbent",
        }
        summary = _ORIGINAL_WRITE_NO_SOLUTION_SUMMARY(model, run_dir)
        (output / "summary.json").unlink(missing_ok=True)
        return _finalize_policy_report(
            run_dir,
            summary,
            constraint_audit=constraint_audit,
            solution_audit=solution_audit,
        )
    except Exception:
        _clear_final_outputs(output)
        raise


@contextmanager
def _patched_cli() -> Iterator[None]:
    """Install process-local CLI overrides and restore them after invocation."""

    replacements = {
        "build_instance": build_instance_migration_cap_one,
        "build_model": build_model_migration_cap_one,
        "write_solution_reports": write_solution_reports_migration_cap_one,
        "write_no_solution_summary": write_no_solution_summary_migration_cap_one,
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
