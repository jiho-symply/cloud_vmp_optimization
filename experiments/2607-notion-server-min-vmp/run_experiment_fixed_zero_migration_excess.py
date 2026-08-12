#!/usr/bin/env python3
"""Run the isolated fixed-zero migration and OD-excess experiment.

The primary migration-v3 formulation remains unchanged.  This entry point
requires an explicit two-part configuration opt-in, builds the ordinary model,
then fixes every destination-indexed migration variable and every positive
OD-excess branch indicator to zero.  Fixing the branch indicator to zero
activates the existing indicator rows that require OD CPU demand not to exceed
online capacity and require the corresponding excess variable to equal zero.
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


POLICY_ID = "notion_server_min_fixed_zero_migration_excess_v5_20260724"
BASE_FORMULATION_ID = "notion_server_min_exact_destination_migration_v3_20260723"
EXTENSION_REVISION = "fixed_zero_migration_excess_v5_20260724"
MIGRATION_CONFIG_KEY = "migration.fixed_zero"
EXCESS_BRANCH_CONFIG_KEY = "excess_load_indicator.fixed_zero"
BOUND_AUDIT_TOLERANCE = 1e-12
PRIMARY_HARD_AUDIT_TOLERANCE = 1e-5
FIDELITY_WARNING_KEY = "fixed_zero_migration_excess_experimental_override"
FIDELITY_WARNING = (
    "Intentional experimental restriction: all on-demand migration variables "
    "and all positive OD CPU-excess branch indicators are fixed to zero. This "
    "is not the unrestricted primary migration-v3 formulation."
)

_ORIGINAL_BUILD_INSTANCE = _cli.build_instance
_ORIGINAL_BUILD_MODEL = _cli.build_model
_ORIGINAL_WRITE_SOLUTION_REPORTS = _cli.write_solution_reports
_ORIGINAL_WRITE_NO_SOLUTION_SUMMARY = _cli.write_no_solution_summary


def _policy_payload(
    *,
    bound_audit_tolerance: float = BOUND_AUDIT_TOLERANCE,
    solution_audit_tolerance: float = PRIMARY_HARD_AUDIT_TOLERANCE,
) -> dict[str, Any]:
    return {
        "policy_id": POLICY_ID,
        "base_formulation_id": BASE_FORMULATION_ID,
        "extension_revision": EXTENSION_REVISION,
        "configuration_guard": (
            f"{MIGRATION_CONFIG_KEY} and {EXCESS_BRANCH_CONFIG_KEY} are both "
            "the literal YAML boolean true"
        ),
        "migration_fixed_zero": True,
        "excess_load_indicator_fixed_zero": True,
        "migration_allowed": False,
        "excess_load_allowed": False,
        "bound_audit_tolerance": float(bound_audit_tolerance),
        "solution_audit_tolerance": float(solution_audit_tolerance),
        "scope": {
            "migration": (
                "all destination-indexed on-demand migration variables over all "
                "modeled VM transitions, destination servers, and scenarios"
            ),
            "excess_load_indicator": (
                "all excess_positive_branch server-time-scenario indicators"
            ),
        },
        "implementation": (
            "after constructing the unchanged primary model, set LB=UB=0 on "
            "every migration[...] and excess_positive_branch[...] variable"
        ),
        "excess_branch_zero_semantics": (
            "branch=0 activates the existing zero-branch indicators imposing "
            "OD CPU demand <= online CPU capacity and od_cpu_excess == 0"
        ),
        "source_formulation_fidelity": "intentional experimental restriction",
        "warning": FIDELITY_WARNING,
    }


def require_fixed_zero_migration_excess_configuration(
    config: dict[str, Any],
) -> None:
    """Require both fixed-zero flags to be literal YAML booleans."""

    migration = config.get("migration")
    migration_value = (
        migration.get("fixed_zero") if isinstance(migration, dict) else None
    )
    if migration_value is not True:
        raise ValueError(
            "run_experiment_fixed_zero_migration_excess.py requires "
            f"{MIGRATION_CONFIG_KEY}: true (the value must be the literal YAML "
            "boolean true)"
        )

    excess_indicator = config.get("excess_load_indicator")
    excess_indicator_value = (
        excess_indicator.get("fixed_zero")
        if isinstance(excess_indicator, dict)
        else None
    )
    if excess_indicator_value is not True:
        raise ValueError(
            "run_experiment_fixed_zero_migration_excess.py requires "
            f"{EXCESS_BRANCH_CONFIG_KEY}: true (the value must be the literal "
            "YAML boolean true)"
        )


def build_instance_fixed_zero_migration_excess(
    config: dict[str, Any], *args: Any, **kwargs: Any
) -> Any:
    """Build the ordinary instance and record the experimental restriction."""

    require_fixed_zero_migration_excess_configuration(config)
    instance = _ORIGINAL_BUILD_INSTANCE(config, *args, **kwargs)
    metadata = instance.metadata
    metadata["fixed_zero_policy"] = copy.deepcopy(_policy_payload())
    warnings = metadata.setdefault("fidelity_warnings", {})
    if not isinstance(warnings, dict):
        raise TypeError("instance metadata fidelity_warnings must be a mapping")
    warnings[FIDELITY_WARNING_KEY] = FIDELITY_WARNING
    return instance


def _variable_group(artifacts: Any, key: str) -> list[Any]:
    mapping = artifacts.variables.get(key)
    if mapping is None:
        raise KeyError(f"model artifacts do not expose the {key!r} variable mapping")
    if not hasattr(mapping, "values"):
        raise TypeError(f"model artifacts variable group {key!r} is not a mapping")
    return list(mapping.values())


def _zero_bound_group_audit(
    variables: list[Any],
    *,
    group: str,
    tolerance: float,
) -> dict[str, Any]:
    deviations: list[float] = []
    for variable in variables:
        lower_bound = float(variable.LB)
        upper_bound = float(variable.UB)
        if not math.isfinite(lower_bound) or not math.isfinite(upper_bound):
            deviations.append(math.inf)
        else:
            deviations.append(max(abs(lower_bound), abs(upper_bound)))
    maximum_deviation = max(deviations, default=0.0)
    return {
        "variable_group": group,
        "fixed_variable_count": len(variables),
        "all_bounds_fixed_zero": maximum_deviation <= tolerance,
        "maximum_bound_deviation_from_zero": maximum_deviation,
    }


def _bound_audit_from_variables(
    migration_variables: list[Any],
    excess_branch_variables: list[Any],
    *,
    tolerance: float = BOUND_AUDIT_TOLERANCE,
) -> dict[str, Any]:
    migration_audit = _zero_bound_group_audit(
        migration_variables,
        group="migration",
        tolerance=tolerance,
    )
    excess_branch_audit = _zero_bound_group_audit(
        excess_branch_variables,
        group="excess_branch",
        tolerance=tolerance,
    )
    all_fixed = bool(
        migration_audit["all_bounds_fixed_zero"]
        and excess_branch_audit["all_bounds_fixed_zero"]
    )
    audit = {
        "tolerance": float(tolerance),
        "all_fixed_bounds_zero": all_fixed,
        "all_required_bounds_fixed_zero": all_fixed,
        "migration": migration_audit,
        "excess_branch": excess_branch_audit,
    }
    if not all_fixed:
        raise AssertionError(
            "fixed-zero model found a required variable whose bounds were not "
            f"fixed to zero: {audit}"
        )
    return audit


def build_model_fixed_zero_migration_excess(
    data: Any, *args: Any, **kwargs: Any
) -> Any:
    """Build migration-v3, then fix migration and excess branches to zero."""

    artifacts = _ORIGINAL_BUILD_MODEL(data, *args, **kwargs)
    migration_variables = _variable_group(artifacts, "migration")
    excess_branch_variables = _variable_group(artifacts, "excess_branch")
    for variable in (*migration_variables, *excess_branch_variables):
        variable.LB = 0.0
        variable.UB = 0.0
    artifacts.model.update()

    bound_audit = _bound_audit_from_variables(
        migration_variables,
        excess_branch_variables,
    )
    artifacts.variables["fixed_zero_bound_audit"] = bound_audit
    artifacts.model._fixed_zero_migration_excess_bound_audit = copy.deepcopy(
        bound_audit
    )
    return artifacts


def _finite_incumbent_values(variables: list[Any], *, group: str) -> list[float]:
    values = [float(variable.X) for variable in variables]
    if any(not math.isfinite(value) for value in values):
        raise AssertionError(f"a {group} variable has a non-finite incumbent value")
    return values


def _solution_audit(
    artifacts: Any,
    *,
    tolerance: float = PRIMARY_HARD_AUDIT_TOLERANCE,
) -> dict[str, Any]:
    """Independently verify both fixed-zero restrictions in an incumbent."""

    data = artifacts.data
    variables = artifacts.variables
    migration_mapping = variables.get("migration")
    excess_branch_mapping = variables.get("excess_branch")
    excess_mapping = variables.get("excess")
    if migration_mapping is None:
        raise KeyError("model artifacts do not expose the migration variable mapping")
    if excess_branch_mapping is None:
        raise KeyError("model artifacts do not expose the excess_branch variable mapping")
    if excess_mapping is None:
        raise KeyError("model artifacts do not expose the excess variable mapping")
    if set(excess_branch_mapping) != set(excess_mapping):
        raise AssertionError(
            "excess_branch and excess mappings do not have identical "
            "server-time-scenario support"
        )

    migration_values = _finite_incumbent_values(
        list(migration_mapping.values()), group="migration"
    )
    excess_branch_values = _finite_incumbent_values(
        list(excess_branch_mapping.values()), group="excess_branch"
    )
    excess_values = _finite_incumbent_values(
        list(excess_mapping.values()), group="excess"
    )

    actual_server_change_count = 0
    maximum_adjacent_assignment_change = 0.0
    transition_keys = {
        (vm_id, t, scenario_id)
        for vm_id, _server_id, t, scenario_id in migration_mapping
    }
    for vm_id, t, scenario_id in transition_keys:
        transition_change = max(
            (
                abs(
                    float(
                        variables["x"][
                            vm_id, server_id, t + 1, scenario_id
                        ].X
                    )
                    - float(
                        variables["x"][vm_id, server_id, t, scenario_id].X
                    )
                )
                for server_id in data.S
            ),
            default=0.0,
        )
        if not math.isfinite(transition_change):
            raise AssertionError(
                "an on-demand assignment has a non-finite incumbent value"
            )
        maximum_adjacent_assignment_change = max(
            maximum_adjacent_assignment_change, transition_change
        )
        if transition_change > tolerance:
            actual_server_change_count += 1

    independently_computed_expected_excess = sum(
        float(data.p[scenario_id])
        * sum(
            float(excess_mapping[server_id, t, scenario_id].X)
            for server_id in data.S
            for t in data.T
        )
        for scenario_id in data.Xi
    )
    if not math.isfinite(independently_computed_expected_excess):
        raise AssertionError("expected OD CPU excess is non-finite")

    audit = {
        "incumbent_available": True,
        "tolerance": float(tolerance),
        "migration_variable_count": len(migration_values),
        "migration_value_sum": float(sum(migration_values)),
        "maximum_absolute_migration_value": max(
            (abs(value) for value in migration_values), default=0.0
        ),
        "actual_server_change_count": actual_server_change_count,
        "maximum_adjacent_assignment_change": (
            maximum_adjacent_assignment_change
        ),
        "excess_branch_variable_count": len(excess_branch_values),
        "excess_branch_value_sum": float(sum(excess_branch_values)),
        "maximum_absolute_excess_branch_value": max(
            (abs(value) for value in excess_branch_values), default=0.0
        ),
        "excess_variable_count": len(excess_values),
        "excess_value_sum": float(sum(excess_values)),
        "maximum_absolute_excess_value": max(
            (abs(value) for value in excess_values), default=0.0
        ),
        "independently_computed_expected_od_cpu_excess": (
            independently_computed_expected_excess
        ),
    }
    violations = {
        "migration value": audit["maximum_absolute_migration_value"],
        "adjacent assignment change": maximum_adjacent_assignment_change,
        "excess branch value": audit["maximum_absolute_excess_branch_value"],
        "excess value": audit["maximum_absolute_excess_value"],
        "expected OD CPU excess": abs(independently_computed_expected_excess),
    }
    failed = {name: value for name, value in violations.items() if value > tolerance}
    if actual_server_change_count or failed:
        raise AssertionError(
            "fixed-zero migration/excess solution audit failed: "
            f"actual changes={actual_server_change_count}, nonzero metrics={failed}"
        )
    return audit


def _reported_number(container: dict[str, Any], key: str, *, scope: str) -> float:
    try:
        value = float(container[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise AssertionError(
            f"primary solution report is missing numeric {scope}.{key}"
        ) from exc
    if not math.isfinite(value):
        raise AssertionError(f"primary solution report has non-finite {scope}.{key}")
    return value


def _audit_primary_report(
    summary: dict[str, Any],
    solution_audit: dict[str, Any],
    *,
    tolerance: float,
) -> None:
    """Require primary aggregate reports to agree with the direct audit."""

    service = summary.get("service")
    operations = summary.get("operations")
    diagnostics = summary.get("model_validation_diagnostics")
    if not isinstance(service, dict):
        raise AssertionError("primary solution report has no service mapping")
    if not isinstance(operations, dict):
        raise AssertionError("primary solution report has no operations mapping")
    if not isinstance(diagnostics, dict):
        raise AssertionError(
            "primary solution report has no model_validation_diagnostics mapping"
        )

    reported = {
        "reported_expected_od_cpu_excess": _reported_number(
            service, "expected_od_cpu_excess", scope="service"
        ),
        "reported_expected_od_cpu_excess_rate": _reported_number(
            service, "expected_od_cpu_excess_rate", scope="service"
        ),
        "reported_operations_actual_server_change_count": _reported_number(
            operations, "actual_server_change_count", scope="operations"
        ),
        "reported_migration_relaxed_value_sum": _reported_number(
            diagnostics,
            "migration_relaxed_value_sum",
            scope="model_validation_diagnostics",
        ),
        "reported_diagnostics_actual_server_change_count": _reported_number(
            diagnostics,
            "actual_server_change_count",
            scope="model_validation_diagnostics",
        ),
    }
    nonzero = {
        name: value for name, value in reported.items() if abs(value) > tolerance
    }
    if nonzero:
        raise AssertionError(
            "primary solution report disagrees with the fixed-zero policy: "
            f"{nonzero}"
        )
    computed_expected = float(
        solution_audit["independently_computed_expected_od_cpu_excess"]
    )
    if (
        abs(reported["reported_expected_od_cpu_excess"] - computed_expected)
        > tolerance
    ):
        raise AssertionError(
            "reported expected OD CPU excess disagrees with the direct audit"
        )
    solution_audit.update(reported)


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
        output / "fixed_zero_policy.json",
    )


def _clear_final_outputs(output: Path) -> None:
    """Remove stale or partially finalized policy-report artifacts."""

    for path in _final_output_paths(output):
        path.unlink(missing_ok=True)


@contextmanager
def _defer_primary_summary_write(reporter: Any) -> Iterator[None]:
    """Let the primary reporter write artifacts without publishing completion.

    The shared sweep treats ``summary.json`` as the completion marker.  The
    primary reporter normally writes that file before this adapter can perform
    its fixed-zero cross-checks, so intercept only that write and leave every
    other primary artifact untouched.  The audited wrapper publishes the final
    summary atomically after diagnostics and policy artifacts are complete.
    """

    reporter_globals = getattr(reporter, "__globals__", None)
    if not isinstance(reporter_globals, dict):
        raise TypeError(
            "primary reporter does not expose a mutable globals mapping"
        )
    original_write_json = reporter_globals.get("_write_json")
    if not callable(original_write_json):
        raise TypeError("primary reporter does not expose its JSON writer")

    def write_json_without_completion_marker(path: Path, payload: Any) -> None:
        if Path(path).name != "summary.json":
            original_write_json(path, payload)

    reporter_globals["_write_json"] = write_json_without_completion_marker
    try:
        yield
    finally:
        reporter_globals["_write_json"] = original_write_json


def _finalize_policy_report(
    run_dir: str | Path,
    summary: dict[str, Any],
    *,
    bound_audit: dict[str, Any],
    solution_audit: dict[str, Any],
) -> dict[str, Any]:
    """Write diagnostics and policy first, then publish summary as the marker."""

    output = Path(run_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = _policy_payload(
        bound_audit_tolerance=float(
            bound_audit.get("tolerance", BOUND_AUDIT_TOLERANCE)
        ),
        solution_audit_tolerance=float(
            solution_audit.get("tolerance", PRIMARY_HARD_AUDIT_TOLERANCE)
        ),
    )
    report["bound_audit"] = bound_audit
    report["solution_audit"] = solution_audit
    report["audit"] = {"bounds": bound_audit, "solution": solution_audit}
    summary["fixed_zero_policy"] = report

    diagnostics = summary.setdefault("model_validation_diagnostics", {})
    if not isinstance(diagnostics, dict):
        raise TypeError(
            "solution summary model_validation_diagnostics must be a mapping"
        )
    diagnostics["fixed_zero_migration_excess_bounds"] = bound_audit
    diagnostics["fixed_zero_migration_excess_solution"] = solution_audit

    operations = summary.get("operations")
    if isinstance(operations, dict):
        operations["migration_allowed"] = False
        operations["excess_load_allowed"] = False
        operations["excess_load_indicator_fixed_zero"] = True

    summary_path, diagnostics_path, policy_path = _final_output_paths(output)
    _atomic_write_json(diagnostics_path, diagnostics)
    _atomic_write_json(policy_path, report)
    # The sweep runner treats summary.json as completion.  Publish it only
    # after the two standalone policy artifacts are safely finalized.
    _atomic_write_json(summary_path, summary)
    return summary


def write_solution_reports_fixed_zero_migration_excess(
    artifacts: Any,
    run_dir: str | Path,
    *,
    tolerance: float = PRIMARY_HARD_AUDIT_TOLERANCE,
) -> dict[str, Any]:
    """Audit the incumbent, then atomically finalize combined policy reports."""

    output = Path(run_dir)
    _clear_final_outputs(output)
    try:
        bound_audit = _bound_audit_from_variables(
            _variable_group(artifacts, "migration"),
            _variable_group(artifacts, "excess_branch"),
        )
        # Validate the direct incumbent values before the primary reporter can
        # publish its summary completion marker.
        solution_audit = _solution_audit(artifacts, tolerance=tolerance)
        with _defer_primary_summary_write(_ORIGINAL_WRITE_SOLUTION_REPORTS):
            summary = _ORIGINAL_WRITE_SOLUTION_REPORTS(
                artifacts, run_dir, tolerance=tolerance
            )
        _audit_primary_report(
            summary,
            solution_audit,
            tolerance=tolerance,
        )
        return _finalize_policy_report(
            run_dir,
            summary,
            bound_audit=bound_audit,
            solution_audit=solution_audit,
        )
    except Exception:
        _clear_final_outputs(output)
        raise


def _bound_audit_from_model(model: Any) -> dict[str, Any]:
    migration_variables = [
        variable
        for variable in model.getVars()
        if str(variable.VarName).startswith("migration[")
    ]
    excess_branch_variables = [
        variable
        for variable in model.getVars()
        if str(variable.VarName).startswith("excess_positive_branch[")
    ]
    audit = _bound_audit_from_variables(
        migration_variables,
        excess_branch_variables,
    )
    construction_audit = getattr(
        model, "_fixed_zero_migration_excess_bound_audit", None
    )
    if isinstance(construction_audit, dict):
        for group in ("migration", "excess_branch"):
            expected_count = int(
                construction_audit[group]["fixed_variable_count"]
            )
            observed_count = int(audit[group]["fixed_variable_count"])
            if observed_count != expected_count:
                raise AssertionError(
                    "fixed-zero no-incumbent audit found an unexpected number "
                    f"of {group} variables: expected={expected_count}, "
                    f"observed={observed_count}"
                )
    return audit


def write_no_solution_summary_fixed_zero_migration_excess(
    model: Any,
    run_dir: str | Path,
) -> dict[str, Any]:
    """Persist the bound policy and explicit no-incumbent audit status."""

    output = Path(run_dir)
    _clear_final_outputs(output)
    try:
        bound_audit = _bound_audit_from_model(model)
        solution_audit = {
            "incumbent_available": False,
            "audit_status": "not_applicable_without_incumbent",
        }
        with _defer_primary_summary_write(_ORIGINAL_WRITE_NO_SOLUTION_SUMMARY):
            summary = _ORIGINAL_WRITE_NO_SOLUTION_SUMMARY(model, run_dir)
        return _finalize_policy_report(
            run_dir,
            summary,
            bound_audit=bound_audit,
            solution_audit=solution_audit,
        )
    except Exception:
        _clear_final_outputs(output)
        raise


@contextmanager
def _patched_cli() -> Iterator[None]:
    """Install process-local CLI overrides and restore them after invocation."""

    replacements = {
        "build_instance": build_instance_fixed_zero_migration_excess,
        "build_model": build_model_fixed_zero_migration_excess,
        "write_solution_reports": write_solution_reports_fixed_zero_migration_excess,
        "write_no_solution_summary": (
            write_no_solution_summary_fixed_zero_migration_excess
        ),
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
