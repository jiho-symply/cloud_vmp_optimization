#!/usr/bin/env python3
"""Run the cumulative fixed-zero migration, excess, and Spot experiment.

The ordinary migration-v3 model is built unchanged and then every migration
variable, positive OD-excess branch indicator, and Spot preempted-state
variable is fixed to zero.  The dedicated configuration guards and result
audits keep this experimental restriction isolated from the primary model.
"""

from __future__ import annotations

import copy
import importlib.util
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator


EXPERIMENT_ROOT = Path(__file__).resolve().parent
V5_ENTRYPOINT = EXPERIMENT_ROOT / "run_experiment_fixed_zero_migration_excess.py"


def _load_v5_entrypoint() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "notion_server_min_fixed_zero_migration_excess_v5_base",
        V5_ENTRYPOINT,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load cumulative fixed-zero base: {V5_ENTRYPOINT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V5 = _load_v5_entrypoint()
_cli = V5._cli

POLICY_ID = (
    "notion_server_min_fixed_zero_migration_excess_preemption_v6_20260725"
)
BASE_FORMULATION_ID = "notion_server_min_exact_destination_migration_v3_20260723"
EXTENSION_REVISION = "fixed_zero_migration_excess_preemption_v6_20260725"
MIGRATION_CONFIG_KEY = "migration.fixed_zero"
EXCESS_BRANCH_CONFIG_KEY = "excess_load_indicator.fixed_zero"
SPOT_PREEMPTION_CONFIG_KEY = "spot_preemption.fixed_zero"
BOUND_AUDIT_TOLERANCE = 1e-12
PRIMARY_HARD_AUDIT_TOLERANCE = 1e-5
FIDELITY_WARNING_KEY = "fixed_zero_migration_excess_preemption_override"
FIDELITY_WARNING = (
    "Intentional experimental restriction: all on-demand migration variables, "
    "positive OD CPU-excess branch indicators, and Spot preempted-state "
    "variables are fixed to zero. This is not the unrestricted primary "
    "migration-v3 formulation."
)


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
            f"{MIGRATION_CONFIG_KEY}, {EXCESS_BRANCH_CONFIG_KEY}, and "
            f"{SPOT_PREEMPTION_CONFIG_KEY} are all the literal YAML boolean true"
        ),
        "migration_fixed_zero": True,
        "excess_load_indicator_fixed_zero": True,
        "spot_preemption_fixed_zero": True,
        "migration_allowed": False,
        "excess_load_allowed": False,
        "spot_preemption_allowed": False,
        "bound_audit_tolerance": float(bound_audit_tolerance),
        "solution_audit_tolerance": float(solution_audit_tolerance),
        "scope": {
            "migration": "all destination-indexed on-demand migration variables",
            "excess_load_indicator": (
                "all excess_positive_branch server-time-scenario indicators"
            ),
            "spot_preemption": (
                "all spot_preempted[j,t,xi] cumulative preempted-state variables "
                "over every modeled Spot active slot and scenario"
            ),
        },
        "spot_preemption_zero_semantics": (
            "for an accepted Spot VM, spot_state plus fixed-server constraints "
            "force service on its initial server in every active slot and scenario"
        ),
        "source_formulation_fidelity": "intentional experimental restriction",
        "warning": FIDELITY_WARNING,
    }


def require_fixed_zero_migration_excess_preemption_configuration(
    config: dict[str, Any],
) -> None:
    """Require all three fixed-zero flags to be literal YAML booleans."""

    V5.require_fixed_zero_migration_excess_configuration(config)
    section = config.get("spot_preemption")
    value = section.get("fixed_zero") if isinstance(section, dict) else None
    if value is not True:
        raise ValueError(
            "run_experiment_fixed_zero_migration_excess_preemption.py requires "
            f"{SPOT_PREEMPTION_CONFIG_KEY}: true (the value must be the literal "
            "YAML boolean true)"
        )


def build_instance_fixed_zero_migration_excess_preemption(
    config: dict[str, Any], *args: Any, **kwargs: Any
) -> Any:
    require_fixed_zero_migration_excess_preemption_configuration(config)
    instance = V5._ORIGINAL_BUILD_INSTANCE(config, *args, **kwargs)
    instance.metadata["fixed_zero_policy"] = copy.deepcopy(_policy_payload())
    warnings = instance.metadata.setdefault("fidelity_warnings", {})
    if not isinstance(warnings, dict):
        raise TypeError("instance metadata fidelity_warnings must be a mapping")
    warnings[FIDELITY_WARNING_KEY] = FIDELITY_WARNING
    return instance


def _variable_group(artifacts: Any, key: str) -> list[Any]:
    return V5._variable_group(artifacts, key)


def _bound_audit_from_variables(
    migration_variables: list[Any],
    excess_branch_variables: list[Any],
    spot_preemption_variables: list[Any],
    *,
    tolerance: float = BOUND_AUDIT_TOLERANCE,
    expected_support_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    migration = V5._zero_bound_group_audit(
        migration_variables,
        group="migration",
        tolerance=tolerance,
    )
    excess_branch = V5._zero_bound_group_audit(
        excess_branch_variables,
        group="excess_branch",
        tolerance=tolerance,
    )
    spot_preemption = V5._zero_bound_group_audit(
        spot_preemption_variables,
        group="spot_preemption",
        tolerance=tolerance,
    )
    all_fixed = bool(
        migration["all_bounds_fixed_zero"]
        and excess_branch["all_bounds_fixed_zero"]
        and spot_preemption["all_bounds_fixed_zero"]
    )
    audit = {
        "tolerance": float(tolerance),
        "all_fixed_bounds_zero": all_fixed,
        "all_required_bounds_fixed_zero": all_fixed,
        "migration": migration,
        "excess_branch": excess_branch,
        "spot_preemption": spot_preemption,
    }
    if not all_fixed:
        raise AssertionError(
            "cumulative fixed-zero model found a required variable whose bounds "
            f"were not fixed to zero: {audit}"
        )
    if expected_support_counts is not None:
        observed_counts = {
            group: int(audit[group]["fixed_variable_count"])
            for group in ("migration", "excess_branch", "spot_preemption")
        }
        expected_counts = {
            group: int(expected_support_counts[group])
            for group in observed_counts
        }
        audit["expected_support_counts"] = expected_counts
        audit["observed_support_counts"] = observed_counts
        audit["semantic_support_verified"] = observed_counts == expected_counts
        if not audit["semantic_support_verified"]:
            raise AssertionError(
                "cumulative fixed-zero variable support drifted: "
                f"expected={expected_counts}, observed={observed_counts}"
            )
    return audit


def _expected_support_counts(data: Any) -> dict[str, int]:
    return {
        "migration": sum(
            max(0, len(data.active_od[vm_id]) - 1)
            * len(data.S)
            * len(data.Xi)
            for vm_id in data.I
        ),
        "excess_branch": len(data.S) * len(data.T) * len(data.Xi),
        "spot_preemption": sum(
            len(data.active_spot[spot_id]) * len(data.Xi)
            for spot_id in data.J
        ),
    }


def build_model_fixed_zero_migration_excess_preemption(
    data: Any, *args: Any, **kwargs: Any
) -> Any:
    """Build migration-v3, then fix all three experimental groups to zero."""

    artifacts = V5._ORIGINAL_BUILD_MODEL(data, *args, **kwargs)
    migration_variables = _variable_group(artifacts, "migration")
    excess_branch_variables = _variable_group(artifacts, "excess_branch")
    spot_preemption_variables = _variable_group(artifacts, "h")
    for variable in (
        *migration_variables,
        *excess_branch_variables,
        *spot_preemption_variables,
    ):
        variable.LB = 0.0
        variable.UB = 0.0
    artifacts.model.update()

    bound_audit = _bound_audit_from_variables(
        migration_variables,
        excess_branch_variables,
        spot_preemption_variables,
        expected_support_counts=_expected_support_counts(data),
    )
    artifacts.variables["fixed_zero_bound_audit"] = bound_audit
    artifacts.model._fixed_zero_migration_excess_preemption_bound_audit = (
        copy.deepcopy(bound_audit)
    )
    return artifacts


def _finite_value(variable: Any, *, group: str) -> float:
    value = float(variable.X)
    if not math.isfinite(value):
        raise AssertionError(f"a {group} variable has a non-finite incumbent value")
    return value


def _solution_audit(
    artifacts: Any,
    *,
    tolerance: float = PRIMARY_HARD_AUDIT_TOLERANCE,
) -> dict[str, Any]:
    """Audit zero migration/excess and uninterrupted accepted Spot service."""

    audit = V5._solution_audit(artifacts, tolerance=tolerance)
    data = artifacts.data
    variables = artifacts.variables
    h = variables.get("h")
    y = variables.get("y")
    y_init = variables.get("y_init")
    u = variables.get("u")
    if h is None or y is None or y_init is None or u is None:
        raise KeyError("model artifacts do not expose all Spot policy variables")

    expected_h_keys = {
        (spot_id, t, scenario_id)
        for spot_id in data.J
        for t in data.active_spot[spot_id]
        for scenario_id in data.Xi
    }
    if set(h) != expected_h_keys:
        raise AssertionError("Spot preemption variables have unexpected support")
    expected_y_keys = {
        (spot_id, server_id, t, scenario_id)
        for spot_id in data.J
        for server_id in data.S
        for t in data.active_spot[spot_id]
        for scenario_id in data.Xi
    }
    if set(y) != expected_y_keys:
        raise AssertionError("Spot placement variables have unexpected support")
    expected_y_init_keys = {
        (spot_id, server_id)
        for spot_id in data.J
        for server_id in data.S
    }
    if set(y_init) != expected_y_init_keys:
        raise AssertionError("Spot admission variables have unexpected support")

    h_values = V5._finite_incumbent_values(
        list(h.values()), group="spot preemption"
    )
    maximum_state_residual = 0.0
    maximum_full_service_deficit = 0.0
    maximum_fixed_server_residual = 0.0
    maximum_spot_server_on_violation = 0.0
    maximum_spot_placement_fractionality = 0.0
    maximum_spot_admission_fractionality = 0.0
    maximum_admitted_sum_fractionality = 0.0
    preempted_state_count = 0
    preemption_event_count = 0
    arrival_preemption_violation_count = 0
    preemption_monotonicity_violation_count = 0
    accepted_deactivation_count = 0
    rejected_activation_count = 0
    accepted_spot_count = 0.0
    raw_spot_admission_sum = 0.0
    accepted_possible_slot_count = 0.0
    active_spot_slot_count = 0.0
    scenario_possible_slots = {scenario_id: 0.0 for scenario_id in data.Xi}
    scenario_active_slots = {scenario_id: 0.0 for scenario_id in data.Xi}

    for spot_id in data.J:
        initial_values = {
            server_id: _finite_value(
                y_init[spot_id, server_id], group="Spot admission"
            )
            for server_id in data.S
        }
        admitted = sum(initial_values.values())
        accepted = round(admitted)
        accepted_spot_count += accepted
        raw_spot_admission_sum += admitted
        maximum_spot_admission_fractionality = max(
            maximum_spot_admission_fractionality,
            *(abs(value - round(value)) for value in initial_values.values()),
        )
        maximum_admitted_sum_fractionality = max(
            maximum_admitted_sum_fractionality,
            abs(admitted - accepted),
        )
        for scenario_id in data.Xi:
            previous_h: float | None = None
            for position, t in enumerate(data.active_spot[spot_id]):
                h_value = _finite_value(
                    h[spot_id, t, scenario_id], group="Spot preemption"
                )
                placement_values = {
                    server_id: _finite_value(
                        y[spot_id, server_id, t, scenario_id],
                        group="Spot placement",
                    )
                    for server_id in data.S
                }
                active_sum = sum(placement_values.values())
                accepted_possible_slot_count += accepted
                active_spot_slot_count += active_sum
                scenario_possible_slots[scenario_id] += accepted
                scenario_active_slots[scenario_id] += active_sum
                maximum_spot_placement_fractionality = max(
                    maximum_spot_placement_fractionality,
                    *(abs(value - round(value)) for value in placement_values.values()),
                )
                maximum_state_residual = max(
                    maximum_state_residual,
                    abs(active_sum + h_value - admitted),
                )
                maximum_full_service_deficit = max(
                    maximum_full_service_deficit,
                    abs(active_sum - admitted),
                )
                if h_value > tolerance:
                    preempted_state_count += 1
                if position == 0 and abs(h_value) > tolerance:
                    arrival_preemption_violation_count += 1
                if previous_h is not None and previous_h - h_value > tolerance:
                    preemption_monotonicity_violation_count += 1
                if (previous_h is None or previous_h <= tolerance) and (
                    h_value > tolerance
                ):
                    preemption_event_count += 1
                if accepted == 1 and active_sum < 1.0 - tolerance:
                    accepted_deactivation_count += 1
                if accepted == 0 and active_sum > tolerance:
                    rejected_activation_count += 1
                previous_h = h_value
                for server_id in data.S:
                    placement = placement_values[server_id]
                    initial = initial_values[server_id]
                    server_on = _finite_value(
                        u[server_id, t], group="server state"
                    )
                    maximum_fixed_server_residual = max(
                        maximum_fixed_server_residual,
                        abs(placement - initial),
                    )
                    maximum_spot_server_on_violation = max(
                        maximum_spot_server_on_violation,
                        placement - server_on,
                    )

    unweighted_aggregate_service_rate = (
        active_spot_slot_count / accepted_possible_slot_count
        if accepted_possible_slot_count > tolerance
        else 0.0
    )
    scenario_service_rates = {
        scenario_id: (
            scenario_active_slots[scenario_id]
            / scenario_possible_slots[scenario_id]
            if scenario_possible_slots[scenario_id] > tolerance
            else 0.0
        )
        for scenario_id in data.Xi
    }
    direct_expected_service_rate = sum(
        float(data.p[scenario_id]) * scenario_service_rates[scenario_id]
        for scenario_id in data.Xi
    )
    spot_audit = {
        "spot_preemption_variable_count": len(h_values),
        "spot_preemption_value_sum": float(sum(h_values)),
        "maximum_absolute_spot_preemption_value": max(
            (abs(value) for value in h_values), default=0.0
        ),
        "preempted_state_count": preempted_state_count,
        "preemption_event_count": preemption_event_count,
        "accepted_spot_count": float(accepted_spot_count),
        "raw_spot_admission_sum": float(raw_spot_admission_sum),
        "accepted_possible_slot_count": float(accepted_possible_slot_count),
        "active_spot_slot_count": float(active_spot_slot_count),
        "scenario_spot_accepted_possible_slots": {
            str(scenario_id): value
            for scenario_id, value in scenario_possible_slots.items()
        },
        "scenario_spot_active_slots": {
            str(scenario_id): value
            for scenario_id, value in scenario_active_slots.items()
        },
        "scenario_spot_service_rates": {
            str(scenario_id): value
            for scenario_id, value in scenario_service_rates.items()
        },
        "direct_expected_spot_service_rate": float(
            direct_expected_service_rate
        ),
        "unweighted_aggregate_spot_service_rate": float(
            unweighted_aggregate_service_rate
        ),
        "maximum_spot_state_equation_residual": maximum_state_residual,
        "maximum_accepted_spot_service_deficit": maximum_full_service_deficit,
        "maximum_fixed_initial_server_residual": maximum_fixed_server_residual,
        "maximum_spot_server_on_violation": maximum_spot_server_on_violation,
        "maximum_spot_placement_fractionality": (
            maximum_spot_placement_fractionality
        ),
        "maximum_spot_admission_fractionality": (
            maximum_spot_admission_fractionality
        ),
        "maximum_admitted_sum_fractionality": (
            maximum_admitted_sum_fractionality
        ),
        "arrival_preemption_violation_count": (
            arrival_preemption_violation_count
        ),
        "preemption_monotonicity_violation_count": (
            preemption_monotonicity_violation_count
        ),
        "accepted_deactivation_count": accepted_deactivation_count,
        "rejected_activation_count": rejected_activation_count,
    }
    audit.update(spot_audit)
    violations = {
        "Spot preemption value": spot_audit[
            "maximum_absolute_spot_preemption_value"
        ],
        "Spot state equation": maximum_state_residual,
        "accepted Spot service deficit": maximum_full_service_deficit,
        "fixed initial Spot server": maximum_fixed_server_residual,
        "Spot server-on implication": maximum_spot_server_on_violation,
        "Spot placement fractionality": maximum_spot_placement_fractionality,
        "Spot admission fractionality": maximum_spot_admission_fractionality,
        "Spot admitted-sum fractionality": maximum_admitted_sum_fractionality,
    }
    failed = {name: value for name, value in violations.items() if value > tolerance}
    lifecycle_counts = {
        "preempted states": preempted_state_count,
        "preemption events": preemption_event_count,
        "arrival preemption violations": arrival_preemption_violation_count,
        "preemption monotonicity violations": (
            preemption_monotonicity_violation_count
        ),
        "accepted deactivations": accepted_deactivation_count,
        "rejected activations": rejected_activation_count,
    }
    nonzero_counts = {
        name: value for name, value in lifecycle_counts.items() if value
    }
    if nonzero_counts or failed:
        raise AssertionError(
            "cumulative fixed-zero Spot solution audit failed: "
            f"counts={nonzero_counts}, metrics={failed}"
        )
    return audit


def _audit_primary_report(
    summary: dict[str, Any],
    solution_audit: dict[str, Any],
    *,
    tolerance: float,
) -> None:
    V5._audit_primary_report(summary, solution_audit, tolerance=tolerance)
    service = summary.get("service")
    diagnostics = summary.get("model_validation_diagnostics")
    if not isinstance(service, dict) or not isinstance(diagnostics, dict):
        raise AssertionError("primary solution report lacks Spot audit mappings")

    reported_acceptance = V5._reported_number(
        service, "spot_acceptance_count", scope="service"
    )
    reported_service_rate = V5._reported_number(
        service, "expected_spot_service_rate", scope="service"
    )
    reported_arrival_violations = V5._reported_number(
        diagnostics, "spot_arrival_violation_count", scope="diagnostics"
    )
    reported_reactivation_violations = V5._reported_number(
        diagnostics, "spot_reactivation_violation_count", scope="diagnostics"
    )
    reported_spot_fractionality = V5._reported_number(
        diagnostics,
        "maximum_spot_relaxation_fractionality",
        scope="diagnostics",
    )
    expected_acceptance = float(solution_audit["accepted_spot_count"])
    expected_service_rate = float(
        solution_audit["direct_expected_spot_service_rate"]
    )
    expected_spot_fractionality = float(
        solution_audit["maximum_spot_placement_fractionality"]
    )
    discrepancies = {
        "spot acceptance count": abs(reported_acceptance - expected_acceptance),
        "expected Spot service rate": abs(
            reported_service_rate - expected_service_rate
        ),
        "Spot arrival violations": abs(reported_arrival_violations),
        "Spot reactivation violations": abs(reported_reactivation_violations),
        "reported Spot placement fractionality": abs(
            reported_spot_fractionality
        ),
        "Spot placement fractionality cross-check": abs(
            reported_spot_fractionality - expected_spot_fractionality
        ),
    }
    failed = {name: value for name, value in discrepancies.items() if value > tolerance}
    if failed:
        raise AssertionError(
            "primary solution report disagrees with zero-preemption policy: "
            f"{failed}"
        )
    solution_audit.update(
        {
            "reported_spot_acceptance_count": reported_acceptance,
            "reported_expected_spot_service_rate": reported_service_rate,
            "reported_spot_arrival_violation_count": (
                reported_arrival_violations
            ),
            "reported_spot_reactivation_violation_count": (
                reported_reactivation_violations
            ),
            "reported_maximum_spot_relaxation_fractionality": (
                reported_spot_fractionality
            ),
        }
    )


def _finalize_policy_report(
    run_dir: str | Path,
    summary: dict[str, Any],
    *,
    bound_audit: dict[str, Any],
    solution_audit: dict[str, Any],
) -> dict[str, Any]:
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
        raise TypeError("solution summary diagnostics must be a mapping")
    diagnostics["fixed_zero_migration_excess_preemption_bounds"] = bound_audit
    diagnostics["fixed_zero_migration_excess_preemption_solution"] = solution_audit

    operations = summary.get("operations")
    if isinstance(operations, dict):
        operations["migration_allowed"] = False
        operations["excess_load_allowed"] = False
        operations["excess_load_indicator_fixed_zero"] = True
        operations["spot_preemption_allowed"] = False
        operations["spot_preemption_fixed_zero"] = True

    summary_path, diagnostics_path, policy_path = V5._final_output_paths(output)
    V5._atomic_write_json(diagnostics_path, diagnostics)
    V5._atomic_write_json(policy_path, report)
    V5._atomic_write_json(summary_path, summary)
    return summary


def write_solution_reports_fixed_zero_migration_excess_preemption(
    artifacts: Any,
    run_dir: str | Path,
    *,
    tolerance: float = PRIMARY_HARD_AUDIT_TOLERANCE,
) -> dict[str, Any]:
    output = Path(run_dir)
    V5._clear_final_outputs(output)
    try:
        bound_audit = _bound_audit_from_variables(
            _variable_group(artifacts, "migration"),
            _variable_group(artifacts, "excess_branch"),
            _variable_group(artifacts, "h"),
            expected_support_counts=_expected_support_counts(artifacts.data),
        )
        solution_audit = _solution_audit(artifacts, tolerance=tolerance)
        with V5._defer_primary_summary_write(V5._ORIGINAL_WRITE_SOLUTION_REPORTS):
            summary = V5._ORIGINAL_WRITE_SOLUTION_REPORTS(
                artifacts, run_dir, tolerance=tolerance
            )
        _audit_primary_report(summary, solution_audit, tolerance=tolerance)
        return _finalize_policy_report(
            run_dir,
            summary,
            bound_audit=bound_audit,
            solution_audit=solution_audit,
        )
    except Exception:
        V5._clear_final_outputs(output)
        raise


def _bound_audit_from_model(model: Any) -> dict[str, Any]:
    groups = {
        "migration": [
            variable
            for variable in model.getVars()
            if str(variable.VarName).startswith("migration[")
        ],
        "excess_branch": [
            variable
            for variable in model.getVars()
            if str(variable.VarName).startswith("excess_positive_branch[")
        ],
        "spot_preemption": [
            variable
            for variable in model.getVars()
            if str(variable.VarName).startswith("spot_preempted[")
        ],
    }
    construction_audit = getattr(
        model,
        "_fixed_zero_migration_excess_preemption_bound_audit",
        None,
    )
    if not isinstance(construction_audit, dict):
        raise AssertionError(
            "no-incumbent audit lacks the construction-time fixed-zero audit"
        )
    if construction_audit.get("semantic_support_verified") is not True:
        raise AssertionError(
            "construction-time fixed-zero audit did not certify semantic support"
        )
    expected_counts = construction_audit.get("expected_support_counts")
    if not isinstance(expected_counts, dict):
        raise AssertionError(
            "construction-time fixed-zero audit lacks expected support counts"
        )
    audit = _bound_audit_from_variables(
        groups["migration"],
        groups["excess_branch"],
        groups["spot_preemption"],
        expected_support_counts=expected_counts,
    )
    for group in groups:
        construction_count = int(
            construction_audit[group]["fixed_variable_count"]
        )
        observed_count = int(audit[group]["fixed_variable_count"])
        expected_count = int(expected_counts[group])
        if not construction_count == observed_count == expected_count:
            raise AssertionError(
                "no-incumbent audit found an unexpected fixed-zero support "
                f"for {group}: construction={construction_count}, "
                f"observed={observed_count}, expected={expected_count}"
            )
    return audit


def write_no_solution_summary_fixed_zero_migration_excess_preemption(
    model: Any,
    run_dir: str | Path,
) -> dict[str, Any]:
    output = Path(run_dir)
    V5._clear_final_outputs(output)
    try:
        bound_audit = _bound_audit_from_model(model)
        solution_audit = {
            "incumbent_available": False,
            "audit_status": "not_applicable_without_incumbent",
        }
        with V5._defer_primary_summary_write(
            V5._ORIGINAL_WRITE_NO_SOLUTION_SUMMARY
        ):
            summary = V5._ORIGINAL_WRITE_NO_SOLUTION_SUMMARY(model, run_dir)
        return _finalize_policy_report(
            run_dir,
            summary,
            bound_audit=bound_audit,
            solution_audit=solution_audit,
        )
    except Exception:
        V5._clear_final_outputs(output)
        raise


@contextmanager
def _patched_cli() -> Iterator[None]:
    replacements = {
        "build_instance": build_instance_fixed_zero_migration_excess_preemption,
        "build_model": build_model_fixed_zero_migration_excess_preemption,
        "write_solution_reports": (
            write_solution_reports_fixed_zero_migration_excess_preemption
        ),
        "write_no_solution_summary": (
            write_no_solution_summary_fixed_zero_migration_excess_preemption
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
