from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .data import CPU, MEM, InstanceData
from .model import ModelArtifacts


def _value(variable: Any) -> float:
    return float(variable.X)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError(f"cannot JSON-serialize {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _weighted_var_cvar(
    losses: dict[int, float], probabilities: dict[int, float], alpha: float
) -> tuple[float, float]:
    ordered = sorted((float(losses[xi]), float(probabilities[xi])) for xi in losses)
    cumulative = 0.0
    var = ordered[-1][0]
    for loss, probability in ordered:
        cumulative += probability
        if cumulative + 1e-12 >= alpha:
            var = loss
            break
    cvar = var + sum(
        probability * max(0.0, loss - var) for loss, probability in ordered
    ) / (1.0 - alpha)
    return var, cvar


def _status_name(status: int) -> str:
    try:
        from gurobipy import GRB

        names = {
            GRB.LOADED: "LOADED",
            GRB.OPTIMAL: "OPTIMAL",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INF_OR_UNBD: "INF_OR_UNBD",
            GRB.UNBOUNDED: "UNBOUNDED",
            GRB.CUTOFF: "CUTOFF",
            GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
            GRB.NODE_LIMIT: "NODE_LIMIT",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
            GRB.INTERRUPTED: "INTERRUPTED",
            GRB.NUMERIC: "NUMERIC",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
            GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
            GRB.WORK_LIMIT: "WORK_LIMIT",
            GRB.MEM_LIMIT: "MEM_LIMIT",
        }
        return names.get(status, f"STATUS_{status}")
    except ImportError:  # pragma: no cover
        return str(status)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _instance_summary(data: InstanceData) -> dict[str, Any]:
    if hasattr(data, "summary"):
        summary = data.summary()
        economics = data.metadata.get("economics", {})
        summary["units_and_fidelity"] = {
            "migration_coefficient": data.c_mig,
            "migration_coefficient_unit": economics.get("migration_coefficient_unit"),
            "warnings": data.metadata.get("fidelity_warnings", {}),
        }
        return summary
    return {
        "counts": {
            "on_demand_vms": len(data.I),
            "spot_vms": len(data.J),
            "batch_families": len(data.K),
            "servers": len(data.S),
            "time_periods": len(data.T),
            "scenarios": len(data.Xi),
        },
        "risk": {"alpha": data.alpha, "epsilon": data.epsilon},
    }


def write_instance_artifacts(data: InstanceData, destination: str | Path) -> Path:
    """Write the generated model inputs and their lineage without solving."""

    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "instance_summary.json", _instance_summary(data))
    source_manifest = []
    for source_value in data.source_files:
        source = Path(source_value)
        source_manifest.append(
            {
                "path": str(source),
                "size_bytes": source.stat().st_size if source.is_file() else None,
                "sha256": _sha256(source) if source.is_file() else None,
                "exists": source.is_file(),
            }
        )
    _write_json(
        output / "data_audit.json",
        {
            "source_files": data.source_files,
            "source_manifest": source_manifest,
            "metadata": data.metadata,
        },
    )

    od_rows = [
        {
            "vm_id": i,
            "arrival_t": data.active_od[i][0],
            "departure_t": data.active_od[i][-1],
            "active_slots": len(data.active_od[i]),
            "q_cpu": data.q_od[i, CPU],
            "q_mem": data.q_od[i, MEM],
            "revenue_usd_per_slot": data.pi_od[i],
            "migration_coefficient": data.c_mig,
        }
        for i in data.I
    ]
    spot_rows = [
        {
            "vm_id": j,
            "arrival_t": data.active_spot[j][0],
            "departure_t": data.active_spot[j][-1],
            "active_slots": len(data.active_spot[j]),
            "q_cpu": data.q_spot[j, CPU],
            "q_mem": data.q_spot[j, MEM],
            "revenue_usd_per_active_slot": data.pi_spot[j],
        }
        for j in data.J
    ]
    batch_rows = [
        {
            "family_id": k,
            "q_cpu": data.q_batch[k, CPU],
            "q_mem": data.q_batch[k, MEM],
            "rho_cpu": data.rho_batch[k, CPU],
            "rho_mem": data.rho_batch[k, MEM],
            "workload": data.W[k],
            "revenue_usd_per_workload_unit": data.pi_batch[k],
            "base_cpu": data.batch_base_cpu,
            "base_mem": data.batch_base_mem,
            "startup_cpu": data.batch_startup_cpu,
            "startup_mem": data.batch_startup_mem,
        }
        for k in data.K
    ]
    price_rows = [
        {
            "scenario_id": xi,
            "probability": data.p[xi],
            "t": t,
            "rt_price_usd_per_kwh": data.p_rt[t, xi],
        }
        for xi in data.Xi
        for t in data.T
    ]
    server_rows = [
        {
            "server_id": s,
            "C_cpu": data.C_cpu,
            "C_mem": data.C_mem,
            "E_idle_kwh_per_slot": data.E_idle,
            "E_cpu_kwh_per_cpu_unit_slot": data.E_cpu,
            "min_on_slots": data.min_on,
            "min_off_slots": data.min_off,
        }
        for s in data.S
    ]
    od_usage_rows = (
        {
            "scenario_id": xi,
            "vm_id": i,
            "t": t,
            "cpu_usage": data.load_od[i, CPU, t, xi],
            "mem_usage": data.load_od[i, MEM, t, xi],
        }
        for xi in data.Xi
        for i in data.I
        for t in data.active_od[i]
    )
    spot_usage_rows = (
        {
            "scenario_id": xi,
            "vm_id": j,
            "t": t,
            "cpu_usage": data.load_spot[j, CPU, t, xi],
            "mem_usage": data.load_spot[j, MEM, t, xi],
        }
        for xi in data.Xi
        for j in data.J
        for t in data.active_spot[j]
    )

    frames: list[tuple[str, Iterable[dict[str, Any]], list[str] | None]] = [
        ("on_demand_parameters.csv", od_rows, None),
        (
            "spot_parameters.csv",
            spot_rows,
            [
                "vm_id",
                "arrival_t",
                "departure_t",
                "active_slots",
                "q_cpu",
                "q_mem",
                "revenue_usd_per_active_slot",
            ],
        ),
        (
            "batch_parameters.csv",
            batch_rows,
            [
                "family_id",
                "q_cpu",
                "q_mem",
                "rho_cpu",
                "rho_mem",
                "workload",
                "revenue_usd_per_workload_unit",
                "base_cpu",
                "base_mem",
                "startup_cpu",
                "startup_mem",
            ],
        ),
        ("server_parameters.csv", server_rows, None),
        ("rt_prices.csv", price_rows, None),
        ("on_demand_usage.csv", od_usage_rows, None),
        (
            "spot_usage.csv",
            spot_usage_rows,
            ["scenario_id", "vm_id", "t", "cpu_usage", "mem_usage"],
        ),
    ]
    written: list[Path] = []
    for filename, rows, columns in frames:
        path = output / filename
        pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
        written.append(path)
    checksums = {path.name: _sha256(path) for path in written}
    _write_json(output / "sha256.json", checksums)
    return output


def write_solution_reports(
    artifacts: ModelArtifacts,
    run_dir: str | Path,
    *,
    tolerance: float = 1e-7,
) -> dict[str, Any]:
    """Extract solution, accounting, exact-excess, and server-time CVaR audits."""

    model, data, variables = artifacts.model, artifacts.data, artifacts.variables
    output = Path(run_dir)
    output.mkdir(parents=True, exist_ok=True)
    if model.SolCount <= 0:
        raise RuntimeError("cannot report a model with no incumbent solution")
    model.write(str(output / "solution.sol"))
    economics_metadata = data.metadata.get("economics", {})
    migration_coefficient = data.c_mig
    migration_unit = economics_metadata.get(
        "migration_coefficient_unit", "unspecified"
    )
    spot_accepted = {
        j: round(sum(_value(variables["y_init"][j, s]) for s in data.S))
        for j in data.J
    }

    server_rows: list[dict[str, Any]] = []
    for s in data.S:
        for t in data.T:
            server_rows.append(
                {
                    "server_id": s,
                    "t": t,
                    "on": round(_value(variables["u"][s, t])),
                    "started_after_t": (
                        round(_value(variables["u_on"][s, t])) if t in data.T[:-1] else 0
                    ),
                    "stopped_after_t": (
                        round(_value(variables["u_off"][s, t])) if t in data.T[:-1] else 0
                    ),
                }
            )
    pd.DataFrame(server_rows).to_csv(output / "server_schedule.csv", index=False)

    od_initial_rows = []
    for i in data.I:
        chosen = next(
            (s for s in data.S if _value(variables["x_init"][i, s]) > 0.5),
            None,
        )
        od_initial_rows.append({"vm_id": i, "server_id": chosen})
    pd.DataFrame(od_initial_rows).to_csv(output / "od_initial_placement.csv", index=False)

    od_schedule_rows: list[dict[str, Any]] = []
    for i in data.I:
        for xi in data.Xi:
            for t in data.active_od[i]:
                server = next(
                    (
                        s
                        for s in data.S
                        if _value(variables["x"][i, s, t, xi]) > 0.5
                    ),
                    None,
                )
                od_schedule_rows.append(
                    {
                        "scenario_id": xi,
                        "vm_id": i,
                        "t": t,
                        "server_id": server,
                        "actual_cpu_demand": data.load_od[i, CPU, t, xi],
                        "actual_mem_demand": data.load_od[i, MEM, t, xi],
                    }
                )
    pd.DataFrame(od_schedule_rows).to_csv(output / "od_schedule.csv", index=False)

    migration_rows: list[dict[str, Any]] = []
    false_migration_count = 0
    migration_indicator_count = 0
    migration_relaxed_value_sum = 0.0
    maximum_migration_fractionality = 0.0
    maximum_migration_definition_residual = 0.0
    actual_migration_count = 0
    for i in data.I:
        for xi in data.Xi:
            for t in data.active_od[i][:-1]:
                destination_values = {
                    s: _value(variables["migration"][i, s, t, xi])
                    for s in data.S
                }
                relaxed_value = sum(destination_values.values())
                maximum_migration_fractionality = max(
                    maximum_migration_fractionality,
                    *(
                        abs(value - round(value))
                        for value in destination_values.values()
                    ),
                )
                old_server = next(
                    (
                        s
                        for s in data.S
                        if _value(variables["x"][i, s, t, xi]) > 0.5
                    ),
                    None,
                )
                new_server = next(
                    (
                        s
                        for s in data.S
                        if _value(variables["x"][i, s, t + 1, xi]) > 0.5
                    ),
                    None,
                )
                actual_change = int(
                    old_server is not None
                    and new_server is not None
                    and old_server != new_server
                )
                destination_residual = max(
                    (
                        abs(
                            value
                            - float(actual_change and destination == new_server)
                        )
                        for destination, value in destination_values.items()
                    ),
                    default=0.0,
                )
                definition_residual = max(
                    destination_residual,
                    abs(relaxed_value - actual_change),
                )
                maximum_migration_definition_residual = max(
                    maximum_migration_definition_residual,
                    definition_residual,
                )
                indicator = round(relaxed_value)
                migration_indicator_count += indicator
                migration_relaxed_value_sum += relaxed_value
                actual_migration_count += actual_change
                if definition_residual > tolerance:
                    false_migration_count += 1
                if relaxed_value > tolerance or actual_change:
                    migration_energy = (
                        data.c_mig
                        * data.load_od[i, MEM, t, xi]
                        * relaxed_value
                    )
                    migration_rows.append(
                        {
                            "scenario_id": xi,
                            "vm_id": i,
                            "from_t": t,
                            "to_t": t + 1,
                            "from_server": old_server,
                            "to_server": new_server,
                            "destination_server": new_server if actual_change else None,
                            "indicator": indicator,
                            "relaxed_value": relaxed_value,
                            "actual_change": actual_change,
                            "definition_residual": definition_residual,
                            "mem_usage": data.load_od[i, MEM, t, xi],
                            "migration_coefficient": migration_coefficient,
                            "migration_coefficient_unit": migration_unit,
                            "migration_energy_kwh": migration_energy,
                            "rt_price_usd_per_kwh": data.p_rt[t, xi],
                            "migration_rt_cost_usd": (
                                migration_energy * data.p_rt[t, xi]
                            ),
                        }
                    )
    pd.DataFrame(
        migration_rows,
        columns=[
            "scenario_id",
            "vm_id",
            "from_t",
            "to_t",
            "from_server",
            "to_server",
            "destination_server",
            "indicator",
            "relaxed_value",
            "actual_change",
            "definition_residual",
            "mem_usage",
            "migration_coefficient",
            "migration_coefficient_unit",
            "migration_energy_kwh",
            "rt_price_usd_per_kwh",
            "migration_rt_cost_usd",
        ],
    ).to_csv(output / "migrations.csv", index=False)

    spot_rows: list[dict[str, Any]] = []
    spot_arrival_violation_count = 0
    spot_reactivation_violation_count = 0
    maximum_spot_fractionality = 0.0
    for j in data.J:
        accepted = spot_accepted[j]
        initial_server = next(
            (s for s in data.S if _value(variables["y_init"][j, s]) > 0.5),
            None,
        )
        for xi in data.Xi:
            for t in data.active_spot[j]:
                active_server = next(
                    (
                        s
                        for s in data.S
                        if _value(variables["y"][j, s, t, xi]) > 0.5
                    ),
                    None,
                )
                spot_rows.append(
                    {
                        "scenario_id": xi,
                        "vm_id": j,
                        "t": t,
                        "accepted": accepted,
                        "initial_server": initial_server,
                        "active_server": active_server,
                        "preempted": round(_value(variables["h"][j, t, xi])),
                        "slot_revenue": data.pi_spot[j] if active_server is not None else 0.0,
                    }
                )
                for s in data.S:
                    spot_value = _value(variables["y"][j, s, t, xi])
                    maximum_spot_fractionality = max(
                        maximum_spot_fractionality,
                        abs(spot_value - round(spot_value)),
                    )
            arrival = data.active_spot[j][0]
            arrival_active = sum(
                _value(variables["y"][j, s, arrival, xi]) for s in data.S
            )
            if abs(arrival_active - accepted) > tolerance:
                spot_arrival_violation_count += 1
            seen_preempted = False
            for t in data.active_spot[j]:
                preempted = _value(variables["h"][j, t, xi]) > 0.5
                active = any(
                    _value(variables["y"][j, s, t, xi]) > 0.5 for s in data.S
                )
                if seen_preempted and active:
                    spot_reactivation_violation_count += 1
                seen_preempted = seen_preempted or preempted
    pd.DataFrame(
        spot_rows,
        columns=[
            "scenario_id",
            "vm_id",
            "t",
            "accepted",
            "initial_server",
            "active_server",
            "preempted",
            "slot_revenue",
        ],
    ).to_csv(output / "spot_schedule.csv", index=False)

    batch_rows: list[dict[str, Any]] = []
    false_startup_count = 0
    startup_indicator_count = 0
    logical_start_count = 0
    for k in data.K:
        for s in data.S:
            for t in data.T:
                prepared = round(_value(variables["z"][k, s, t]))
                started = round(_value(variables["z_on"][k, s, t]))
                previous = 0 if t == data.T[0] else round(_value(variables["z"][k, s, t - 1]))
                logical_start = int(prepared == 1 and previous == 0)
                startup_indicator_count += started
                logical_start_count += logical_start
                if started and not logical_start:
                    false_startup_count += 1
                for xi in data.Xi:
                    workload = _value(variables["batch_work"][k, s, t, xi])
                    if prepared or started or workload > tolerance:
                        batch_rows.append(
                            {
                                "scenario_id": xi,
                                "family_id": k,
                                "server_id": s,
                                "t": t,
                                "prepared": prepared,
                                "started": started,
                                "logical_start": logical_start,
                                "workload": workload,
                                "cpu_workload": data.rho_batch[k, CPU] * workload,
                                "mem_workload": data.rho_batch[k, MEM] * workload,
                            }
                        )
    pd.DataFrame(
        batch_rows,
        columns=[
            "scenario_id",
            "family_id",
            "server_id",
            "t",
            "prepared",
            "started",
            "logical_start",
            "workload",
            "cpu_workload",
            "mem_workload",
        ],
    ).to_csv(output / "batch_schedule.csv", index=False)
    batch_completion_rows = []
    maximum_batch_completion_residual = 0.0
    for k in data.K:
        for xi in data.Xi:
            processed = sum(
                _value(variables["batch_work"][k, s, t, xi])
                for s in data.S
                for t in data.T
            )
            residual = processed - data.W[k]
            maximum_batch_completion_residual = max(
                maximum_batch_completion_residual, abs(residual)
            )
            batch_completion_rows.append(
                {
                    "scenario_id": xi,
                    "family_id": k,
                    "required_workload": data.W[k],
                    "processed_workload": processed,
                    "completion_residual": residual,
                }
            )
    pd.DataFrame(
        batch_completion_rows,
        columns=[
            "scenario_id",
            "family_id",
            "required_workload",
            "processed_workload",
            "completion_residual",
        ],
    ).to_csv(output / "batch_completion_audit.csv", index=False)

    load_rows: list[dict[str, Any]] = []
    energy_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    maximum_positive_part_residual = 0.0
    maximum_energy_definition_residual = 0.0
    maximum_cpu_capacity_violation = 0.0
    maximum_mem_capacity_violation = 0.0
    maximum_excess_branch_server_on_violation = 0.0
    for xi in data.Xi:
        scenario_spot_revenue = 0.0
        scenario_energy = 0.0
        scenario_energy_cost = 0.0
        scenario_excess = 0.0
        scenario_od_demand = 0.0
        scenario_spot_slots = 0.0
        accepted_spot_slots = sum(
            spot_accepted[j] * len(data.active_spot[j]) for j in data.J
        )
        for t in data.T:
            energy = _value(variables["energy"][t, xi])
            idle_component = sum(
                data.E_idle * _value(variables["u"][s, t]) for s in data.S
            )
            cpu_component = sum(
                data.E_cpu * _value(variables["total_load"][s, CPU, t, xi])
                for s in data.S
            )
            migration_component = sum(
                data.c_mig
                * data.load_od[i, MEM, t, xi]
                * _value(variables["migration"][i, s, t, xi])
                for i in data.I
                for s in data.S
                if (i, s, t, xi) in variables["migration"]
            )
            energy_definition_residual = energy - (
                idle_component + cpu_component + migration_component
            )
            maximum_energy_definition_residual = max(
                maximum_energy_definition_residual, abs(energy_definition_residual)
            )
            energy_rows.append(
                {
                    "scenario_id": xi,
                    "t": t,
                    "migration_coefficient": migration_coefficient,
                    "migration_coefficient_unit": migration_unit,
                    "idle_energy_kwh": idle_component,
                    "served_cpu_energy_kwh": cpu_component,
                    "migration_energy_kwh": migration_component,
                    "energy_kwh": energy,
                    "energy_definition_residual": energy_definition_residual,
                    "rt_price_usd_per_kwh": data.p_rt[t, xi],
                    "rt_energy_cost_usd": data.p_rt[t, xi] * energy,
                }
            )
            scenario_energy += energy
            scenario_energy_cost += data.p_rt[t, xi] * energy
            for j in data.J:
                if t in data.active_spot[j]:
                    active = sum(_value(variables["y"][j, s, t, xi]) for s in data.S)
                    scenario_spot_slots += active
                    scenario_spot_revenue += data.pi_spot[j] * active
            for s in data.S:
                od_cpu_demand = sum(
                    data.load_od[i, CPU, t, xi] * _value(variables["x"][i, s, t, xi])
                    for i in data.I
                    if t in data.active_od[i]
                )
                server_on = _value(variables["u"][s, t])
                branch_value = _value(variables["excess_branch"][s, t, xi])
                maximum_excess_branch_server_on_violation = max(
                    maximum_excess_branch_server_on_violation,
                    branch_value - server_on,
                )
                actual_excess = _value(variables["excess"][s, t, xi])
                exact_positive_part = max(0.0, od_cpu_demand - data.C_cpu * server_on)
                residual = actual_excess - exact_positive_part
                maximum_positive_part_residual = max(
                    maximum_positive_part_residual, abs(residual)
                )
                scenario_excess += actual_excess
                scenario_od_demand += od_cpu_demand
                total_cpu_load = _value(variables["total_load"][s, CPU, t, xi])
                total_mem_load = _value(variables["total_load"][s, MEM, t, xi])
                maximum_cpu_capacity_violation = max(
                    maximum_cpu_capacity_violation,
                    total_cpu_load - data.C_cpu * server_on,
                )
                maximum_mem_capacity_violation = max(
                    maximum_mem_capacity_violation,
                    total_mem_load - data.C_mem * server_on,
                )
                load_rows.append(
                    {
                        "scenario_id": xi,
                        "server_id": s,
                        "t": t,
                        "server_on": server_on,
                        "positive_branch": round(branch_value),
                        "od_cpu_demand": od_cpu_demand,
                        "od_cpu_served": _value(variables["od_served"][s, CPU, t, xi]),
                        "od_mem_served": _value(variables["od_served"][s, MEM, t, xi]),
                        "cpu_excess": actual_excess,
                        "exact_positive_part": exact_positive_part,
                        "positive_part_residual": residual,
                        "total_cpu_load": total_cpu_load,
                        "total_mem_load": total_mem_load,
                    }
                )
        scenario_profit = (
            artifacts.objective_constants["on_demand_revenue"]
            + artifacts.objective_constants["batch_revenue"]
            + scenario_spot_revenue
            - scenario_energy_cost
        )
        scenario_rows.append(
            {
                "scenario_id": xi,
                "probability": data.p[xi],
                "profit_usd": scenario_profit,
                "constant_on_demand_revenue_usd": artifacts.objective_constants[
                    "on_demand_revenue"
                ],
                "constant_batch_revenue_usd": artifacts.objective_constants[
                    "batch_revenue"
                ],
                "spot_revenue_usd": scenario_spot_revenue,
                "rt_energy_cost_usd": scenario_energy_cost,
                "energy_kwh": scenario_energy,
                "od_cpu_demand": scenario_od_demand,
                "od_cpu_excess": scenario_excess,
                "od_cpu_excess_rate": (
                    scenario_excess / scenario_od_demand
                    if scenario_od_demand > tolerance
                    else 0.0
                ),
                "spot_active_slots": scenario_spot_slots,
                "spot_accepted_possible_slots": accepted_spot_slots,
                "spot_service_rate": (
                    scenario_spot_slots / accepted_spot_slots
                    if accepted_spot_slots > tolerance
                    else 0.0
                ),
            }
        )
    load_frame = pd.DataFrame(load_rows)
    scenario_frame = pd.DataFrame(scenario_rows)
    load_frame.to_csv(output / "server_scenario_load.csv", index=False)
    pd.DataFrame(energy_rows).to_csv(output / "energy_rt_cost.csv", index=False)
    scenario_frame.to_csv(output / "scenario_metrics.csv", index=False)

    risk_rows: list[dict[str, Any]] = []
    for s in data.S:
        for t in data.T:
            losses = {
                xi: _value(variables["excess"][s, t, xi]) / data.C_cpu
                for xi in data.Xi
            }
            empirical_var, empirical_cvar = _weighted_var_cvar(
                losses, data.p, data.alpha
            )
            model_expression = _value(variables["eta"][s, t]) + sum(
                data.p[xi] * _value(variables["zeta"][s, t, xi])
                for xi in data.Xi
            ) / (1.0 - data.alpha)
            risk_rows.append(
                {
                    "server_id": s,
                    "t": t,
                    "model_eta": _value(variables["eta"][s, t]),
                    "model_cvar_expression": model_expression,
                    "empirical_var": empirical_var,
                    "empirical_cvar": empirical_cvar,
                    "epsilon": data.epsilon,
                    "maximum_excess_ratio": max(losses.values()),
                    "expected_excess_ratio": sum(
                        data.p[xi] * losses[xi] for xi in data.Xi
                    ),
                }
            )
    risk_frame = pd.DataFrame(risk_rows)
    risk_frame.to_csv(output / "cvar_by_server_time.csv", index=False)

    expected = lambda column: float(
        (scenario_frame[column] * scenario_frame["probability"]).sum()
    )
    expected_spot_revenue = expected("spot_revenue_usd")
    expected_energy_cost = expected("rt_energy_cost_usd")
    constant_revenue_offset = (
        artifacts.objective_constants["on_demand_revenue"]
        + artifacts.objective_constants["batch_revenue"]
    )
    reconstructed_model_objective = expected_spot_revenue - expected_energy_cost
    reconstructed_total_profit = (
        constant_revenue_offset + reconstructed_model_objective
    )
    expected_scenario_total_profit = expected("profit_usd")
    model_objective = float(model.ObjVal)
    model_best_bound = float(model.ObjBound)
    diagnostics = {
        "page_fidelity_policy": (
            "Migration uses the attached page's exact destination-entry indicator "
            "formulation with continuous [0,1] relaxation; batch startup remains "
            "a lower-bound-only event indicator."
        ),
        "maximum_exact_positive_part_residual": maximum_positive_part_residual,
        "maximum_energy_definition_residual": maximum_energy_definition_residual,
        "maximum_batch_completion_residual": maximum_batch_completion_residual,
        "maximum_cpu_capacity_violation": maximum_cpu_capacity_violation,
        "maximum_mem_capacity_violation": maximum_mem_capacity_violation,
        "maximum_excess_branch_server_on_violation": (
            maximum_excess_branch_server_on_violation
        ),
        "spot_arrival_violation_count": spot_arrival_violation_count,
        "spot_reactivation_violation_count": spot_reactivation_violation_count,
        "false_migration_indicator_count": false_migration_count,
        "maximum_exact_migration_definition_residual": (
            maximum_migration_definition_residual
        ),
        "false_batch_startup_indicator_count": false_startup_count,
        "migration_indicator_count": migration_indicator_count,
        "migration_relaxed_value_sum": migration_relaxed_value_sum,
        "maximum_migration_relaxation_fractionality": (
            maximum_migration_fractionality
        ),
        "maximum_spot_relaxation_fractionality": maximum_spot_fractionality,
        "initial_assignment_packing_orbitope": variables["symmetry_audit"],
        "actual_server_change_count": actual_migration_count,
        "batch_startup_indicator_count": startup_indicator_count,
        "logical_batch_start_count": logical_start_count,
        "initial_servers_on_without_initial_state_constraint": sum(
            round(_value(variables["u"][s, data.T[0]])) for s in data.S
        ),
        "migration_coefficient": migration_coefficient,
        "migration_coefficient_unit": migration_unit,
        "model_fidelity_warnings": data.metadata.get("fidelity_warnings", {}),
    }
    _write_json(output / "model_validation_diagnostics.json", diagnostics)

    summary = {
        "solver": {
            "status_code": int(model.Status),
            "status": _status_name(int(model.Status)),
            "solution_count": int(model.SolCount),
            "objective": model_objective,
            "best_bound": model_best_bound,
            "mip_gap": float(model.MIPGap),
            "objective_scope": (
                "expected_spot_revenue_minus_expected_rt_energy_cost_"
                "excluding_constant_revenue"
            ),
            "objective_sense": "MAXIMIZE",
            "objective_excludes_constant_revenue": True,
            "constant_revenue_offset": constant_revenue_offset,
            "total_profit_including_constant_revenue": (
                model_objective + constant_revenue_offset
            ),
            "best_bound_including_constant_revenue": (
                model_best_bound + constant_revenue_offset
            ),
            "runtime_seconds": float(model.Runtime),
            "node_count": float(model.NodeCount),
            "simplex_iterations": float(model.IterCount),
            "barrier_iterations": int(model.BarIterCount),
            "num_variables": int(model.NumVars),
            "num_binary_variables": int(model.NumBinVars),
            "num_constraints": int(model.NumConstrs),
            "num_general_constraints": int(model.NumGenConstrs),
            "num_nonzeros": int(model.NumNZs),
            "effective_parameters": {
                "mip_gap_target": float(model.Params.MIPGap),
                "threads": int(model.Params.Threads),
                "time_limit_seconds": float(model.Params.TimeLimit),
                "no_rel_heur_time_seconds": float(
                    model.Params.NoRelHeurTime
                ),
                "seed": int(model.Params.Seed),
                "numeric_focus": int(model.Params.NumericFocus),
                "presolve": int(model.Params.Presolve),
            },
        },
        "objective_components": {
            "constant_on_demand_revenue": artifacts.objective_constants[
                "on_demand_revenue"
            ],
            "constant_batch_revenue": artifacts.objective_constants["batch_revenue"],
            "expected_spot_revenue": expected_spot_revenue,
            "expected_rt_energy_cost": expected_energy_cost,
            "operating_margin_without_constant_revenue": expected_spot_revenue
            - expected_energy_cost,
            "reconstructed_objective": reconstructed_model_objective,
            "reconstructed_total_profit": reconstructed_total_profit,
            "expected_scenario_total_profit": expected_scenario_total_profit,
            "objective_reconstruction_error": reconstructed_model_objective
            - model_objective,
            "total_profit_reconstruction_error": reconstructed_total_profit
            - (model_objective + constant_revenue_offset),
            "scenario_profit_reconstruction_error": (
                expected_scenario_total_profit - reconstructed_total_profit
            ),
        },
        "service": {
            "spot_acceptance_count": int(
                sum(spot_accepted.values())
            ),
            "spot_candidate_count": len(data.J),
            "expected_spot_service_rate": expected("spot_service_rate"),
            "expected_od_cpu_excess": expected("od_cpu_excess"),
            "expected_od_cpu_excess_rate": expected("od_cpu_excess_rate"),
            "maximum_empirical_server_time_cvar": float(
                risk_frame["empirical_cvar"].max()
            ),
            "maximum_model_server_time_cvar_expression": float(
                risk_frame["model_cvar_expression"].max()
            ),
            "cvar_bound": data.epsilon,
        },
        "operations": {
            "available_servers": len(data.S),
            "average_active_servers": float(
                pd.DataFrame(server_rows).groupby("t")["on"].sum().mean()
            ),
            "peak_active_servers": int(
                pd.DataFrame(server_rows).groupby("t")["on"].sum().max()
            ),
            "actual_server_change_count": actual_migration_count,
            "logical_batch_start_count": logical_start_count,
            "expected_energy_kwh": expected("energy_kwh"),
        },
        "units_and_fidelity": {
            "migration_coefficient": migration_coefficient,
            "migration_coefficient_unit": migration_unit,
            "energy_quantity_unit": "kWh",
        },
        "model_validation_diagnostics": diagnostics,
    }
    error = summary["objective_components"]["objective_reconstruction_error"]
    if abs(error) > 1e-5 * max(1.0, abs(model_objective)):
        raise AssertionError(
            f"reported components do not reconstruct the Gurobi objective: {error}"
        )
    total_profit_error = summary["objective_components"][
        "scenario_profit_reconstruction_error"
    ]
    if abs(total_profit_error) > 1e-5 * max(1.0, abs(reconstructed_total_profit)):
        raise AssertionError(
            "probability-weighted scenario profit does not reconstruct total profit: "
            f"{total_profit_error}"
        )
    if maximum_positive_part_residual > 1e-5:
        raise AssertionError(
            "indicator formulation failed exact positive-part audit: "
            f"max residual={maximum_positive_part_residual}"
        )
    if maximum_energy_definition_residual > 1e-5:
        raise AssertionError(
            "reported energy components do not reconstruct the model energy: "
            f"max residual={maximum_energy_definition_residual}"
        )
    if maximum_migration_definition_residual > 1e-5:
        raise AssertionError(
            "destination-indexed migration indicator failed exact-definition audit: "
            f"max residual={maximum_migration_definition_residual}"
        )
    if maximum_batch_completion_residual > 1e-5:
        raise AssertionError(
            "batch completion audit failed: "
            f"max residual={maximum_batch_completion_residual}"
        )
    if max(maximum_cpu_capacity_violation, maximum_mem_capacity_violation) > 1e-5:
        raise AssertionError(
            "hard server capacity audit failed: "
            f"cpu={maximum_cpu_capacity_violation}, mem={maximum_mem_capacity_violation}"
        )
    if maximum_excess_branch_server_on_violation > 1e-5:
        raise AssertionError(
            "excess positive-branch variable is active while its server is off: "
            f"max violation={maximum_excess_branch_server_on_violation}"
        )
    if maximum_spot_fractionality > 1e-5:
        raise AssertionError(
            "relaxed spot placement lost its structural integrality: "
            f"max fractionality={maximum_spot_fractionality}"
        )
    if spot_arrival_violation_count or spot_reactivation_violation_count:
        raise AssertionError(
            "spot lifecycle audit failed: "
            f"arrival={spot_arrival_violation_count}, reactivation={spot_reactivation_violation_count}"
        )
    maximum_empirical_cvar = float(risk_frame["empirical_cvar"].max())
    if maximum_empirical_cvar > data.epsilon + 1e-5:
        raise AssertionError(
            "empirical server-time CVaR exceeds epsilon: "
            f"max={maximum_empirical_cvar}, epsilon={data.epsilon}"
        )
    _write_json(output / "summary.json", summary)
    return summary


def write_no_solution_summary(model: Any, run_dir: str | Path) -> dict[str, Any]:
    output = Path(run_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "solver": {
            "status_code": int(model.Status),
            "status": _status_name(int(model.Status)),
            "solution_count": int(model.SolCount),
            "runtime_seconds": float(model.Runtime),
            "node_count": float(model.NodeCount),
            "num_variables": int(model.NumVars),
            "num_constraints": int(model.NumConstrs),
            "num_general_constraints": int(model.NumGenConstrs),
        }
    }
    _write_json(output / "summary.json", summary)
    return summary
