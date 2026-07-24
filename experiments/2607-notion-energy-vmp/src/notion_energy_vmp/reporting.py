from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import CPU, MEM
from .model import ModelArtifacts


def _x(var: Any) -> float:
    return float(var.X)


def _weighted_var_cvar(losses: dict[int, float], probabilities: dict[int, float], alpha: float) -> tuple[float, float]:
    ordered = sorted((float(losses[xi]), float(probabilities[xi])) for xi in losses)
    cumulative = 0.0
    var = ordered[-1][0]
    for loss, prob in ordered:
        cumulative += prob
        if cumulative + 1e-12 >= alpha:
            var = loss
            break
    cvar = var + sum(prob * max(0.0, loss - var) for loss, prob in ordered) / (1.0 - alpha)
    return var, cvar


def _status_name(status: int) -> str:
    try:
        from gurobipy import GRB

        mapping = {
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
        return mapping.get(status, f"STATUS_{status}")
    except ImportError:
        return str(status)


def write_solution_reports(artifacts: ModelArtifacts, run_dir: str | Path, *, tolerance: float = 1e-7) -> dict[str, Any]:
    model, d, v = artifacts.model, artifacts.data, artifacts.v
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    if model.SolCount <= 0:
        raise RuntimeError("Cannot write solution reports because no incumbent solution exists")

    model.write(str(out / "solution.sol"))

    server_rows = []
    for s in d.S:
        for t in d.T:
            server_rows.append(
                {
                    "server_id": s,
                    "t": t,
                    "on": round(_x(v["u"][s, t])),
                    "started": round(_x(v["u_on"][s, t])) if t in d.T[:-1] else 0,
                    "stopped": round(_x(v["u_off"][s, t])) if t in d.T[:-1] else 0,
                }
            )
    pd.DataFrame(server_rows).to_csv(out / "server_schedule.csv", index=False)

    od_initial_rows = []
    for i in d.I:
        chosen = [s for s in d.S if _x(v["x_init"][i, s]) > 0.5]
        od_initial_rows.append({"vm_id": i, "server_id": chosen[0] if chosen else None})
    pd.DataFrame(od_initial_rows).to_csv(out / "od_initial_placement.csv", index=False)

    migration_rows = []
    for (i, t, xi), var in v["mig"].items():
        if _x(var) > 0.5:
            old = next((s for s in d.S if _x(v["x"][i, s, t, xi]) > 0.5), None)
            new = next((s for s in d.S if _x(v["x"][i, s, t + 1, xi]) > 0.5), None)
            migration_rows.append(
                {
                    "scenario_id": xi,
                    "vm_id": i,
                    "from_t": t,
                    "to_t": t + 1,
                    "from_server": old,
                    "to_server": new,
                    "actual_server_change": int(old is not None and new is not None and old != new),
                    "mem_usage": d.load_od[(i, MEM, t, xi)],
                    "migration_energy": d.c_mig[i] * d.load_od[(i, MEM, t, xi)],
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
            "actual_server_change",
            "mem_usage",
            "migration_energy",
        ],
    ).to_csv(out / "migrations.csv", index=False)
    migration_indicator_count = {
        xi: sum(1 for row in migration_rows if row["scenario_id"] == xi) for xi in d.Xi
    }
    actual_migration_count = {
        xi: sum(row["actual_server_change"] for row in migration_rows if row["scenario_id"] == xi) for xi in d.Xi
    }

    spot_rows = []
    for j in d.J:
        accepted = round(_x(v["w"][j]))
        init_server = next((s for s in d.S if _x(v["y_init"][j, s]) > 0.5), None)
        for xi in d.Xi:
            for t in d.active_spot[j]:
                active_server = next((s for s in d.S if _x(v["y"][j, s, t, xi]) > 0.5), None)
                spot_rows.append(
                    {
                        "scenario_id": xi,
                        "vm_id": j,
                        "t": t,
                        "accepted": accepted,
                        "initial_server": init_server,
                        "active_server": active_server,
                        "preempted": round(_x(v["h"][j, t, xi])),
                        "data_available": d.spot_available[(j, t, xi)],
                        "slot_revenue": d.pi_spot[j] if active_server is not None else 0.0,
                    }
                )
    pd.DataFrame(spot_rows).to_csv(out / "spot_schedule.csv", index=False)

    batch_rows = []
    for k in d.K:
        for s in d.S:
            for t in d.T:
                prepared = _x(v["z"][k, s, t])
                started = _x(v["z_on"][k, s, t])
                works = {xi: _x(v["b"][k, s, t, xi]) for xi in d.Xi}
                if prepared > 0.5 or started > 0.5 or max(works.values(), default=0.0) > tolerance:
                    for xi in d.Xi:
                        batch_rows.append(
                            {
                                "scenario_id": xi,
                                "family_id": k,
                                "server_id": s,
                                "t": t,
                                "prepared": round(prepared),
                                "started": round(started),
                                "workload": works[xi],
                                "cpu_workload": d.rho_batch[(k, CPU)] * works[xi],
                                "mem_workload": d.rho_batch[(k, MEM)] * works[xi],
                            }
                        )
    pd.DataFrame(batch_rows).to_csv(out / "batch_schedule.csv", index=False)

    false_startup_rows = []
    for k in d.K:
        for s in d.S:
            for t in d.T:
                current = round(_x(v["z"][k, s, t]))
                previous = 0 if t == d.T[0] else round(_x(v["z"][k, s, t - 1]))
                startup = round(_x(v["z_on"][k, s, t]))
                logical_startup = int(current == 1 and previous == 0)
                if startup == 1 and logical_startup == 0:
                    false_startup_rows.append({"family_id": k, "server_id": s, "t": t, "prepared": current, "previous_prepared": previous})

    excess_rows = []
    energy_rows = []
    time_rows = []
    scenario_rows = []
    da_cost = sum(d.p_da[t] * _x(v["g_da"][t]) for t in d.T)

    for xi in d.Xi:
        scenario_spot_revenue = 0.0
        scenario_rt_cost = 0.0
        scenario_sale_revenue = 0.0
        scenario_excess_penalty = 0.0
        scenario_energy = 0.0
        scenario_da_energy = 0.0
        scenario_rt_energy = 0.0
        scenario_renewable_used = 0.0
        scenario_renewable_sold = 0.0
        scenario_renewable_available = 0.0
        scenario_solar_available = 0.0
        scenario_wind_available = 0.0
        scenario_charge = 0.0
        scenario_discharge = 0.0
        scenario_cpu_demand = 0.0
        scenario_cpu_served = 0.0
        scenario_cpu_excess = 0.0
        scenario_migrations = float(migration_indicator_count[xi])
        scenario_actual_migrations = float(actual_migration_count[xi])
        scenario_spot_slots = 0.0
        scenario_spot_possible_slots = sum(_x(v["w"][j]) * len(d.active_spot[j]) for j in d.J)
        peak_cpu_util = 0.0
        peak_mem_util = 0.0

        for t in d.T:
            active_servers = sum(_x(v["u"][s, t]) for s in d.S)
            cpu_load = sum(_x(v["load_total"][s, CPU, t, xi]) for s in d.S)
            mem_load = sum(_x(v["load_total"][s, MEM, t, xi]) for s in d.S)
            cpu_excess = sum(_x(v["excess"][s, t, xi]) for s in d.S)
            od_cpu_served = sum(_x(v["load_od"][s, CPU, t, xi]) for s in d.S)
            od_cpu_demand = od_cpu_served + cpu_excess
            spot_revenue = sum(
                d.pi_spot[j] * _x(v["y"][j, s, t, xi])
                for j in d.J
                if t in d.active_spot[j]
                for s in d.S
            )
            spot_slots = sum(
                _x(v["y"][j, s, t, xi]) for j in d.J if t in d.active_spot[j] for s in d.S
            )
            e = _x(v["energy"][t, xi])
            grt = _x(v["g_rt"][t, xi])
            gda = _x(v["g_da"][t])
            used = _x(v["r_used"][t, xi])
            sold = _x(v["r_sold"][t, xi])
            char = _x(v["charge"][t, xi])
            dis = _x(v["discharge"][t, xi])
            available = d.renewable[(t, xi)]
            solar_available = d.solar_renewable[(t, xi)]
            wind_available = d.wind_renewable[(t, xi)]
            rt_cost = d.p_rt[(t, xi)] * grt
            sale_revenue = d.p_sell[t] * sold
            excess_penalty = d.lambda_exc * cpu_excess
            peak_cpu_util = max(peak_cpu_util, max((_x(v["load_total"][s, CPU, t, xi]) / d.C[CPU] for s in d.S), default=0.0))
            peak_mem_util = max(peak_mem_util, max((_x(v["load_total"][s, MEM, t, xi]) / d.C[MEM] for s in d.S), default=0.0))

            for s in d.S:
                excess_rows.append(
                    {
                        "scenario_id": xi,
                        "server_id": s,
                        "t": t,
                        "cpu_excess": _x(v["excess"][s, t, xi]),
                        "od_cpu_served": _x(v["load_od"][s, CPU, t, xi]),
                        "total_cpu_load": _x(v["load_total"][s, CPU, t, xi]),
                        "total_mem_load": _x(v["load_total"][s, MEM, t, xi]),
                    }
                )
            energy_rows.append(
                {
                    "scenario_id": xi,
                    "t": t,
                    "energy_kwh": e,
                    "day_ahead_kwh": gda,
                    "real_time_kwh": grt,
                    "renewable_available_kwh": available,
                    "solar_available_kwh": solar_available,
                    "wind_available_kwh": wind_available,
                    "renewable_used_kwh": used,
                    "renewable_sold_kwh": sold,
                    "renewable_curtailed_kwh": available - used - sold,
                    "ess_charge_kwh": char,
                    "ess_discharge_kwh": dis,
                    "soc_start_kwh": _x(v["soc"][t, xi]),
                    "soc_end_kwh": _x(v["soc"][t + 1, xi]),
                    "da_price_usd_per_kwh": d.p_da[t],
                    "rt_price_usd_per_kwh": d.p_rt[(t, xi)],
                    "sell_price_usd_per_kwh": d.p_sell[t],
                }
            )
            time_rows.append(
                {
                    "scenario_id": xi,
                    "probability": d.p[xi],
                    "t": t,
                    "active_servers": active_servers,
                    "cpu_load": cpu_load,
                    "mem_load": mem_load,
                    "od_cpu_demand": od_cpu_demand,
                    "od_cpu_served": od_cpu_served,
                    "od_cpu_excess": cpu_excess,
                    "spot_active_slots": spot_slots,
                    "energy_kwh": e,
                    "grid_da_kwh": gda,
                    "grid_rt_kwh": grt,
                    "renewable_used_kwh": used,
                    "renewable_sold_kwh": sold,
                    "ess_charge_kwh": char,
                    "ess_discharge_kwh": dis,
                    "spot_revenue": spot_revenue,
                    "rt_cost": rt_cost,
                    "sale_revenue": sale_revenue,
                    "excess_penalty": excess_penalty,
                }
            )

            scenario_spot_revenue += spot_revenue
            scenario_rt_cost += rt_cost
            scenario_sale_revenue += sale_revenue
            scenario_excess_penalty += excess_penalty
            scenario_energy += e
            scenario_da_energy += gda
            scenario_rt_energy += grt
            scenario_renewable_used += used
            scenario_renewable_sold += sold
            scenario_renewable_available += available
            scenario_solar_available += solar_available
            scenario_wind_available += wind_available
            scenario_charge += char
            scenario_discharge += dis
            scenario_cpu_demand += od_cpu_demand
            scenario_cpu_served += od_cpu_served
            scenario_cpu_excess += cpu_excess
            scenario_spot_slots += spot_slots

        scenario_rows.append(
            {
                "scenario_id": xi,
                "probability": d.p[xi],
                "profit_including_common_da": scenario_spot_revenue - scenario_rt_cost + scenario_sale_revenue - scenario_excess_penalty - da_cost,
                "spot_revenue": scenario_spot_revenue,
                "day_ahead_cost_common": da_cost,
                "real_time_cost": scenario_rt_cost,
                "renewable_sale_revenue": scenario_sale_revenue,
                "cpu_excess_penalty": scenario_excess_penalty,
                "energy_kwh": scenario_energy,
                "grid_da_kwh": scenario_da_energy,
                "grid_rt_kwh": scenario_rt_energy,
                "renewable_available_kwh": scenario_renewable_available,
                "solar_available_kwh": scenario_solar_available,
                "wind_available_kwh": scenario_wind_available,
                "renewable_used_kwh": scenario_renewable_used,
                "renewable_sold_kwh": scenario_renewable_sold,
                "renewable_curtailed_kwh": scenario_renewable_available - scenario_renewable_used - scenario_renewable_sold,
                "ess_charge_kwh": scenario_charge,
                "ess_discharge_kwh": scenario_discharge,
                "terminal_soc_kwh": _x(v["soc"][len(d.T), xi]),
                "od_cpu_demand": scenario_cpu_demand,
                "od_cpu_served": scenario_cpu_served,
                "od_cpu_excess": scenario_cpu_excess,
                "od_cpu_excess_rate": scenario_cpu_excess / scenario_cpu_demand if scenario_cpu_demand > tolerance else 0.0,
                "migrations": scenario_migrations,
                "actual_migrations": scenario_actual_migrations,
                "nonmovement_migration_indicators": scenario_migrations - scenario_actual_migrations,
                "spot_active_slots": scenario_spot_slots,
                "spot_possible_slots": scenario_spot_possible_slots,
                "spot_service_rate": scenario_spot_slots / scenario_spot_possible_slots if scenario_spot_possible_slots > tolerance else 0.0,
                "peak_server_cpu_utilization": peak_cpu_util,
                "peak_server_mem_utilization": peak_mem_util,
            }
        )

    pd.DataFrame(excess_rows).to_csv(out / "excess_load.csv", index=False)
    pd.DataFrame(energy_rows).to_csv(out / "energy_dispatch.csv", index=False)
    time_df = pd.DataFrame(time_rows)
    scenario_df = pd.DataFrame(scenario_rows)
    time_df.to_csv(out / "time_scenario_metrics.csv", index=False)
    scenario_df.to_csv(out / "scenario_metrics.csv", index=False)

    simultaneous_ess_rows = [
        {
            "scenario_id": row["scenario_id"],
            "t": row["t"],
            "charge_kwh": row["ess_charge_kwh"],
            "discharge_kwh": row["ess_discharge_kwh"],
        }
        for row in energy_rows
        if row["ess_charge_kwh"] > tolerance and row["ess_discharge_kwh"] > tolerance
    ]
    spot_after_observed_preemption = [
        row for row in spot_rows if row["accepted"] == 1 and row["active_server"] is not None and row["data_available"] == 0
    ]
    endogenous_early_preemption = [
        row for row in spot_rows if row["accepted"] == 1 and row["preempted"] == 1 and row["data_available"] == 1
    ]
    validation_diagnostics = {
        "policy": (
            "Detected only. Spot interruption data is an external reference trace; endogenous preemption is a model decision."
        ),
        "false_batch_startup_count": len(false_startup_rows),
        "simultaneous_ess_charge_discharge_slot_count": len(simultaneous_ess_rows),
        "spot_active_after_observed_unavailability_slot_count": len(spot_after_observed_preemption),
        "endogenous_preemption_before_observed_unavailability_slot_count": len(endogenous_early_preemption),
        "initial_servers_on_without_initial_state_constraint": int(sum(_x(v["u"][s, d.T[0]]) > 0.5 for s in d.S)),
        "nonmovement_migration_indicator_count": int(
            sum(row["actual_server_change"] == 0 for row in migration_rows)
        ),
        "migration_coefficient_units": d.metadata["model_fidelity_warnings"]["migration_coefficient_units"],
        "details": {
            "false_batch_startups": false_startup_rows,
            "simultaneous_ess_charge_discharge": simultaneous_ess_rows,
            "spot_active_after_observed_unavailability": spot_after_observed_preemption,
            "endogenous_early_preemption": endogenous_early_preemption,
        },
    }
    (out / "model_validation_diagnostics.json").write_text(
        json.dumps(validation_diagnostics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    risk_rows = []
    for t in d.T:
        losses = {
            xi: float(time_df.loc[(time_df["scenario_id"] == xi) & (time_df["t"] == t), "od_cpu_excess"].iloc[0])
            for xi in d.Xi
        }
        empirical_var, empirical_cvar = _weighted_var_cvar(losses, d.p, d.alpha)
        risk_rows.append(
            {
                "t": t,
                "model_eta": _x(v["var_eta"][t]),
                "model_cvar_expression": _x(v["var_eta"][t])
                + sum(d.p[xi] * _x(v["cvar_zeta"][t, xi]) for xi in d.Xi) / (1.0 - d.alpha),
                "empirical_var": empirical_var,
                "empirical_cvar": empirical_cvar,
                "epsilon": d.epsilon,
                "max_excess": max(losses.values()),
                "expected_excess": sum(d.p[xi] * losses[xi] for xi in d.Xi),
            }
        )
    risk_df = pd.DataFrame(risk_rows)
    risk_df.to_csv(out / "cvar_by_time.csv", index=False)

    expected = lambda column: float((scenario_df[column] * scenario_df["probability"]).sum())
    expected_spot_revenue = expected("spot_revenue")
    expected_rt_cost = expected("real_time_cost")
    expected_sale = expected("renewable_sale_revenue")
    expected_excess_penalty = expected("cpu_excess_penalty")
    reconstructed_objective = expected_spot_revenue - da_cost - expected_rt_cost + expected_sale - expected_excess_penalty

    summary = {
        "solver": {
            "status_code": int(model.Status),
            "status": _status_name(int(model.Status)),
            "solution_count": int(model.SolCount),
            "objective": float(model.ObjVal),
            "best_bound": float(model.ObjBound),
            "mip_gap": float(model.MIPGap),
            "runtime_seconds": float(model.Runtime),
            "node_count": float(model.NodeCount),
            "simplex_iterations": float(model.IterCount),
            "barrier_iterations": int(model.BarIterCount),
            "num_variables": int(model.NumVars),
            "num_binary_variables": int(model.NumBinVars),
            "num_integer_variables": int(model.NumIntVars),
            "num_constraints": int(model.NumConstrs),
            "num_nonzeros": int(model.NumNZs),
        },
        "objective_components": {
            "expected_spot_revenue": expected_spot_revenue,
            "day_ahead_cost": da_cost,
            "expected_real_time_cost": expected_rt_cost,
            "expected_renewable_sale_revenue": expected_sale,
            "expected_cpu_excess_penalty": expected_excess_penalty,
            "reconstructed_objective": reconstructed_objective,
            "objective_reconstruction_error": reconstructed_objective - float(model.ObjVal),
        },
        "service": {
            "spot_acceptance_count": int(round(sum(_x(v["w"][j]) for j in d.J))),
            "spot_candidate_count": len(d.J),
            "expected_spot_service_rate": expected("spot_service_rate"),
            "expected_od_cpu_excess": expected("od_cpu_excess"),
            "expected_od_cpu_excess_rate": expected("od_cpu_excess_rate"),
            "max_time_empirical_cvar": float(risk_df["empirical_cvar"].max()),
            "max_time_model_cvar_expression": float(risk_df["model_cvar_expression"].max()),
            "cvar_bound": d.epsilon,
        },
        "operations": {
            "expected_migrations": expected("migrations"),
            "expected_actual_migrations": expected("actual_migrations"),
            "expected_nonmovement_migration_indicators": expected("nonmovement_migration_indicators"),
            "average_active_servers": float(pd.DataFrame(server_rows)["on"].mean() * len(d.S)),
            "peak_active_servers": int(pd.DataFrame(server_rows).groupby("t")["on"].sum().max()),
            "batch_prepared_server_slots": int(round(sum(_x(v["z"][k, s, t]) for k in d.K for s in d.S for t in d.T))),
            "batch_startups": int(round(sum(_x(v["z_on"][k, s, t]) for k in d.K for s in d.S for t in d.T))),
        },
        "energy": {
            "expected_consumption_kwh": expected("energy_kwh"),
            "expected_grid_da_kwh": expected("grid_da_kwh"),
            "expected_grid_rt_kwh": expected("grid_rt_kwh"),
            "expected_renewable_used_kwh": expected("renewable_used_kwh"),
            "expected_renewable_sold_kwh": expected("renewable_sold_kwh"),
            "expected_renewable_curtailed_kwh": expected("renewable_curtailed_kwh"),
            "expected_solar_available_kwh": expected("solar_available_kwh"),
            "expected_wind_available_kwh": expected("wind_available_kwh"),
            "expected_ess_charge_kwh": expected("ess_charge_kwh"),
            "expected_ess_discharge_kwh": expected("ess_discharge_kwh"),
            "renewable_coverage_ratio": expected("renewable_used_kwh") / expected("energy_kwh") if expected("energy_kwh") > tolerance else 0.0,
        },
        "model_validation_diagnostics": {
            key: value for key, value in validation_diagnostics.items() if key != "details"
        },
    }
    if abs(summary["objective_components"]["objective_reconstruction_error"]) > 1e-5 * max(1.0, abs(model.ObjVal)):
        raise AssertionError("Extracted objective components do not reconstruct Gurobi objective")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def write_no_solution_summary(model: Any, run_dir: str | Path) -> dict[str, Any]:
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "solver": {
            "status_code": int(model.Status),
            "status": _status_name(int(model.Status)),
            "solution_count": int(model.SolCount),
            "runtime_seconds": float(model.Runtime),
            "node_count": float(model.NodeCount),
            "num_variables": int(model.NumVars),
            "num_constraints": int(model.NumConstrs),
        }
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
