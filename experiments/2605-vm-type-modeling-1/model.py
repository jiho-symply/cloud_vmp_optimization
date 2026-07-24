"""
Gurobi implementation of the Notion page "VM type modeling (1)".

The model is intentionally written as a readable research prototype.  It is
not decomposed into many tiny helpers, and most constraints appear in the same
order as the Notion page:

1. First-stage placement/reservation decisions
   - x[i,s]: initial on-demand VM placement
   - y[j,s]: spot VM placement
   - b[k,s,t]: reserved batch-VM slot for job k
   - u[s,t], u_used[s]: server power state and horizon-level use flag

2. Second-stage scenario decisions
   - xR[i,s,t,xi]: on-demand VM's actual server after optional migration
   - m[i,t,xi]: 1 if on-demand VM i changes server at time t
   - yR[j,s,t,xi]: spot VM active state after server-level suspension
   - z[k,s,t,xi]: processed batch workload volume
   - gamma/phi/eta/delta: suspension and SLA violation indicators

Important modeling choices made from the latest Notion page:

- The current page uses x[i,s] as first-stage on-demand placement and
  xR[i,s,t,xi] as realized scenario placement.  Migration is represented by
  a change detector m[i,t,xi], not by explicit server-to-server arcs.
- The page states that migration is limited to at most once per VM.  The code
  implements this as sum_t m[i,t,xi] <= 1 for each VM and scenario.  Migration
  variables are created only from the second active period onward, because the
  first active period is fixed to the first-stage placement.
- Batch z is linked to both reservation b and spot-like suspension gamma:
  z[k,s,t,xi] <= r_B[k] * b[k,s,t]
  z[k,s,t,xi] <= r_B[k] * (1 - gamma[s,t,xi])
- Implementation-required: xR[i,s,t,xi] <= u[s,t] is kept so realized
  on-demand placement cannot use a powered-off server.
- Implementation-required: the current Notion energy objective uses capped
  load barL[s,t,xi].  Therefore this file defines barL = min(L, C u);
  config selects either the indicator or big-M implementation.
- Scenario probabilities are used in the chance constraints and in the CPU
  energy term exactly as shown in the updated Notion objective.
- The migration-energy term is counted once per positive m[i,t,xi].  This
  avoids multiplying the same migration by every server-time pair.
- Homogeneous-server symmetry breaking is controlled by config.  The available
  modes are none, horizon-level use ordering, powered-slot ordering, or both.
- Additional model-meaningfulness constraints are marked below.  They make
  loose indicator/state variables exact where possible, prevent post-recourse
  low-priority workload from exceeding server capacity, and remove dominated
  idle-on server states.  These are not copied from the Notion page verbatim.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB


EXPERIMENT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = EXPERIMENT_DIR.parent
REPO_ROOT = EXPERIMENTS_DIR.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from config_2605 import (  # noqa: E402
    apply_solver_config,
    formulation_config,
    load_config,
    resolve_path,
    solver_config,
)
DEFAULT_CONFIG = EXPERIMENT_DIR / "config.json"
DEFAULT_INSTANCE = (
    REPO_ROOT
    / "data"
    / "processed"
    / "2605-vm-type-modeling-1"
    / "notion_vm_type_24vm_od8_sp8_bj8_sc10_cap8"
    / "vm_type_instance.json"
)
DEFAULT_RESULTS_DIR = REPO_ROOT / "experiments" / "2605-vm-type-modeling-1" / "results"


def parse_args():
    parser = argparse.ArgumentParser(description="Solve the Notion VM type modeling (1) MILP.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--instance", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--time-limit", type=float, default=None)
    parser.add_argument("--mip-gap", type=float, default=None)
    parser.add_argument("--threads", type=int, default=None)
    return parser.parse_args()


def load_data(path):
    with open(path, "r", encoding="utf-8") as file:
        raw = json.load(file)

    params = raw["parameters"]
    scenarios = raw["sets"]["scenarios"]
    scenario_prob = {row["id"]: float(row["probability"]) for row in raw["scenarios"]}
    d_od = {(row["id"], row["time"], row["scenario"]): row["demand"] for row in raw["demands"]["on_demand"]}
    d_sp = {(row["id"], row["time"], row["scenario"]): row["demand"] for row in raw["demands"]["spot"]}
    batch = {row["id"]: row for row in raw["batch_jobs"]}

    return {
        "raw": raw,
        "servers": raw["sets"]["servers"],
        "times": raw["sets"]["times"],
        "on_demand": raw["sets"]["on_demand"],
        "spot": raw["sets"]["spot"],
        "batch": raw["sets"]["batch"],
        "scenarios": scenarios,
        "scenario_prob": scenario_prob,
        "od_active": {key: value for key, value in raw["active_periods"]["on_demand"].items()},
        "spot_active": {key: value for key, value in raw["active_periods"]["spot"].items()},
        "batch_info": batch,
        "d_od": d_od,
        "d_sp": d_sp,
        "capacity": float(params["capacity"]),
        "big_m": float(params["big_m"]),
        "epsilon_od": float(params["epsilon_od"]),
        "epsilon_sp": float(params["epsilon_sp"]),
        "rho": float(params["rho"]),
        "lambda_migration": float(params["lambda_migration"]),
        "objective_type": params["objective_type"],
        "energy_idle": float(params["energy_idle"]),
        "energy_cpu": float(params["energy_cpu"]),
        "energy_migration": float(params["energy_migration"]),
    }


def status_name(status):
    names = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INTERRUPTED: "INTERRUPTED",
    }
    return names.get(status, str(status))


def build_model(data, formulation=None):
    formulation = formulation or {}
    bar_load_mode = formulation.get("bar_load_mode", "indicator")
    server_symmetry = formulation.get("server_symmetry", "used_and_powered_slots")
    if bar_load_mode not in {"indicator", "big_m"}:
        raise ValueError("model.formulation.bar_load_mode must be 'indicator' or 'big_m'.")
    if server_symmetry not in {"none", "used", "powered_slots", "used_and_powered_slots"}:
        raise ValueError(
            "model.formulation.server_symmetry must be one of "
            "'none', 'used', 'powered_slots', 'used_and_powered_slots'."
        )

    S = data["servers"]
    T = data["times"]
    I = data["on_demand"]
    J = data["spot"]
    K = data["batch"]
    Xi = data["scenarios"]
    T_i = data["od_active"]
    T_j = data["spot_active"]
    prob = data["scenario_prob"]
    C = data["capacity"]
    M = data["big_m"]

    model = gp.Model("vm_type_modeling_1")

    # First-stage variables.
    u = model.addVars(S, T, vtype=GRB.BINARY, name="u")
    u_used = model.addVars(S, vtype=GRB.BINARY, name="u_used")
    x = model.addVars(I, S, vtype=GRB.BINARY, name="x")
    y = model.addVars(J, S, vtype=GRB.BINARY, name="y")
    b = model.addVars(K, S, T, vtype=GRB.BINARY, name="b")

    # Second-stage variables.
    xR = model.addVars(
        [(i, s, t, xi) for i in I for s in S for t in T_i[i] for xi in Xi],
        vtype=GRB.BINARY,
        name="xR",
    )
    m = model.addVars(
        [(i, t, xi) for i in I for t in T_i[i][1:] for xi in Xi],
        vtype=GRB.BINARY,
        name="m",
    )
    yR = model.addVars(
        [(j, s, t, xi) for j in J for s in S for t in T_j[j] for xi in Xi],
        vtype=GRB.BINARY,
        name="yR",
    )
    z = model.addVars(K, S, T, Xi, lb=0.0, name="z")
    gamma = model.addVars(S, T, Xi, vtype=GRB.BINARY, name="gamma")
    phi = model.addVars(S, T, Xi, vtype=GRB.BINARY, name="phi")
    eta = model.addVars(S, Xi, vtype=GRB.BINARY, name="eta")
    delta = model.addVars(J, Xi, vtype=GRB.BINARY, name="delta")
    od_load = model.addVars(S, T, Xi, lb=0.0, name="od_load")
    total_load = model.addVars(S, T, Xi, lb=0.0, name="total_load")
    bar_load = model.addVars(S, T, Xi, lb=0.0, name="barL")
    cap_select = model.addVars(S, T, Xi, vtype=GRB.BINARY, name="cap_select")

    max_od_load = sum(max(data["d_od"].get((i, t, xi), 0.0) for t in T_i[i] for xi in Xi) for i in I)
    max_spot_load = sum(max(data["d_sp"].get((j, t, xi), 0.0) for t in T_j[j] for xi in Xi) for j in J)
    max_batch_load = sum(data["batch_info"][k]["reserved_cpu"] for k in K)
    cap_m = max(M, max_od_load + max_spot_load + max_batch_load + C)

    # First-stage placement and reservation constraints.
    model.addConstrs(gp.quicksum(x[i, s] for s in S) == 1 for i in I)
    model.addConstrs(gp.quicksum(y[j, s] for s in S) == 1 for j in J)
    model.addConstrs(y[j, s] <= u[s, t] for j in J for s in S for t in T_j[j])
    model.addConstrs(b[k, s, t] <= u[s, t] for k in K for s in S for t in T)
    model.addConstrs(
        gp.quicksum(b[k, s, t] for s in S for t in T) <= data["batch_info"][k]["processing_slots"]
        for k in K
    )
    model.addConstrs(u_used[s] >= u[s, t] for s in S for t in T)
    model.addConstrs(u_used[s] <= gp.quicksum(u[s, t] for t in T) for s in S)

    # On-demand migration.  The latest Notion notation uses m[i,t,xi] as a
    # server-change detector between consecutive active periods.
    model.addConstrs(
        gp.quicksum(xR[i, s, t, xi] for s in S) == 1
        for i in I
        for t in T_i[i]
        for xi in Xi
    )
    for i in I:
        first_t = T_i[i][0]
        model.addConstrs(xR[i, s, first_t, xi] == x[i, s] for s in S for xi in Xi)
        for prev_t, t in zip(T_i[i][:-1], T_i[i][1:]):
            for s in S:
                for xi in Xi:
                    model.addConstr(m[i, t, xi] >= xR[i, s, t, xi] - xR[i, s, prev_t, xi])
                    model.addConstr(m[i, t, xi] >= xR[i, s, prev_t, xi] - xR[i, s, t, xi])
                    # Optional exactness upper bound, intentionally disabled to
                    # match the current formulation.  The active lower bounds
                    # force m=1 on a placement change, but allow m=1 without a
                    # change when migration has zero objective cost.
                    # model.addConstr(m[i, t, xi] <= 2 - xR[i, s, prev_t, xi] - xR[i, s, t, xi])
        model.addConstrs(
            gp.quicksum(m[i, t, xi] for t in T_i[i][1:]) <= 1
            for xi in Xi
        )
    model.addConstrs(xR[i, s, t, xi] <= u[s, t] for i in I for s in S for t in T_i[i] for xi in Xi)

    # Spot active linking, using the Notion symbol yR.
    model.addConstrs(yR[j, s, t, xi] <= y[j, s] for j in J for s in S for t in T_j[j] for xi in Xi)
    model.addConstrs(yR[j, s, t, xi] <= 1 - gamma[s, t, xi] for j in J for s in S for t in T_j[j] for xi in Xi)
    model.addConstrs(yR[j, s, t, xi] >= y[j, s] - gamma[s, t, xi] for j in J for s in S for t in T_j[j] for xi in Xi)

    # Batch processing is allowed only on reserved slots and is stopped by gamma.
    for k in K:
        r_b = data["batch_info"][k]["reserved_cpu"]
        model.addConstrs(z[k, s, t, xi] <= r_b * b[k, s, t] for s in S for t in T for xi in Xi)
        model.addConstrs(z[k, s, t, xi] <= r_b * (1 - gamma[s, t, xi]) for s in S for t in T for xi in Xi)
        model.addConstrs(gp.quicksum(z[k, s, t, xi] for s in S for t in T) >= data["batch_info"][k]["workload"] for xi in Xi)

    # Server load and chance constraints.
    for s in S:
        for t in T:
            for xi in Xi:
                model.addConstr(
                    od_load[s, t, xi]
                    == gp.quicksum(data["d_od"].get((i, t, xi), 0.0) * xR[i, s, t, xi] for i in I if t in T_i[i])
                )
                spot_load = gp.quicksum(data["d_sp"].get((j, t, xi), 0.0) * yR[j, s, t, xi] for j in J if t in T_j[j])
                batch_load = gp.quicksum(z[k, s, t, xi] for k in K)
                model.addConstr(total_load[s, t, xi] == od_load[s, t, xi] + spot_load + batch_load)
                model.addConstr(od_load[s, t, xi] <= C * u[s, t] + M * phi[s, t, xi])
                model.addConstr(eta[s, xi] >= phi[s, t, xi])
                model.addConstr(gamma[s, t, xi] >= phi[s, t, xi])
                # Additional model-meaningfulness constraint:
                # If there is no on-demand violation, the realized workload
                # after spot/batch suspension must fit within physical server
                # capacity.  If phi=1, gamma=1 already removes low-priority
                # work and the remaining excess is interpreted as OD SLA loss.
                model.addConstr(total_load[s, t, xi] <= C * u[s, t] + M * phi[s, t, xi])

                # Additional model-meaningfulness constraint:
                # gamma should not float to 1 on a server-time with no
                # low-priority work unless OD violation forces it.
                low_priority_planned = (
                    gp.quicksum(y[j, s] for j in J if t in T_j[j])
                    + gp.quicksum(b[k, s, t] for k in K)
                )
                model.addConstr(gamma[s, t, xi] <= phi[s, t, xi] + low_priority_planned)

                # Implementation-required: Notion's updated energy objective
                # uses barL, so the capped-load variable must be defined.
                # cap_select=0 selects barL=total_load; cap_select=1 selects
                # barL=C*u.  The active formulation is selected by config.
                model.addConstr(bar_load[s, t, xi] <= total_load[s, t, xi])
                model.addConstr(bar_load[s, t, xi] <= C * u[s, t])
                if bar_load_mode == "indicator":
                    model.addGenConstrIndicator(
                        cap_select[s, t, xi],
                        0,
                        bar_load[s, t, xi] - total_load[s, t, xi],
                        GRB.EQUAL,
                        0.0,
                    )
                    model.addGenConstrIndicator(
                        cap_select[s, t, xi],
                        1,
                        bar_load[s, t, xi] - C * u[s, t],
                        GRB.EQUAL,
                        0.0,
                    )
                else:
                    model.addConstr(bar_load[s, t, xi] >= total_load[s, t, xi] - cap_m * cap_select[s, t, xi])
                    model.addConstr(bar_load[s, t, xi] >= C * u[s, t] - cap_m * (1 - cap_select[s, t, xi]))
    # Additional model-meaningfulness constraint:
    # eta is the exact "any OD SLA violation on server s in scenario xi" flag.
    # The lower bound eta >= phi is above; this upper bound prevents eta from
    # taking arbitrary 1 values when all phi values are 0.
    model.addConstrs(
        eta[s, xi] <= gp.quicksum(phi[s, t, xi] for t in T)
        for s in S
        for xi in Xi
    )
    model.addConstrs(gp.quicksum(prob[xi] * eta[s, xi] for xi in Xi) <= data["epsilon_od"] for s in S)

    # Spot chance constraint and per-scenario minimum service ratio.
    model.addConstrs(delta[j, xi] >= 1 - gp.quicksum(yR[j, s, t, xi] for s in S) for j in J for t in T_j[j] for xi in Xi)
    # Additional model-meaningfulness constraint:
    # delta is the exact "spot VM j is suspended at least once" flag.
    model.addConstrs(
        delta[j, xi]
        <= gp.quicksum(1 - gp.quicksum(yR[j, s, t, xi] for s in S) for t in T_j[j])
        for j in J
        for xi in Xi
    )
    model.addConstrs(gp.quicksum(prob[xi] * delta[j, xi] for xi in Xi) <= data["epsilon_sp"] for j in J)
    model.addConstrs(
        gp.quicksum(yR[j, s, t, xi] for s in S for t in T_j[j]) >= data["rho"] * len(T_j[j])
        for j in J
        for xi in Xi
    )

    if server_symmetry in {"used", "used_and_powered_slots"}:
        model.addConstrs(u_used[s] >= u_used[S[index + 1]] for index, s in enumerate(S[:-1]))
    if server_symmetry in {"powered_slots", "used_and_powered_slots"}:
        # Safe for homogeneous servers: any feasible solution can be relabeled
        # in nonincreasing powered-slot order without changing objective value.
        model.addConstrs(
            gp.quicksum(u[s, t] for t in T) >= gp.quicksum(u[S[index + 1], t] for t in T)
            for index, s in enumerate(S[:-1])
        )

    migration_count = gp.quicksum(prob[xi] * m[i, t, xi] for i in I for t in T_i[i][1:] for xi in Xi)
    if data["objective_type"] == "energy":
        idle_energy = data["energy_idle"] * gp.quicksum(u[s, t] for s in S for t in T)
        cpu_energy = (data["energy_cpu"] / C) * gp.quicksum(
            prob[xi] * bar_load[s, t, xi]
            for s in S
            for t in T
            for xi in Xi
        )
        mig_energy = data["energy_migration"] * migration_count
        model.setObjective(idle_energy + cpu_energy + mig_energy, GRB.MINIMIZE)
    else:
        model.setObjective(
            gp.quicksum(u_used[s] for s in S)
            + data["lambda_migration"] * migration_count / max(1, len(I) * len(Xi)),
            GRB.MINIMIZE,
        )

    return model


def write_solution(model, results_dir):
    if model.SolCount <= 0:
        return None

    results_dir.mkdir(parents=True, exist_ok=True)
    model.write(str(results_dir / "solution.sol"))

    csv_path = results_dir / "solution_nonzero_variables.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["variable", "value"])
        for var in model.getVars():
            if abs(var.X) > 1e-6:
                writer.writerow([var.VarName, var.X])

    return {
        "solution_file": "solution.sol",
        "nonzero_variable_file": "solution_nonzero_variables.csv",
    }


def write_summary(model, data, results_dir, solution_files=None, run_config=None):
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": status_name(model.Status),
        "objective": model.ObjVal if model.SolCount else None,
        "bound": model.ObjBound if model.SolCount else None,
        "gap": model.MIPGap if model.SolCount else None,
        "runtime": model.Runtime,
        "nodes": model.NodeCount,
        "variables": model.NumVars,
        "constraints": model.NumConstrs,
        "instance": data["raw"]["name"],
        "solution_files": solution_files or {},
        "run_config": run_config or {},
    }
    with open(results_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def main():
    args = parse_args()
    config = load_config(args.config)
    instance_path = args.instance or resolve_path(
        config.get("model", {}).get("instance"),
        REPO_ROOT,
        DEFAULT_INSTANCE,
    )
    results_dir = args.results_dir or resolve_path(
        config.get("model", {}).get("results_dir"),
        REPO_ROOT,
        DEFAULT_RESULTS_DIR,
    )
    formulation = formulation_config(
        config,
        defaults={"bar_load_mode": "indicator", "server_symmetry": "used_and_powered_slots"},
    )
    solver = solver_config(config, defaults={"mip_gap": 0.001, "method": 2})

    data = load_data(instance_path)
    model = build_model(data, formulation=formulation)
    results_dir.mkdir(parents=True, exist_ok=True)
    effective_solver = apply_solver_config(
        model,
        solver,
        results_dir,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        threads=args.threads,
    )

    model.optimize()
    solution_files = write_solution(model, results_dir)
    write_summary(
        model,
        data,
        results_dir,
        solution_files,
        run_config={
            "config_file": str(args.config),
            "instance": str(instance_path),
            "results_dir": str(results_dir),
            "formulation": formulation,
            "solver": effective_solver,
        },
    )


if __name__ == "__main__":
    main()
