"""
No-risk-control robust variant of the 2605 VM type model.

This model keeps the first-stage placement/reservation structure and the
on-demand migration notation from "VM type modeling (1)", but removes the
chance-control machinery:

- No OD violation indicators phi/eta.
- No spot suspension indicators delta.
- No low-priority suspension indicator gamma.
- No scenario active-state variable yR; spot VMs remain active on their
  first-stage server in every scenario and active period.
- No capped-load selector cap_select and no barL variable.  Since every
  scenario must satisfy total_load[s,t,xi] <= C u[s,t], the capped load in the
  energy objective is exactly total_load.

The remaining binary variables are the core placement/reservation/migration
decisions required by the VMP MILP: u, u_used, x, y, b, xR, and m.
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
    / "2605-no-risk-control"
    / "notion_vm_type_24vm_od8_sp8_bj8_sc10_cap8"
    / "vm_type_instance.json"
)
DEFAULT_RESULTS_DIR = REPO_ROOT / "experiments" / "2605-no-risk-control" / "results"


def parse_args():
    parser = argparse.ArgumentParser(description="Solve the 2605 no-risk-control robust MILP.")
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
    server_symmetry = formulation.get("server_symmetry", "used_and_powered_slots")
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

    model = gp.Model("vm_type_modeling_1_no_risk_control")

    # Core first-stage binary decisions.
    u = model.addVars(S, T, vtype=GRB.BINARY, name="u")
    u_used = model.addVars(S, vtype=GRB.BINARY, name="u_used")
    x = model.addVars(I, S, vtype=GRB.BINARY, name="x")
    y = model.addVars(J, S, vtype=GRB.BINARY, name="y")
    b = model.addVars(K, S, T, vtype=GRB.BINARY, name="b")

    # Core second-stage on-demand migration decisions.
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

    # Continuous recourse workload variables.
    z = model.addVars(K, S, T, Xi, lb=0.0, name="z")
    od_load = model.addVars(S, T, Xi, lb=0.0, name="od_load")
    total_load = model.addVars(S, T, Xi, lb=0.0, name="total_load")

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

    # On-demand migration using m[i,t,xi] as a server-change detector.
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
        model.addConstrs(
            gp.quicksum(m[i, t, xi] for t in T_i[i][1:]) <= 1
            for xi in Xi
        )
    model.addConstrs(xR[i, s, t, xi] <= u[s, t] for i in I for s in S for t in T_i[i] for xi in Xi)

    # Batch processing is allowed only on reserved slots and must finish in
    # every scenario.  There is no gamma suspension in this robust variant.
    for k in K:
        r_b = data["batch_info"][k]["reserved_cpu"]
        model.addConstrs(z[k, s, t, xi] <= r_b * b[k, s, t] for s in S for t in T for xi in Xi)
        model.addConstrs(
            gp.quicksum(z[k, s, t, xi] for s in S for t in T) >= data["batch_info"][k]["workload"]
            for xi in Xi
        )

    # All scenario-time loads must physically fit.  Spot load is tied directly
    # to first-stage y because spot VMs are not suspended in this variant.
    for s in S:
        for t in T:
            for xi in Xi:
                model.addConstr(
                    od_load[s, t, xi]
                    == gp.quicksum(data["d_od"].get((i, t, xi), 0.0) * xR[i, s, t, xi] for i in I if t in T_i[i])
                )
                spot_load = gp.quicksum(data["d_sp"].get((j, t, xi), 0.0) * y[j, s] for j in J if t in T_j[j])
                batch_load = gp.quicksum(z[k, s, t, xi] for k in K)
                model.addConstr(total_load[s, t, xi] == od_load[s, t, xi] + spot_load + batch_load)
                model.addConstr(total_load[s, t, xi] <= C * u[s, t])

    if server_symmetry in {"used", "used_and_powered_slots"}:
        model.addConstrs(u_used[s] >= u_used[S[index + 1]] for index, s in enumerate(S[:-1]))
    if server_symmetry in {"powered_slots", "used_and_powered_slots"}:
        model.addConstrs(
            gp.quicksum(u[s, t] for t in T) >= gp.quicksum(u[S[index + 1], t] for t in T)
            for index, s in enumerate(S[:-1])
        )

    migration_count = gp.quicksum(prob[xi] * m[i, t, xi] for i in I for t in T_i[i][1:] for xi in Xi)
    if data["objective_type"] == "energy":
        idle_energy = data["energy_idle"] * gp.quicksum(u[s, t] for s in S for t in T)
        # barL is exactly total_load because total_load <= C u is enforced in
        # every scenario, so the capped-load objective uses total_load directly.
        cpu_energy = (data["energy_cpu"] / C) * gp.quicksum(
            prob[xi] * total_load[s, t, xi]
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
        defaults={"server_symmetry": "used_and_powered_slots"},
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
            "risk_control": "none; all scenario-time total loads satisfy capacity",
        },
    )


if __name__ == "__main__":
    main()
