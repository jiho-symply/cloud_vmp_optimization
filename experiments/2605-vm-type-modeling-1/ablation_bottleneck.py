"""
Constraint-family ablation runner for 2605 VM type modeling.

This script intentionally does not modify model.py.  It rebuilds the same
model with a small set of switches so we can measure which constraint family
dominates solve time on smaller instances.
"""

import argparse
import csv
import importlib.util
import json
from dataclasses import dataclass, replace
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(__file__).resolve().parent / "model.py"
DEFAULT_INSTANCE = (
    REPO_ROOT
    / "data"
    / "processed"
    / "2605-vm-type-modeling-1"
    / "ablation_12vm_od4_sp4_bj4_sc5_cap8"
    / "vm_type_instance.json"
)
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "ablation_results"


def load_base_model_module():
    spec = importlib.util.spec_from_file_location("model2605", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE_MODEL = load_base_model_module()
load_data = BASE_MODEL.load_data
status_name = BASE_MODEL.status_name


@dataclass(frozen=True)
class Variant:
    name: str
    barload_mode: str = "bigm"  # bigm, indicator, none
    exact_indicator_uppers: bool = True
    od_chance_budget: bool = True
    spot_chance_budget: bool = True
    spot_service_ratio: bool = True
    migration: bool = True
    batch_completion: bool = True
    slot_symmetry: bool = False


def make_variants():
    base = Variant("base")
    return [
        base,
        replace(base, name="barload_none", barload_mode="none"),
        replace(base, name="barload_indicator", barload_mode="indicator"),
        replace(base, name="barload_indicator_slot_symmetry", barload_mode="indicator", slot_symmetry=True),
        replace(base, name="no_exact_indicator_uppers", exact_indicator_uppers=False),
        replace(base, name="no_od_chance_budget", od_chance_budget=False),
        replace(base, name="no_spot_chance_budget", spot_chance_budget=False),
        replace(base, name="no_spot_service_ratio", spot_service_ratio=False),
        replace(base, name="no_migration", migration=False),
        replace(base, name="no_batch_completion", batch_completion=False),
        replace(base, name="slot_symmetry", slot_symmetry=True),
    ]


def quicksum(items):
    return gp.quicksum(items)


def build_ablation_model(data, variant):
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

    model = gp.Model(f"vm_type_ablation_{variant.name}")

    u = model.addVars(S, T, vtype=GRB.BINARY, name="u")
    u_used = model.addVars(S, vtype=GRB.BINARY, name="u_used")
    x = model.addVars(I, S, vtype=GRB.BINARY, name="x")
    y = model.addVars(J, S, vtype=GRB.BINARY, name="y")
    b = model.addVars(K, S, T, vtype=GRB.BINARY, name="b")

    xR = model.addVars(
        [(i, s, t, xi) for i in I for s in S for t in T_i[i] for xi in Xi],
        vtype=GRB.BINARY,
        name="xR",
    )
    if variant.migration:
        m = model.addVars(
            [(i, t, xi) for i in I for t in T_i[i][1:] for xi in Xi],
            vtype=GRB.BINARY,
            name="m",
        )
    else:
        m = {}
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

    if variant.barload_mode in {"bigm", "indicator"}:
        bar_load = model.addVars(S, T, Xi, lb=0.0, name="barL")
        cap_select = model.addVars(S, T, Xi, vtype=GRB.BINARY, name="cap_select")
    else:
        bar_load = None
        cap_select = None

    max_od_load = sum(max(data["d_od"].get((i, t, xi), 0.0) for t in T_i[i] for xi in Xi) for i in I)
    max_spot_load = sum(max(data["d_sp"].get((j, t, xi), 0.0) for t in T_j[j] for xi in Xi) for j in J)
    max_batch_load = sum(data["batch_info"][k]["reserved_cpu"] for k in K)
    cap_m = max(M, max_od_load + max_spot_load + max_batch_load + C)

    model.addConstrs(quicksum(x[i, s] for s in S) == 1 for i in I)
    model.addConstrs(quicksum(y[j, s] for s in S) == 1 for j in J)
    model.addConstrs(y[j, s] <= u[s, t] for j in J for s in S for t in T_j[j])
    model.addConstrs(b[k, s, t] <= u[s, t] for k in K for s in S for t in T)
    model.addConstrs(
        quicksum(b[k, s, t] for s in S for t in T) <= data["batch_info"][k]["processing_slots"]
        for k in K
    )
    model.addConstrs(u_used[s] >= u[s, t] for s in S for t in T)
    model.addConstrs(u_used[s] <= quicksum(u[s, t] for t in T) for s in S)

    model.addConstrs(quicksum(xR[i, s, t, xi] for s in S) == 1 for i in I for t in T_i[i] for xi in Xi)
    for i in I:
        first_t = T_i[i][0]
        model.addConstrs(xR[i, s, first_t, xi] == x[i, s] for s in S for xi in Xi)
        if variant.migration:
            for prev_t, t in zip(T_i[i][:-1], T_i[i][1:]):
                model.addConstrs(m[i, t, xi] >= xR[i, s, t, xi] - xR[i, s, prev_t, xi] for s in S for xi in Xi)
                model.addConstrs(m[i, t, xi] >= xR[i, s, prev_t, xi] - xR[i, s, t, xi] for s in S for xi in Xi)
            model.addConstrs(quicksum(m[i, t, xi] for t in T_i[i][1:]) <= 1 for xi in Xi)
        else:
            model.addConstrs(xR[i, s, t, xi] == x[i, s] for s in S for t in T_i[i][1:] for xi in Xi)
    model.addConstrs(xR[i, s, t, xi] <= u[s, t] for i in I for s in S for t in T_i[i] for xi in Xi)

    model.addConstrs(yR[j, s, t, xi] <= y[j, s] for j in J for s in S for t in T_j[j] for xi in Xi)
    model.addConstrs(yR[j, s, t, xi] <= 1 - gamma[s, t, xi] for j in J for s in S for t in T_j[j] for xi in Xi)
    model.addConstrs(yR[j, s, t, xi] >= y[j, s] - gamma[s, t, xi] for j in J for s in S for t in T_j[j] for xi in Xi)

    for k in K:
        r_b = data["batch_info"][k]["reserved_cpu"]
        model.addConstrs(z[k, s, t, xi] <= r_b * b[k, s, t] for s in S for t in T for xi in Xi)
        model.addConstrs(z[k, s, t, xi] <= r_b * (1 - gamma[s, t, xi]) for s in S for t in T for xi in Xi)
        if variant.batch_completion:
            model.addConstrs(
                quicksum(z[k, s, t, xi] for s in S for t in T) >= data["batch_info"][k]["workload"]
                for xi in Xi
            )

    for s in S:
        for t in T:
            for xi in Xi:
                model.addConstr(
                    od_load[s, t, xi]
                    == quicksum(data["d_od"].get((i, t, xi), 0.0) * xR[i, s, t, xi] for i in I if t in T_i[i])
                )
                spot_load = quicksum(data["d_sp"].get((j, t, xi), 0.0) * yR[j, s, t, xi] for j in J if t in T_j[j])
                batch_load = quicksum(z[k, s, t, xi] for k in K)
                model.addConstr(total_load[s, t, xi] == od_load[s, t, xi] + spot_load + batch_load)
                model.addConstr(od_load[s, t, xi] <= C * u[s, t] + M * phi[s, t, xi])
                model.addConstr(eta[s, xi] >= phi[s, t, xi])
                model.addConstr(gamma[s, t, xi] >= phi[s, t, xi])
                model.addConstr(total_load[s, t, xi] <= C * u[s, t] + M * phi[s, t, xi])

                if variant.exact_indicator_uppers:
                    low_priority_planned = (
                        quicksum(y[j, s] for j in J if t in T_j[j])
                        + quicksum(b[k, s, t] for k in K)
                    )
                    model.addConstr(gamma[s, t, xi] <= phi[s, t, xi] + low_priority_planned)

                if variant.barload_mode == "bigm":
                    model.addConstr(bar_load[s, t, xi] <= total_load[s, t, xi])
                    model.addConstr(bar_load[s, t, xi] <= C * u[s, t])
                    model.addConstr(bar_load[s, t, xi] >= total_load[s, t, xi] - cap_m * cap_select[s, t, xi])
                    model.addConstr(bar_load[s, t, xi] >= C * u[s, t] - cap_m * (1 - cap_select[s, t, xi]))
                elif variant.barload_mode == "indicator":
                    model.addConstr(bar_load[s, t, xi] <= total_load[s, t, xi])
                    model.addConstr(bar_load[s, t, xi] <= C * u[s, t])
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

    if variant.exact_indicator_uppers:
        model.addConstrs(eta[s, xi] <= quicksum(phi[s, t, xi] for t in T) for s in S for xi in Xi)
    if variant.od_chance_budget:
        model.addConstrs(quicksum(prob[xi] * eta[s, xi] for xi in Xi) <= data["epsilon_od"] for s in S)

    if variant.spot_chance_budget:
        model.addConstrs(
            delta[j, xi] >= 1 - quicksum(yR[j, s, t, xi] for s in S)
            for j in J
            for t in T_j[j]
            for xi in Xi
        )
        if variant.exact_indicator_uppers:
            model.addConstrs(
                delta[j, xi] <= quicksum(1 - quicksum(yR[j, s, t, xi] for s in S) for t in T_j[j])
                for j in J
                for xi in Xi
            )
        model.addConstrs(quicksum(prob[xi] * delta[j, xi] for xi in Xi) <= data["epsilon_sp"] for j in J)

    if variant.spot_service_ratio:
        model.addConstrs(
            quicksum(yR[j, s, t, xi] for s in S for t in T_j[j]) >= data["rho"] * len(T_j[j])
            for j in J
            for xi in Xi
        )

    model.addConstrs(u_used[s] >= u_used[S[index + 1]] for index, s in enumerate(S[:-1]))
    if variant.slot_symmetry:
        model.addConstrs(
            quicksum(u[s, t] for t in T) >= quicksum(u[S[index + 1], t] for t in T)
            for index, s in enumerate(S[:-1])
        )

    if variant.migration:
        migration_count = quicksum(prob[xi] * m[i, t, xi] for i in I for t in T_i[i][1:] for xi in Xi)
    else:
        migration_count = 0.0

    load_for_objective = bar_load if variant.barload_mode in {"bigm", "indicator"} else total_load
    if data["objective_type"] == "energy":
        idle_energy = data["energy_idle"] * quicksum(u[s, t] for s in S for t in T)
        cpu_energy = (data["energy_cpu"] / C) * quicksum(
            prob[xi] * load_for_objective[s, t, xi] for s in S for t in T for xi in Xi
        )
        mig_energy = data["energy_migration"] * migration_count
        model.setObjective(idle_energy + cpu_energy + mig_energy, GRB.MINIMIZE)
    else:
        model.setObjective(
            quicksum(u_used[s] for s in S)
            + data["lambda_migration"] * migration_count / max(1, len(I) * len(Xi)),
            GRB.MINIMIZE,
        )

    return model


def safe_attr(model, attr, default=None):
    try:
        return getattr(model, attr)
    except gp.GurobiError:
        return default


def solve_variant(data, variant, args, log_dir):
    model = build_ablation_model(data, variant)
    model.setParam("TimeLimit", args.time_limit)
    model.setParam("MIPGap", args.mip_gap)
    model.setParam("Threads", args.threads)
    model.setParam("Seed", args.seed)
    model.setParam("LogFile", str(log_dir / f"{variant.name}.log"))
    if not args.verbose:
        model.setParam("OutputFlag", 0)
    model.optimize()
    return {
        "variant": variant.name,
        "status": status_name(model.Status),
        "runtime": float(model.Runtime),
        "nodes": float(model.NodeCount),
        "iter_count": float(safe_attr(model, "IterCount", 0.0)),
        "bar_iter_count": float(safe_attr(model, "BarIterCount", 0.0)),
        "variables": int(model.NumVars),
        "binary_vars": int(model.NumBinVars),
        "constraints": int(model.NumConstrs),
        "objective": float(model.ObjVal) if model.SolCount else "",
        "bound": float(model.ObjBound) if model.SolCount else "",
        "gap": float(model.MIPGap) if model.SolCount else "",
        "sol_count": int(model.SolCount),
        "barload_mode": variant.barload_mode,
        "exact_indicator_uppers": int(variant.exact_indicator_uppers),
        "od_chance_budget": int(variant.od_chance_budget),
        "spot_chance_budget": int(variant.spot_chance_budget),
        "spot_service_ratio": int(variant.spot_service_ratio),
        "migration": int(variant.migration),
        "batch_completion": int(variant.batch_completion),
        "slot_symmetry": int(variant.slot_symmetry),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run 2605 constraint-family ablation study.")
    parser.add_argument("--instance", type=Path, default=DEFAULT_INSTANCE)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--mip-gap", type=float, default=0.01)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    data = load_data(args.instance)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.results_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    variants = make_variants()
    if args.variants:
        wanted = set(args.variants)
        variants = [variant for variant in variants if variant.name in wanted]
        missing = wanted - {variant.name for variant in variants}
        if missing:
            raise ValueError(f"Unknown variants: {sorted(missing)}")

    rows = []
    for variant in variants:
        print(f"running {variant.name}...")
        rows.append(solve_variant(data, variant, args, log_dir))

    csv_path = args.results_dir / "ablation_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = args.results_dir / "ablation_summary.json"
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2)

    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
