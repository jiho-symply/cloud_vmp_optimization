#!/usr/bin/env python3
"""
Benchmark alternative migration formulations for stochastic VM placement.

Formulations compared
---------------------
1. arc    : pairwise migration arc q[i, s_from, s_to, t, xi]
2. change : location xR[i,s,t,xi] + weak change detector m[i,t,xi]
3. flow   : location xR[i,s,t,xi] + transition variables mu_plus/mu_minus + m[i,t,xi]
4. event  : migration time m[i,t,xi] + post-migration destination xprime[i,s,xi]
            + active location a[i,s,t,xi]

The common model is intentionally minimal:
- first-stage initial VM placement x[i,s]
- first-stage server on/off schedule u[s,t]
- second-stage migration decisions by scenario xi
- per-server capacity constraints in every scenario/time
- objective = server-on cost + expected migration penalty

Requires:
    pip install gurobipy pandas

Example:
    python benchmark_migration.py \
        --formulations arc change flow event \
        --n-vm 30 --n-server 10 --n-time 12 --n-scen 5 \
        --seeds 0 1 2 --time-limit 300 --threads 16 \
        --root-relax --out migration_benchmark.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Any, Optional

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError as e:
    raise SystemExit(
        "This script requires gurobipy. Install it and make sure a Gurobi license is available."
    ) from e


@dataclass(frozen=True)
class Instance:
    n_vm: int
    n_server: int
    n_time: int
    n_scen: int
    capacity: int
    demand: Dict[Tuple[int, int, int], int]  # (i,t,xi) -> CPU demand
    prob: Dict[int, float]


def generate_instance(
    n_vm: int,
    n_server: int,
    n_time: int,
    n_scen: int,
    seed: int,
    capacity: int = 0,
    load_low: int = 8,
    load_high: int = 28,
    target_total_util: float = 0.68,
) -> Instance:
    """
    Generate a synthetic all-active VM instance.

    capacity=0 means auto-scale C so that all servers can feasibly cover the
    maximum scenario-time aggregate load with approximately target_total_util.
    """
    rng = random.Random(seed)

    base = [rng.randint(load_low, load_high) for _ in range(n_vm)]
    time_factor = [rng.uniform(0.75, 1.35) for _ in range(n_time)]
    scen_factor = [rng.uniform(0.85, 1.25) for _ in range(n_scen)]

    demand: Dict[Tuple[int, int, int], int] = {}
    max_total = 0
    max_single = 0

    for xi in range(n_scen):
        for t in range(n_time):
            total = 0
            for i in range(n_vm):
                noise = rng.uniform(0.85, 1.15)
                val = max(1, int(round(base[i] * time_factor[t] * scen_factor[xi] * noise)))
                demand[(i, t, xi)] = val
                total += val
                max_single = max(max_single, val)
            max_total = max(max_total, total)

    if capacity <= 0:
        capacity = max(
            max_single,
            int(math.ceil(max_total / max(1.0, n_server * target_total_util))),
        )

    if max_total > n_server * capacity:
        raise ValueError(
            f"Infeasible generated instance: max total load {max_total} > "
            f"n_server * capacity = {n_server * capacity}. Increase --n-server or --capacity."
        )

    prob = {xi: 1.0 / n_scen for xi in range(n_scen)}
    return Instance(n_vm, n_server, n_time, n_scen, capacity, demand, prob)


def status_name(status: int) -> str:
    names = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }
    return names.get(status, str(status))


def safe_attr(model: gp.Model, attr: str) -> Optional[float]:
    try:
        return getattr(model, attr)
    except gp.GurobiError:
        return None
    except AttributeError:
        return None


def add_common_first_stage(
    model: gp.Model,
    inst: Instance,
    symmetry: bool,
) -> Tuple[Any, Any]:
    I = range(inst.n_vm)
    S = range(inst.n_server)
    T = range(inst.n_time)

    x = model.addVars(I, S, vtype=GRB.BINARY, name="x")       # initial placement
    u = model.addVars(S, T, vtype=GRB.BINARY, name="u")       # server-on schedule

    model.addConstrs(
        (gp.quicksum(x[i, s] for s in S) == 1 for i in I),
        name="initial_assignment",
    )

    if symmetry:
        # Server indices are homogeneous. This removes many equivalent server permutations.
        model.addConstrs(
            (u[s, t] >= u[s + 1, t] for s in range(inst.n_server - 1) for t in T),
            name="server_order",
        )

    return x, u


def add_capacity_constraints(
    model: gp.Model,
    inst: Instance,
    u: Any,
    loc: Dict[Tuple[int, int, int, int], Any],
) -> None:
    I = range(inst.n_vm)
    S = range(inst.n_server)
    T = range(inst.n_time)
    Xi = range(inst.n_scen)

    model.addConstrs(
        (
            gp.quicksum(inst.demand[(i, t, xi)] * loc[(i, s, t, xi)] for i in I)
            <= inst.capacity * u[s, t]
            for s in S for t in T for xi in Xi
        ),
        name="capacity",
    )

    model.addConstrs(
        (
            loc[(i, s, t, xi)] <= u[s, t]
            for i in I for s in S for t in T for xi in Xi
        ),
        name="loc_on_link",
    )


def build_arc_model(inst: Instance, symmetry: bool, mig_cost: float) -> gp.Model:
    """
    Arc formulation:
        q[i,s,r,t,xi] = 1 if VM i migrates from s to r at time t in scenario xi.
    Since at most one migration is allowed, the departure server must be the initial
    first-stage server.
    """
    I = range(inst.n_vm)
    S = range(inst.n_server)
    T = range(inst.n_time)
    Tp = range(1, inst.n_time)
    Xi = range(inst.n_scen)

    model = gp.Model("migration_arc")
    x, u = add_common_first_stage(model, inst, symmetry)

    arcs = [
        (i, s, r, t, xi)
        for i in I for s in S for r in S if s != r
        for t in Tp for xi in Xi
    ]

    q = model.addVars(arcs, vtype=GRB.BINARY, name="q_arc")
    xR = model.addVars(I, S, T, Xi, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="xR")

    model.addConstrs(
        (xR[i, s, 0, xi] == x[i, s] for i in I for s in S for xi in Xi),
        name="initial_realized_loc",
    )

    model.addConstrs(
        (
            gp.quicksum(q[i, s, r, t, xi] for s in S for r in S if s != r for t in Tp) <= 1
            for i in I for xi in Xi
        ),
        name="at_most_one_migration",
    )

    model.addConstrs(
        (
            q[i, s, r, t, xi] <= x[i, s]
            for (i, s, r, t, xi) in arcs
        ),
        name="depart_from_initial_server",
    )

    for i in I:
        for s in S:
            for t in Tp:
                for xi in Xi:
                    outflow = gp.quicksum(
                        q[i, s, r, tau, xi]
                        for r in S if r != s
                        for tau in range(1, t + 1)
                    )
                    inflow = gp.quicksum(
                        q[i, r, s, tau, xi]
                        for r in S if r != s
                        for tau in range(1, t + 1)
                    )
                    model.addConstr(
                        xR[i, s, t, xi] == x[i, s] - outflow + inflow,
                        name=f"loc_from_arc[{i},{s},{t},{xi}]",
                    )

    loc = {(i, s, t, xi): xR[i, s, t, xi] for i in I for s in S for t in T for xi in Xi}
    add_capacity_constraints(model, inst, u, loc)

    server_cost = gp.quicksum(u[s, t] for s in S for t in T)
    migration_cost = gp.quicksum(
        inst.prob[xi] * mig_cost * q[i, s, r, t, xi]
        for (i, s, r, t, xi) in arcs
    )
    model.setObjective(server_cost + migration_cost, GRB.MINIMIZE)
    return model


def build_change_model(inst: Instance, symmetry: bool, mig_cost: float) -> gp.Model:
    """
    Weak change-detection formulation:
        xR[i,s,t,xi] tracks location.
        m[i,t,xi] is forced to 1 if any server-position bit changes.
    """
    I = range(inst.n_vm)
    S = range(inst.n_server)
    T = range(inst.n_time)
    Tp = range(1, inst.n_time)
    Xi = range(inst.n_scen)

    model = gp.Model("migration_change")
    x, u = add_common_first_stage(model, inst, symmetry)

    xR = model.addVars(I, S, T, Xi, vtype=GRB.BINARY, name="xR")
    mvar = model.addVars(I, Tp, Xi, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="m")

    model.addConstrs(
        (xR[i, s, 0, xi] == x[i, s] for i in I for s in S for xi in Xi),
        name="initial_realized_loc",
    )

    model.addConstrs(
        (gp.quicksum(xR[i, s, t, xi] for s in S) == 1 for i in I for t in T for xi in Xi),
        name="unique_realized_loc",
    )

    model.addConstrs(
        (
            mvar[i, t, xi] >= xR[i, s, t, xi] - xR[i, s, t - 1, xi]
            for i in I for s in S for t in Tp for xi in Xi
        ),
        name="change_pos",
    )

    model.addConstrs(
        (
            mvar[i, t, xi] >= xR[i, s, t - 1, xi] - xR[i, s, t, xi]
            for i in I for s in S for t in Tp for xi in Xi
        ),
        name="change_neg",
    )

    model.addConstrs(
        (gp.quicksum(mvar[i, t, xi] for t in Tp) <= 1 for i in I for xi in Xi),
        name="at_most_one_migration",
    )

    loc = {(i, s, t, xi): xR[i, s, t, xi] for i in I for s in S for t in T for xi in Xi}
    add_capacity_constraints(model, inst, u, loc)

    server_cost = gp.quicksum(u[s, t] for s in S for t in T)
    migration_cost = gp.quicksum(
        inst.prob[xi] * mig_cost * mvar[i, t, xi]
        for i in I for t in Tp for xi in Xi
    )
    model.setObjective(server_cost + migration_cost, GRB.MINIMIZE)
    return model


def build_flow_model(inst: Instance, symmetry: bool, mig_cost: float) -> gp.Model:
    """
    Recommended state-transition formulation:
        xR tracks location.
        mu_plus / mu_minus represent arrival/departure at each server.
        m is total migration event at time t.
    """
    I = range(inst.n_vm)
    S = range(inst.n_server)
    T = range(inst.n_time)
    Tp = range(1, inst.n_time)
    Xi = range(inst.n_scen)

    model = gp.Model("migration_flow")
    x, u = add_common_first_stage(model, inst, symmetry)

    xR = model.addVars(I, S, T, Xi, vtype=GRB.BINARY, name="xR")
    mvar = model.addVars(I, Tp, Xi, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="m")
    mu_p = model.addVars(I, S, Tp, Xi, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="mu_plus")
    mu_m = model.addVars(I, S, Tp, Xi, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="mu_minus")

    model.addConstrs(
        (xR[i, s, 0, xi] == x[i, s] for i in I for s in S for xi in Xi),
        name="initial_realized_loc",
    )

    model.addConstrs(
        (gp.quicksum(xR[i, s, t, xi] for s in S) == 1 for i in I for t in T for xi in Xi),
        name="unique_realized_loc",
    )

    model.addConstrs(
        (
            xR[i, s, t, xi] - xR[i, s, t - 1, xi]
            == mu_p[i, s, t, xi] - mu_m[i, s, t, xi]
            for i in I for s in S for t in Tp for xi in Xi
        ),
        name="state_transition",
    )

    model.addConstrs(
        (
            gp.quicksum(mu_p[i, s, t, xi] for s in S) == mvar[i, t, xi]
            for i in I for t in Tp for xi in Xi
        ),
        name="one_arrival_if_migrate",
    )

    model.addConstrs(
        (
            gp.quicksum(mu_m[i, s, t, xi] for s in S) == mvar[i, t, xi]
            for i in I for t in Tp for xi in Xi
        ),
        name="one_departure_if_migrate",
    )

    # Strengthening constraints. They are redundant for integer xR but useful in the LP relaxation.
    model.addConstrs(
        (
            mu_p[i, s, t, xi] <= xR[i, s, t, xi]
            for i in I for s in S for t in Tp for xi in Xi
        ),
        name="arrive_only_if_present_now",
    )

    model.addConstrs(
        (
            mu_p[i, s, t, xi] <= 1 - xR[i, s, t - 1, xi]
            for i in I for s in S for t in Tp for xi in Xi
        ),
        name="arrive_only_if_absent_before",
    )

    model.addConstrs(
        (
            mu_m[i, s, t, xi] <= xR[i, s, t - 1, xi]
            for i in I for s in S for t in Tp for xi in Xi
        ),
        name="depart_only_if_present_before",
    )

    model.addConstrs(
        (
            mu_m[i, s, t, xi] <= 1 - xR[i, s, t, xi]
            for i in I for s in S for t in Tp for xi in Xi
        ),
        name="depart_only_if_absent_now",
    )

    model.addConstrs(
        (gp.quicksum(mvar[i, t, xi] for t in Tp) <= 1 for i in I for xi in Xi),
        name="at_most_one_migration",
    )

    loc = {(i, s, t, xi): xR[i, s, t, xi] for i in I for s in S for t in T for xi in Xi}
    add_capacity_constraints(model, inst, u, loc)

    server_cost = gp.quicksum(u[s, t] for s in S for t in T)
    migration_cost = gp.quicksum(
        inst.prob[xi] * mig_cost * mvar[i, t, xi]
        for i in I for t in Tp for xi in Xi
    )
    model.setObjective(server_cost + migration_cost, GRB.MINIMIZE)
    return model


def build_event_model(inst: Instance, symmetry: bool, mig_cost: float) -> gp.Model:
    """
    Event-destination formulation:
        m[i,t,xi] decides migration time.
        xprime[i,s,xi] decides destination after migration.
        a[i,s,t,xi] is active location used in load/energy constraints.

    This implements the user's proposed m_it + x'_is idea, with active-location
    variables needed for a linear MILP.
    """
    I = range(inst.n_vm)
    S = range(inst.n_server)
    T = range(inst.n_time)
    Tp = range(1, inst.n_time)
    Xi = range(inst.n_scen)

    model = gp.Model("migration_event")
    x, u = add_common_first_stage(model, inst, symmetry)

    mvar = model.addVars(I, Tp, Xi, vtype=GRB.BINARY, name="m")
    xprime = model.addVars(I, S, Xi, vtype=GRB.BINARY, name="xprime")
    a = model.addVars(I, S, T, Xi, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="a")

    model.addConstrs(
        (gp.quicksum(mvar[i, t, xi] for t in Tp) <= 1 for i in I for xi in Xi),
        name="at_most_one_migration",
    )

    model.addConstrs(
        (
            gp.quicksum(xprime[i, s, xi] for s in S)
            == gp.quicksum(mvar[i, t, xi] for t in Tp)
            for i in I for xi in Xi
        ),
        name="choose_destination_iff_migrate",
    )

    model.addConstrs(
        (
            xprime[i, s, xi] <= 1 - x[i, s]
            for i in I for s in S for xi in Xi
        ),
        name="no_fake_same_server_migration",
    )

    model.addConstrs(
        (gp.quicksum(a[i, s, t, xi] for s in S) == 1 for i in I for t in T for xi in Xi),
        name="unique_active_loc",
    )

    # If cumulative migration H_it = 0, force a = x.
    # If H_it = 1, force a = xprime.
    # H_it is a linear expression because at most one migration is allowed.
    for i in I:
        for s in S:
            for t in T:
                for xi in Xi:
                    h = gp.quicksum(mvar[i, tau, xi] for tau in Tp if tau <= t)
                    model.addConstr(a[i, s, t, xi] - x[i, s] <= h, name=f"pre_link_1[{i},{s},{t},{xi}]")
                    model.addConstr(x[i, s] - a[i, s, t, xi] <= h, name=f"pre_link_2[{i},{s},{t},{xi}]")
                    model.addConstr(a[i, s, t, xi] - xprime[i, s, xi] <= 1 - h, name=f"post_link_1[{i},{s},{t},{xi}]")
                    model.addConstr(xprime[i, s, xi] - a[i, s, t, xi] <= 1 - h, name=f"post_link_2[{i},{s},{t},{xi}]")

    loc = {(i, s, t, xi): a[i, s, t, xi] for i in I for s in S for t in T for xi in Xi}
    add_capacity_constraints(model, inst, u, loc)

    server_cost = gp.quicksum(u[s, t] for s in S for t in T)
    migration_cost = gp.quicksum(
        inst.prob[xi] * mig_cost * mvar[i, t, xi]
        for i in I for t in Tp for xi in Xi
    )
    model.setObjective(server_cost + migration_cost, GRB.MINIMIZE)
    return model


BUILDERS = {
    "arc": build_arc_model,
    "change": build_change_model,
    "flow": build_flow_model,
    "event": build_event_model,
}


def solve_root_relaxation(model: gp.Model, time_limit: float, output_flag: int) -> Dict[str, Any]:
    model.update()
    relaxed = model.relax()
    relaxed.Params.OutputFlag = output_flag
    relaxed.Params.TimeLimit = min(time_limit, 120.0)
    relaxed.optimize()

    out = {
        "root_status": status_name(relaxed.Status),
        "root_runtime": safe_attr(relaxed, "Runtime"),
        "root_obj": None,
    }
    if relaxed.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL) and relaxed.SolCount > 0:
        out["root_obj"] = relaxed.ObjVal
    return out


def run_one(
    formulation: str,
    inst: Instance,
    seed: int,
    time_limit: float,
    mip_gap: float,
    threads: int,
    output_flag: int,
    symmetry: bool,
    mig_cost: float,
    root_relax: bool,
) -> Dict[str, Any]:
    builder = BUILDERS[formulation]
    t0 = time.time()
    model = builder(inst, symmetry=symmetry, mig_cost=mig_cost)
    build_time = time.time() - t0

    model.Params.TimeLimit = time_limit
    model.Params.MIPGap = mip_gap
    model.Params.Threads = threads
    model.Params.OutputFlag = output_flag
    model.Params.Seed = seed

    model.update()

    root_info = {
        "root_status": None,
        "root_runtime": None,
        "root_obj": None,
    }
    if root_relax:
        root_info = solve_root_relaxation(model, time_limit=time_limit, output_flag=0)

    model.optimize()

    result = {
        "formulation": formulation,
        "seed": seed,
        "n_vm": inst.n_vm,
        "n_server": inst.n_server,
        "n_time": inst.n_time,
        "n_scen": inst.n_scen,
        "capacity": inst.capacity,
        "status": status_name(model.Status),
        "build_time": build_time,
        "runtime": safe_attr(model, "Runtime"),
        "num_vars": model.NumVars,
        "num_bin_vars": model.NumBinVars,
        "num_int_vars": model.NumIntVars,
        "num_constrs": model.NumConstrs,
        "num_nz": model.NumNZs,
        "node_count": safe_attr(model, "NodeCount"),
        "obj_val": None,
        "obj_bound": safe_attr(model, "ObjBound"),
        "mip_gap": None,
        **root_info,
    }

    if model.SolCount > 0:
        result["obj_val"] = model.ObjVal
        result["mip_gap"] = safe_attr(model, "MIPGap")

    return result


def write_results(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formulations", nargs="+", default=["arc", "change", "flow", "event"], choices=list(BUILDERS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--n-vm", type=int, default=30)
    parser.add_argument("--n-server", type=int, default=10)
    parser.add_argument("--n-time", type=int, default=12)
    parser.add_argument("--n-scen", type=int, default=5)
    parser.add_argument("--capacity", type=int, default=0, help="0 means auto capacity")
    parser.add_argument("--target-total-util", type=float, default=0.68)
    parser.add_argument("--mig-cost", type=float, default=0.02)
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--root-relax", action="store_true")
    parser.add_argument("--no-symmetry", action="store_true")
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("migration_benchmark.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows: List[Dict[str, Any]] = []
    for seed in args.seeds:
        inst = generate_instance(
            n_vm=args.n_vm,
            n_server=args.n_server,
            n_time=args.n_time,
            n_scen=args.n_scen,
            seed=seed,
            capacity=args.capacity,
            target_total_util=args.target_total_util,
        )

        for formulation in args.formulations:
            print(f"[run] formulation={formulation}, seed={seed}, C={inst.capacity}")
            row = run_one(
                formulation=formulation,
                inst=inst,
                seed=seed,
                time_limit=args.time_limit,
                mip_gap=args.mip_gap,
                threads=args.threads,
                output_flag=1 if args.log else 0,
                symmetry=not args.no_symmetry,
                mig_cost=args.mig_cost,
                root_relax=args.root_relax,
            )
            rows.append(row)
            print(
                f"  status={row['status']}, obj={row['obj_val']}, "
                f"bound={row['obj_bound']}, gap={row['mip_gap']}, "
                f"runtime={row['runtime']}, nodes={row['node_count']}, "
                f"vars={row['num_vars']}, bin={row['num_bin_vars']}, constrs={row['num_constrs']}"
            )

    write_results(args.out, rows)
    print(f"\nSaved results to: {args.out}")


if __name__ == "__main__":
    main()
