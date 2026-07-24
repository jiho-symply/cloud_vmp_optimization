#!/usr/bin/env python3
"""
Benchmark migration formulations for stochastic VM placement.

This version focuses on the user's pre/post migration placement idea:

    x[i,s]              : placement before migration
    xprime[i,s,xi]     : placement after migration under scenario xi
    m[i,t,xi]           : VM i migrates between t-1 and t under scenario xi
    H[i,t,xi]           : cumulative migration indicator, sum_{tau <= t} m[i,tau,xi]
    xR[i,s,t,xi]        : realized location at time t

New formulations
----------------
event_linear:
    Linear big-M-free switching inequalities:
        if H=0, xR = x
        if H=1, xR = xprime
    implemented with linear constraints.

event_indicator:
    Indicator constraints:
        H = 0  => xR = x
        H = 1  => xR = xprime

event_bilinear:
    Direct bilinear/quadratic equality:
        xR = (1-H) x + H xprime
    This is not MISOCP in general. It is a nonconvex MIQCP-like formulation.
    Gurobi solves it with NonConvex=2.

Reference formulations retained for comparison
----------------------------------------------
change:
    xR location variables plus weak change-detection m variables.

flow:
    xR location variables plus transition-flow variables mu_plus/mu_minus.

arc:
    full server-pair migration arc variables q[i,s,r,t,xi].
    Expensive; use only for small instances.

Requires:
    pip install gurobipy

Example on a 96-thread server:
    .venv/bin/python benchmark_migration_v2.py \
      --formulations change event_linear event_indicator event_bilinear flow \
      --n-vm 30 --n-server 10 --n-time 24 --n-scen 20 \
      --seeds 0 1 2 \
      --time-limit 300 \
      --parallel-jobs 4 --threads 24 \
      --out migration_benchmark_v2.csv

Fair sequential benchmark:
    .venv/bin/python benchmark_migration_v2.py \
      --formulations change event_linear event_indicator event_bilinear flow \
      --n-vm 20 --n-server 8 --n-time 16 --n-scen 8 \
      --seeds 0 1 2 3 4 \
      --time-limit 300 \
      --parallel-jobs 1 --threads 96 \
      --root-bound \
      --out migration_benchmark_v2_fair.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError as e:
    raise SystemExit(
        "This script requires gurobipy. Install it and make sure a Gurobi license is available."
    ) from e


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Instance:
    n_vm: int
    n_server: int
    n_time: int
    n_scen: int
    capacity: int
    demand: Dict[Tuple[int, int, int], int]  # (i, t, xi) -> CPU demand
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
        GRB.NODE_LIMIT: "NODE_LIMIT",
    }
    return names.get(status, str(status))


def safe_attr(model: gp.Model, attr: str) -> Optional[float]:
    try:
        return getattr(model, attr)
    except Exception:
        return None


def add_common_first_stage(
    model: gp.Model,
    inst: Instance,
    symmetry: bool,
) -> Tuple[Any, Any]:
    I = range(inst.n_vm)
    S = range(inst.n_server)
    T = range(inst.n_time)

    x = model.addVars(I, S, vtype=GRB.BINARY, name="x")       # pre-migration placement
    u = model.addVars(S, T, vtype=GRB.BINARY, name="u")       # server-on schedule

    model.addConstrs(
        (gp.quicksum(x[i, s] for s in S) == 1 for i in I),
        name="initial_assignment",
    )

    if symmetry:
        # Homogeneous-server labels can be permuted once, globally. Ordering
        # aggregate powered slots is therefore valid; ordering every time slot
        # independently is not, because crossing on/off schedules may require
        # different permutations at different times.
        model.addConstrs(
            (
                gp.quicksum(u[s, t] for t in T) >= gp.quicksum(u[s + 1, t] for t in T)
                for s in range(inst.n_server - 1)
            ),
            name="server_order_aggregate",
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


def add_objective(
    model: gp.Model,
    inst: Instance,
    u: Any,
    mig_expr: Any,
    mig_cost: float,
) -> None:
    S = range(inst.n_server)
    T = range(inst.n_time)
    server_cost = gp.quicksum(u[s, t] for s in S for t in T)
    model.setObjective(server_cost + mig_cost * mig_expr, GRB.MINIMIZE)


def migration_budget_sense(exact_one_migration: bool) -> str:
    return "==" if exact_one_migration else "<="


# ---------------------------------------------------------------------------
# Reference formulations
# ---------------------------------------------------------------------------

def build_change_model(inst: Instance, cfg: Dict[str, Any]) -> gp.Model:
    """
    Weak change-detection formulation:
        xR[i,s,t,xi] tracks location.
        m[i,t,xi] is forced to 1 if any server-position bit changes.

    The reverse implication is absent by design, so exact-one migration is not
    supported for this reference formulation.
    """
    if cfg["exact_one_migration"]:
        raise ValueError(
            "The weak 'change' formulation does not support exact-one migration; "
            "use flow, arc, or an event formulation instead."
        )

    I = range(inst.n_vm)
    S = range(inst.n_server)
    T = range(inst.n_time)
    Tp = range(1, inst.n_time)
    Xi = range(inst.n_scen)

    model = gp.Model("migration_change")
    x, u = add_common_first_stage(model, inst, cfg["symmetry"])

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

    mig_expr = gp.quicksum(
        inst.prob[xi] * mvar[i, t, xi]
        for i in I for t in Tp for xi in Xi
    )
    add_objective(model, inst, u, mig_expr, cfg["mig_cost"])
    return model


def build_flow_model(inst: Instance, cfg: Dict[str, Any]) -> gp.Model:
    """
    State-transition formulation:
        xR tracks location.
        mu_plus / mu_minus represent arrival/departure at each server.
        m is the migration event at time t.
    """
    I = range(inst.n_vm)
    S = range(inst.n_server)
    T = range(inst.n_time)
    Tp = range(1, inst.n_time)
    Xi = range(inst.n_scen)

    model = gp.Model("migration_flow")
    x, u = add_common_first_stage(model, inst, cfg["symmetry"])

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

    if cfg["exact_one_migration"]:
        model.addConstrs(
            (gp.quicksum(mvar[i, t, xi] for t in Tp) == 1 for i in I for xi in Xi),
            name="exactly_one_migration",
        )
    else:
        model.addConstrs(
            (gp.quicksum(mvar[i, t, xi] for t in Tp) <= 1 for i in I for xi in Xi),
            name="at_most_one_migration",
        )

    loc = {(i, s, t, xi): xR[i, s, t, xi] for i in I for s in S for t in T for xi in Xi}
    add_capacity_constraints(model, inst, u, loc)

    mig_expr = gp.quicksum(
        inst.prob[xi] * mvar[i, t, xi]
        for i in I for t in Tp for xi in Xi
    )
    add_objective(model, inst, u, mig_expr, cfg["mig_cost"])
    return model


def build_arc_model(inst: Instance, cfg: Dict[str, Any]) -> gp.Model:
    """
    Full server-pair migration arc formulation:
        q[i,s,r,t,xi] = 1 if VM i migrates from s to r between t-1 and t.
    """
    I = range(inst.n_vm)
    S = range(inst.n_server)
    T = range(inst.n_time)
    Tp = range(1, inst.n_time)
    Xi = range(inst.n_scen)

    model = gp.Model("migration_arc")
    x, u = add_common_first_stage(model, inst, cfg["symmetry"])

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

    if cfg["exact_one_migration"]:
        model.addConstrs(
            (
                gp.quicksum(q[i, s, r, t, xi] for s in S for r in S if s != r for t in Tp) == 1
                for i in I for xi in Xi
            ),
            name="exactly_one_migration",
        )
    else:
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

    mig_expr = gp.quicksum(
        inst.prob[xi] * q[i, s, r, t, xi]
        for (i, s, r, t, xi) in arcs
    )
    add_objective(model, inst, u, mig_expr, cfg["mig_cost"])
    return model


# ---------------------------------------------------------------------------
# New event formulations: x, xprime, m, H
# ---------------------------------------------------------------------------

def add_event_core_variables(
    model: gp.Model,
    inst: Instance,
    x: Any,
    force_h_var: bool,
) -> Tuple[Any, Any, Any, Optional[Any]]:
    """
    Common variables for event formulations.

    m[i,t,xi] is defined for t=1,...,T-1 and means migration between t-1 and t.
    xprime[i,s,xi] is the post-migration placement.
    xR[i,s,t,xi] is the realized location used in capacity/energy constraints.
    h[i,t,xi] is optional cumulative migration indicator H_it.
    """
    I = range(inst.n_vm)
    S = range(inst.n_server)
    T = range(inst.n_time)
    Tp = range(1, inst.n_time)
    Xi = range(inst.n_scen)

    mvar = model.addVars(I, Tp, Xi, vtype=GRB.BINARY, name="m")
    xprime = model.addVars(I, S, Xi, vtype=GRB.BINARY, name="xprime")
    xR = model.addVars(I, S, T, Xi, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="xR")

    hvar = None
    if force_h_var:
        hvar = model.addVars(I, T, Xi, vtype=GRB.BINARY, name="H")

    # At most/exactly one migration.
    if model._cfg["exact_one_migration"]:
        model.addConstrs(
            (gp.quicksum(mvar[i, t, xi] for t in Tp) == 1 for i in I for xi in Xi),
            name="exactly_one_migration",
        )
    else:
        model.addConstrs(
            (gp.quicksum(mvar[i, t, xi] for t in Tp) <= 1 for i in I for xi in Xi),
            name="at_most_one_migration",
        )

    # Destination exists iff migration occurs.
    model.addConstrs(
        (
            gp.quicksum(xprime[i, s, xi] for s in S)
            == gp.quicksum(mvar[i, t, xi] for t in Tp)
            for i in I for xi in Xi
        ),
        name="choose_destination_iff_migrate",
    )

    # Do not migrate to the initial server.
    model.addConstrs(
        (
            xprime[i, s, xi] <= 1 - x[i, s]
            for i in I for s in S for xi in Xi
        ),
        name="no_same_server_destination",
    )

    # Unique realized location. This is redundant when the switching constraints are exact,
    # but helps numerical stability and catches modeling bugs.
    model.addConstrs(
        (
            gp.quicksum(xR[i, s, t, xi] for s in S) == 1
            for i in I for t in T for xi in Xi
        ),
        name="unique_realized_location",
    )

    if hvar is not None:
        model.addConstrs(
            (hvar[i, 0, xi] == 0 for i in I for xi in Xi),
            name="H_initial_zero",
        )
        model.addConstrs(
            (
                hvar[i, t, xi] == gp.quicksum(mvar[i, tau, xi] for tau in Tp if tau <= t)
                for i in I for t in Tp for xi in Xi
            ),
            name="H_cumulative_migration",
        )

    return mvar, xprime, xR, hvar


def h_expr(mvar: Any, i: int, t: int, xi: int) -> Any:
    """Cumulative migration expression H_it = sum_{tau <= t} m_i,tau."""
    if t == 0:
        return 0.0
    return gp.quicksum(mvar[i, tau, xi] for tau in range(1, t + 1))


def build_event_linear_model(inst: Instance, cfg: Dict[str, Any]) -> gp.Model:
    """
    Linear switching formulation:
        H=0 -> xR=x
        H=1 -> xR=xprime

    Implemented as:
        xR - x      <= H
        x - xR      <= H
        xR - xprime <= 1-H
        xprime - xR <= 1-H
    """
    I = range(inst.n_vm)
    S = range(inst.n_server)
    T = range(inst.n_time)
    Xi = range(inst.n_scen)

    model = gp.Model("migration_event_linear")
    model._cfg = cfg
    x, u = add_common_first_stage(model, inst, cfg["symmetry"])
    mvar, xprime, xR, _ = add_event_core_variables(model, inst, x, force_h_var=False)

    for i in I:
        for s in S:
            for t in T:
                for xi in Xi:
                    h = h_expr(mvar, i, t, xi)
                    model.addConstr(xR[i, s, t, xi] - x[i, s] <= h, name=f"pre_switch_ub[{i},{s},{t},{xi}]")
                    model.addConstr(x[i, s] - xR[i, s, t, xi] <= h, name=f"pre_switch_lb[{i},{s},{t},{xi}]")
                    model.addConstr(xR[i, s, t, xi] - xprime[i, s, xi] <= 1 - h, name=f"post_switch_ub[{i},{s},{t},{xi}]")
                    model.addConstr(xprime[i, s, xi] - xR[i, s, t, xi] <= 1 - h, name=f"post_switch_lb[{i},{s},{t},{xi}]")

    loc = {(i, s, t, xi): xR[i, s, t, xi] for i in I for s in S for t in T for xi in Xi}
    add_capacity_constraints(model, inst, u, loc)

    mig_expr = gp.quicksum(
        inst.prob[xi] * mvar[i, t, xi]
        for i in I for t in range(1, inst.n_time) for xi in Xi
    )
    add_objective(model, inst, u, mig_expr, cfg["mig_cost"])
    return model


def build_event_indicator_model(inst: Instance, cfg: Dict[str, Any]) -> gp.Model:
    """
    Indicator formulation:
        H[i,t,xi] = 0 => xR[i,s,t,xi] = x[i,s]
        H[i,t,xi] = 1 => xR[i,s,t,xi] = xprime[i,s,xi]

    hvar is binary and linked by:
        H[i,t,xi] = sum_{tau <= t} m[i,tau,xi]
    """
    I = range(inst.n_vm)
    S = range(inst.n_server)
    T = range(inst.n_time)
    Xi = range(inst.n_scen)

    model = gp.Model("migration_event_indicator")
    model._cfg = cfg
    x, u = add_common_first_stage(model, inst, cfg["symmetry"])
    mvar, xprime, xR, H = add_event_core_variables(model, inst, x, force_h_var=True)
    assert H is not None

    for i in I:
        for s in S:
            for t in T:
                for xi in Xi:
                    model.addGenConstrIndicator(
                        H[i, t, xi], 0,
                        xR[i, s, t, xi] - x[i, s],
                        GRB.EQUAL,
                        0.0,
                        name=f"H0_pre[{i},{s},{t},{xi}]",
                    )
                    model.addGenConstrIndicator(
                        H[i, t, xi], 1,
                        xR[i, s, t, xi] - xprime[i, s, xi],
                        GRB.EQUAL,
                        0.0,
                        name=f"H1_post[{i},{s},{t},{xi}]",
                    )

    loc = {(i, s, t, xi): xR[i, s, t, xi] for i in I for s in S for t in T for xi in Xi}
    add_capacity_constraints(model, inst, u, loc)

    mig_expr = gp.quicksum(
        inst.prob[xi] * mvar[i, t, xi]
        for i in I for t in range(1, inst.n_time) for xi in Xi
    )
    add_objective(model, inst, u, mig_expr, cfg["mig_cost"])
    return model


def build_event_bilinear_model(inst: Instance, cfg: Dict[str, Any]) -> gp.Model:
    """
    Direct bilinear formulation:
        xR[i,s,t,xi] = (1-H[i,t,xi]) * x[i,s] + H[i,t,xi] * xprime[i,s,xi]

    Since H is linear in binary m variables, this creates bilinear terms:
        H*x and H*xprime

    This is a nonconvex MIQCP-like model, not an SOCP in general.
    Gurobi requires NonConvex=2.
    """
    I = range(inst.n_vm)
    S = range(inst.n_server)
    T = range(inst.n_time)
    Xi = range(inst.n_scen)

    model = gp.Model("migration_event_bilinear")
    model._cfg = cfg
    model._requires_nonconvex = True
    x, u = add_common_first_stage(model, inst, cfg["symmetry"])
    mvar, xprime, xR, _ = add_event_core_variables(model, inst, x, force_h_var=False)

    for i in I:
        for s in S:
            for t in T:
                for xi in Xi:
                    if t == 0:
                        model.addConstr(
                            xR[i, s, t, xi] == x[i, s],
                            name=f"bilinear_initial[{i},{s},{t},{xi}]",
                        )
                    else:
                        h = h_expr(mvar, i, t, xi)
                        # xR = (1-H)*x + H*xprime = x - H*x + H*xprime
                        model.addQConstr(
                            xR[i, s, t, xi]
                            == x[i, s] - h * x[i, s] + h * xprime[i, s, xi],
                            name=f"bilinear_switch[{i},{s},{t},{xi}]",
                        )

    loc = {(i, s, t, xi): xR[i, s, t, xi] for i in I for s in S for t in T for xi in Xi}
    add_capacity_constraints(model, inst, u, loc)

    mig_expr = gp.quicksum(
        inst.prob[xi] * mvar[i, t, xi]
        for i in I for t in range(1, inst.n_time) for xi in Xi
    )
    add_objective(model, inst, u, mig_expr, cfg["mig_cost"])
    return model


BUILDERS = {
    "change": build_change_model,
    "flow": build_flow_model,
    "arc": build_arc_model,
    "event_linear": build_event_linear_model,
    "event_indicator": build_event_indicator_model,
    "event_bilinear": build_event_bilinear_model,
}


# ---------------------------------------------------------------------------
# Solving and benchmarking
# ---------------------------------------------------------------------------

def solve_root_bound(model: gp.Model, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Estimate root-node bound by solving a copied model with NodeLimit=0.
    This is often more meaningful than model.relax() for indicator constraints
    and nonconvex quadratic constraints.
    """
    out = {
        "root_bound_status": None,
        "root_bound_runtime": None,
        "root_bound": None,
        "root_incumbent": None,
    }

    try:
        root = model.copy()
        root.Params.OutputFlag = 0
        root.Params.Threads = cfg["threads"]
        root.Params.TimeLimit = min(float(cfg["time_limit"]), float(cfg["root_time_limit"]))
        root.Params.NodeLimit = 0
        root.Params.Heuristics = 0
        if getattr(model, "_requires_nonconvex", False):
            root.Params.NonConvex = 2

        root.optimize()
        out["root_bound_status"] = status_name(root.Status)
        out["root_bound_runtime"] = safe_attr(root, "Runtime")
        out["root_bound"] = safe_attr(root, "ObjBound")
        if root.SolCount > 0:
            out["root_incumbent"] = root.ObjVal
    except Exception as e:
        out["root_bound_status"] = f"ERROR: {type(e).__name__}: {e}"

    return out


def solve_root_relaxation(model: gp.Model, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Solve continuous relaxation. For event_bilinear this remains a nonconvex
    continuous QCP, so it is not directly comparable to an LP relaxation.
    """
    out = {
        "relax_status": None,
        "relax_runtime": None,
        "relax_obj": None,
    }

    try:
        model.update()
        relaxed = model.relax()
        relaxed.Params.OutputFlag = 0
        relaxed.Params.Threads = cfg["threads"]
        relaxed.Params.TimeLimit = min(float(cfg["time_limit"]), float(cfg["root_time_limit"]))
        if getattr(model, "_requires_nonconvex", False):
            relaxed.Params.NonConvex = 2

        relaxed.optimize()
        out["relax_status"] = status_name(relaxed.Status)
        out["relax_runtime"] = safe_attr(relaxed, "Runtime")
        if relaxed.SolCount > 0:
            out["relax_obj"] = relaxed.ObjVal
    except Exception as e:
        out["relax_status"] = f"ERROR: {type(e).__name__}: {e}"

    return out


def run_one_job(job: Dict[str, Any]) -> Dict[str, Any]:
    formulation = job["formulation"]
    seed = job["seed"]

    cfg = {
        "symmetry": job["symmetry"],
        "mig_cost": job["mig_cost"],
        "exact_one_migration": job["exact_one_migration"],
        "threads": job["threads"],
        "time_limit": job["time_limit"],
        "root_time_limit": job["root_time_limit"],
    }

    started = time.time()
    try:
        inst = generate_instance(
            n_vm=job["n_vm"],
            n_server=job["n_server"],
            n_time=job["n_time"],
            n_scen=job["n_scen"],
            seed=seed,
            capacity=job["capacity"],
            target_total_util=job["target_total_util"],
            load_low=job["load_low"],
            load_high=job["load_high"],
        )

        build_started = time.time()
        model = BUILDERS[formulation](inst, cfg)
        build_time = time.time() - build_started

        model.Params.TimeLimit = job["time_limit"]
        model.Params.MIPGap = job["mip_gap"]
        model.Params.Threads = job["threads"]
        model.Params.OutputFlag = 1 if job["log"] else 0
        model.Params.Seed = seed

        if getattr(model, "_requires_nonconvex", False):
            model.Params.NonConvex = 2

        model.update()

        root_bound_info = {
            "root_bound_status": None,
            "root_bound_runtime": None,
            "root_bound": None,
            "root_incumbent": None,
        }
        relax_info = {
            "relax_status": None,
            "relax_runtime": None,
            "relax_obj": None,
        }

        if job["root_bound"]:
            root_bound_info = solve_root_bound(model, cfg)

        if job["root_relax"]:
            relax_info = solve_root_relaxation(model, cfg)

        solve_started = time.time()
        model.optimize()
        solve_time = time.time() - solve_started

        result = {
            "formulation": formulation,
            "seed": seed,
            "n_vm": inst.n_vm,
            "n_server": inst.n_server,
            "n_time": inst.n_time,
            "n_scen": inst.n_scen,
            "capacity": inst.capacity,
            "exact_one_migration": job["exact_one_migration"],
            "status": status_name(model.Status),
            "build_time": build_time,
            "solve_wall_time": solve_time,
            "total_wall_time": time.time() - started,
            "runtime": safe_attr(model, "Runtime"),
            "num_vars": model.NumVars,
            "num_bin_vars": model.NumBinVars,
            "num_int_vars": model.NumIntVars,
            "num_constrs": model.NumConstrs,
            "num_qconstrs": getattr(model, "NumQConstrs", None),
            "num_genconstrs": getattr(model, "NumGenConstrs", None),
            "num_nz": model.NumNZs,
            "num_qnz": getattr(model, "NumQNZs", None),
            "node_count": safe_attr(model, "NodeCount"),
            "obj_val": None,
            "obj_bound": safe_attr(model, "ObjBound"),
            "mip_gap": None,
            "error": None,
            **root_bound_info,
            **relax_info,
        }

        if model.SolCount > 0:
            result["obj_val"] = model.ObjVal
            result["mip_gap"] = safe_attr(model, "MIPGap")

        return result

    except Exception:
        return {
            "formulation": formulation,
            "seed": seed,
            "n_vm": job["n_vm"],
            "n_server": job["n_server"],
            "n_time": job["n_time"],
            "n_scen": job["n_scen"],
            "capacity": job["capacity"],
            "exact_one_migration": job["exact_one_migration"],
            "status": "ERROR",
            "build_time": None,
            "solve_wall_time": None,
            "total_wall_time": time.time() - started,
            "runtime": None,
            "num_vars": None,
            "num_bin_vars": None,
            "num_int_vars": None,
            "num_constrs": None,
            "num_qconstrs": None,
            "num_genconstrs": None,
            "num_nz": None,
            "num_qnz": None,
            "node_count": None,
            "obj_val": None,
            "obj_bound": None,
            "mip_gap": None,
            "root_bound_status": None,
            "root_bound_runtime": None,
            "root_bound": None,
            "root_incumbent": None,
            "relax_status": None,
            "relax_runtime": None,
            "relax_obj": None,
            "error": traceback.format_exc(),
        }


def write_header_if_needed(path: Path, fieldnames: List[str]) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def append_row(path: Path, fieldnames: List[str], row: Dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def default_fieldnames() -> List[str]:
    return [
        "formulation",
        "seed",
        "n_vm",
        "n_server",
        "n_time",
        "n_scen",
        "capacity",
        "exact_one_migration",
        "status",
        "build_time",
        "solve_wall_time",
        "total_wall_time",
        "runtime",
        "num_vars",
        "num_bin_vars",
        "num_int_vars",
        "num_constrs",
        "num_qconstrs",
        "num_genconstrs",
        "num_nz",
        "num_qnz",
        "node_count",
        "obj_val",
        "obj_bound",
        "mip_gap",
        "root_bound_status",
        "root_bound_runtime",
        "root_bound",
        "root_incumbent",
        "relax_status",
        "relax_runtime",
        "relax_obj",
        "error",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--formulations",
        nargs="+",
        default=["change", "event_linear", "event_indicator", "event_bilinear", "flow"],
        choices=list(BUILDERS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])

    parser.add_argument("--n-vm", type=int, default=30)
    parser.add_argument("--n-server", type=int, default=10)
    parser.add_argument("--n-time", type=int, default=24)
    parser.add_argument("--n-scen", type=int, default=20)
    parser.add_argument("--capacity", type=int, default=0, help="0 means auto capacity")
    parser.add_argument("--target-total-util", type=float, default=0.68)
    parser.add_argument("--load-low", type=int, default=8)
    parser.add_argument("--load-high", type=int, default=28)

    parser.add_argument("--mig-cost", type=float, default=0.02)
    parser.add_argument(
        "--exact-one-migration",
        action="store_true",
        help="Require one physical move per VM/scenario; unsupported with the weak 'change' formulation.",
    )

    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument(
        "--parallel-jobs",
        type=int,
        default=1,
        help=(
            "Number of independent Gurobi solves to run concurrently. "
            "For a 96-thread server, use e.g. --parallel-jobs 4 --threads 24."
        ),
    )

    parser.add_argument(
        "--root-bound",
        action="store_true",
        help="Also solve a NodeLimit=0 copy to estimate root-node bound.",
    )
    parser.add_argument(
        "--root-relax",
        action="store_true",
        help=(
            "Also solve continuous relaxation. For event_bilinear this is a nonconvex "
            "continuous QCP, not a plain LP relaxation."
        ),
    )
    parser.add_argument("--root-time-limit", type=float, default=120.0)

    parser.add_argument("--no-symmetry", action="store_true")
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("migration_benchmark_v2.csv"))

    args = parser.parse_args()
    if args.exact_one_migration and "change" in args.formulations:
        parser.error(
            "--exact-one-migration cannot be combined with the weak 'change' formulation; "
            "choose flow, arc, or an event formulation"
        )
    return args


def make_jobs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    for seed in args.seeds:
        for formulation in args.formulations:
            jobs.append(
                {
                    "formulation": formulation,
                    "seed": seed,
                    "n_vm": args.n_vm,
                    "n_server": args.n_server,
                    "n_time": args.n_time,
                    "n_scen": args.n_scen,
                    "capacity": args.capacity,
                    "target_total_util": args.target_total_util,
                    "load_low": args.load_low,
                    "load_high": args.load_high,
                    "mig_cost": args.mig_cost,
                    "exact_one_migration": args.exact_one_migration,
                    "time_limit": args.time_limit,
                    "mip_gap": args.mip_gap,
                    "threads": args.threads,
                    "parallel_jobs": args.parallel_jobs,
                    "root_bound": args.root_bound,
                    "root_relax": args.root_relax,
                    "root_time_limit": args.root_time_limit,
                    "symmetry": not args.no_symmetry,
                    "log": args.log,
                }
            )
    return jobs


def print_row_summary(row: Dict[str, Any]) -> None:
    print(
        "[done] "
        f"form={row['formulation']}, seed={row['seed']}, status={row['status']}, "
        f"obj={row['obj_val']}, bound={row['obj_bound']}, gap={row['mip_gap']}, "
        f"runtime={row['runtime']}, nodes={row['node_count']}, "
        f"vars={row['num_vars']}, bin={row['num_bin_vars']}, "
        f"constrs={row['num_constrs']}, qconstrs={row['num_qconstrs']}, "
        f"genconstrs={row['num_genconstrs']}"
    )
    if row.get("error"):
        print(row["error"])


def main() -> None:
    args = parse_args()

    if args.parallel_jobs < 1:
        raise ValueError("--parallel-jobs must be at least 1")

    if args.parallel_jobs * args.threads > 96:
        print(
            f"[warning] parallel_jobs * threads = {args.parallel_jobs * args.threads}. "
            "This may oversubscribe a 96-thread server."
        )

    jobs = make_jobs(args)
    fieldnames = default_fieldnames()
    write_header_if_needed(args.out, fieldnames)

    print(f"[info] formulations={args.formulations}")
    print(f"[info] jobs={len(jobs)}, parallel_jobs={args.parallel_jobs}, threads_per_job={args.threads}")
    print(f"[info] output={args.out}")

    if args.parallel_jobs == 1:
        for job in jobs:
            print(f"[run] formulation={job['formulation']}, seed={job['seed']}")
            row = run_one_job(job)
            append_row(args.out, fieldnames, row)
            print_row_summary(row)
    else:
        with ProcessPoolExecutor(max_workers=args.parallel_jobs) as ex:
            future_to_job = {ex.submit(run_one_job, job): job for job in jobs}
            for fut in as_completed(future_to_job):
                job = future_to_job[fut]
                try:
                    row = fut.result()
                except Exception:
                    row = {
                        "formulation": job["formulation"],
                        "seed": job["seed"],
                        "n_vm": job["n_vm"],
                        "n_server": job["n_server"],
                        "n_time": job["n_time"],
                        "n_scen": job["n_scen"],
                        "capacity": job["capacity"],
                        "exact_one_migration": job["exact_one_migration"],
                        "status": "ERROR",
                        "build_time": None,
                        "solve_wall_time": None,
                        "total_wall_time": None,
                        "runtime": None,
                        "num_vars": None,
                        "num_bin_vars": None,
                        "num_int_vars": None,
                        "num_constrs": None,
                        "num_qconstrs": None,
                        "num_genconstrs": None,
                        "num_nz": None,
                        "num_qnz": None,
                        "node_count": None,
                        "obj_val": None,
                        "obj_bound": None,
                        "mip_gap": None,
                        "root_bound_status": None,
                        "root_bound_runtime": None,
                        "root_bound": None,
                        "root_incumbent": None,
                        "relax_status": None,
                        "relax_runtime": None,
                        "relax_obj": None,
                        "error": traceback.format_exc(),
                    }
                append_row(args.out, fieldnames, row)
                print_row_summary(row)

    print(f"\nSaved results to: {args.out}")


if __name__ == "__main__":
    main()
