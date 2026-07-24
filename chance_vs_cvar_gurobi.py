# chance_vs_cvar_gurobi.py
#
# Compare:
#   1) Exact empirical chance constraint via MILP
#   2) CVaR surrogate via LP
#
# Model:
#   maximize c^T x
#   subject to P(a(xi)^T x <= b) >= 1 - eps
#   0 <= x <= ub
#
# Run:
#   python chance_vs_cvar_gurobi.py --N 800 --n 40 --trials 3

import argparse
import math
import numpy as np
import gurobipy as gp
from gurobipy import GRB


def status_name(status):
    names = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }
    return names.get(status, str(status))


def set_common_params(model, time_limit, mip_gap, output_flag, threads, seed):
    model.Params.OutputFlag = int(output_flag)
    model.Params.TimeLimit = float(time_limit)
    model.Params.Seed = int(seed)
    if mip_gap is not None:
        model.Params.MIPGap = float(mip_gap)
    if threads is not None and threads > 0:
        model.Params.Threads = int(threads)


def generate_instance(n, n_train, n_test, seed, cap_factor, sigma):
    rng = np.random.default_rng(seed)

    # Positive random resource coefficients.
    # lognormal mean is approximately base_mean.
    base_mean = rng.uniform(0.7, 1.3, size=n)
    log_mean = np.log(base_mean) - 0.5 * sigma**2

    A_train = rng.lognormal(mean=log_mean, sigma=sigma, size=(n_train, n))
    A_test = rng.lognormal(mean=log_mean, sigma=sigma, size=(n_test, n))

    ub = rng.uniform(0.5, 1.5, size=n)

    # Profit-to-resource tradeoff is nontrivial.
    c = rng.uniform(0.8, 2.0, size=n) / base_mean

    # Capacity: cap_factor times expected full-upper-bound consumption.
    b = cap_factor * float(base_mean @ ub)

    return A_train, A_test, b, c, ub


def solve_exact_chance_mip(A, b, c, ub, eps, time_limit, mip_gap, output_flag, threads, seed):
    N, n = A.shape
    k = int(math.floor(eps * N + 1e-9))

    # Valid tight Big-M for this generated instance:
    # A >= 0 and 0 <= x <= ub, so max_x A_s x - b = A_s ub - b.
    M = np.maximum(A @ ub - b, 0.0)

    model = gp.Model("exact_empirical_chance_mip")
    set_common_params(model, time_limit, mip_gap, output_flag, threads, seed)

    x = model.addVars(
        n,
        lb=0.0,
        ub={j: float(ub[j]) for j in range(n)},
        name="x",
    )
    z = model.addVars(N, vtype=GRB.BINARY, name="viol")

    model.addConstrs(
        (
            gp.quicksum(float(A[s, j]) * x[j] for j in range(n))
            <= float(b) + float(M[s]) * z[s]
            for s in range(N)
        ),
        name="scenario_chance",
    )

    model.addConstr(gp.quicksum(z[s] for s in range(N)) <= k, name="violation_budget")

    model.setObjective(
        gp.quicksum(float(c[j]) * x[j] for j in range(n)),
        GRB.MAXIMIZE,
    )

    model.optimize()

    x_val = None
    obj = float("nan")
    if model.SolCount > 0:
        x_val = np.array([x[j].X for j in range(n)])
        obj = float(model.ObjVal)

    try:
        gap = float(model.MIPGap)
    except Exception:
        gap = float("nan")

    try:
        nodes = float(model.NodeCount)
    except Exception:
        nodes = float("nan")

    return {
        "model": "chance_mip",
        "status": status_name(model.Status),
        "runtime": float(model.Runtime),
        "obj": obj,
        "gap": gap,
        "nodes": nodes,
        "x": x_val,
        "num_bin": int(model.NumBinVars),
        "num_constr": int(model.NumConstrs),
    }


def solve_cvar_surrogate_lp(A, b, c, ub, eps, time_limit, mip_gap, output_flag, threads, seed):
    N, n = A.shape

    model = gp.Model("cvar_surrogate_lp")
    set_common_params(model, time_limit, mip_gap, output_flag, threads, seed)

    x = model.addVars(
        n,
        lb=0.0,
        ub={j: float(ub[j]) for j in range(n)},
        name="x",
    )

    eta = model.addVar(lb=-GRB.INFINITY, name="eta")
    u = model.addVars(N, lb=0.0, name="excess")

    # u_s >= a_s^T x - b - eta
    model.addConstrs(
        (
            u[s]
            >= gp.quicksum(float(A[s, j]) * x[j] for j in range(n)) - float(b) - eta
            for s in range(N)
        ),
        name="cvar_excess",
    )

    # eta + (1 / (eps * N)) * sum_s u_s <= 0
    model.addConstr(
        eta + (1.0 / (eps * N)) * gp.quicksum(u[s] for s in range(N)) <= 0.0,
        name="cvar_limit",
    )

    model.setObjective(
        gp.quicksum(float(c[j]) * x[j] for j in range(n)),
        GRB.MAXIMIZE,
    )

    model.optimize()

    x_val = None
    obj = float("nan")
    if model.SolCount > 0:
        x_val = np.array([x[j].X for j in range(n)])
        obj = float(model.ObjVal)

    return {
        "model": "cvar_lp",
        "status": status_name(model.Status),
        "runtime": float(model.Runtime),
        "obj": obj,
        "gap": float("nan"),
        "nodes": float("nan"),
        "x": x_val,
        "num_bin": int(model.NumBinVars),
        "num_constr": int(model.NumConstrs),
    }


def empirical_tail_cvar(losses, eps):
    losses = np.asarray(losses, dtype=float)

    try:
        eta = np.quantile(losses, 1.0 - eps, method="higher")
    except TypeError:
        eta = np.quantile(losses, 1.0 - eps, interpolation="higher")

    return float(eta + np.maximum(losses - eta, 0.0).mean() / eps)


def evaluate_solution(x, A, b, c, eps):
    if x is None:
        return {
            "objective": float("nan"),
            "viol_rate": float("nan"),
            "mean_excess": float("nan"),
            "max_loss": float("nan"),
            "tail_cvar": float("nan"),
        }

    losses = A @ x - b
    violations = losses > 1e-7

    return {
        "objective": float(c @ x),
        "viol_rate": float(np.mean(violations)),
        "mean_excess": float(np.maximum(losses, 0.0).mean()),
        "max_loss": float(np.max(losses)),
        "tail_cvar": empirical_tail_cvar(losses, eps),
    }


def fmt(x, digits=4):
    if x is None:
        return "nan"
    try:
        if not np.isfinite(x):
            return "nan"
    except TypeError:
        return str(x)
    return f"{x:.{digits}g}"


def run_trial(args, seed):
    A_train, A_test, b, c, ub = generate_instance(
        n=args.n,
        n_train=args.N,
        n_test=args.Ntest,
        seed=seed,
        cap_factor=args.cap_factor,
        sigma=args.sigma,
    )

    chance = solve_exact_chance_mip(
        A_train,
        b,
        c,
        ub,
        args.eps,
        args.time_limit,
        args.mip_gap,
        args.output,
        args.threads,
        seed,
    )

    cvar = solve_cvar_surrogate_lp(
        A_train,
        b,
        c,
        ub,
        args.eps,
        args.time_limit,
        args.mip_gap,
        args.output,
        args.threads,
        seed,
    )

    rows = []
    for result in [chance, cvar]:
        train_eval = evaluate_solution(result["x"], A_train, b, c, args.eps)
        test_eval = evaluate_solution(result["x"], A_test, b, c, args.eps)

        rows.append(
            {
                "seed": seed,
                "model": result["model"],
                "status": result["status"],
                "runtime": result["runtime"],
                "obj": result["obj"],
                "gap": result["gap"],
                "nodes": result["nodes"],
                "num_bin": result["num_bin"],
                "num_constr": result["num_constr"],
                "train_viol": train_eval["viol_rate"],
                "test_viol": test_eval["viol_rate"],
                "train_cvar": train_eval["tail_cvar"],
                "test_cvar": test_eval["tail_cvar"],
                "train_mean_excess": train_eval["mean_excess"],
                "test_mean_excess": test_eval["mean_excess"],
            }
        )

    return rows


def average(values):
    vals = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def print_rows(rows):
    header = (
        "seed", "model", "status", "time", "obj", "gap", "nodes",
        "bin", "train_v", "test_v", "train_cvar", "test_cvar"
    )
    print(
        f"{header[0]:>5} {header[1]:<11} {header[2]:<11} "
        f"{header[3]:>9} {header[4]:>12} {header[5]:>9} {header[6]:>9} "
        f"{header[7]:>5} {header[8]:>9} {header[9]:>9} "
        f"{header[10]:>12} {header[11]:>12}"
    )

    for r in rows:
        print(
            f"{r['seed']:>5} {r['model']:<11} {r['status']:<11} "
            f"{fmt(r['runtime']):>9} {fmt(r['obj']):>12} {fmt(r['gap']):>9} "
            f"{fmt(r['nodes']):>9} {r['num_bin']:>5} "
            f"{fmt(r['train_viol']):>9} {fmt(r['test_viol']):>9} "
            f"{fmt(r['train_cvar']):>12} {fmt(r['test_cvar']):>12}"
        )


def print_summary(rows):
    print("\nSummary by model")
    print("-" * 80)

    for model_name in ["chance_mip", "cvar_lp"]:
        subset = [r for r in rows if r["model"] == model_name]

        solved = sum(np.isfinite(r["obj"]) for r in subset)
        total = len(subset)

        print(f"{model_name}: solved {solved}/{total}")
        print(f"  avg runtime       = {fmt(average([r['runtime'] for r in subset]))}")
        print(f"  avg objective     = {fmt(average([r['obj'] for r in subset]))}")
        print(f"  avg train viol    = {fmt(average([r['train_viol'] for r in subset]))}")
        print(f"  avg test viol     = {fmt(average([r['test_viol'] for r in subset]))}")
        print(f"  avg train CVaR    = {fmt(average([r['train_cvar'] for r in subset]))}")
        print(f"  avg test CVaR     = {fmt(average([r['test_cvar'] for r in subset]))}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--N", type=int, default=800)
    parser.add_argument("--Ntest", type=int, default=10000)
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--trials", type=int, default=3)

    parser.add_argument("--cap-factor", type=float, default=0.45)
    parser.add_argument("--sigma", type=float, default=0.35)

    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--mip-gap", type=float, default=1e-4)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--output", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)

    args = parser.parse_args()

    if not (0.0 < args.eps < 1.0):
        raise ValueError("--eps must be in (0, 1).")

    all_rows = []
    for t in range(args.trials):
        seed = args.seed + t
        all_rows.extend(run_trial(args, seed))

    print_rows(all_rows)
    print_summary(all_rows)


if __name__ == "__main__":
    main()