from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data import CPU, MEM, InstanceData


@dataclass
class ModelArtifacts:
    model: Any
    data: InstanceData
    v: dict[str, Any]


def build_model(data: InstanceData) -> ModelArtifacts:
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise RuntimeError("gurobipy is required to build and solve the model") from exc

    m = gp.Model("notion_energy_two_stage_vmp")
    I, J, K, S, T, Xi = data.I, data.J, data.K, data.S, data.T, data.Xi
    phi = [CPU, MEM]

    od_keys = [(i, s, t, xi) for i in I for s in S for t in data.active_od[i] for xi in Xi]
    spot_keys = [(j, s, t, xi) for j in J for s in S for t in data.active_spot[j] for xi in Xi]
    mig_keys = [
        (i, t, xi)
        for i in I
        for t in data.active_od[i][:-1]
        for xi in Xi
    ]
    h_keys = [(j, t, xi) for j in J for t in data.active_spot[j] for xi in Xi]
    batch_keys = [(k, s, t, xi) for k in K for s in S for t in T for xi in Xi]
    load_keys = [(s, r, t, xi) for s in S for r in phi for t in T for xi in Xi]
    exc_keys = [(s, t, xi) for s in S for t in T for xi in Xi]
    energy_keys = [(t, xi) for t in T for xi in Xi]

    # First-stage decisions.
    x_init = m.addVars(I, S, vtype=GRB.BINARY, name="x_init")
    y_init = m.addVars(J, S, vtype=GRB.BINARY, name="y_init")
    z = m.addVars(K, S, T, vtype=GRB.BINARY, name="z_batch")
    z_on = m.addVars(K, S, T, vtype=GRB.BINARY, name="z_batch_on")
    w = m.addVars(J, vtype=GRB.BINARY, name="w_spot_accept")
    u = m.addVars(S, T, vtype=GRB.BINARY, name="u_server")
    transition_periods = T[:-1]
    u_on = m.addVars(S, transition_periods, vtype=GRB.BINARY, name="u_server_on")
    u_off = m.addVars(S, transition_periods, vtype=GRB.BINARY, name="u_server_off")
    g_da = m.addVars(T, lb=0.0, name="grid_da")

    # Second-stage decisions.
    x = m.addVars(od_keys, vtype=GRB.BINARY, name="x_od")
    mig = m.addVars(mig_keys, vtype=GRB.BINARY, name="migration")
    y = m.addVars(spot_keys, vtype=GRB.BINARY, name="y_spot")
    h = m.addVars(h_keys, vtype=GRB.BINARY, name="spot_preempted")
    b = m.addVars(batch_keys, lb=0.0, name="batch_work")
    load_od = m.addVars(load_keys, lb=0.0, name="load_od_served")
    load_total = m.addVars(load_keys, lb=0.0, name="load_total")
    excess = m.addVars(exc_keys, lb=0.0, name="cpu_excess")
    var_eta = m.addVars(T, lb=0.0, name="cvar_eta")
    cvar_zeta = m.addVars(T, Xi, lb=0.0, name="cvar_zeta")
    energy = m.addVars(energy_keys, lb=0.0, name="energy")
    g_rt = m.addVars(energy_keys, lb=0.0, name="grid_rt")
    r_used = m.addVars(energy_keys, lb=0.0, name="renewable_used")
    r_sold = m.addVars(energy_keys, lb=0.0, name="renewable_sold")
    charge = m.addVars(energy_keys, lb=0.0, name="ess_charge")
    discharge = m.addVars(energy_keys, lb=0.0, name="ess_discharge")
    soc = m.addVars([(t, xi) for t in range(len(T) + 1) for xi in Xi], lb=0.0, ub=data.ess_capacity, name="soc")
    # Server provisioning follows the displayed Notion indexing exactly.
    for s in S:
        for t in transition_periods:
            m.addConstr(u[s, t + 1] - u[s, t] == u_on[s, t] - u_off[s, t], name=f"server_transition[{s},{t}]")
            m.addConstr(u_on[s, t] + u_off[s, t] <= 1, name=f"server_switch_exclusive[{s},{t}]")
            for tau in range(t + 1, min(len(T) - 1, t + data.min_on[s]) + 1):
                m.addConstr(u[s, tau] >= u_on[s, t], name=f"min_up[{s},{t},{tau}]")
            for tau in range(t + 1, min(len(T) - 1, t + data.min_off[s]) + 1):
                m.addConstr(u[s, tau] <= 1 - u_off[s, t], name=f"min_down[{s},{t},{tau}]")

    # On-demand placement and migration.
    for i in I:
        m.addConstr(gp.quicksum(x_init[i, s] for s in S) == 1, name=f"od_init_one[{i}]")
        a = data.active_od[i][0]
        for xi in Xi:
            for s in S:
                m.addConstr(x[i, s, a, xi] == x_init[i, s], name=f"od_start_link[{i},{s},{xi}]")
            for t in data.active_od[i]:
                m.addConstr(gp.quicksum(x[i, s, t, xi] for s in S) == 1, name=f"od_one[{i},{t},{xi}]")
                for s in S:
                    m.addConstr(x[i, s, t, xi] <= u[s, t], name=f"od_on_server[{i},{s},{t},{xi}]")
            for t in data.active_od[i][:-1]:
                for s in S:
                    m.addConstr(mig[i, t, xi] >= x[i, s, t + 1, xi] - x[i, s, t, xi], name=f"mig_pos[{i},{s},{t},{xi}]")
                    m.addConstr(mig[i, t, xi] >= x[i, s, t, xi] - x[i, s, t + 1, xi], name=f"mig_neg[{i},{s},{t},{xi}]")

    # Spot admission, fixed initial server, and irreversible preemption.
    for j in J:
        m.addConstr(gp.quicksum(y_init[j, s] for s in S) == w[j], name=f"spot_init[{j}]")
        periods = data.active_spot[j]
        for xi in Xi:
            for pos, t in enumerate(periods):
                if pos > 0:
                    m.addConstr(h[j, periods[pos - 1], xi] <= h[j, t, xi], name=f"spot_irreversible[{j},{t},{xi}]")
                m.addConstr(
                    gp.quicksum(y[j, s, t, xi] for s in S) + h[j, t, xi] == w[j],
                    name=f"spot_state[{j},{t},{xi}]",
                )
                for s in S:
                    m.addConstr(y[j, s, t, xi] <= y_init[j, s], name=f"spot_fixed_server[{j},{s},{t},{xi}]")
                    m.addConstr(y[j, s, t, xi] <= u[s, t], name=f"spot_on_server[{j},{s},{t},{xi}]")

    # Batch preparation follows the Notion equality at t=1 and lower bound thereafter.
    for k in K:
        for s in S:
            for t in T:
                m.addConstr(z[k, s, t] <= u[s, t], name=f"batch_on_server[{k},{s},{t}]")
                if t == T[0]:
                    m.addConstr(z_on[k, s, t] == z[k, s, t], name=f"batch_start_first[{k},{s}]")
                else:
                    m.addConstr(z_on[k, s, t] >= z[k, s, t] - z[k, s, t - 1], name=f"batch_start_lb[{k},{s},{t}]")

    for k in K:
        for xi in Xi:
            m.addConstr(
                gp.quicksum(b[k, s, t, xi] for s in S for t in T) == data.W[k],
                name=f"batch_complete[{k},{xi}]",
            )
            for s in S:
                for t in T:
                    for r in phi:
                        m.addConstr(
                            data.rho_batch[(k, r)] * b[k, s, t, xi] <= data.q_batch[(k, r)] * z[k, s, t],
                            name=f"batch_limit[{k},{s},{r},{t},{xi}]",
                        )

    # On-demand served load, total served load, and hard server capacity.
    for s in S:
        for t in T:
            for xi in Xi:
                m.addConstr(
                    load_od[s, CPU, t, xi] + excess[s, t, xi]
                    == gp.quicksum(data.load_od[(i, CPU, t, xi)] * x[i, s, t, xi] for i in I if t in data.active_od[i]),
                    name=f"od_cpu_split[{s},{t},{xi}]",
                )
                m.addConstr(
                    load_od[s, MEM, t, xi]
                    == gp.quicksum(data.load_od[(i, MEM, t, xi)] * x[i, s, t, xi] for i in I if t in data.active_od[i]),
                    name=f"od_mem_exact[{s},{t},{xi}]",
                )
                for r in phi:
                    spot_term = gp.quicksum(
                        data.load_spot[(j, r, t, xi)] * y[j, s, t, xi]
                        for j in J
                        if t in data.active_spot[j]
                    )
                    batch_term = gp.quicksum(
                        data.batch_base[r] * z[k, s, t]
                        + data.batch_startup[r] * z_on[k, s, t]
                        + data.rho_batch[(k, r)] * b[k, s, t, xi]
                        for k in K
                    )
                    m.addConstr(
                        load_total[s, r, t, xi] == load_od[s, r, t, xi] + spot_term + batch_term,
                        name=f"total_load[{s},{r},{t},{xi}]",
                    )
                    m.addConstr(load_total[s, r, t, xi] <= data.C[r] * u[s, t], name=f"capacity[{s},{r},{t},{xi}]")

    # Time-indexed CVaR of aggregate on-demand CPU excess.
    for t in T:
        for xi in Xi:
            m.addConstr(
                gp.quicksum(excess[s, t, xi] for s in S) <= var_eta[t] + cvar_zeta[t, xi],
                name=f"cvar_tail[{t},{xi}]",
            )
        m.addConstr(
            var_eta[t] + (1.0 / (1.0 - data.alpha)) * gp.quicksum(data.p[xi] * cvar_zeta[t, xi] for xi in Xi)
            <= data.epsilon,
            name=f"cvar_bound[{t}]",
        )

    # Energy, renewable dispatch, and ESS operation.
    last_t = T[-1]
    for xi in Xi:
        m.addConstr(soc[0, xi] == data.soc_init, name=f"soc_initial[{xi}]")
        m.addConstr(soc[len(T), xi] >= data.soc_init, name=f"soc_terminal[{xi}]")
        for t in T:
            migration_energy = gp.quicksum(
                data.c_mig[i] * data.load_od[(i, MEM, t, xi)] * mig[i, t, xi]
                for i in I
                if t in data.active_od[i][:-1]
            )
            m.addConstr(
                energy[t, xi]
                == gp.quicksum(data.E_idle * u[s, t] + data.E_cpu * load_total[s, CPU, t, xi] for s in S)
                + migration_energy,
                name=f"energy_definition[{t},{xi}]",
            )
            m.addConstr(r_used[t, xi] + r_sold[t, xi] <= data.renewable[(t, xi)], name=f"renewable_limit[{t},{xi}]")
            m.addConstr(
                g_da[t] + g_rt[t, xi] + r_used[t, xi] + discharge[t, xi] - charge[t, xi] == energy[t, xi],
                name=f"energy_balance[{t},{xi}]",
            )
            m.addConstr(
                soc[t + 1, xi] - soc[t, xi]
                == data.eta_charge * charge[t, xi] - discharge[t, xi] / data.eta_discharge,
                name=f"soc_transition[{t},{xi}]",
            )
            m.addConstr(data.eta_charge * charge[t, xi] <= data.ess_capacity - soc[t, xi], name=f"soc_charge_room[{t},{xi}]")
            m.addConstr(discharge[t, xi] / data.eta_discharge <= soc[t, xi], name=f"soc_discharge_room[{t},{xi}]")
            m.addConstr(charge[t, xi] <= data.ess_charge_max, name=f"charge_limit[{t},{xi}]")
            m.addConstr(discharge[t, xi] <= data.ess_discharge_max, name=f"discharge_limit[{t},{xi}]")

    expected_spot_revenue = gp.quicksum(
        data.p[xi] * data.pi_spot[j] * y[j, s, t, xi]
        for j in J
        for s in S
        for t in data.active_spot[j]
        for xi in Xi
    )
    da_cost = gp.quicksum(data.p_da[t] * g_da[t] for t in T)
    expected_rt_cost = gp.quicksum(data.p[xi] * data.p_rt[(t, xi)] * g_rt[t, xi] for t in T for xi in Xi)
    expected_sale = gp.quicksum(data.p[xi] * data.p_sell[t] * r_sold[t, xi] for t in T for xi in Xi)
    expected_excess_penalty = gp.quicksum(
        data.p[xi] * data.lambda_exc * excess[s, t, xi] for s in S for t in T for xi in Xi
    )
    m.setObjective(
        expected_spot_revenue - da_cost - expected_rt_cost + expected_sale - expected_excess_penalty,
        GRB.MAXIMIZE,
    )

    v = {
        "x_init": x_init,
        "y_init": y_init,
        "z": z,
        "z_on": z_on,
        "w": w,
        "u": u,
        "u_on": u_on,
        "u_off": u_off,
        "g_da": g_da,
        "x": x,
        "mig": mig,
        "y": y,
        "h": h,
        "b": b,
        "load_od": load_od,
        "load_total": load_total,
        "excess": excess,
        "var_eta": var_eta,
        "cvar_zeta": cvar_zeta,
        "energy": energy,
        "g_rt": g_rt,
        "r_used": r_used,
        "r_sold": r_sold,
        "charge": charge,
        "discharge": discharge,
        "soc": soc,
    }
    m.update()
    return ModelArtifacts(model=m, data=data, v=v)


def configure_solver(model: Any, solver_cfg: dict[str, Any], run_dir: str | Path) -> None:
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.Params.OutputFlag = 1
    model.Params.LogToConsole = 1
    model.Params.LogFile = str(out / "solver.log")
    if solver_cfg.get("time_limit_seconds") is not None:
        model.Params.TimeLimit = float(solver_cfg["time_limit_seconds"])
    model.Params.MIPGap = float(solver_cfg["mip_gap"])
    model.Params.Threads = int(solver_cfg.get("threads", 0))
    model.Params.Seed = int(solver_cfg.get("seed", 42))
    model.Params.NumericFocus = int(solver_cfg.get("numeric_focus", 1))
    model.Params.Presolve = int(solver_cfg.get("presolve", 2))
    model.Params.MIPFocus = int(solver_cfg.get("mip_focus", 1))
    model.Params.DisplayInterval = int(solver_cfg.get("display_interval_seconds", 1))
    if solver_cfg.get("soft_mem_limit_gb") is not None:
        model.Params.SoftMemLimit = float(solver_cfg["soft_mem_limit_gb"])
    if solver_cfg.get("nodefile_start_gb") is not None:
        model.Params.NodefileStart = float(solver_cfg["nodefile_start_gb"])
    node_dir_value = solver_cfg.get("nodefile_dir")
    node_dir = Path(node_dir_value) if node_dir_value else Path(run_dir) / "nodefiles"
    if node_dir_value:
        if not node_dir.is_absolute():
            node_dir = Path.cwd() / node_dir
    node_dir.mkdir(parents=True, exist_ok=True)
    model.Params.NodefileDir = str(node_dir)
