from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data import CPU, MEM, InstanceData, validate_instance


@dataclass(frozen=True)
class ModelArtifacts:
    """Gurobi model plus the indexed variables needed by reporting/tests."""

    model: Any
    data: InstanceData
    variables: dict[str, Any]
    objective_constants: dict[str, float]

    @property
    def v(self) -> dict[str, Any]:
        """Backward-compatible short alias used by earlier experiment tooling."""

        return self.variables


def _active_transitions(periods: list[int] | tuple[int, ...]) -> list[int]:
    transitions: list[int] = []
    for left, right in zip(periods[:-1], periods[1:]):
        if right != left + 1:
            raise ValueError(f"active periods must be consecutive; found {left} -> {right}")
        transitions.append(left)
    return transitions


def _add_initial_assignment_packing_orbitope(
    model: Any,
    *,
    I: list[str],
    J: list[str],
    S: list[str],
    x_init: Any,
    y_init: Any,
    gp: Any,
) -> dict[str, int]:
    """Break homogeneous-server column symmetry in the initial assignments.

    Rows contain mandatory on-demand assignments first and optional spot
    assignments second, exactly as specified by the attached packing-orbitope
    formulation.  The implementation translates the page's one-based
    ``(r, s)`` indices to Python's zero-based list positions without reordering
    the deterministic VM or server sets supplied by the data layer.
    """

    rows: list[tuple[str, str, Any]] = [
        ("od", str(vm_id), x_init) for vm_id in I
    ]
    rows.extend(("spot", str(vm_id), y_init) for vm_id in J)

    def assignment(row_index: int, server_index: int) -> Any:
        _kind, vm_id, variables = rows[row_index]
        return variables[vm_id, S[server_index]]

    triangular_count = 0
    introduction_count = 0
    row_count = len(rows)
    server_count = len(S)

    # a[r,s] = 0 for one-based s > r.
    for row_index, (kind, vm_id, _variables) in enumerate(rows):
        for server_index in range(row_index + 1, server_count):
            model.addConstr(
                assignment(row_index, server_index) == 0,
                name=(
                    "symmetry_assignment_triangular"
                    f"[{kind},{vm_id},{S[server_index]}]"
                ),
            )
            triangular_count += 1

    # sum_{ell=s}^{min(r,|S|)} a[r,ell]
    #   <= sum_{q=1}^{r-1} a[q,s-1]
    # for one-based r=2..R and s=2..min(r,|S|).
    for row_index in range(1, row_count):
        one_based_row = row_index + 1
        row_server_limit = min(one_based_row, server_count)
        for server_index in range(1, row_server_limit):
            model.addConstr(
                gp.quicksum(
                    assignment(row_index, column_index)
                    for column_index in range(server_index, row_server_limit)
                )
                <= gp.quicksum(
                    assignment(previous_row, server_index - 1)
                    for previous_row in range(row_index)
                ),
                name=(
                    "symmetry_assignment_introduction"
                    f"[row{one_based_row},server{server_index + 1}]"
                ),
            )
            introduction_count += 1

    return {
        "rows": row_count,
        "servers": server_count,
        "triangular_constraints": triangular_count,
        "introduction_constraints": introduction_count,
    }


def build_model(data: InstanceData, *, name: str = "notion_server_min_two_stage_vmp") -> ModelArtifacts:
    """Build the deterministic equivalent of the attached Notion formulation.

    Migration is the page's exact destination-entry formulation.  The
    destination-indexed variables remain continuous in ``[0, 1]`` as prescribed
    by the optimization section, while binary placement makes them exact at an
    integer incumbent.  Batch startup intentionally retains the page's
    lower-bound-only event definition.
    """

    validate_instance(data)
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:  # pragma: no cover - depends on licensed runtime
        raise RuntimeError("gurobipy is required to build and solve this experiment") from exc

    model = gp.Model(name)
    I, J, K = data.I, data.J, data.K
    S, T, Xi = data.S, data.T, data.Xi
    resources = (CPU, MEM)
    transition_periods = T[:-1]

    od_keys = [
        (i, s, t, xi)
        for i in I
        for s in S
        for t in data.active_od[i]
        for xi in Xi
    ]
    spot_keys = [
        (j, s, t, xi)
        for j in J
        for s in S
        for t in data.active_spot[j]
        for xi in Xi
    ]
    migration_keys = [
        (i, s, t, xi)
        for i in I
        for s in S
        for t in _active_transitions(data.active_od[i])
        for xi in Xi
    ]
    preemption_keys = [
        (j, t, xi)
        for j in J
        for t in data.active_spot[j]
        for xi in Xi
    ]
    batch_keys = [(k, s, t, xi) for k in K for s in S for t in T for xi in Xi]
    load_keys = [(s, resource, t, xi) for s in S for resource in resources for t in T for xi in Xi]
    server_scenario_keys = [(s, t, xi) for s in S for t in T for xi in Xi]
    energy_keys = [(t, xi) for t in T for xi in Xi]

    # First-stage decisions.
    x_init = model.addVars(I, S, vtype=GRB.BINARY, name="x_init")
    y_init = model.addVars(J, S, vtype=GRB.BINARY, name="y_init")
    z = model.addVars(K, S, T, vtype=GRB.BINARY, name="z_batch")
    z_on = model.addVars(K, S, T, vtype=GRB.BINARY, name="z_batch_on")
    u = model.addVars(S, T, vtype=GRB.BINARY, name="u_server")
    u_on = model.addVars(S, transition_periods, vtype=GRB.BINARY, name="u_server_on")
    u_off = model.addVars(S, transition_periods, vtype=GRB.BINARY, name="u_server_off")

    # Second-stage decisions.
    x = model.addVars(od_keys, vtype=GRB.BINARY, name="x_od")
    migration = model.addVars(
        migration_keys,
        lb=0.0,
        ub=1.0,
        vtype=GRB.CONTINUOUS,
        name="migration",
    )
    y = model.addVars(
        spot_keys,
        lb=0.0,
        ub=1.0,
        vtype=GRB.CONTINUOUS,
        name="y_spot",
    )
    h = model.addVars(preemption_keys, vtype=GRB.BINARY, name="spot_preempted")
    batch_work = model.addVars(batch_keys, lb=0.0, name="batch_work")
    od_served = model.addVars(load_keys, lb=0.0, name="od_served")
    total_load = model.addVars(load_keys, lb=0.0, name="total_served_load")
    excess = model.addVars(server_scenario_keys, lb=0.0, name="od_cpu_excess")
    excess_branch = model.addVars(server_scenario_keys, vtype=GRB.BINARY, name="excess_positive_branch")
    eta = model.addVars(S, T, lb=0.0, name="cvar_eta")
    zeta = model.addVars(server_scenario_keys, lb=0.0, name="cvar_zeta")
    energy = model.addVars(energy_keys, lb=0.0, name="energy")

    # Server provisioning and minimum on/off time. The page does not define an
    # initial state, so transitions are indexed only from t to t+1.
    for s in S:
        for t in transition_periods:
            model.addConstr(
                u[s, t + 1] - u[s, t] == u_on[s, t] - u_off[s, t],
                name=f"server_transition[{s},{t}]",
            )
            model.addConstr(
                u_on[s, t] + u_off[s, t] <= 1,
                name=f"server_switch_exclusive[{s},{t}]",
            )
            for tau in range(t + 1, min(T[-1], t + data.min_on) + 1):
                model.addConstr(u[s, tau] >= u_on[s, t], name=f"min_on[{s},{t},{tau}]")
            for tau in range(t + 1, min(T[-1], t + data.min_off) + 1):
                model.addConstr(u[s, tau] <= 1 - u_off[s, t], name=f"min_off[{s},{t},{tau}]")

    # On-demand initial placement, recourse placement, and migration.
    for i in I:
        periods = data.active_od[i]
        arrival = periods[0]
        model.addConstr(gp.quicksum(x_init[i, s] for s in S) == 1, name=f"od_init_one[{i}]")
        for xi in Xi:
            for s in S:
                model.addConstr(
                    x[i, s, arrival, xi] == x_init[i, s],
                    name=f"od_start_link[{i},{s},{xi}]",
                )
            for t in periods:
                model.addConstr(
                    gp.quicksum(x[i, s, t, xi] for s in S) == 1,
                    name=f"od_one_server[{i},{t},{xi}]",
                )
                for s in S:
                    model.addConstr(x[i, s, t, xi] <= u[s, t], name=f"od_server_on[{i},{s},{t},{xi}]")
            for t in _active_transitions(periods):
                for s in S:
                    model.addConstr(
                        migration[i, s, t, xi]
                        >= x[i, s, t + 1, xi] - x[i, s, t, xi],
                        name=f"migration_entry_lower[{i},{s},{t},{xi}]",
                    )
                    model.addConstr(
                        migration[i, s, t, xi] <= x[i, s, t + 1, xi],
                        name=f"migration_entry_next_upper[{i},{s},{t},{xi}]",
                    )
                    model.addConstr(
                        migration[i, s, t, xi] <= 1 - x[i, s, t, xi],
                        name=f"migration_entry_previous_upper[{i},{s},{t},{xi}]",
                    )

    # Spot admission, fixed server, mandatory execution at arrival, and
    # irreversible preemption.
    for j in J:
        periods = data.active_spot[j]
        admitted = gp.quicksum(y_init[j, s] for s in S)
        model.addConstr(admitted <= 1, name=f"spot_init_at_most_one[{j}]")
        for xi in Xi:
            model.addConstr(h[j, periods[0], xi] == 0, name=f"spot_not_preempted_at_arrival[{j},{xi}]")
            for position, t in enumerate(periods):
                if position:
                    model.addConstr(
                        h[j, periods[position - 1], xi] <= h[j, t, xi],
                        name=f"spot_preemption_irreversible[{j},{t},{xi}]",
                    )
                model.addConstr(
                    gp.quicksum(y[j, s, t, xi] for s in S) + h[j, t, xi]
                    == admitted,
                    name=f"spot_state[{j},{t},{xi}]",
                )
                for s in S:
                    model.addConstr(y[j, s, t, xi] <= y_init[j, s], name=f"spot_fixed_server[{j},{s},{t},{xi}]")
                    model.addConstr(y[j, s, t, xi] <= u[s, t], name=f"spot_server_on[{j},{s},{t},{xi}]")

    symmetry_audit = _add_initial_assignment_packing_orbitope(
        model,
        I=I,
        J=J,
        S=S,
        x_init=x_init,
        y_init=y_init,
        gp=gp,
    )

    # Batch preparation/startup and completion in every scenario.
    for k in K:
        for s in S:
            for t in T:
                model.addConstr(z[k, s, t] <= u[s, t], name=f"batch_server_on[{k},{s},{t}]")
                if t == T[0]:
                    model.addConstr(z_on[k, s, t] == z[k, s, t], name=f"batch_start_first[{k},{s}]")
                else:
                    model.addConstr(
                        z_on[k, s, t] >= z[k, s, t] - z[k, s, t - 1],
                        name=f"batch_start_lower_bound[{k},{s},{t}]",
                    )
        for xi in Xi:
            model.addConstr(
                gp.quicksum(batch_work[k, s, t, xi] for s in S for t in T) == data.W[k],
                name=f"batch_complete[{k},{xi}]",
            )
            for s in S:
                for t in T:
                    for resource in resources:
                        model.addConstr(
                            data.rho_batch[k, resource] * batch_work[k, s, t, xi]
                            <= data.q_batch[k, resource] * z[k, s, t],
                            name=f"batch_resource_limit[{k},{s},{resource},{t},{xi}]",
                        )

    # Exact positive-part OD CPU excess, served OD load, strict priority, and
    # hard CPU/MEM capacity.
    for s in S:
        for t in T:
            active_i = [i for i in I if t in data.active_od[i]]
            active_j = [j for j in J if t in data.active_spot[j]]
            for xi in Xi:
                od_cpu_demand = gp.quicksum(
                    data.load_od[i, CPU, t, xi] * x[i, s, t, xi] for i in active_i
                )
                capacity_if_on = data.C_cpu * u[s, t]
                branch = excess_branch[s, t, xi]

                model.addConstr(
                    branch <= u[s, t],
                    name=f"excess_branch_server_on[{s},{t},{xi}]",
                )

                model.addGenConstrIndicator(
                    branch,
                    0,
                    od_cpu_demand <= capacity_if_on,
                    name=f"excess_zero_branch_demand[{s},{t},{xi}]",
                )
                model.addGenConstrIndicator(
                    branch,
                    0,
                    excess[s, t, xi] == 0.0,
                    name=f"excess_zero_branch_value[{s},{t},{xi}]",
                )
                model.addGenConstrIndicator(
                    branch,
                    1,
                    od_cpu_demand >= capacity_if_on,
                    name=f"excess_positive_branch_demand[{s},{t},{xi}]",
                )
                model.addGenConstrIndicator(
                    branch,
                    1,
                    excess[s, t, xi] == od_cpu_demand - capacity_if_on,
                    name=f"excess_positive_branch_value[{s},{t},{xi}]",
                )
                model.addConstr(
                    od_served[s, CPU, t, xi] == od_cpu_demand - excess[s, t, xi],
                    name=f"od_cpu_served[{s},{t},{xi}]",
                )
                model.addConstr(
                    od_served[s, MEM, t, xi]
                    == gp.quicksum(data.load_od[i, MEM, t, xi] * x[i, s, t, xi] for i in active_i),
                    name=f"od_mem_exact[{s},{t},{xi}]",
                )

                spot_cpu = gp.quicksum(
                    data.load_spot[j, CPU, t, xi] * y[j, s, t, xi] for j in active_j
                )
                spot_mem = gp.quicksum(
                    data.load_spot[j, MEM, t, xi] * y[j, s, t, xi] for j in active_j
                )
                batch_cpu = gp.quicksum(
                    data.batch_startup_cpu * z_on[k, s, t]
                    + data.rho_batch[k, CPU] * batch_work[k, s, t, xi]
                    for k in K
                )
                batch_mem = gp.quicksum(
                    data.batch_base_mem * z[k, s, t]
                    + data.rho_batch[k, MEM] * batch_work[k, s, t, xi]
                    for k in K
                )
                model.addConstr(
                    total_load[s, CPU, t, xi]
                    == od_served[s, CPU, t, xi] + spot_cpu + batch_cpu,
                    name=f"total_cpu_load[{s},{t},{xi}]",
                )
                model.addConstr(
                    total_load[s, MEM, t, xi]
                    == od_served[s, MEM, t, xi] + spot_mem + batch_mem,
                    name=f"total_mem_load[{s},{t},{xi}]",
                )
                model.addConstr(
                    total_load[s, CPU, t, xi] <= data.C_cpu * u[s, t],
                    name=f"cpu_capacity[{s},{t},{xi}]",
                )
                model.addConstr(
                    total_load[s, MEM, t, xi] <= data.C_mem * u[s, t],
                    name=f"mem_capacity[{s},{t},{xi}]",
                )

    # Separate normalized CVaR bound for every server-time pair.
    for s in S:
        for t in T:
            for xi in Xi:
                model.addConstr(
                    excess[s, t, xi] / data.C_cpu <= eta[s, t] + zeta[s, t, xi],
                    name=f"cvar_tail[{s},{t},{xi}]",
                )
            model.addConstr(
                eta[s, t]
                + gp.quicksum(data.p[xi] * zeta[s, t, xi] for xi in Xi) / (1.0 - data.alpha)
                <= data.epsilon,
                name=f"cvar_bound[{s},{t}]",
            )

    # All electricity is purchased in the real-time market. E_idle and E_cpu
    # are already expressed per 30-minute bucket by the data layer.
    for t in T:
        for xi in Xi:
            migration_energy = gp.quicksum(
                data.c_mig
                * data.load_od[i, MEM, t, xi]
                * migration[i, s, t, xi]
                for i in I
                for s in S
                if (i, s, t, xi) in migration
            )
            model.addConstr(
                energy[t, xi]
                == gp.quicksum(
                    data.E_idle * u[s, t] + data.E_cpu * total_load[s, CPU, t, xi]
                    for s in S
                )
                + migration_energy,
                name=f"energy_definition[{t},{xi}]",
            )

    constant_od_revenue = float(sum(data.pi_od[i] * len(data.active_od[i]) for i in I))
    constant_batch_revenue = float(sum(data.pi_batch[k] * data.W[k] for k in K))
    expected_spot_revenue = gp.quicksum(
        data.p[xi] * data.pi_spot[j] * y[j, s, t, xi]
        for j in J
        for s in S
        for t in data.active_spot[j]
        for xi in Xi
    )
    expected_rt_energy_cost = gp.quicksum(
        data.p[xi] * data.p_rt[t, xi] * energy[t, xi] for t in T for xi in Xi
    )
    # On-demand and batch revenue are decision-independent constants.  The
    # revised page excludes them from the solver objective and adds them back
    # only in post-solve total-profit accounting.
    model.setObjective(expected_spot_revenue - expected_rt_energy_cost, GRB.MAXIMIZE)

    variables = {
        "x_init": x_init,
        "y_init": y_init,
        "z": z,
        "z_on": z_on,
        "u": u,
        "u_on": u_on,
        "u_off": u_off,
        "x": x,
        "migration": migration,
        "mig": migration,
        "y": y,
        "h": h,
        "batch_work": batch_work,
        "b": batch_work,
        "od_served": od_served,
        "total_load": total_load,
        "excess": excess,
        "excess_branch": excess_branch,
        "eta": eta,
        "zeta": zeta,
        "energy": energy,
        "symmetry_audit": symmetry_audit,
    }
    model.update()
    return ModelArtifacts(
        model=model,
        data=data,
        variables=variables,
        objective_constants={
            "on_demand_revenue": constant_od_revenue,
            "batch_revenue": constant_batch_revenue,
        },
    )


def configure_solver(model: Any, solver_config: dict[str, Any], run_dir: str | Path) -> None:
    """Apply reproducible baseline solver settings without imposing a time limit."""

    output = Path(run_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.Params.OutputFlag = int(solver_config.get("output_flag", 1))
    model.Params.LogToConsole = int(solver_config.get("log_to_console", 1))
    model.Params.LogFile = str(output / "solver.log")
    model.Params.MIPGap = float(solver_config.get("mip_gap", 0.001))
    model.Params.Threads = int(solver_config.get("threads", 8))
    model.Params.Seed = int(solver_config.get("seed", 42))
    model.Params.NumericFocus = int(solver_config.get("numeric_focus", 1))
    model.Params.Presolve = int(solver_config.get("presolve", 2))
    if solver_config.get("mip_focus") is not None:
        model.Params.MIPFocus = int(solver_config["mip_focus"])
    if solver_config.get("display_interval_seconds") is not None:
        model.Params.DisplayInterval = int(solver_config["display_interval_seconds"])
    if solver_config.get("no_rel_heur_time_seconds") is not None:
        model.Params.NoRelHeurTime = float(
            solver_config["no_rel_heur_time_seconds"]
        )
    if solver_config.get("time_limit_seconds") is not None:
        model.Params.TimeLimit = float(solver_config["time_limit_seconds"])
    if solver_config.get("soft_mem_limit_gb") is not None:
        model.Params.SoftMemLimit = float(solver_config["soft_mem_limit_gb"])
    if solver_config.get("nodefile_start_gb") is not None:
        model.Params.NodefileStart = float(solver_config["nodefile_start_gb"])
    if solver_config.get("nodefile_dir"):
        nodefile_dir = Path(solver_config["nodefile_dir"])
        if not nodefile_dir.is_absolute():
            nodefile_dir = output / nodefile_dir
        nodefile_dir.mkdir(parents=True, exist_ok=True)
        model.Params.NodefileDir = str(nodefile_dir.resolve())
