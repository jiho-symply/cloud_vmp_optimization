from __future__ import annotations

import pytest

from notion_server_min_vmp.data import CPU, MEM, validate_instance

from conftest import build_tiny_model, make_instance, optimize_tiny


def test_variable_indices_indicator_count_and_removed_energy_system(
    solved_full_artifacts,
) -> None:
    gp = pytest.importorskip("gurobipy")
    GRB = gp.GRB
    artifacts = solved_full_artifacts
    data, model, variables = artifacts.data, artifacts.model, artifacts.variables
    server_scenario_count = len(data.S) * len(data.T) * len(data.Xi)

    assert len(variables["excess_branch"]) == server_scenario_count
    assert len(variables["eta"]) == len(data.S) * len(data.T)
    assert len(variables["zeta"]) == server_scenario_count
    assert model.NumGenConstrs == 4 * server_scenario_count
    assert all(
        constraint.GenConstrType == GRB.GENCONSTR_INDICATOR
        for constraint in model.getGenConstrs()
    )
    names = {constraint.GenConstrName for constraint in model.getGenConstrs()}
    assert {
        "excess_zero_branch_demand[server0,0,0]",
        "excess_zero_branch_value[server0,0,0]",
        "excess_positive_branch_demand[server0,0,0]",
        "excess_positive_branch_value[server0,0,0]",
    }.issubset(names)

    removed = {
        "g_da",
        "g_rt",
        "grid_da",
        "grid_rt",
        "renewable_used",
        "renewable_sold",
        "charge",
        "discharge",
        "soc",
    }
    assert removed.isdisjoint(variables)


def test_revised_recourse_domains_and_removed_spot_admission_variable(
    solved_full_artifacts,
) -> None:
    gp = pytest.importorskip("gurobipy")
    GRB = gp.GRB
    variables = solved_full_artifacts.variables

    assert "w" not in variables
    assert variables["y"]
    assert variables["migration"]
    assert all(
        variable.VType == GRB.CONTINUOUS
        and variable.LB == pytest.approx(0.0)
        and variable.UB == pytest.approx(1.0)
        for variable in variables["y"].values()
    )
    assert all(
        variable.VType == GRB.CONTINUOUS
        and variable.LB == pytest.approx(0.0)
        and variable.UB == pytest.approx(1.0)
        for variable in variables["migration"].values()
    )
    assert set(variables["migration"]) == {
        (i, server, t, xi)
        for i in solved_full_artifacts.data.I
        for server in solved_full_artifacts.data.S
        for t in solved_full_artifacts.data.active_od[i][:-1]
        for xi in solved_full_artifacts.data.Xi
    }
    assert all(variable.VType == GRB.BINARY for variable in variables["h"].values())
    assert all(
        variable.VType == GRB.BINARY
        for variable in variables["excess_branch"].values()
    )


def test_destination_indexed_migration_exact_linearization_coefficients(
    solved_full_artifacts,
) -> None:
    artifacts = solved_full_artifacts
    model, variables = artifacts.model, artifacts.variables
    i, server, t, xi = "od0", "server1", 0, 0
    migration = variables["migration"][i, server, t, xi]
    previous = variables["x"][i, server, t, xi]
    following = variables["x"][i, server, t + 1, xi]

    lower = model.getConstrByName(
        f"migration_entry_lower[{i},{server},{t},{xi}]"
    )
    next_upper = model.getConstrByName(
        f"migration_entry_next_upper[{i},{server},{t},{xi}]"
    )
    previous_upper = model.getConstrByName(
        f"migration_entry_previous_upper[{i},{server},{t},{xi}]"
    )

    assert lower.Sense == ">"
    assert lower.RHS == pytest.approx(0.0)
    assert model.getCoeff(lower, migration) == pytest.approx(1.0)
    assert model.getCoeff(lower, following) == pytest.approx(-1.0)
    assert model.getCoeff(lower, previous) == pytest.approx(1.0)

    assert next_upper.Sense == "<"
    assert next_upper.RHS == pytest.approx(0.0)
    assert model.getCoeff(next_upper, migration) == pytest.approx(1.0)
    assert model.getCoeff(next_upper, following) == pytest.approx(-1.0)

    assert previous_upper.Sense == "<"
    assert previous_upper.RHS == pytest.approx(1.0)
    assert model.getCoeff(previous_upper, migration) == pytest.approx(1.0)
    assert model.getCoeff(previous_upper, previous) == pytest.approx(1.0)
    assert model.getConstrByName(
        f"migration_neg[{i},{server},{t},{xi}]"
    ) is None


def test_destination_indexed_migration_is_exact_for_move_stay_and_zero_cost() -> None:
    gp = pytest.importorskip("gurobipy")
    moving = build_tiny_model(
        make_instance(
            spot_ids=(),
            batch_ids=(),
            periods=(0, 1),
            scenarios=(0,),
        )
    )
    x = moving.variables["x"]
    x["od0", "server0", 0, 0].LB = x["od0", "server0", 0, 0].UB = 1.0
    x["od0", "server1", 1, 0].LB = x["od0", "server1", 1, 0].UB = 1.0
    optimize_tiny(moving)

    assert moving.model.Status == gp.GRB.OPTIMAL
    assert moving.variables["migration"]["od0", "server0", 0, 0].X == pytest.approx(0.0)
    assert moving.variables["migration"]["od0", "server1", 0, 0].X == pytest.approx(1.0)
    assert sum(
        moving.variables["migration"]["od0", server, 0, 0].X
        for server in moving.data.S
    ) == pytest.approx(1.0)

    zero_cost_data = make_instance(
        spot_ids=(),
        batch_ids=(),
        periods=(0, 1),
        scenarios=(0,),
    )
    zero_cost_data.c_mig = 0.0
    staying = build_tiny_model(zero_cost_data)
    x = staying.variables["x"]
    for t in staying.data.T:
        x["od0", "server0", t, 0].LB = x["od0", "server0", t, 0].UB = 1.0
    optimize_tiny(staying)

    assert staying.model.Status == gp.GRB.OPTIMAL
    assert all(
        staying.variables["migration"]["od0", server, 0, 0].X
        == pytest.approx(0.0)
        for server in staying.data.S
    )


def test_spot_admission_and_state_use_initial_assignment_sum(
    solved_full_artifacts,
) -> None:
    artifacts = solved_full_artifacts
    data, model, variables = artifacts.data, artifacts.model, artifacts.variables

    admission = model.getConstrByName("spot_init_at_most_one[spot0]")
    assert admission is not None
    assert admission.Sense == "<"
    assert admission.RHS == pytest.approx(1.0)
    assert {
        variable.VarName: model.getCoeff(admission, variable)
        for variable in variables["y_init"].values()
        if model.getCoeff(admission, variable)
    } == {
        "y_init[spot0,server0]": pytest.approx(1.0),
        "y_init[spot0,server1]": pytest.approx(1.0),
    }

    state = model.getConstrByName("spot_state[spot0,0,0]")
    assert state is not None
    assert state.Sense == "="
    assert state.RHS == pytest.approx(0.0)
    for server in data.S:
        assert model.getCoeff(state, variables["y"]["spot0", server, 0, 0]) == pytest.approx(1.0)
        assert model.getCoeff(state, variables["y_init"]["spot0", server]) == pytest.approx(-1.0)
    assert model.getCoeff(state, variables["h"]["spot0", 0, 0]) == pytest.approx(1.0)


def test_excess_branch_is_disabled_when_server_is_off(
    solved_full_artifacts,
) -> None:
    artifacts = solved_full_artifacts
    data, model, variables = artifacts.data, artifacts.model, artifacts.variables
    constraints = [
        constraint
        for constraint in model.getConstrs()
        if constraint.ConstrName.startswith("excess_branch_server_on[")
    ]
    assert len(constraints) == len(data.S) * len(data.T) * len(data.Xi)

    constraint = model.getConstrByName("excess_branch_server_on[server0,0,0]")
    assert constraint.Sense == "<"
    assert constraint.RHS == pytest.approx(0.0)
    assert model.getCoeff(
        constraint, variables["excess_branch"]["server0", 0, 0]
    ) == pytest.approx(1.0)
    assert model.getCoeff(constraint, variables["u"]["server0", 0]) == pytest.approx(-1.0)


def test_initial_assignment_orbitope_counts_and_coefficient_support() -> None:
    data = make_instance(
        on_demand_ids=("od0", "od1"),
        spot_ids=("spot0", "spot1"),
        batch_ids=(),
        servers=("server0", "server1", "server2"),
        periods=(0,),
        scenarios=(0,),
    )
    artifacts = build_tiny_model(data)
    model, variables = artifacts.model, artifacts.variables

    assert variables["symmetry_audit"] == {
        "rows": 4,
        "servers": 3,
        "triangular_constraints": 3,
        "introduction_constraints": 5,
    }
    triangular = model.getConstrByName(
        "symmetry_assignment_triangular[od,od0,server2]"
    )
    assert triangular is not None
    assert triangular.Sense == "="
    assert triangular.RHS == pytest.approx(0.0)
    assert model.getCoeff(triangular, variables["x_init"]["od0", "server2"]) == pytest.approx(1.0)

    introduction = model.getConstrByName(
        "symmetry_assignment_introduction[row3,server2]"
    )
    expected_coefficients = {
        variables["y_init"]["spot0", "server1"].VarName: 1.0,
        variables["y_init"]["spot0", "server2"].VarName: 1.0,
        variables["x_init"]["od0", "server0"].VarName: -1.0,
        variables["x_init"]["od1", "server0"].VarName: -1.0,
    }
    observed_coefficients = {
        variable.VarName: model.getCoeff(introduction, variable)
        for variable in model.getVars()
        if abs(model.getCoeff(introduction, variable)) > 0.0
    }
    assert introduction.Sense == "<"
    assert introduction.RHS == pytest.approx(0.0)
    assert observed_coefficients == pytest.approx(expected_coefficients)


def test_initial_assignment_orbitope_empty_rows_and_semantic_feasibility() -> None:
    empty = build_tiny_model(
        make_instance(
            on_demand_ids=(),
            spot_ids=(),
            batch_ids=(),
            servers=("server0", "server1", "server2"),
            periods=(0,),
            scenarios=(0,),
        )
    )
    assert empty.variables["symmetry_audit"] == {
        "rows": 0,
        "servers": 3,
        "triangular_constraints": 0,
        "introduction_constraints": 0,
    }

    gp = pytest.importorskip("gurobipy")
    feasible = build_tiny_model(
        make_instance(
            on_demand_ids=("od0",),
            spot_ids=("spot0",),
            batch_ids=(),
            servers=("server0", "server1"),
            periods=(0,),
            scenarios=(0,),
        )
    )
    # The first mandatory OD row opens column 1, so the second row may
    # introduce column 2. This exercises the packing-orbitope implication
    # without relying only on coefficient inspection.
    feasible.variables["y_init"]["spot0", "server1"].LB = 1.0
    feasible.variables["y_init"]["spot0", "server1"].UB = 1.0
    optimize_tiny(feasible)
    assert feasible.model.Status == gp.GRB.OPTIMAL
    assert feasible.variables["x_init"]["od0", "server0"].X == pytest.approx(1.0)
    assert feasible.variables["y_init"]["spot0", "server1"].X == pytest.approx(1.0)


def test_cvar_is_indexed_by_server_time_and_normalized_by_cpu_capacity(
    solved_full_artifacts,
) -> None:
    artifacts = solved_full_artifacts
    data, model, variables = artifacts.data, artifacts.model, artifacts.variables

    assert set(variables["eta"].keys()) == {
        (server, t) for server in data.S for t in data.T
    }
    assert set(variables["zeta"].keys()) == {
        (server, t, xi)
        for server in data.S
        for t in data.T
        for xi in data.Xi
    }
    tail_constraints = [
        constraint
        for constraint in model.getConstrs()
        if constraint.ConstrName.startswith("cvar_tail[")
    ]
    bound_constraints = [
        constraint
        for constraint in model.getConstrs()
        if constraint.ConstrName.startswith("cvar_bound[")
    ]
    assert len(tail_constraints) == len(data.S) * len(data.T) * len(data.Xi)
    assert len(bound_constraints) == len(data.S) * len(data.T)

    tail = model.getConstrByName("cvar_tail[server0,0,0]")
    assert model.getCoeff(tail, variables["excess"]["server0", 0, 0]) == pytest.approx(
        1.0 / data.C_cpu
    )
    assert model.getCoeff(tail, variables["eta"]["server0", 0]) == pytest.approx(-1.0)
    assert model.getCoeff(tail, variables["zeta"]["server0", 0, 0]) == pytest.approx(-1.0)


def test_spot_is_not_preempted_at_arrival_when_accepted(
    solved_full_artifacts,
) -> None:
    gp = pytest.importorskip("gurobipy")
    artifacts = solved_full_artifacts
    model, variables = artifacts.model, artifacts.variables
    assert model.Status == gp.GRB.OPTIMAL

    for xi in artifacts.data.Xi:
        arrival = artifacts.data.active_spot["spot0"][0]
        assert variables["h"]["spot0", arrival, xi].X == pytest.approx(0.0)
        assert sum(
            variables["y"]["spot0", server, arrival, xi].X
            for server in artifacts.data.S
        ) == pytest.approx(1.0)


def test_batch_load_uses_only_startup_cpu_and_prepared_memory(
    solved_full_artifacts,
) -> None:
    artifacts = solved_full_artifacts
    data, model, variables = artifacts.data, artifacts.model, artifacts.variables
    family, server, t, xi = "batch0", "server0", 0, 0
    cpu_definition = model.getConstrByName(f"total_cpu_load[{server},{t},{xi}]")
    mem_definition = model.getConstrByName(f"total_mem_load[{server},{t},{xi}]")

    assert data.batch_base_cpu == 0.0
    assert data.batch_startup_mem == 0.0
    assert model.getCoeff(cpu_definition, variables["z"][family, server, t]) == 0.0
    assert model.getCoeff(
        cpu_definition, variables["z_on"][family, server, t]
    ) == pytest.approx(-data.batch_startup_cpu)
    assert model.getCoeff(
        mem_definition, variables["z"][family, server, t]
    ) == pytest.approx(-data.batch_base_mem)
    assert model.getCoeff(mem_definition, variables["z_on"][family, server, t]) == 0.0


def test_model_objective_excludes_constant_revenue_but_retains_accounting_values(
    solved_full_artifacts,
) -> None:
    artifacts = solved_full_artifacts
    data, model, variables = artifacts.data, artifacts.model, artifacts.variables
    expected_od_constant = sum(data.pi_od[i] * len(data.active_od[i]) for i in data.I)
    expected_batch_constant = sum(data.pi_batch[k] * data.W[k] for k in data.K)

    assert artifacts.objective_constants == {
        "on_demand_revenue": pytest.approx(expected_od_constant),
        "batch_revenue": pytest.approx(expected_batch_constant),
    }
    assert model.getObjective().getConstant() == pytest.approx(0.0)
    assert variables["y"]["spot0", "server0", 0, 0].Obj == pytest.approx(
        data.p[0] * data.pi_spot["spot0"]
    )
    assert variables["energy"][0, 1].Obj == pytest.approx(
        -data.p[1] * data.p_rt[0, 1]
    )


def test_migration_energy_uses_one_constant_coefficient(
    solved_full_artifacts,
) -> None:
    artifacts = solved_full_artifacts
    data, model, variables = artifacts.data, artifacts.model, artifacts.variables
    i, t, xi = "od0", 0, 0
    energy_definition = model.getConstrByName(f"energy_definition[{t},{xi}]")

    assert data.c_mig == pytest.approx(0.05)
    for server in data.S:
        assert model.getCoeff(
            energy_definition, variables["migration"][i, server, t, xi]
        ) == pytest.approx(-data.c_mig * data.load_od[i, MEM, t, xi])


def test_validate_instance_rejects_forbidden_cross_batch_overheads() -> None:
    base_cpu = make_instance(
        periods=range(48),
        scenarios=range(40),
        batch_base_cpu=0.01,
    )
    with pytest.raises(ValueError, match="base CPU|batch_base_cpu"):
        validate_instance(base_cpu)

    startup_mem = make_instance(
        periods=range(48),
        scenarios=range(40),
        batch_startup_mem=0.01,
    )
    with pytest.raises(ValueError, match="startup memory|batch_startup_mem"):
        validate_instance(startup_mem)


def test_validate_instance_rejects_negative_migration_coefficient() -> None:
    instance = make_instance()
    instance.c_mig = -0.01

    with pytest.raises(ValueError, match="c_mig"):
        validate_instance(instance)


def test_minimum_on_and_off_time_exclude_short_state_runs() -> None:
    gp = pytest.importorskip("gurobipy")
    GRB = gp.GRB
    data = make_instance(
        on_demand_ids=(),
        spot_ids=(),
        batch_ids=(),
        servers=("server0",),
        periods=(0, 1, 2, 3),
        scenarios=(0,),
        min_on=2,
        min_off=2,
    )
    artifacts = build_tiny_model(data)
    model, u = artifacts.model, artifacts.variables["u"]
    model.Params.DualReductions = 0

    cases = [
        ((0, 1, 0, 0), GRB.INFEASIBLE),
        ((0, 1, 1, 0), GRB.OPTIMAL),
        ((1, 0, 1, 1), GRB.INFEASIBLE),
        ((1, 0, 0, 1), GRB.OPTIMAL),
    ]
    for pattern, expected_status in cases:
        for t, state in enumerate(pattern):
            u["server0", t].LB = float(state)
            u["server0", t].UB = float(state)
        model.update()
        optimize_tiny(artifacts)
        assert model.Status == expected_status, pattern


def test_exact_positive_part_below_above_and_at_capacity(
    positive_part_artifacts,
) -> None:
    gp = pytest.importorskip("gurobipy")
    artifacts = positive_part_artifacts
    variables = artifacts.variables
    assert artifacts.model.Status == gp.GRB.OPTIMAL

    expected = {
        0: (0.6, 0.0, 0.6, 0.0),
        1: (1.0, 0.2, 0.8, 1.0),
        2: (0.8, 0.0, 0.8, None),
    }
    for xi, (_demand, excess, served, branch) in expected.items():
        assert variables["excess"]["server0", 0, xi].X == pytest.approx(excess)
        assert variables["od_served"]["server0", CPU, 0, xi].X == pytest.approx(served)
        if branch is not None:
            assert variables["excess_branch"]["server0", 0, xi].X == pytest.approx(branch)
