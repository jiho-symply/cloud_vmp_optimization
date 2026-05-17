DEFAULT_FLAGS = {
    "activation_upper": False,
    "spot_completion_server": False,
    "spot_delta_server": False,
    "spot_time_lower": False,
    "delta_gamma_link": False,
    "delta_mass_gamma": False,
    "spot_gamma_aggregate": False,
    "spot_bridge_lower": False,
    "spot_bridge_aggregate": False,
    "gamma_probability_cap": False,
    "state_link": False,
    "phi_scenario_mass": False,
    "eta_aggregate_load": False,
    "eta_cover_general": False,
    "eta_cover_fixed": False,
    "pairwise_cover": False,
    "triple_cover": False,
    "minimal_cover_general": False,
    "minimal_cover_fixed": False,
    "aggregate_fixed_load": False,
    "uptime_symmetry": False,
}

BARRIER_NO_CROSSOVER = {
    "Method": 2,
    "NodeMethod": 2,
    "Crossover": 0,
}


def profile(*, solver_params=None, **flags):
    values = dict(DEFAULT_FLAGS)
    values.update(flags)
    values["solver_params"] = dict(solver_params or {})
    return values


CUT_PROFILES = {
    "baseline": profile(),
    "activation": profile(activation_upper=True),
    "spot_completion_server_only": profile(spot_completion_server=True),
    "spot_delta_server_only": profile(spot_delta_server=True),
    "spot_time_lower_only": profile(spot_time_lower=True),
    "delta_gamma_link_only": profile(delta_gamma_link=True),
    "delta_mass_gamma": profile(delta_mass_gamma=True),
    "spot_gamma_aggregate": profile(spot_gamma_aggregate=True),
    "spot_bridge_lower": profile(spot_bridge_lower=True),
    "spot_bridge_aggregate": profile(spot_bridge_aggregate=True),
    "gamma_probability_cap": profile(gamma_probability_cap=True),
    "state_link": profile(state_link=True),
    "state_link_phi_mass": profile(
        state_link=True,
        phi_scenario_mass=True,
    ),
    "state_phi_eta_cover_general": profile(
        state_link=True,
        phi_scenario_mass=True,
        eta_cover_general=True,
    ),
    "state_phi_eta_cover_fixed": profile(
        state_link=True,
        phi_scenario_mass=True,
        eta_cover_fixed=True,
    ),
    "state_phi_aggregate_fixed_load": profile(
        state_link=True,
        phi_scenario_mass=True,
        aggregate_fixed_load=True,
    ),
    "state_phi_delta_mass": profile(
        state_link=True,
        phi_scenario_mass=True,
        delta_mass_gamma=True,
    ),
    "phi_scenario_mass": profile(phi_scenario_mass=True),
    "eta_aggregate_load": profile(eta_aggregate_load=True),
    "eta_cover_general": profile(eta_cover_general=True),
    "eta_cover_fixed": profile(eta_cover_fixed=True),
    "minimal_cover_general": profile(minimal_cover_general=True),
    "minimal_cover_fixed": profile(minimal_cover_fixed=True),
    "aggregate_fixed_load": profile(aggregate_fixed_load=True),
    "pairwise_cover": profile(pairwise_cover=True),
    "triple_cover": profile(pairwise_cover=True, triple_cover=True),
    "uptime_symmetry": profile(uptime_symmetry=True),
    "spot_server_link": profile(
        spot_completion_server=True,
        spot_delta_server=True,
    ),
    "spot_time_link": profile(
        spot_time_lower=True,
        delta_gamma_link=True,
    ),
    "strategy_mipfocus_feasible": profile(solver_params={"MIPFocus": 1}),
    "strategy_mipfocus_balance": profile(solver_params={"MIPFocus": 2}),
    "strategy_mipfocus_bound": profile(solver_params={"MIPFocus": 3}),
    "strategy_method_dual": profile(solver_params={"Method": 1}),
    "strategy_method_barrier": profile(solver_params={"Method": 2}),
    "strategy_barrier_nocrossover": profile(solver_params=BARRIER_NO_CROSSOVER),
    "strategy_heuristics_high": profile(solver_params={"Heuristics": 0.5}),
    "strategy_integrality_focus": profile(solver_params={"IntegralityFocus": 1}),
    "strategy_symmetry_aggressive": profile(solver_params={"Symmetry": 2}),
    "strategy_rins": profile(solver_params={"RINS": 10}),
    "strategy_presolve_aggressive": profile(
        solver_params={
            "Presolve": 2,
            "Aggregate": 2,
        }
    ),
    "combo_barrier_uptime": profile(
        solver_params=BARRIER_NO_CROSSOVER,
        uptime_symmetry=True,
    ),
    "combo_barrier_delta_gamma": profile(
        solver_params=BARRIER_NO_CROSSOVER,
        delta_gamma_link=True,
    ),
    "combo_barrier_state_link": profile(
        solver_params=BARRIER_NO_CROSSOVER,
        state_link=True,
    ),
    "combo_barrier_state_phi_mass": profile(
        solver_params=BARRIER_NO_CROSSOVER,
        state_link=True,
        phi_scenario_mass=True,
    ),
    "combo_barrier_state_uptime": profile(
        solver_params=BARRIER_NO_CROSSOVER,
        state_link=True,
        uptime_symmetry=True,
    ),
    "combo_barrier_state_uptime_phi_mass": profile(
        solver_params=BARRIER_NO_CROSSOVER,
        state_link=True,
        uptime_symmetry=True,
        phi_scenario_mass=True,
    ),
    "combo_barrier_state_uptime_delta_mass": profile(
        solver_params=BARRIER_NO_CROSSOVER,
        state_link=True,
        uptime_symmetry=True,
        delta_mass_gamma=True,
    ),
    "combo_barrier_state_uptime_eta_cover_fixed": profile(
        solver_params=BARRIER_NO_CROSSOVER,
        state_link=True,
        uptime_symmetry=True,
        eta_cover_fixed=True,
    ),
    "combo_barrier_eta_aggregate": profile(
        solver_params=BARRIER_NO_CROSSOVER,
        eta_aggregate_load=True,
    ),
    "combo_barrier_spot_bridge_lower": profile(
        solver_params=BARRIER_NO_CROSSOVER,
        spot_bridge_lower=True,
    ),
    "combo_barrier_eta_cover_general": profile(
        solver_params=BARRIER_NO_CROSSOVER,
        eta_cover_general=True,
    ),
    "solver_cover_focus": profile(
        solver_params={
            "Cuts": 1,
            "CoverCuts": 2,
            "FlowCoverCuts": 2,
            "GUBCoverCuts": 2,
        }
    ),
    "solver_clique_focus": profile(
        solver_params={
            "Cuts": 1,
            "CliqueCuts": 2,
            "ZeroHalfCuts": 2,
        }
    ),
    "solver_implied_focus": profile(
        solver_params={
            "Cuts": 1,
            "ImpliedCuts": 2,
            "ProjImpliedCuts": 2,
            "DualImpliedCuts": 2,
        }
    ),
    "solver_lift_focus": profile(
        solver_params={
            "Cuts": 1,
            "MIRCuts": 2,
            "StrongCGCuts": 2,
            "RelaxLiftCuts": 2,
            "LiftProjectCuts": 2,
        }
    ),
    "builtin_aggressive": profile(
        solver_params={
            "Cuts": 2,
            "CliqueCuts": 2,
            "CoverCuts": 2,
            "FlowCoverCuts": 2,
            "GUBCoverCuts": 2,
            "ImpliedCuts": 2,
            "MIRCuts": 2,
            "StrongCGCuts": 2,
            "ZeroHalfCuts": 2,
            "InfProofCuts": 2,
            "ProjImpliedCuts": 2,
            "DualImpliedCuts": 2,
            "RelaxLiftCuts": 2,
            "LiftProjectCuts": 2,
        }
    ),
    "combined_light": profile(
        activation_upper=True,
        spot_completion_server=True,
        spot_delta_server=True,
        spot_time_lower=True,
        delta_gamma_link=True,
        pairwise_cover=True,
        uptime_symmetry=True,
    ),
    "combined_full": profile(
        activation_upper=True,
        spot_completion_server=True,
        spot_delta_server=True,
        spot_time_lower=True,
        delta_gamma_link=True,
        pairwise_cover=True,
        triple_cover=True,
        uptime_symmetry=True,
        solver_params={
            "Cuts": 2,
            "CliqueCuts": 2,
            "CoverCuts": 2,
            "FlowCoverCuts": 2,
            "GUBCoverCuts": 2,
            "ImpliedCuts": 2,
            "MIRCuts": 2,
            "StrongCGCuts": 2,
            "ZeroHalfCuts": 2,
            "InfProofCuts": 2,
            "ProjImpliedCuts": 2,
            "DualImpliedCuts": 2,
            "RelaxLiftCuts": 2,
            "LiftProjectCuts": 2,
        },
    ),
}
