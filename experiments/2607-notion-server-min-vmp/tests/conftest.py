from __future__ import annotations

import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

import pytest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_SRC = EXPERIMENT_ROOT / "src"
if str(EXPERIMENT_SRC) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_SRC))

from notion_server_min_vmp.data import CPU, MEM, InstanceData  # noqa: E402


def make_instance(
    *,
    on_demand_ids: Iterable[str] = ("od0",),
    spot_ids: Iterable[str] = ("spot0",),
    batch_ids: Iterable[str] = ("batch0",),
    servers: Iterable[str] = ("server0", "server1"),
    periods: Iterable[int] = (0, 1, 2, 3),
    scenarios: Iterable[int] = (0, 1),
    active_od: Mapping[str, Iterable[int]] | None = None,
    active_spot: Mapping[str, Iterable[int]] | None = None,
    od_cpu_by_scenario: Mapping[int, float] | None = None,
    C_cpu: float = 2.5,
    C_mem: float = 2.5,
    min_on: int = 2,
    min_off: int = 2,
    alpha: float = 0.5,
    epsilon: float = 0.25,
    batch_base_cpu: float = 0.0,
    batch_base_mem: float = 0.05,
    batch_startup_cpu: float = 0.05,
    batch_startup_mem: float = 0.0,
) -> InstanceData:
    """Return a deliberately small, fully explicit model instance.

    Formulation tests use smaller contiguous index sets so semantic solves stay
    well below one second while still passing the model-facing validator.
    """

    I = list(on_demand_ids)
    J = list(spot_ids)
    K = list(batch_ids)
    S = list(servers)
    T = list(periods)
    Xi = list(scenarios)
    if not T or not Xi or not S:
        raise ValueError("tiny instances require nonempty S, T, and Xi")

    probability = 1.0 / len(Xi)
    p = {xi: probability for xi in Xi}
    default_active = T[: min(2, len(T))]
    active_od_map = {
        vm_id: list((active_od or {}).get(vm_id, default_active)) for vm_id in I
    }
    active_spot_map = {
        vm_id: list((active_spot or {}).get(vm_id, default_active)) for vm_id in J
    }

    q_od = {(vm_id, CPU): 1.0 for vm_id in I}
    q_od.update({(vm_id, MEM): 1.0 for vm_id in I})
    q_spot = {(vm_id, CPU): 0.5 for vm_id in J}
    q_spot.update({(vm_id, MEM): 0.5 for vm_id in J})

    load_od: dict[tuple[str, str, int, int], float] = {}
    for vm_id in I:
        for t in active_od_map[vm_id]:
            for xi in Xi:
                cpu = (
                    float(od_cpu_by_scenario[xi])
                    if od_cpu_by_scenario is not None
                    else 0.4
                )
                load_od[vm_id, CPU, t, xi] = cpu
                load_od[vm_id, MEM, t, xi] = 0.1

    load_spot: dict[tuple[str, str, int, int], float] = {}
    for vm_id in J:
        for t in active_spot_map[vm_id]:
            for xi in Xi:
                load_spot[vm_id, CPU, t, xi] = 0.1
                load_spot[vm_id, MEM, t, xi] = 0.1

    q_batch = {(family, CPU): 0.5 for family in K}
    q_batch.update({(family, MEM): 0.5 for family in K})
    rho_batch = {(family, CPU): 0.1 for family in K}
    rho_batch.update({(family, MEM): 0.1 for family in K})
    W = {family: 0.5 for family in K}

    return InstanceData(
        I=I,
        J=J,
        K=K,
        S=S,
        T=T,
        Xi=Xi,
        p=p,
        active_od=active_od_map,
        active_spot=active_spot_map,
        q_od=q_od,
        q_spot=q_spot,
        load_od=load_od,
        load_spot=load_spot,
        q_batch=q_batch,
        rho_batch=rho_batch,
        W=W,
        batch_base_cpu=batch_base_cpu,
        batch_base_mem=batch_base_mem,
        batch_startup_cpu=batch_startup_cpu,
        batch_startup_mem=batch_startup_mem,
        C_cpu=C_cpu,
        C_mem=C_mem,
        E_idle=0.1,
        E_cpu=0.2,
        min_on=min_on,
        min_off=min_off,
        p_rt={(t, xi): 0.02 + 0.01 * xi for t in T for xi in Xi},
        pi_od={vm_id: 2.0 for vm_id in I},
        pi_spot={vm_id: 1.5 for vm_id in J},
        pi_batch={family: 4.0 for family in K},
        c_mig=0.05,
        alpha=alpha,
        epsilon=epsilon,
        source_files=[],
        metadata={
            "economics": {
                "migration_coefficient": 0.05,
                "migration_coefficient_unit": "kWh per normalized memory unit migrated",
            },
            "fidelity_warnings": {},
        },
    )


def build_tiny_model(data: InstanceData):
    pytest.importorskip("gurobipy")
    import notion_server_min_vmp.model as model_module

    artifacts = model_module.build_model(data, name="server_min_unit_test")

    model = artifacts.model
    model.Params.OutputFlag = 0
    model.Params.LogToConsole = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = 5
    model.Params.Seed = 1
    return artifacts


def optimize_tiny(artifacts):
    model = artifacts.model
    model.optimize()
    return artifacts


@pytest.fixture(scope="session")
def solved_full_artifacts():
    artifacts = build_tiny_model(make_instance())
    # Exercise the accepted-spot arrival contract rather than allowing the
    # optimizer to make admission vacuous.
    artifacts.variables["y_init"]["spot0", "server0"].LB = 1.0
    artifacts.variables["y_init"]["spot0", "server0"].UB = 1.0
    artifacts.variables["y_init"]["spot0", "server1"].UB = 0.0
    return optimize_tiny(artifacts)


@pytest.fixture(scope="session")
def positive_part_artifacts():
    data = make_instance(
        spot_ids=(),
        batch_ids=(),
        servers=("server0",),
        periods=(0,),
        scenarios=(0, 1, 2),
        active_od={"od0": (0,)},
        # Keep every synthetic realization within q_cpu=1.0 while still
        # exercising demand below, above, and exactly at server capacity.
        od_cpu_by_scenario={0: 0.6, 1: 1.0, 2: 0.8},
        C_cpu=0.8,
        C_mem=1.0,
        epsilon=0.2,
    )
    return optimize_tiny(build_tiny_model(data))
