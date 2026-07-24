from __future__ import annotations

from types import SimpleNamespace

from notion_energy_vmp.model import configure_solver


def test_none_time_limit_leaves_gurobi_default_unchanged(tmp_path) -> None:
    params = SimpleNamespace(TimeLimit="UNSET")
    model = SimpleNamespace(Params=params)
    solver_cfg = {
        "time_limit_seconds": None,
        "mip_gap": 0.001,
        "threads": 8,
        "seed": 42,
        "numeric_focus": 1,
        "presolve": 2,
        "mip_focus": 1,
        "display_interval_seconds": 1,
        "soft_mem_limit_gb": 32,
        "nodefile_start_gb": 4,
        "nodefile_dir": None,
    }

    configure_solver(model, solver_cfg, tmp_path)

    assert params.TimeLimit == "UNSET"
    assert params.MIPGap == 0.001
    assert params.Threads == 8
