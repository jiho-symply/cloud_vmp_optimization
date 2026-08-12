from __future__ import annotations

from types import SimpleNamespace

from notion_server_min_vmp.model import configure_solver


def test_configure_solver_applies_norel_heuristic_time(tmp_path) -> None:
    model = SimpleNamespace(Params=SimpleNamespace())

    configure_solver(
        model,
        {
            "output_flag": 0,
            "threads": 16,
            "time_limit_seconds": 10_800,
            "no_rel_heur_time_seconds": 1_800,
        },
        tmp_path,
    )

    assert model.Params.Threads == 16
    assert model.Params.TimeLimit == 10_800
    assert model.Params.NoRelHeurTime == 1_800
