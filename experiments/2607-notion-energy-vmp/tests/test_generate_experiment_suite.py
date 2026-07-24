from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_generator_module():
    path = Path(__file__).resolve().parents[1] / "scripts/generate_experiment_suite.py"
    spec = importlib.util.spec_from_file_location("generate_experiment_suite", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_renewable_mix_grid_is_full_cross_product() -> None:
    module = _load_generator_module()
    designs = list(
        module._renewable_mix_designs(
            {
                "solar_capacity_multiplier": [0.0, 0.5, 1.0, 2.0],
                "wind_capacity_multiplier": [0.0, 0.5, 1.0, 2.0],
            }
        )
    )

    assert len(designs) == 16
    assert designs[0] == (
        "solar_0p0_wind_0p0",
        {
            "energy_data.solar_capacity_multiplier": 0.0,
            "energy_data.wind_capacity_multiplier": 0.0,
        },
    )
    assert designs[-1] == (
        "solar_2p0_wind_2p0",
        {
            "energy_data.solar_capacity_multiplier": 2.0,
            "energy_data.wind_capacity_multiplier": 2.0,
        },
    )


def test_lhs_uses_every_stratum_once_per_dimension() -> None:
    module = _load_generator_module()
    design = module._lhs(128, 8, 20260710)

    assert design.shape == (128, 8)
    for column in range(design.shape[1]):
        strata = (design[:, column] * 128).astype(int)
        assert sorted(strata.tolist()) == list(range(128))
