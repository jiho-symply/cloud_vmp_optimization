from __future__ import annotations

from pathlib import Path

import pytest

from notion_server_min_vmp.data import load_config


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


def test_baseline_uses_scalar_migration_energy_coefficient() -> None:
    config = load_config(EXPERIMENT_ROOT / "configs" / "baseline.yaml")

    assert config["migration"] == {"coefficient": pytest.approx(0.05)}
    assert "mode" not in config["migration"]
    assert "kappa" not in config["migration"]
