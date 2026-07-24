from __future__ import annotations

from pathlib import Path

import notion_server_min_vmp
import notion_server_min_vmp.model as model_module

from conftest import EXPERIMENT_ROOT


def test_unique_package_is_loaded_from_the_new_experiment() -> None:
    expected_package = (
        EXPERIMENT_ROOT / "src" / "notion_server_min_vmp" / "__init__.py"
    ).resolve()
    expected_model = (
        EXPERIMENT_ROOT / "src" / "notion_server_min_vmp" / "model.py"
    ).resolve()

    assert Path(notion_server_min_vmp.__file__).resolve() == expected_package
    assert Path(model_module.__file__).resolve() == expected_model
    assert "2607-notion-energy-vmp" not in str(expected_package)
