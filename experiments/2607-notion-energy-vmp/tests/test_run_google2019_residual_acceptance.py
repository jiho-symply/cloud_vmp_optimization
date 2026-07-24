from __future__ import annotations

import copy
import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[3] / "scripts/run_google2019_residual_acceptance.py"
SPEC = importlib.util.spec_from_file_location(
    "run_google2019_residual_acceptance",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
acceptance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(acceptance)


def test_parser_and_config_override_are_diagnostics_only(tmp_path) -> None:
    processed_dir = tmp_path / "processed"
    baseline = {
        "data": {"google_dir": "original/processed"},
        "experiment": {
            "num_scenarios": 40,
            "num_servers": 6,
            "class_counts": {
                "on_demand": 100,
                "spot": 100,
                "batch_candidate": 100,
            },
        },
        "solver": {
            "time_limit_seconds": None,
            "mip_gap": 0.001,
            "threads": 8,
        },
    }
    original = copy.deepcopy(baseline)
    args = acceptance.parse_args(
        [
            "--processed-dir",
            str(processed_dir),
            "--mode",
            "solve",
            "--on-demand-count",
            "3",
            "--spot-count",
            "4",
            "--batch-job-count",
            "5",
            "--num-scenarios",
            "2",
            "--num-servers",
            "7",
            "--time-limit",
            "60",
            "--mip-gap",
            "0.02",
            "--threads",
            "6",
        ]
    )

    effective = acceptance._apply_overrides(
        baseline,
        args,
        processed_dir=processed_dir.resolve(),
    )

    assert baseline == original
    assert effective["data"]["google_dir"] == str(processed_dir.resolve())
    assert effective["experiment"]["class_counts"] == {
        "on_demand": 3,
        "spot": 4,
        "batch_jobs": 5,
    }
    assert effective["experiment"]["num_scenarios"] == 2
    assert effective["experiment"]["num_servers"] == 7
    assert effective["solver"]["time_limit_seconds"] == 60.0
    assert effective["solver"]["mip_gap"] == 0.02
    assert effective["solver"]["threads"] == 6


def test_relative_cli_paths_are_resolved_from_repository_root() -> None:
    resolved = acceptance._resolve_repo_path("data/processed/example")

    assert resolved == (acceptance.REPO_ROOT / "data/processed/example").resolve()
