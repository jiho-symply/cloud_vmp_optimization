from __future__ import annotations

import os
from pathlib import Path

import importlib.util


def _load_runner_module():
    path = Path(__file__).resolve().parents[1] / "scripts/run_experiment_suite.py"
    spec = importlib.util.spec_from_file_location("run_experiment_suite", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_child_environment_limits_blas_and_openmp_threads_to_one() -> None:
    runner = _load_runner_module()
    env = runner._child_environment(os.environ.copy())
    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["OMP_NUM_THREADS"] == "1"
    assert env["MKL_NUM_THREADS"] == "1"
    assert env["OPENBLAS_NUM_THREADS"] == "1"
