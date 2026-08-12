#!/usr/bin/env python3
"""Run a generic sweep with the migration-prohibited experiment entry point."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


EXPERIMENT_ROOT = Path(__file__).resolve().parent
BASE_RUNNER_PATH = EXPERIMENT_ROOT / "run_sweep.py"
SINGLE_RUN_ENTRYPOINT = EXPERIMENT_ROOT / "run_experiment_no_migration.py"


def _load_base_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "notion_server_min_no_migration_shared_sweep", BASE_RUNNER_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load shared sweep runner: {BASE_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base_runner()
BASE.SINGLE_RUN_ENTRYPOINT = SINGLE_RUN_ENTRYPOINT


def main(argv: list[str] | None = None) -> int:
    if not SINGLE_RUN_ENTRYPOINT.is_file():
        raise FileNotFoundError(SINGLE_RUN_ENTRYPOINT)
    return BASE.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
