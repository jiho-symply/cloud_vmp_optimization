#!/usr/bin/env python3
"""Build or solve the server-min baseline against a processed Google 2019 dataset.

This is a diagnostics-only acceptance runner.  It loads the experiment's current
baseline YAML, replaces only ``data.google_dir`` plus explicitly requested CLI
overrides, and invokes the experiment's existing ``build_instance``,
``build_model``, and ``configure_solver`` functions.  It never edits the
baseline configuration or the model source.

Examples (paths may be absolute or repository-relative)::

    .venv/bin/python scripts/run_google2019_residual_acceptance.py \
      --processed-dir data/processed/notion_toy_google2019_v3_deterministic_union_coverage \
      --mode build \
      --run-dir runs/google2019_residual_build

    .venv/bin/python scripts/run_google2019_residual_acceptance.py \
      --processed-dir data/processed/notion_toy_google2019_v3_deterministic_union_coverage \
      --mode solve --on-demand-count 3 --spot-count 3 --batch-job-count 3 \
      --num-scenarios 2 --num-servers 6 --time-limit 60 --threads 8 \
      --run-dir runs/google2019_residual_reduced_solve
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = REPO_ROOT / "experiments/2607-notion-server-min-vmp"
EXPERIMENT_SRC = EXPERIMENT_ROOT / "src"
DEFAULT_BASELINE_CONFIG = EXPERIMENT_ROOT / "configs/baseline.yaml"
STATUS_FILENAME = "acceptance_status.json"

# The server-min experiment is intentionally isolated from the root-level
# compatibility package.  Put its source first so this script imports exactly
# the implementation whose baseline is being accepted.
if str(EXPERIMENT_SRC) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_SRC))

from notion_server_min_vmp.data import build_instance, load_config  # noqa: E402
from notion_server_min_vmp.model import (  # noqa: E402
    build_model,
    configure_solver,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a nonnegative integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("value must be finite and nonnegative")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build or solve the unchanged server-min model using a processed "
            "Google ClusterData 2019 directory, then write acceptance_status.json."
        )
    )
    parser.add_argument(
        "--processed-dir",
        required=True,
        help="Processed Google 2019 directory (absolute or repository-relative)",
    )
    parser.add_argument(
        "--baseline-config",
        default=str(DEFAULT_BASELINE_CONFIG),
        help="Server-min baseline YAML (absolute or repository-relative)",
    )
    parser.add_argument(
        "--mode",
        choices=("build", "solve"),
        default="build",
        help="Build validates and constructs the MILP without optimizing; solve also optimizes",
    )
    parser.add_argument(
        "--run-dir",
        help="Diagnostic output directory (absolute or repository-relative)",
    )
    parser.add_argument("--on-demand-count", type=_positive_int)
    parser.add_argument("--spot-count", type=_nonnegative_int)
    parser.add_argument("--batch-job-count", type=_nonnegative_int)
    parser.add_argument("--num-scenarios", type=_positive_int)
    parser.add_argument("--num-servers", type=_positive_int)
    parser.add_argument("--time-limit", type=_positive_float)
    parser.add_argument("--mip-gap", type=_nonnegative_float)
    parser.add_argument("--threads", type=_positive_int)
    return parser.parse_args(argv)


def _resolve_repo_path(value: str | Path) -> Path:
    """Resolve CLI paths consistently, independent of the caller's cwd."""

    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (REPO_ROOT / candidate).resolve()


def _default_run_dir(mode: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")
    return (REPO_ROOT / "runs" / f"google2019_residual_{mode}_{stamp}").resolve()


def _apply_overrides(
    baseline: Mapping[str, Any],
    args: argparse.Namespace,
    *,
    processed_dir: Path,
) -> dict[str, Any]:
    """Return an isolated effective config; never mutate the loaded baseline."""

    config = copy.deepcopy(dict(baseline))
    data = config.setdefault("data", {})
    experiment = config.setdefault("experiment", {})
    solver = config.setdefault("solver", {})
    if not isinstance(data, dict) or not isinstance(experiment, dict) or not isinstance(solver, dict):
        raise ValueError("baseline data, experiment, and solver sections must be mappings")

    class_counts = experiment.setdefault("class_counts", {})
    if not isinstance(class_counts, dict):
        raise ValueError("baseline experiment.class_counts must be a mapping")

    # An absolute override prevents the baseline's workspace_root from changing
    # the meaning of a repository-relative --processed-dir.
    data["google_dir"] = str(processed_dir)
    if args.on_demand_count is not None:
        class_counts["on_demand"] = args.on_demand_count
    if args.spot_count is not None:
        class_counts["spot"] = args.spot_count
    if args.batch_job_count is not None:
        class_counts.pop("batch_candidate", None)
        class_counts["batch_jobs"] = args.batch_job_count
    if args.num_scenarios is not None:
        experiment["num_scenarios"] = args.num_scenarios
    if args.num_servers is not None:
        experiment["num_servers"] = args.num_servers
    if args.time_limit is not None:
        solver["time_limit_seconds"] = args.time_limit
    if args.mip_gap is not None:
        solver["mip_gap"] = args.mip_gap
    if args.threads is not None:
        solver["threads"] = args.threads
    return config


def _effective_overrides(config: Mapping[str, Any]) -> dict[str, Any]:
    experiment = config["experiment"]
    solver = config["solver"]
    counts = experiment["class_counts"]
    return {
        "google_dir": str(config["data"]["google_dir"]),
        "class_counts": {
            "on_demand": int(counts["on_demand"]),
            "spot": int(counts["spot"]),
            "batch_jobs": int(counts.get("batch_jobs", counts.get("batch_candidate", 0))),
        },
        "num_scenarios": int(experiment["num_scenarios"]),
        "num_servers": int(experiment["num_servers"]),
        "solver": {
            "time_limit_seconds": solver.get("time_limit_seconds"),
            "mip_gap": solver.get("mip_gap"),
            "threads": solver.get("threads"),
        },
    }


def _model_sizes(model: Any) -> dict[str, int]:
    model.update()
    fields = {
        "num_variables": "NumVars",
        "num_binary_variables": "NumBinVars",
        "num_integer_variables": "NumIntVars",
        "num_constraints": "NumConstrs",
        "num_general_constraints": "NumGenConstrs",
        "num_nonzeros": "NumNZs",
    }
    return {label: int(getattr(model, attribute)) for label, attribute in fields.items()}


def _solver_status_name(status_code: int) -> str:
    try:
        from gurobipy import GRB
    except ImportError:  # pragma: no cover - build_model already gives the actionable error
        return f"STATUS_{status_code}"

    names = (
        "LOADED",
        "OPTIMAL",
        "INFEASIBLE",
        "INF_OR_UNBD",
        "UNBOUNDED",
        "CUTOFF",
        "ITERATION_LIMIT",
        "NODE_LIMIT",
        "TIME_LIMIT",
        "SOLUTION_LIMIT",
        "INTERRUPTED",
        "NUMERIC",
        "SUBOPTIMAL",
        "INPROGRESS",
        "USER_OBJ_LIMIT",
        "WORK_LIMIT",
        "MEM_LIMIT",
    )
    for name in names:
        if hasattr(GRB, name) and status_code == int(getattr(GRB, name)):
            return name
    return f"STATUS_{status_code}"


def _optional_finite_attribute(model: Any, attribute: str) -> float | None:
    try:
        value = float(getattr(model, attribute))
    # Gurobi may raise its own exception type when a solve attribute is not
    # defined for the terminal status (for example ObjVal with no incumbent).
    # These fields are explicitly optional diagnostics, so omit only that field.
    except Exception:  # noqa: BLE001
        return None
    return value if math.isfinite(value) else None


def _solve_summary(model: Any) -> dict[str, Any]:
    status_code = int(model.Status)
    solution_count = int(model.SolCount)
    result: dict[str, Any] = {
        "status": _solver_status_name(status_code),
        "status_code": status_code,
        "solution_count": solution_count,
    }
    optional_attributes = {
        "objective": "ObjVal",
        "objective_bound": "ObjBound",
        "relative_mip_gap": "MIPGap",
        "runtime_seconds": "Runtime",
        "node_count": "NodeCount",
    }
    for label, attribute in optional_attributes.items():
        value = _optional_finite_attribute(model, attribute)
        if value is not None:
            result[label] = value
    return result


def _write_status(run_dir: Path, payload: Mapping[str, Any]) -> Path:
    path = run_dir / STATUS_FILENAME
    path.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def run(args: argparse.Namespace) -> dict[str, Any]:
    baseline_path = _resolve_repo_path(args.baseline_config)
    processed_dir = _resolve_repo_path(args.processed_dir)
    run_dir = _resolve_repo_path(args.run_dir) if args.run_dir else _default_run_dir(args.mode)
    run_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "mode": args.mode,
        "status": "STARTED",
        "baseline_config": str(baseline_path),
        "processed_dir": str(processed_dir),
        "run_dir": str(run_dir),
    }
    try:
        if not baseline_path.is_file():
            raise FileNotFoundError(baseline_path)
        if not processed_dir.is_dir():
            raise FileNotFoundError(processed_dir)

        config = _apply_overrides(
            load_config(baseline_path),
            args,
            processed_dir=processed_dir,
        )
        payload["effective_overrides"] = _effective_overrides(config)

        instance = build_instance(config, config_path=baseline_path)
        payload["instance_summary"] = instance.summary()

        artifacts = build_model(instance)
        model = artifacts.model
        configure_solver(model, config.get("solver", {}), run_dir)
        payload["model_sizes"] = _model_sizes(model)

        if args.mode == "build":
            payload["status"] = "BUILT_NOT_SOLVED"
            payload["solution_count"] = 0
        else:
            model.optimize()
            solve = _solve_summary(model)
            payload["status"] = solve["status"]
            payload["solution_count"] = solve["solution_count"]
            payload["solve"] = solve
    except Exception as exc:
        payload["status"] = "ERROR"
        payload["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        _write_status(run_dir, payload)
        raise

    _write_status(run_dir, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = run(args)
    except Exception as exc:
        print(f"Acceptance run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
