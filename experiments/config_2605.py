import importlib.util
import json
import math
from pathlib import Path


WORKLOAD_TYPES = ("on_demand", "spot", "batch")


def load_config(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_config(path, config):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)


def resolve_path(value, repo_root, default=None):
    if value is None or value == "":
        return default
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path


def get_nested(mapping, keys, default=None):
    current = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def as_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def resolve_workload_counts(data_config):
    counts_config = data_config.get("vm_counts") or {}
    explicit_counts = {
        key: counts_config.get(key)
        for key in WORKLOAD_TYPES
        if counts_config.get(key) is not None
    }

    if explicit_counts:
        missing = [key for key in WORKLOAD_TYPES if key not in explicit_counts]
        if missing:
            raise ValueError(f"vm_counts must set all workload types when used. Missing: {missing}")
        counts = {key: int(explicit_counts[key]) for key in WORKLOAD_TYPES}
        configured_total = data_config.get("vm_count")
        if configured_total is not None and int(configured_total) != sum(counts.values()):
            raise ValueError(
                f"vm_count={configured_total} does not match vm_counts total={sum(counts.values())}."
            )
        return counts

    total = int(data_config.get("vm_count", 24))
    if total <= 0:
        raise ValueError("vm_count must be positive.")

    ratios = data_config.get("vm_type_ratios") or {
        "on_demand": 1.0,
        "spot": 1.0,
        "batch": 1.0,
    }
    normalized = {key: max(0.0, float(ratios.get(key, 0.0))) for key in WORKLOAD_TYPES}
    ratio_sum = sum(normalized.values())
    if ratio_sum <= 0:
        raise ValueError("vm_type_ratios must contain at least one positive value.")

    exact = {key: total * normalized[key] / ratio_sum for key in WORKLOAD_TYPES}
    counts = {key: int(math.floor(exact[key])) for key in WORKLOAD_TYPES}
    remainder = total - sum(counts.values())
    order = sorted(WORKLOAD_TYPES, key=lambda key: (exact[key] - counts[key], key), reverse=True)
    for key in order[:remainder]:
        counts[key] += 1
    return counts


def default_source_instance_name(counts, scenario_count, server_capacity):
    total = sum(counts.values())
    return (
        f"chance_2sp_toy_{total}vm_"
        f"od{counts['on_demand']}_sp{counts['spot']}_bj{counts['batch']}_"
        f"sc{scenario_count}_cap{int(server_capacity)}"
    )


def default_output_instance_name(counts, server_capacity):
    total = sum(counts.values())
    return (
        f"notion_vm_type_{total}vm_"
        f"od{counts['on_demand']}_sp{counts['spot']}_bj{counts['batch']}_"
        f"cap{int(server_capacity)}"
    )


def load_2604_builder(repo_root):
    script_path = repo_root / "experiments" / "2604-chance-2sp-toy" / "build_dataset.py"
    spec = importlib.util.spec_from_file_location("chance_2sp_build_dataset", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_parameters(config):
    parameters = config.get("parameters", {})
    chance = parameters.get("chance_constraints", {})
    batch = parameters.get("batch", {})
    objective = parameters.get("objective", {})
    constants = parameters.get("constants", {})
    return {
        "epsilon_od": float(chance.get("epsilon_od", 0.10)),
        "epsilon_sp": float(chance.get("epsilon_sp", 0.20)),
        "rho": float(chance.get("rho", 0.80)),
        "kappa": float(batch.get("kappa", 0.20)),
        "objective_type": objective.get("type", "energy"),
        "lambda_migration": float(objective.get("lambda_migration", 0.10)),
        "energy_idle": float(objective.get("energy_idle", 100.0)),
        "energy_cpu": float(objective.get("energy_cpu", 300.0)),
        "energy_migration": float(objective.get("energy_migration", 50.0)),
        "big_m": constants.get("big_m"),
        "server_capacity": constants.get("server_capacity"),
        "server_count": constants.get("server_count"),
    }


def formulation_config(config, defaults=None):
    formulation = dict(defaults or {})
    formulation.update(get_nested(config, ["model", "formulation"], {}) or {})
    return formulation


def solver_config(config, defaults=None):
    solver = dict(defaults or {})
    solver.update(get_nested(config, ["model", "solver"], {}) or {})
    return solver


def apply_solver_config(model, solver, results_dir, time_limit=None, mip_gap=None, threads=None):
    effective = dict(solver)
    if time_limit is not None:
        effective["time_limit"] = time_limit
    if mip_gap is not None:
        effective["mip_gap"] = mip_gap
    if threads is not None:
        effective["threads"] = threads

    param_map = {
        "time_limit": "TimeLimit",
        "mip_gap": "MIPGap",
        "threads": "Threads",
        "seed": "Seed",
        "method": "Method",
        "node_method": "NodeMethod",
        "crossover": "Crossover",
        "no_rel_heur_time": "NoRelHeurTime",
        "heuristics": "Heuristics",
        "presolve": "Presolve",
        "cuts": "Cuts",
        "mip_focus": "MIPFocus",
        "numeric_focus": "NumericFocus",
        "output_flag": "OutputFlag",
    }
    for config_key, gurobi_key in param_map.items():
        value = effective.get(config_key)
        if value is not None:
            model.setParam(gurobi_key, value)

    log_file = effective.get("log_file", "solver.log")
    model.setParam("LogFile", str(results_dir / log_file))
    effective["log_file"] = str(results_dir / log_file)
    return effective
