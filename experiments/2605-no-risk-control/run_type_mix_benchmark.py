"""
Benchmark no-risk-control difficulty by VM type mix.

Masks use OD/SP/BJ order:

- 100: OD only
- 110: OD + SP
- 101: OD + BJ
- 111: OD + SP + BJ

For masks that include BJ, the script sweeps kappa values.  For masks without
BJ, kappa is recorded as "na" and the source/instance is generated once.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import gurobipy as gp


EXPERIMENT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = EXPERIMENT_DIR.parent
REPO_ROOT = EXPERIMENTS_DIR.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from config_2605 import (  # noqa: E402
    apply_solver_config,
    default_output_instance_name,
    load_config,
    model_parameters,
    resolve_path,
    solver_config,
    write_config,
)


WORKLOAD_KEYS = ("on_demand", "spot", "batch")
MASK_BITS = {
    "on_demand": 0,
    "spot": 1,
    "batch": 2,
}
DEFAULT_CONFIG = EXPERIMENT_DIR / "config.json"
DEFAULT_RESULTS_ROOT = EXPERIMENT_DIR / "type_mix_results"


def load_module(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mask_counts(mask, total_vms):
    if len(mask) != 3 or any(bit not in {"0", "1"} for bit in mask):
        raise ValueError(f"Invalid mask {mask!r}; expected a 3-bit OD/SP/BJ mask.")

    included = [key for key in WORKLOAD_KEYS if mask[MASK_BITS[key]] == "1"]
    if not included:
        raise ValueError("At least one VM type must be included.")

    base = total_vms // len(included)
    remainder = total_vms % len(included)
    counts = {key: 0 for key in WORKLOAD_KEYS}
    for index, key in enumerate(included):
        counts[key] = base + (1 if index < remainder else 0)
    return counts


def kappa_jobs(mask, kappa_values, default_kappa):
    if mask[2] == "1":
        return [(float(value), str(value)) for value in kappa_values]
    return [(float(default_kappa), "na")]


def tag_number(value):
    return str(value).replace(".", "p")


def case_name(mask, counts, total_vms, kappa_label):
    parts = [
        f"mix{mask}",
        f"{total_vms}vm",
        f"od{counts['on_demand']}",
        f"sp{counts['spot']}",
        f"bj{counts['batch']}",
    ]
    if kappa_label != "na":
        parts.append(f"k{tag_number(kappa_label)}")
    return "_".join(parts)


def source_name(mask, counts, scenario_count, server_capacity, seed, scenario_seed):
    total = sum(counts.values())
    return (
        f"no_risk_interactive_source_mix{mask}_{total}vm_"
        f"od{counts['on_demand']}_sp{counts['spot']}_bj{counts['batch']}_"
        f"sc{scenario_count}_cap{int(server_capacity)}_seed{seed}_sseed{scenario_seed}"
    )


def set_nested(mapping, keys, value):
    current = mapping
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value


def config_path(path):
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def safe_attr(model, name):
    try:
        return getattr(model, name)
    except gp.GurobiError:
        return None
    except AttributeError:
        return None


def solve_case(model_module, instance_path, results_dir, config, args):
    data = model_module.load_data(instance_path)
    formulation = config.get("model", {}).get("formulation", {})
    solver = solver_config(config, defaults={"mip_gap": args.mip_gap, "method": args.method})
    solver["time_limit"] = args.time_limit
    solver["threads"] = args.threads
    solver["seed"] = args.gurobi_seed
    solver["method"] = args.method
    solver["output_flag"] = 1 if args.output_flag else 0

    build_start = time.perf_counter()
    model = model_module.build_model(data, formulation=formulation)
    model.update()
    build_wall = time.perf_counter() - build_start

    results_dir.mkdir(parents=True, exist_ok=True)
    effective_solver = apply_solver_config(model, solver, results_dir)

    solve_start = time.perf_counter()
    model.optimize()
    solve_wall = time.perf_counter() - solve_start

    solution_files = None
    if args.write_solutions:
        solution_files = model_module.write_solution(model, results_dir)

    summary = {
        "status": model_module.status_name(model.Status),
        "objective": model.ObjVal if model.SolCount else None,
        "bound": safe_attr(model, "ObjBound"),
        "gap": model.MIPGap if model.SolCount else None,
        "runtime": model.Runtime,
        "build_wall": build_wall,
        "solve_wall": solve_wall,
        "nodes": model.NodeCount,
        "variables": model.NumVars,
        "binary_variables": model.NumBinVars,
        "constraints": model.NumConstrs,
        "instance": data["raw"]["name"],
        "solution_files": solution_files or {},
        "run_config": {
            "config": config,
            "effective_solver": effective_solver,
            "instance": str(instance_path),
            "results_dir": str(results_dir),
        },
    }
    with open(results_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return summary


def prepare_case(prepare_module, config, counts, kappa, source_dir, output_dir, source_instance_name, output_instance_name):
    params = model_parameters(config)
    prepare_module.build_source_instance(config, source_dir, source_instance_name, counts, params)
    prepare_module.validate_source_counts(source_dir, counts)
    instance = prepare_module.build_instance(
        source_dir=source_dir,
        kappa=kappa,
        objective_type=params["objective_type"],
        lambda_migration=params["lambda_migration"],
        energy_idle=params["energy_idle"],
        energy_cpu=params["energy_cpu"],
        energy_migration=params["energy_migration"],
        epsilon_od=params["epsilon_od"],
        epsilon_sp=params["epsilon_sp"],
        rho=params["rho"],
        capacity=params["server_capacity"],
        big_m=params["big_m"],
        server_count=params["server_count"],
        config_metadata={
            "source_instance_name": source_instance_name,
            "output_instance_name": output_instance_name,
            "workload_counts": counts,
            "benchmark": "no-risk-control type mix",
        },
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    instance_path = output_dir / "vm_type_instance.json"
    with open(instance_path, "w", encoding="utf-8") as file:
        json.dump(instance, file, indent=2)
    write_config(output_dir / "config_used.json", config)
    return instance_path


def write_summary_files(rows, results_root):
    results_root.mkdir(parents=True, exist_ok=True)
    csv_path = results_root / "summary.csv"
    fieldnames = [
        "mask",
        "kappa",
        "on_demand",
        "spot",
        "batch",
        "status",
        "objective",
        "bound",
        "gap",
        "runtime",
        "build_wall",
        "solve_wall",
        "nodes",
        "variables",
        "binary_variables",
        "constraints",
        "instance_path",
        "results_dir",
        "solution_file",
        "nonzero_variable_file",
        "error",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item.get("case_index", 0)):
            writer.writerow({key: row.get(key) for key in fieldnames})

    md_path = results_root / "summary.md"
    with open(md_path, "w", encoding="utf-8") as file:
        file.write("# No-Risk-Control Type Mix Benchmark\n\n")
        file.write("| mask | kappa | counts | status | runtime | solve wall | gap | objective | vars | bin vars | constrs | solution |\n")
        file.write("|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for row in sorted(rows, key=lambda item: item.get("case_index", 0)):
            counts = f"od={row['on_demand']}, sp={row['spot']}, bj={row['batch']}"
            gap = "" if row.get("gap") is None else f"{row['gap']:.4g}"
            objective = "" if row.get("objective") is None else f"{row['objective']:.6g}"
            runtime = "" if row.get("runtime") is None else f"{row['runtime']:.2f}"
            solve_wall = "" if row.get("solve_wall") is None else f"{row['solve_wall']:.2f}"
            solution = "yes" if row.get("solution_file") else ""
            file.write(
                f"| {row['mask']} | {row['kappa']} | {counts} | {row['status']} | "
                f"{runtime} | {solve_wall} | {gap} | {objective} | "
                f"{row.get('variables', '')} | {row.get('binary_variables', '')} | {row.get('constraints', '')} | {solution} |\n"
            )
    return csv_path, md_path


def build_case_specs(args, base_config):
    specs = []
    case_index = 0
    for mask in args.masks:
        counts = mask_counts(mask, args.total_vms)
        for kappa, kappa_label in kappa_jobs(mask, args.kappa_values, args.default_kappa):
            name = case_name(mask, counts, args.total_vms, kappa_label)
            base_source_instance = source_name(
                mask,
                counts,
                args.scenario_count,
                args.server_capacity,
                args.seed,
                args.scenario_seed,
            )
            # Keep source folders case-local.  Several kappa variants share the
            # same VM mix, and rebuilding a shared 2604 source folder in
            # parallel can race with readers of its CSV files.
            source_instance = f"{base_source_instance}_{name}"
            source_dir = REPO_ROOT / "data" / "processed" / "2604-chance-2sp-toy" / source_instance
            output_instance = default_output_instance_name(counts, args.server_capacity)
            output_instance = f"{output_instance}_{name}"
            output_dir = REPO_ROOT / "data" / "processed" / "2605-no-risk-control" / args.data_subdir / name
            results_dir = resolve_path(args.results_root, REPO_ROOT) / name

            config = copy.deepcopy(base_config)
            set_nested(config, ["data", "build_source"], True)
            set_nested(config, ["data", "random_seed"], args.seed)
            set_nested(config, ["data", "scenario_seed"], args.scenario_seed)
            set_nested(config, ["data", "scenario_count"], args.scenario_count)
            set_nested(config, ["data", "vm_count"], args.total_vms)
            set_nested(config, ["data", "vm_counts"], counts)
            set_nested(config, ["data", "max_vcpu"], args.max_vcpu)
            set_nested(config, ["data", "min_avg_cpu"], args.min_avg_cpu)
            set_nested(config, ["data", "server_capacity"], args.server_capacity)
            set_nested(config, ["data", "source_instance_name"], source_instance)
            set_nested(config, ["data", "source_dir"], config_path(source_dir))
            set_nested(config, ["data", "output_instance_name"], output_instance)
            set_nested(config, ["data", "output_dir"], config_path(output_dir))
            set_nested(config, ["parameters", "batch", "kappa"], kappa)
            set_nested(config, ["parameters", "constants", "server_capacity"], args.server_capacity)
            set_nested(config, ["model", "instance"], config_path(output_dir / "vm_type_instance.json"))
            set_nested(config, ["model", "results_dir"], config_path(results_dir))
            set_nested(config, ["model", "solver", "time_limit"], args.time_limit)
            set_nested(config, ["model", "solver", "threads"], args.threads)
            set_nested(config, ["model", "solver", "seed"], args.gurobi_seed)
            set_nested(config, ["model", "solver", "method"], args.method)

            specs.append(
                {
                    "case_index": case_index,
                    "name": name,
                    "mask": mask,
                    "kappa": kappa,
                    "kappa_label": kappa_label,
                    "counts": counts,
                    "source_instance": source_instance,
                    "output_instance": output_instance,
                    "source_dir": source_dir,
                    "output_dir": output_dir,
                    "results_dir": results_dir,
                    "config": config,
                    "run_options": {
                        "time_limit": args.time_limit,
                        "mip_gap": args.mip_gap,
                        "threads": args.threads,
                        "method": args.method,
                        "gurobi_seed": args.gurobi_seed,
                        "output_flag": args.output_flag,
                        "write_solutions": args.write_solutions,
                    },
                }
            )
            case_index += 1
    return specs


def run_case_worker(spec):
    prepare_module = load_module(EXPERIMENT_DIR / "prepare_data.py", "no_risk_prepare_data")
    model_module = load_module(EXPERIMENT_DIR / "model.py", "no_risk_model")
    args = argparse.Namespace(**spec["run_options"])
    counts = spec["counts"]
    row = {
        "case_index": spec["case_index"],
        "mask": spec["mask"],
        "kappa": spec["kappa_label"],
        "on_demand": counts["on_demand"],
        "spot": counts["spot"],
        "batch": counts["batch"],
        "instance_path": str(spec["output_dir"] / "vm_type_instance.json"),
        "results_dir": str(spec["results_dir"]),
        "error": "",
    }
    try:
        prepare_start = time.perf_counter()
        instance_path = prepare_case(
            prepare_module,
            spec["config"],
            counts,
            spec["kappa"],
            spec["source_dir"],
            spec["output_dir"],
            spec["source_instance"],
            spec["output_instance"],
        )
        prepare_wall = time.perf_counter() - prepare_start
        summary = solve_case(model_module, instance_path, spec["results_dir"], spec["config"], args)
        solution_files = summary.get("solution_files", {})
        row.update(
            {
                "status": summary["status"],
                "objective": summary["objective"],
                "bound": summary["bound"],
                "gap": summary["gap"],
                "runtime": summary["runtime"],
                "build_wall": summary["build_wall"] + prepare_wall,
                "solve_wall": summary["solve_wall"],
                "nodes": summary["nodes"],
                "variables": summary["variables"],
                "binary_variables": summary["binary_variables"],
                "constraints": summary["constraints"],
                "solution_file": solution_files.get("solution_file"),
                "nonzero_variable_file": solution_files.get("nonzero_variable_file"),
            }
        )
    except Exception as exc:  # Keep the full sweep running.
        row.update({"status": "ERROR", "error": repr(exc)})
    return row


def parse_args():
    parser = argparse.ArgumentParser(description="Run 12VM no-risk-control type-mix benchmark.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--data-subdir", default="type_mix")
    parser.add_argument("--total-vms", type=int, default=12)
    parser.add_argument("--masks", nargs="+", default=["100", "110", "101", "111"])
    parser.add_argument("--kappa-values", nargs="+", type=float, default=[0.2, 0.5, 1.0])
    parser.add_argument("--default-kappa", type=float, default=0.2)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--no-time-limit", action="store_true")
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--method", type=int, default=2)
    parser.add_argument("--gurobi-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario-seed", type=int, default=42)
    parser.add_argument("--scenario-count", type=int, default=10)
    parser.add_argument("--server-capacity", type=float, default=8.0)
    parser.add_argument("--min-avg-cpu", type=float, default=20.0)
    parser.add_argument("--max-vcpu", type=int, default=8)
    parser.add_argument("--output-flag", action="store_true")
    parser.add_argument("--write-solutions", dest="write_solutions", action="store_true", default=True)
    parser.add_argument("--no-write-solutions", dest="write_solutions", action="store_false")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.no_time_limit:
        args.time_limit = None
    base_config = load_config(args.config)
    specs = build_case_specs(args, base_config)
    results_root = resolve_path(args.results_root, REPO_ROOT)
    rows = []

    if args.parallel <= 1:
        for spec in specs:
            print(f"[run] {spec['name']}: counts={spec['counts']}, kappa={spec['kappa_label']}", flush=True)
            row = run_case_worker(spec)
            print(f"[done] {spec['name']}: status={row.get('status')}", flush=True)
            rows.append(row)
            write_summary_files(rows, results_root)
    else:
        print(f"[parallel] running {len(specs)} cases with max_workers={args.parallel}", flush=True)
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {}
            for spec in specs:
                print(f"[submit] {spec['name']}: counts={spec['counts']}, kappa={spec['kappa_label']}", flush=True)
                futures[executor.submit(run_case_worker, spec)] = spec
            for future in as_completed(futures):
                spec = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    counts = spec["counts"]
                    row = {
                        "case_index": spec["case_index"],
                        "mask": spec["mask"],
                        "kappa": spec["kappa_label"],
                        "on_demand": counts["on_demand"],
                        "spot": counts["spot"],
                        "batch": counts["batch"],
                        "status": "ERROR",
                        "instance_path": str(spec["output_dir"] / "vm_type_instance.json"),
                        "results_dir": str(spec["results_dir"]),
                        "error": repr(exc),
                    }
                print(f"[done] {spec['name']}: status={row.get('status')}", flush=True)
                rows.append(row)
                write_summary_files(rows, results_root)

    csv_path, md_path = write_summary_files(rows, results_root)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
