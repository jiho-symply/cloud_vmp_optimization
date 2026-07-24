#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _set(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    node = cfg
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def _slug(value: Any) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def _lhs(n: int, dimensions: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    design = np.empty((n, dimensions), dtype=float)
    for j in range(dimensions):
        design[:, j] = (rng.permutation(n) + rng.random(n)) / n
    return design


def _renewable_mix_designs(section: dict[str, Any]):
    for solar in section["solar_capacity_multiplier"]:
        for wind in section["wind_capacity_multiplier"]:
            solar_value = float(solar)
            wind_value = float(wind)
            yield (
                f"solar_{_slug(solar_value)}_wind_{_slug(wind_value)}",
                {
                    "energy_data.solar_capacity_multiplier": solar_value,
                    "energy_data.wind_capacity_multiplier": wind_value,
                },
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="experiments/sweep_plan.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    plan_path = Path(args.plan).resolve()
    repo_root = plan_path.parents[1]
    plan = _load(plan_path)
    base_path = Path(plan["base_config"])
    if not base_path.is_absolute():
        base_path = repo_root / base_path
    base = _load(base_path)
    workspace_root = Path(base["workspace_root"])
    output = Path(args.output_dir or f"experiments/generated/{plan['suite_name']}")
    if not output.is_absolute():
        output = repo_root / output
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Generated suite is not empty: {output}; pass --overwrite intentionally")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    defaults = plan["defaults"]
    parallel = plan["parallel"]
    rows: list[dict[str, Any]] = []
    run_ids: set[str] = set()

    def add_run(phase: str, name: str, overrides: dict[str, Any], *, time_limit: float | None = None) -> None:
        run_id = f"{phase}__{name}"
        if run_id in run_ids:
            raise ValueError(f"Duplicate run id: {run_id}")
        run_ids.add(run_id)
        cfg = copy.deepcopy(base)
        cfg["experiment"]["name"] = run_id
        cfg["experiment"]["class_counts"] = None
        cfg["solver"]["threads"] = int(parallel["threads_per_run"])
        cfg["solver"]["soft_mem_limit_gb"] = float(parallel["soft_mem_limit_gb_per_run"])
        cfg["solver"]["nodefile_start_gb"] = float(parallel["nodefile_start_gb_per_run"])
        resolved_time_limit = time_limit if time_limit is not None else defaults.get("time_limit_seconds")
        cfg["solver"]["time_limit_seconds"] = (
            float(resolved_time_limit) if resolved_time_limit is not None else None
        )
        cfg["solver"]["mip_gap"] = float(defaults["mip_gap"])
        cfg["solver"]["write_lp"] = bool(defaults["write_lp"])
        cfg["solver"]["write_mps"] = bool(defaults["write_mps"])
        for key, value in overrides.items():
            _set(cfg, key, value)
        n_scenarios = int(cfg["experiment"]["num_scenarios"])
        cfg["energy_data"]["scenario_dates"] = [str(x) for x in cfg["energy_data"]["scenario_dates"][:n_scenarios]]
        cfg_path = output / f"{run_id}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        run_dir = workspace_root / plan["output_root"] / run_id
        rows.append(
            {
                "run_id": run_id,
                "phase": phase,
                "config_path": str(cfg_path),
                "run_dir": str(run_dir),
                "threads": cfg["solver"]["threads"],
                "time_limit_seconds": cfg["solver"]["time_limit_seconds"],
                "overrides_json": json.dumps(overrides, sort_keys=True),
            }
        )

    phases = plan["phases"]
    smoke = phases["smoke"]
    if smoke.get("enabled"):
        add_run(
            "smoke",
            "v40_xi2",
            {
                "experiment.target_total_vms": int(smoke["target_total_vms"]),
                "experiment.num_scenarios": int(smoke["num_scenarios"]),
                "experiment.num_servers": int(smoke["num_servers"]),
            },
            time_limit=(float(smoke["time_limit_seconds"]) if smoke.get("time_limit_seconds") is not None else None),
        )

    sampling = phases["sampling_replicates"]
    if sampling.get("enabled"):
        for seed in sampling["seeds"]:
            add_run("sampling", f"seed{seed}", {"experiment.seed": int(seed), "solver.seed": int(seed)})

    scaling = phases["size_scaling"]
    if scaling.get("enabled"):
        for design in scaling["designs"]:
            add_run(
                "scale",
                f"v{design['target_total_vms']}_xi{design['num_scenarios']}",
                {
                    "experiment.target_total_vms": int(design["target_total_vms"]),
                    "experiment.num_scenarios": int(design["num_scenarios"]),
                    "experiment.num_servers": int(design["num_servers"]),
                },
            )

    economics = phases["economics_ofat"]
    if economics.get("enabled"):
        mappings = {
            "kappa_sla": "economics.kappa_sla",
            "spot_discount_ratio": "economics.spot_discount_ratio",
            "kappa_migration": "economics.kappa_migration",
        }
        for label, dotted in mappings.items():
            for value in economics[label]:
                add_run("econ", f"{label}_{_slug(value)}", {dotted: float(value)})

    risk = phases["risk_ofat"]
    if risk.get("enabled"):
        for value in risk["alpha"]:
            add_run("risk", f"alpha_{_slug(value)}", {"risk.alpha": float(value)})
        for value in risk["epsilon"]:
            add_run("risk", f"epsilon_{_slug(value)}", {"risk.epsilon": float(value)})

    energy = phases["energy_ofat"]
    if energy.get("enabled"):
        for value in energy["renewable_ratio"]:
            add_run("energy", f"renewable_{_slug(value)}", {"energy_data.renewable_to_reference_demand_ratio": float(value)})
        for value in energy["ess_capacity_ratio"]:
            add_run("energy", f"ess_{_slug(value)}", {"ess.capacity_ratio_to_full_server_hour": float(value)})

    renewable_mix = phases.get("renewable_mix_grid", {})
    if renewable_mix.get("enabled"):
        for name, overrides in _renewable_mix_designs(renewable_mix):
            add_run("renewable_mix", name, overrides)

    seasonal = phases["seasonal_real_data"]
    if seasonal.get("enabled"):
        for season, dates in seasonal["blocks"].items():
            add_run("season", season, {"energy_data.scenario_dates": [str(x) for x in dates]})

    interaction = phases["latin_hypercube_interactions"]
    if interaction.get("enabled"):
        ranges = interaction["ranges"]
        labels = list(ranges)
        design = _lhs(int(interaction["runs"]), len(labels), int(interaction["seed"]))
        dotted = {
            "kappa_sla": "economics.kappa_sla",
            "spot_discount_ratio": "economics.spot_discount_ratio",
            "kappa_migration": "economics.kappa_migration",
            "alpha": "risk.alpha",
            "epsilon": "risk.epsilon",
            "renewable_ratio": "energy_data.renewable_to_reference_demand_ratio",
            "solar_capacity_multiplier": "energy_data.solar_capacity_multiplier",
            "wind_capacity_multiplier": "energy_data.wind_capacity_multiplier",
            "ess_capacity_ratio": "ess.capacity_ratio_to_full_server_hour",
        }
        for n in range(design.shape[0]):
            overrides = {}
            for j, label in enumerate(labels):
                low, high = map(float, ranges[label])
                overrides[dotted[label]] = float(low + design[n, j] * (high - low))
            add_run("lhs", f"{n:02d}", overrides)

    manifest = output / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "suite_metadata.json").write_text(
        json.dumps(
            {
                "suite_name": plan["suite_name"],
                "run_count": len(rows),
                "parallel": parallel,
                "base_config": str(base_path),
                "manifest": str(manifest),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Generated {len(rows)} runs")
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
