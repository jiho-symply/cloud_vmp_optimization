from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from notion_energy_vmp.data import load_instance, write_prepared_data


def _write_fixture(root: Path) -> Path:
    google = root / "data/processed/notion_toy_google2019_v1_pool"
    nyiso = root / "data/processed/nyiso"
    nrel = root / "data/processed/nrel"
    google.mkdir(parents=True)
    nyiso.mkdir(parents=True)
    nrel.mkdir(parents=True)

    vm_rows = []
    for cls, prefix in (("on_demand", "o"), ("spot", "s"), ("batch_candidate", "b")):
        for n in range(2):
            vm_rows.append(
                {
                    "vm_id": f"{prefix}{n}",
                    "class": cls,
                    "q_cpu": 0.10 + 0.01 * n,
                    "q_mem": 0.12 + 0.01 * n,
                    "arrival_t5": 0,
                    "departure_t5": 288,
                }
            )
    pd.DataFrame(vm_rows).to_csv(google / "vm_requests.csv", index=False)

    usage = []
    for xi in range(2):
        for vm_id in ["o0", "o1", "s0", "s1"]:
            for t in range(24):
                usage.append(
                    {
                        "scenario_id": xi,
                        "vm_id": vm_id,
                        "t_hour": t,
                        "cpu_usage": 0.04 + 0.002 * xi,
                        "mem_usage": 0.05,
                    }
                )
    pd.DataFrame(usage).to_csv(google / "vm_usage_hourly_scenarios.csv", index=False)

    preemption = []
    for xi in range(2):
        for vm_id in ["s0", "s1"]:
            for t in range(24):
                preemption.append(
                    {
                        "scenario_id": xi,
                        "vm_id": vm_id,
                        "t_hour": t,
                        "active": int(not (xi == 1 and t >= 20)),
                        "preempted": int(xi == 1 and t >= 20),
                    }
                )
    pd.DataFrame(preemption).to_csv(google / "spot_preemption_scenarios.csv", index=False)
    pd.DataFrame(
        [
            {
                "family_id": "k0",
                "q_cpu_B": 0.5,
                "q_mem_B": 0.5,
                "W_k": 1.0,
                "rho_cpu_B": 0.02,
                "rho_mem_B": 0.02,
                "base_cpu": 0.01,
                "base_mem": 0.01,
                "startup_cpu": 0.01,
                "startup_mem": 0.01,
            },
            {
                "family_id": "k1",
                "q_cpu_B": 0.6,
                "q_mem_B": 0.6,
                "W_k": 2.0,
                "rho_cpu_B": 0.03,
                "rho_mem_B": 0.03,
                "base_cpu": 0.02,
                "base_mem": 0.03,
                "startup_cpu": 0.04,
                "startup_mem": 0.05,
            }
        ]
    ).to_csv(google / "batch_families.csv", index=False)
    pd.DataFrame(
        [
            {
                "server_id": f"server_{n}",
                "C_cpu": 1.0,
                "C_mem": 1.0,
                "E_idle": 0.35,
                "E_cpu": 0.65,
                "min_on_time": 1,
                "min_off_time": 1,
            }
            for n in range(3)
        ]
    ).to_csv(google / "servers.csv", index=False)
    pd.DataFrame([{"scenario_id": 0, "probability": 0.5}, {"scenario_id": 1, "probability": 0.5}]).to_csv(
        google / "scenario_probabilities.csv", index=False
    )

    price_rows, renewable_rows = [], []
    for day_index, day in enumerate(("2019-07-01", "2019-07-02")):
        local_grid = pd.date_range(day, periods=48, freq="30min", tz="America/New_York")
        for timestamp in local_grid:
            price_rows.append(
                {
                    "timestamp_utc": timestamp.tz_convert("UTC"),
                    "site_id": "NYC",
                    "da_lbmp_usd_per_mwh": 40.0,
                    "rt_lbmp_usd_per_mwh": -5.0 if day_index == 1 and timestamp.hour == 3 else 45.0,
                }
            )
            solar_power = 1.5 if 6 <= timestamp.hour <= 18 else 0.0
            wind_power = 0.5 if 6 <= timestamp.hour <= 18 else 0.2
            renewable_rows.append(
                {
                    "timestamp_utc": timestamp.tz_convert("UTC"),
                    "site_id": "VA",
                    "solar_power_mw": solar_power,
                    "wind_power_mw": wind_power,
                    "renewable_power_mw": solar_power + wind_power,
                    "renewable_energy_mwh": 0.5 * (solar_power + wind_power),
                }
            )
    pd.DataFrame(price_rows).to_csv(nyiso / "prices.csv", index=False)
    pd.DataFrame(renewable_rows).to_csv(nrel / "renewable.csv", index=False)

    cfg = {
        "workspace_root": str(root),
        "experiment": {
            "name": "test",
            "seed": 42,
            "horizon_hours": 24,
            "num_scenarios": 2,
            "target_total_vms": 6,
            "class_counts": None,
            "num_servers": 3,
        },
        "data": {
            "google_dir": "data/processed/notion_toy_google2019_v1_pool",
            "vm_requests": "vm_requests.csv",
            "hourly_usage": "vm_usage_hourly_scenarios.csv",
            "spot_preemption": "spot_preemption_scenarios.csv",
            "batch_families": "batch_families.csv",
            "servers": "servers.csv",
            "scenario_probabilities": "scenario_probabilities.csv",
            "model_params": "model_params.json",
            "nyiso_prices": "data/processed/nyiso/prices.csv",
            "nrel_power": "data/processed/nrel/renewable.csv",
        },
        "energy_data": {
            "timezone": "America/New_York",
            "nyiso_zone": "NYC",
            "nrel_site": "VA",
            "scenario_dates": ["2019-07-01", "2019-07-02"],
            "day_ahead_mode": "scenario_mean",
            "electricity_price_floor_usd_per_kwh": 0.01,
            "sell_price_ratio": 0.85,
            "renewable_to_reference_demand_ratio": 1.0,
            "solar_capacity_multiplier": 1.0,
            "wind_capacity_multiplier": 1.0,
        },
        "ess": {
            "capacity_ratio_to_full_server_hour": 1.0,
            "charge_limit_fraction": 0.1,
            "discharge_limit_fraction": 0.1,
            "charge_efficiency": 0.95,
            "discharge_efficiency": 0.895,
            "initial_soc_fraction": 0.5,
        },
        "economics": {
            "kappa_on_demand": 10.0,
            "spot_discount_ratio": 0.3,
            "kappa_sla": 0.2,
            "kappa_migration": 0.1,
        },
        "risk": {"alpha": 0.95, "epsilon": 0.05},
        "solver": {},
    }
    path = root / "test.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_safe_preparation_preserves_values_and_writes_checksums(tmp_path: Path) -> None:
    config_path = _write_fixture(tmp_path)
    instance = load_instance(config_path)
    assert len(instance.I) == 2
    assert len(instance.J) == 2
    assert instance.metadata["sampling"]["target_class_counts"]["batch_candidate"] == 2
    assert min(instance.p_rt.values()) == 0.01
    assert max(instance.p_rt.values()) == 0.045
    assert abs(sum(instance.p.values()) - 1.0) < 1e-12
    assert instance.metadata["energy_alignment"]["renewable_scale"] > 0
    assert all(
        abs(instance.renewable[key] - instance.solar_renewable[key] - instance.wind_renewable[key]) < 1e-12
        for key in instance.renewable
    )
    expected_migration_coefficient = 0.1 * (instance.E_idle + instance.E_cpu * instance.C["CPU"]) / instance.C["MEM"]
    assert set(instance.c_mig.values()) == {expected_migration_coefficient}

    destination = write_prepared_data(instance, tmp_path / "prepared")
    expected = {
        "selected_vm_manifest.csv",
        "on_demand_usage.csv",
        "spot_usage.csv",
        "spot_observed_availability.csv",
        "batch_parameters.csv",
        "server_parameters.csv",
        "energy_scenarios.csv",
        "scenario_probabilities.csv",
        "scalar_parameters.json",
        "data_audit.json",
        "checksums.sha256",
    }
    assert expected == {p.name for p in destination.iterdir()}
    for line in (destination / "checksums.sha256").read_text().splitlines():
        digest, filename = line.split("  ", 1)
        assert hashlib.sha256((destination / filename).read_bytes()).hexdigest() == digest

    audit = json.loads((destination / "data_audit.json").read_text(encoding="utf-8"))
    assert audit["sampling"]["source_class_counts"] == {
        "on_demand": 2,
        "spot": 2,
        "batch_candidate": 2,
    }
    assert audit["sampling"]["selected_class_counts"] == {
        "on_demand": 2,
        "spot": 2,
        "batch_candidate": 2,
    }
    assert audit["data_quality"]["active_hour_coverage_summary"]["missing_active_hour_count"] == 0
    assert audit["data_quality"]["active_hour_coverage_summary"]["duplicate_active_hour_count"] == 0
    assert len(audit["data_quality"]["active_hour_coverage_by_vm_scenario"]) == 8
    assert audit["batch_scaling"][0] == {
        "family_id": "k0",
        "source_W_k": 1.0,
        "applied_scale": 1.0,
        "scaled_W_k": 1.0,
    }
    assert audit["batch_scaling"][1] == {
        "family_id": "k1",
        "source_W_k": 2.0,
        "applied_scale": 1.0,
        "scaled_W_k": 2.0,
    }
    assert audit["batch_parameter_canonicalization"]["base_cpu"] == {
        "source_unique_count": 2,
        "policy": "max_source_value",
        "canonical_value": 0.02,
        "source_min": 0.01,
        "source_max": 0.02,
    }
    assert audit["batch_parameter_canonicalization"]["startup_mem"] == {
        "source_unique_count": 2,
        "policy": "max_source_value",
        "canonical_value": 0.05,
        "source_min": 0.01,
        "source_max": 0.05,
    }
    assert audit["unit_conversions"] == {
        "price_usd_per_mwh_to_usd_per_kwh": 0.001,
        "electricity_purchase_price_floor": "hourly DA and RT prices are floored after USD/MWh-to-USD/kWh conversion; no upper cap is applied",
        "renewable_mwh_to_kwh": 1000.0,
        "day_ahead_first_stage": "hour-wise mean across selected historical scenario dates",
        "real_time_and_renewable": "one selected historical local date mapped to one workload scenario in order",
        "sell_price": "model assumption: max(0, sell_price_ratio * day_ahead_price), not an NYISO observation",
        "renewable_global_capacity_scaling": "model assumption applied after MWh-to-kWh conversion, not an observed capacity",
        "renewable_component_scaling": "solar and wind capacity multipliers are applied after one common reference-demand scale",
    }
    assert audit["energy_alignment"]["nyiso_daily_audit"][0]["hours_present"] == 24
    assert audit["energy_alignment"]["nrel_daily_audit"][0]["hours_present"] == 24
    assert audit["energy_alignment"]["electricity_price_floor_usd_per_kwh"] == 0.01
    assert audit["energy_alignment"]["price_floor_adjustment"]["real_time_adjusted_slot_count"] == 1
    assert audit["energy_alignment"]["price_floor_adjustment"]["day_ahead_adjusted_slot_count"] == 0
    assert audit["parameter_values"]["ess"]["capacity"] == instance.ess_capacity
    assert audit["parameter_values"]["spot_revenue"]["count"] == len(instance.J)
    assert audit["parameter_values"]["migration_coefficient"]["count"] == len(instance.I)
    assert audit["parameter_values"]["migration_coefficient_unit"] == "kWh per normalized memory unit migrated"


def test_solar_and_wind_capacity_multipliers_are_independent(tmp_path: Path) -> None:
    config_path = _write_fixture(tmp_path)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    cfg["energy_data"]["solar_capacity_multiplier"] = 1.0
    cfg["energy_data"]["wind_capacity_multiplier"] = 0.0
    solar_path = tmp_path / "solar.yaml"
    solar_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    solar_only = load_instance(solar_path)

    cfg["energy_data"]["solar_capacity_multiplier"] = 0.0
    cfg["energy_data"]["wind_capacity_multiplier"] = 1.0
    wind_path = tmp_path / "wind.yaml"
    wind_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    wind_only = load_instance(wind_path)

    cfg["energy_data"]["solar_capacity_multiplier"] = 1.0
    cfg["energy_data"]["wind_capacity_multiplier"] = 1.0
    combined_path = tmp_path / "combined.yaml"
    combined_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    combined = load_instance(combined_path)

    for key in combined.renewable:
        assert solar_only.renewable[key] == solar_only.solar_renewable[key]
        assert solar_only.wind_renewable[key] == 0.0
        assert wind_only.renewable[key] == wind_only.wind_renewable[key]
        assert wind_only.solar_renewable[key] == 0.0
        assert abs(combined.renewable[key] - solar_only.renewable[key] - wind_only.renewable[key]) < 1e-12
