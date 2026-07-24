from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml


CPU = "CPU"
MEM = "MEM"


@dataclass
class InstanceData:
    I: list[str]
    J: list[str]
    K: list[str]
    S: list[str]
    T: list[int]
    Xi: list[int]
    p: dict[int, float]
    active_od: dict[str, list[int]]
    active_spot: dict[str, list[int]]
    q_od: dict[tuple[str, str], float]
    q_spot: dict[tuple[str, str], float]
    load_od: dict[tuple[str, str, int, int], float]
    load_spot: dict[tuple[str, str, int, int], float]
    spot_available: dict[tuple[str, int, int], int]
    q_batch: dict[tuple[str, str], float]
    rho_batch: dict[tuple[str, str], float]
    W: dict[str, float]
    batch_base: dict[str, float]
    batch_startup: dict[str, float]
    C: dict[str, float]
    E_idle: float
    E_cpu: float
    min_on: dict[str, int]
    min_off: dict[str, int]
    p_da: dict[int, float]
    p_rt: dict[tuple[int, int], float]
    p_sell: dict[int, float]
    renewable: dict[tuple[int, int], float]
    solar_renewable: dict[tuple[int, int], float]
    wind_renewable: dict[tuple[int, int], float]
    ess_capacity: float
    ess_charge_max: float
    ess_discharge_max: float
    eta_charge: float
    eta_discharge: float
    soc_init: float
    pi_spot: dict[str, float]
    c_mig: dict[str, float]
    lambda_exc: float
    alpha: float
    epsilon: float
    source_files: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "sets": {
                "on_demand_vms": len(self.I),
                "spot_vms": len(self.J),
                "batch_families": len(self.K),
                "servers": len(self.S),
                "periods": len(self.T),
                "scenarios": len(self.Xi),
            },
            "scenario_probabilities": self.p,
            "capacity": self.C,
            "server_energy": {"E_idle": self.E_idle, "E_cpu": self.E_cpu},
            "ess": {
                "capacity": self.ess_capacity,
                "charge_max": self.ess_charge_max,
                "discharge_max": self.ess_discharge_max,
                "charge_efficiency": self.eta_charge,
                "discharge_efficiency": self.eta_discharge,
                "initial_soc": self.soc_init,
            },
            "risk": {"alpha": self.alpha, "epsilon": self.epsilon},
            "economics": {
                "lambda_exc": self.lambda_exc,
                "mean_day_ahead_usd_per_kwh": float(np.mean(list(self.p_da.values()))),
                "mean_spot_revenue_per_slot": float(np.mean(list(self.pi_spot.values()))) if self.J else 0.0,
            },
            **self.metadata,
        }


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration root must be a mapping")
    return cfg


def _resolve(root: Path, path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def _required(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _stable_order(values: Iterable[str], seed: int) -> list[str]:
    def key(v: str) -> str:
        return hashlib.sha256(f"{seed}:{v}".encode()).hexdigest()

    return sorted((str(v) for v in values), key=key)


def _largest_remainder_counts(source: dict[str, int], target: int) -> dict[str, int]:
    positive = {k: v for k, v in source.items() if v > 0}
    if target < len(positive):
        raise ValueError(f"target_total_vms={target} is smaller than the number of nonempty classes")
    total = sum(positive.values())
    raw = {k: target * v / total for k, v in positive.items()}
    out = {k: min(v, max(1, int(math.floor(raw[k])))) for k, v in positive.items()}
    while sum(out.values()) < target:
        candidates = [k for k in positive if out[k] < positive[k]]
        k = max(candidates, key=lambda x: (raw[x] - math.floor(raw[x]), positive[x] - out[x], x))
        out[k] += 1
    while sum(out.values()) > target:
        candidates = [k for k in positive if out[k] > 1]
        k = min(candidates, key=lambda x: (raw[x] - math.floor(raw[x]), -(out[x] - 1), x))
        out[k] -= 1
    return out


def _read_filtered_csv(
    path: Path,
    *,
    vm_ids: set[str] | None = None,
    scenarios: set[int] | None = None,
    chunksize: int = 400_000,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, chunksize=chunksize):
        if "vm_id" in chunk:
            chunk["vm_id"] = chunk["vm_id"].astype(str)
        mask = pd.Series(True, index=chunk.index)
        if vm_ids is not None:
            mask &= chunk["vm_id"].isin(vm_ids)
        if scenarios is not None:
            mask &= chunk["scenario_id"].isin(scenarios)
        filtered = chunk.loc[mask]
        if not filtered.empty:
            parts.append(filtered.copy())
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _hourly_active_periods(row: pd.Series, horizon: int) -> list[int]:
    if {"arrival_t5", "departure_t5"}.issubset(row.index):
        a = int(row["arrival_t5"]) // 12
        d = (int(row["departure_t5"]) - 1) // 12
    elif {"arrival_hour", "departure_hour"}.issubset(row.index):
        a = int(row["arrival_hour"])
        d = int(row["departure_hour"]) - 1
    else:
        a, d = 0, horizon - 1
    a, d = max(0, a), min(horizon - 1, d)
    if a > d:
        raise ValueError(f"Invalid active interval [{a}, {d}]")
    return list(range(a, d + 1))


def _constant_column(df: pd.DataFrame, column: str, *, atol: float = 1e-10) -> float:
    values = pd.to_numeric(df[column], errors="raise").to_numpy(float)
    if not np.allclose(values, values[0], atol=atol, rtol=0):
        raise ValueError(f"Notion model requires homogeneous {column}, found heterogeneous values")
    return float(values[0])


def _canonical_scalar_column(df: pd.DataFrame, column: str, *, atol: float = 1e-10) -> tuple[float, dict[str, Any]]:
    values = pd.to_numeric(df[column], errors="raise").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"{column} must contain finite numeric values")
    unique_count = int(pd.Series(values).round(12).nunique())
    if np.allclose(values, values[0], atol=atol, rtol=0):
        value = float(values[0])
        policy = "source_value"
    else:
        value = float(values.max())
        policy = "max_source_value"
    return value, {
        "source_unique_count": unique_count,
        "policy": policy,
        "canonical_value": value,
        "source_min": float(values.min()),
        "source_max": float(values.max()),
    }


def _load_hourly_energy(
    price_path: Path,
    renewable_path: Path,
    dates: list[str],
    timezone: str,
    nyiso_zone: str,
    nrel_site: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    def read_table(path: Path) -> pd.DataFrame:
        if path.suffix.lower() in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        if path.suffix.lower() in {".csv", ".gz"}:
            return pd.read_csv(path)
        raise ValueError(f"Unsupported table format: {path}")

    prices = read_table(price_path)
    renew = read_table(renewable_path)
    price_site_col = "site_id" if "site_id" in prices else "nyiso_zone"
    prices = prices.loc[prices[price_site_col].astype(str) == nyiso_zone].copy()
    renew = renew.loc[renew["site_id"].astype(str) == nrel_site].copy()
    if prices.empty:
        raise ValueError(f"No NYISO rows for zone/site {nyiso_zone}")
    if renew.empty:
        raise ValueError(f"No NREL rows for site {nrel_site}")
    component_columns = {"solar_power_mw", "wind_power_mw", "renewable_energy_mwh"}
    if not component_columns.issubset(renew.columns):
        raise ValueError(
            "NREL power data must contain solar_power_mw, wind_power_mw, and renewable_energy_mwh "
            "for independent renewable-capacity scaling"
        )
    for column in component_columns:
        renew[column] = pd.to_numeric(renew[column], errors="raise")
        values = renew[column].to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"NREL {column} must be finite and nonnegative")
    component_power = renew["solar_power_mw"] + renew["wind_power_mw"]
    if "renewable_power_mw" in renew:
        reported_power = pd.to_numeric(renew["renewable_power_mw"], errors="raise").to_numpy(float)
        if not np.allclose(component_power.to_numpy(float), reported_power, atol=1e-8, rtol=1e-8):
            raise ValueError("NREL renewable_power_mw must equal solar_power_mw + wind_power_mw")
    impossible_energy = (component_power <= 0) & (renew["renewable_energy_mwh"] > 1e-12)
    if impossible_energy.any():
        raise ValueError("NREL renewable energy is positive where both component powers are zero")
    solar_share = np.divide(
        renew["solar_power_mw"].to_numpy(float),
        component_power.to_numpy(float),
        out=np.zeros(len(renew), dtype=float),
        where=component_power.to_numpy(float) > 0,
    )
    renew["solar_energy_mwh"] = renew["renewable_energy_mwh"].to_numpy(float) * solar_share
    renew["wind_energy_mwh"] = renew["renewable_energy_mwh"] - renew["solar_energy_mwh"]

    for df in (prices, renew):
        ts = pd.to_datetime(df["timestamp_utc"], utc=True)
        local = ts.dt.tz_convert(timezone)
        df["local_date"] = local.dt.strftime("%Y-%m-%d")
        df["hour"] = local.dt.hour.astype(int)

    selected_prices = prices.loc[prices["local_date"].isin(dates)].copy()
    selected_renew = renew.loc[renew["local_date"].isin(dates)].copy()
    price_hourly = (
        selected_prices
        .groupby(["local_date", "hour"], as_index=False)[["da_lbmp_usd_per_mwh", "rt_lbmp_usd_per_mwh"]]
        .mean()
    )
    renewable_energy_columns = ["renewable_energy_mwh", "solar_energy_mwh", "wind_energy_mwh"]
    renew_hourly = selected_renew.groupby(["local_date", "hour"], as_index=False)[renewable_energy_columns].sum()

    da = np.zeros((len(dates), 24), dtype=float)
    rt = np.zeros_like(da)
    renewable_kwh = np.zeros_like(da)
    solar_kwh = np.zeros_like(da)
    wind_kwh = np.zeros_like(da)
    price_daily_audit: list[dict[str, Any]] = []
    renewable_daily_audit: list[dict[str, Any]] = []
    for n, day in enumerate(dates):
        pday = price_hourly.loc[price_hourly["local_date"] == day].sort_values("hour")
        rday = renew_hourly.loc[renew_hourly["local_date"] == day].sort_values("hour")
        if pday["hour"].tolist() != list(range(24)):
            raise ValueError(f"NYISO date {day} does not contain exactly hours 0..23; avoid DST transition days")
        if rday["hour"].tolist() != list(range(24)):
            raise ValueError(f"NREL date {day} does not contain exactly hours 0..23; avoid DST transition days")
        da[n] = pday["da_lbmp_usd_per_mwh"].to_numpy(float) / 1000.0
        rt[n] = pday["rt_lbmp_usd_per_mwh"].to_numpy(float) / 1000.0
        renewable_kwh[n] = rday["renewable_energy_mwh"].to_numpy(float) * 1000.0
        solar_kwh[n] = rday["solar_energy_mwh"].to_numpy(float) * 1000.0
        wind_kwh[n] = rday["wind_energy_mwh"].to_numpy(float) * 1000.0
        if not np.allclose(renewable_kwh[n], solar_kwh[n] + wind_kwh[n], atol=1e-8, rtol=1e-10):
            raise ValueError(f"NREL component energy does not reconstruct combined energy for {day}")

        raw_price_day = selected_prices.loc[selected_prices["local_date"] == day]
        raw_renew_day = selected_renew.loc[selected_renew["local_date"] == day]
        price_hour_counts = raw_price_day.groupby("hour").size()
        renew_hour_counts = raw_renew_day.groupby("hour").size()
        price_daily_audit.append(
            {
                "zone": nyiso_zone,
                "local_date": day,
                "hours_present": int(pday["hour"].nunique()),
                "missing_hours": [h for h in range(24) if h not in set(pday["hour"].astype(int))],
                "source_30min_rows": int(len(raw_price_day)),
                "source_rows_per_hour_min": int(price_hour_counts.min()) if not price_hour_counts.empty else 0,
                "source_rows_per_hour_max": int(price_hour_counts.max()) if not price_hour_counts.empty else 0,
                "da_usd_per_kwh": {
                    "min": float(da[n].min()),
                    "mean": float(da[n].mean()),
                    "max": float(da[n].max()),
                    "negative_count": int((da[n] < 0).sum()),
                    "extreme_abs_gt_1_count": int((np.abs(da[n]) > 1.0).sum()),
                },
                "rt_usd_per_kwh": {
                    "min": float(rt[n].min()),
                    "mean": float(rt[n].mean()),
                    "max": float(rt[n].max()),
                    "negative_count": int((rt[n] < 0).sum()),
                    "extreme_abs_gt_1_count": int((np.abs(rt[n]) > 1.0).sum()),
                },
            }
        )
        renewable_daily_audit.append(
            {
                "site": nrel_site,
                "local_date": day,
                "hours_present": int(rday["hour"].nunique()),
                "missing_hours": [h for h in range(24) if h not in set(rday["hour"].astype(int))],
                "source_30min_rows": int(len(raw_renew_day)),
                "source_rows_per_hour_min": int(renew_hour_counts.min()) if not renew_hour_counts.empty else 0,
                "source_rows_per_hour_max": int(renew_hour_counts.max()) if not renew_hour_counts.empty else 0,
                "raw_energy_kwh": {
                    "min": float(renewable_kwh[n].min()),
                    "mean": float(renewable_kwh[n].mean()),
                    "max": float(renewable_kwh[n].max()),
                },
                "raw_solar_energy_kwh": _numeric_summary(solar_kwh[n]),
                "raw_wind_energy_kwh": _numeric_summary(wind_kwh[n]),
            }
        )
    return da, rt, renewable_kwh, solar_kwh, wind_kwh, price_daily_audit, renewable_daily_audit


def _numeric_summary(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "mean": float(array.mean()),
        "max": float(array.max()),
    }


def _file_manifest(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        stat = path.stat()
        rows.append(
            {
                "path": str(path),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": _sha256(path),
            }
        )
    return rows


def load_instance(config: dict[str, Any] | str | Path) -> InstanceData:
    cfg = load_yaml(config) if isinstance(config, (str, Path)) else config
    root = Path(cfg["workspace_root"]).expanduser().resolve()
    exp = cfg["experiment"]
    data_cfg = cfg["data"]
    energy_cfg = cfg["energy_data"]
    seed = int(exp["seed"])
    horizon = int(exp["horizon_hours"])
    if horizon != 24:
        raise ValueError("The supplied Google day trace and real-data energy alignment currently require horizon_hours=24")

    google_dir = _resolve(root, data_cfg["google_dir"])
    paths = {
        "requests": _required(google_dir / data_cfg["vm_requests"]),
        "usage": _required(google_dir / data_cfg["hourly_usage"]),
        "preemption": _required(google_dir / data_cfg["spot_preemption"]),
        "batch": _required(google_dir / data_cfg["batch_families"]),
        "servers": _required(google_dir / data_cfg["servers"]),
        "probabilities": _required(google_dir / data_cfg["scenario_probabilities"]),
        "prices": _required(_resolve(root, data_cfg["nyiso_prices"])),
        "renewable": _required(_resolve(root, data_cfg["nrel_power"])),
    }

    requests = pd.read_csv(paths["requests"])
    requests["vm_id"] = requests["vm_id"].astype(str)
    if not {"on_demand", "spot", "batch_candidate"}.issubset(set(requests["class"])):
        raise ValueError("vm_requests.csv must contain on_demand, spot, and batch_candidate proxy classes")
    source_counts = requests["class"].value_counts().astype(int).to_dict()
    configured_counts = exp.get("class_counts")
    if configured_counts:
        target_counts = {k: int(v) for k, v in configured_counts.items()}
        if sum(target_counts.values()) != int(exp["target_total_vms"]):
            raise ValueError("experiment.class_counts must sum to target_total_vms")
    else:
        target_counts = _largest_remainder_counts(source_counts, int(exp["target_total_vms"]))
    for cls, count in target_counts.items():
        if count > source_counts.get(cls, 0):
            raise ValueError(f"Requested {count} {cls} VMs but only {source_counts.get(cls, 0)} exist")

    probabilities = pd.read_csv(paths["probabilities"]).sort_values("scenario_id")
    scenario_ids = probabilities["scenario_id"].astype(int).tolist()[: int(exp["num_scenarios"])]
    if len(scenario_ids) != int(exp["num_scenarios"]):
        raise ValueError("Not enough workload scenarios")
    p_raw = probabilities.set_index("scenario_id")["probability"].loc[scenario_ids].astype(float)
    if (p_raw < 0).any() or float(p_raw.sum()) <= 0:
        raise ValueError("Selected scenario probabilities must be nonnegative with positive sum")
    p_raw /= p_raw.sum()
    p = {int(x): float(p_raw.loc[x]) for x in scenario_ids}

    # Select only service VMs with complete hourly observations for every requested scenario.
    # This avoids silently imputing missing usage inside the Notion active interval.
    service_candidate_ids = set(
        requests.loc[requests["class"].isin(["on_demand", "spot"]), "vm_id"].astype(str)
    )
    usage = _read_filtered_csv(paths["usage"], vm_ids=service_candidate_ids, scenarios=set(scenario_ids))
    required_usage = {"scenario_id", "vm_id", "t_hour", "cpu_usage", "mem_usage"}
    if usage.empty or not required_usage.issubset(usage.columns):
        raise ValueError(f"Hourly usage is empty or missing columns {required_usage}")
    for column in ("cpu_usage", "mem_usage"):
        usage[column] = pd.to_numeric(usage[column], errors="raise")
        if not np.isfinite(usage[column].to_numpy(float)).all() or (usage[column] < 0).any():
            raise ValueError(f"{column} must be finite and nonnegative; no clipping/imputation is allowed")
    if usage.duplicated(["scenario_id", "vm_id", "t_hour"]).any():
        raise ValueError("Hourly usage must be unique by scenario_id, vm_id, t_hour")

    requests_by_id = requests.set_index("vm_id")
    duplicate_key_count = int(usage.duplicated(["scenario_id", "vm_id", "t_hour"]).sum())
    observed_hours = {
        (str(vm_id), int(xi)): set(group["t_hour"].astype(int))
        for (vm_id, xi), group in usage.groupby(["vm_id", "scenario_id"], sort=False)
    }
    eligible_by_class: dict[str, list[str]] = {}
    missing_pattern_by_class: dict[str, list[dict[str, Any]]] = {}
    for cls in ("on_demand", "spot"):
        eligible = []
        for vm_id in requests.loc[requests["class"] == cls, "vm_id"].astype(str):
            periods = set(_hourly_active_periods(requests_by_id.loc[vm_id], horizon))
            missing_by_scenario = {
                int(xi): sorted(periods - observed_hours.get((vm_id, xi), set()))
                for xi in scenario_ids
            }
            missing_total = sum(len(hours) for hours in missing_by_scenario.values())
            if missing_total == 0:
                eligible.append(vm_id)
            elif len(missing_pattern_by_class.setdefault(cls, [])) < 20:
                missing_pattern_by_class[cls].append(
                    {
                        "vm_id": vm_id,
                        "active_hours": sorted(periods),
                        "missing_total": int(missing_total),
                        "missing_by_scenario": missing_by_scenario,
                    }
                )
        eligible_by_class[cls] = eligible
        if len(eligible) < target_counts[cls]:
            raise ValueError(
                f"Only {len(eligible)} {cls} VMs have complete hourly observations for all scenarios; "
                f"{target_counts[cls]} requested; missing_pattern_examples={missing_pattern_by_class.get(cls, [])}"
            )

    selected_rows = []
    for cls in ("on_demand", "spot"):
        ids = set(
            _stable_order(eligible_by_class[cls], seed + (0 if cls == "on_demand" else 1))[: target_counts[cls]]
        )
        selected_rows.append(requests.loc[requests["vm_id"].isin(ids)].copy())
    selected = pd.concat(selected_rows, ignore_index=True)
    I = selected.loc[selected["class"] == "on_demand", "vm_id"].tolist()
    J = selected.loc[selected["class"] == "spot", "vm_id"].tolist()
    selected_ids = set(I + J)
    usage = usage.loc[usage["vm_id"].isin(selected_ids)].copy()

    selected_by_id = selected.set_index("vm_id")
    active = {v: _hourly_active_periods(selected_by_id.loc[v], horizon) for v in I + J}
    selected_coverage: list[dict[str, Any]] = []
    missing_active_hour_count = 0
    duplicate_active_hour_count = 0
    selected_usage_counts = (
        usage.groupby(["scenario_id", "vm_id", "t_hour"], as_index=False).size()
        if not usage.empty
        else pd.DataFrame(columns=["scenario_id", "vm_id", "t_hour", "size"])
    )
    selected_usage_count_idx = (
        selected_usage_counts.set_index(["scenario_id", "vm_id", "t_hour"])["size"]
        if not selected_usage_counts.empty
        else pd.Series(dtype=int)
    )
    usage_idx = usage.set_index(["scenario_id", "vm_id", "t_hour"])
    load_od: dict[tuple[str, str, int, int], float] = {}
    load_spot: dict[tuple[str, str, int, int], float] = {}
    missing_usage: list[tuple[int, str, int]] = []
    for v, target in [(v, load_od) for v in I] + [(v, load_spot) for v in J]:
        for xi in scenario_ids:
            observed_count = 0
            vm_duplicate_count = 0
            vm_missing_hours = []
            for t in active[v]:
                key = (xi, v, t)
                if key not in usage_idx.index:
                    missing_usage.append(key)
                    vm_missing_hours.append(t)
                    continue
                observed_count += 1
                key_count = int(selected_usage_count_idx.loc[key])
                if key_count > 1:
                    vm_duplicate_count += key_count - 1
                row = usage_idx.loc[key]
                target[(v, CPU, t, xi)] = float(row["cpu_usage"])
                target[(v, MEM, t, xi)] = float(row["mem_usage"])
            missing_active_hour_count += len(vm_missing_hours)
            duplicate_active_hour_count += vm_duplicate_count
            selected_coverage.append(
                {
                    "scenario_id": int(xi),
                    "vm_id": v,
                    "class": str(selected_by_id.loc[v, "class"]),
                    "active_hour_count": len(active[v]),
                    "observed_active_hour_count": int(observed_count),
                    "missing_active_hour_count": int(len(vm_missing_hours)),
                    "duplicate_active_hour_count": int(vm_duplicate_count),
                    "missing_hours": vm_missing_hours,
                }
            )
    if missing_usage:
        preview = missing_usage[:10]
        raise ValueError(f"Missing {len(missing_usage)} active hourly usage rows; examples={preview}")

    preempt = _read_filtered_csv(paths["preemption"], vm_ids=set(J), scenarios=set(scenario_ids))
    if preempt.empty:
        raise ValueError("No spot preemption rows for selected spot VMs")
    if "active" not in preempt and "preempted" in preempt:
        preempt["active"] = 1 - preempt["preempted"].astype(int)
    if not set(pd.to_numeric(preempt["active"], errors="raise").dropna().astype(int)).issubset({0, 1}):
        raise ValueError("Spot active path must be binary")
    availability_df = (
        preempt.groupby(["scenario_id", "vm_id", "t_hour"], as_index=False)["active"].min()
    )
    availability_idx = availability_df.set_index(["scenario_id", "vm_id", "t_hour"])["active"]
    spot_available: dict[tuple[str, int, int], int] = {}
    missing_availability = []
    for j in J:
        for xi in scenario_ids:
            for t in active[j]:
                key = (xi, j, t)
                if key not in availability_idx.index:
                    missing_availability.append(key)
                else:
                    spot_available[(j, t, xi)] = int(availability_idx.loc[key] > 0)
    if missing_availability:
        raise ValueError(f"Incomplete spot availability path; examples={missing_availability[:10]}")

    q_od = {(i, CPU): float(selected_by_id.loc[i, "q_cpu"]) for i in I}
    q_od.update({(i, MEM): float(selected_by_id.loc[i, "q_mem"]) for i in I})
    q_spot = {(j, CPU): float(selected_by_id.loc[j, "q_cpu"]) for j in J}
    q_spot.update({(j, MEM): float(selected_by_id.loc[j, "q_mem"]) for j in J})
    if min([*q_od.values(), *q_spot.values()]) <= 0:
        raise ValueError("All selected nominal VM resource requirements must be positive")

    batch = pd.read_csv(paths["batch"])
    batch["family_id"] = batch["family_id"].astype(str)
    K = batch["family_id"].tolist()
    batch_scale = target_counts["batch_candidate"] / source_counts["batch_candidate"]
    q_batch = {(r.family_id, CPU): float(r.q_cpu_B) for r in batch.itertuples()}
    q_batch.update({(r.family_id, MEM): float(r.q_mem_B) for r in batch.itertuples()})
    rho_batch = {(r.family_id, CPU): float(r.rho_cpu_B) for r in batch.itertuples()}
    rho_batch.update({(r.family_id, MEM): float(r.rho_mem_B) for r in batch.itertuples()})
    W = {r.family_id: float(r.W_k) * batch_scale for r in batch.itertuples()}
    batch_base_cpu, batch_base_cpu_audit = _canonical_scalar_column(batch, "base_cpu")
    batch_base_mem, batch_base_mem_audit = _canonical_scalar_column(batch, "base_mem")
    batch_startup_cpu, batch_startup_cpu_audit = _canonical_scalar_column(batch, "startup_cpu")
    batch_startup_mem, batch_startup_mem_audit = _canonical_scalar_column(batch, "startup_mem")
    batch_base = {CPU: batch_base_cpu, MEM: batch_base_mem}
    batch_startup = {CPU: batch_startup_cpu, MEM: batch_startup_mem}
    batch_parameter_canonicalization = {
        "base_cpu": batch_base_cpu_audit,
        "base_mem": batch_base_mem_audit,
        "startup_cpu": batch_startup_cpu_audit,
        "startup_mem": batch_startup_mem_audit,
    }
    if min(q_batch.values()) <= 0 or min(rho_batch.values()) <= 0 or min(W.values()) < 0:
        raise ValueError("Batch q/rho must be positive and scaled workload must be nonnegative")
    if min([*batch_base.values(), *batch_startup.values()]) < 0:
        raise ValueError("Batch base/startup resource consumption must be nonnegative")

    servers = pd.read_csv(paths["servers"]).head(int(exp["num_servers"])).copy()
    if len(servers) != int(exp["num_servers"]):
        raise ValueError("Not enough rows in servers.csv")
    servers["server_id"] = servers["server_id"].astype(str)
    S = servers["server_id"].tolist()
    C = {CPU: _constant_column(servers, "C_cpu"), MEM: _constant_column(servers, "C_mem")}
    E_idle = _constant_column(servers, "E_idle")
    E_cpu = _constant_column(servers, "E_cpu")
    min_on = {r.server_id: int(r.min_on_time) for r in servers.itertuples()}
    min_off = {r.server_id: int(r.min_off_time) for r in servers.itertuples()}
    if min(C.values()) <= 0 or E_idle < 0 or E_cpu < 0:
        raise ValueError("Server capacities must be positive and energy coefficients nonnegative")

    dates = [str(x) for x in energy_cfg["scenario_dates"]]
    if len(dates) != len(scenario_ids):
        raise ValueError("energy_data.scenario_dates length must equal num_scenarios")
    (
        da_matrix,
        rt_matrix,
        renewable_raw,
        solar_raw,
        wind_raw,
        price_daily_audit,
        renewable_daily_audit,
    ) = _load_hourly_energy(
        paths["prices"],
        paths["renewable"],
        dates,
        str(energy_cfg["timezone"]),
        str(energy_cfg["nyiso_zone"]),
        str(energy_cfg["nrel_site"]),
    )
    if energy_cfg.get("day_ahead_mode", "scenario_mean") != "scenario_mean":
        raise ValueError("Only day_ahead_mode=scenario_mean is currently implemented")
    if not np.isfinite(da_matrix).all() or not np.isfinite(rt_matrix).all():
        raise ValueError("NYISO hourly price matrices contain non-finite values")
    if not np.isfinite(renewable_raw).all() or (renewable_raw < 0).any():
        raise ValueError("NREL hourly renewable energy must be finite and nonnegative")
    price_floor = float(energy_cfg.get("electricity_price_floor_usd_per_kwh", 0.0))
    if price_floor < 0:
        raise ValueError("electricity_price_floor_usd_per_kwh must be nonnegative")
    da_source = da_matrix.copy()
    rt_source = rt_matrix.copy()
    da_adjusted_count = int((da_source < price_floor).sum())
    rt_adjusted_count = int((rt_source < price_floor).sum())
    da_matrix = np.maximum(da_source, price_floor)
    rt_matrix = np.maximum(rt_source, price_floor)
    p_da = {t: float(da_matrix[:, t].mean()) for t in range(horizon)}
    p_rt = {(t, xi): float(rt_matrix[n, t]) for n, xi in enumerate(scenario_ids) for t in range(horizon)}
    sell_ratio = float(energy_cfg["sell_price_ratio"])
    if sell_ratio < 0:
        raise ValueError("sell_price_ratio must be nonnegative")
    p_sell = {t: max(0.0, sell_ratio * p_da[t]) for t in range(horizon)}

    # Estimate a fixed reference demand only for capacity scaling of the NREL trace.
    ref_energy = np.zeros((len(scenario_ids), horizon), dtype=float)
    batch_cpu_avg = sum(rho_batch[(k, CPU)] * W[k] for k in K) / horizon
    batch_mem_avg = sum(rho_batch[(k, MEM)] * W[k] for k in K) / horizon
    for n, xi in enumerate(scenario_ids):
        for t in range(horizon):
            cpu = sum(load_od.get((i, CPU, t, xi), 0.0) for i in I) + sum(
                load_spot.get((j, CPU, t, xi), 0.0) for j in J
            ) + batch_cpu_avg
            mem = sum(load_od.get((i, MEM, t, xi), 0.0) for i in I) + sum(
                load_spot.get((j, MEM, t, xi), 0.0) for j in J
            ) + batch_mem_avg
            needed = min(len(S), max(1, math.ceil(max(cpu / C[CPU], mem / C[MEM]) - 1e-12)))
            ref_energy[n, t] = needed * E_idle + cpu * E_cpu
    raw_mean = float(renewable_raw.mean())
    if raw_mean <= 0:
        raise ValueError("Selected NREL dates have nonpositive average renewable energy")
    target_ratio = float(energy_cfg["renewable_to_reference_demand_ratio"])
    solar_multiplier = float(energy_cfg.get("solar_capacity_multiplier", 1.0))
    wind_multiplier = float(energy_cfg.get("wind_capacity_multiplier", 1.0))
    if target_ratio < 0 or solar_multiplier < 0 or wind_multiplier < 0:
        raise ValueError("Renewable target ratio and component capacity multipliers must be nonnegative")
    renewable_scale = target_ratio * float(ref_energy.mean()) / raw_mean
    solar_renewable = {
        (t, xi): float(solar_raw[n, t] * renewable_scale * solar_multiplier)
        for n, xi in enumerate(scenario_ids)
        for t in range(horizon)
    }
    wind_renewable = {
        (t, xi): float(wind_raw[n, t] * renewable_scale * wind_multiplier)
        for n, xi in enumerate(scenario_ids)
        for t in range(horizon)
    }
    renewable = {
        (t, xi): solar_renewable[(t, xi)] + wind_renewable[(t, xi)]
        for n, xi in enumerate(scenario_ids)
        for t in range(horizon)
    }
    for n, row in enumerate(renewable_daily_audit):
        scaled_solar = solar_raw[n] * renewable_scale * solar_multiplier
        scaled_wind = wind_raw[n] * renewable_scale * wind_multiplier
        scaled = scaled_solar + scaled_wind
        row["scaled_energy_kwh"] = {
            "min": float(scaled.min()),
            "mean": float(scaled.mean()),
            "max": float(scaled.max()),
        }
        row["scaled_solar_energy_kwh"] = _numeric_summary(scaled_solar)
        row["scaled_wind_energy_kwh"] = _numeric_summary(scaled_wind)

    ess_cfg = cfg["ess"]
    E_full = E_idle + E_cpu * C[CPU]
    ess_capacity = float(ess_cfg["capacity_ratio_to_full_server_hour"]) * len(S) * E_full
    ess_charge_max = float(ess_cfg["charge_limit_fraction"]) * ess_capacity
    ess_discharge_max = float(ess_cfg["discharge_limit_fraction"]) * ess_capacity
    eta_charge = float(ess_cfg["charge_efficiency"])
    eta_discharge = float(ess_cfg["discharge_efficiency"])
    if not (0 < eta_charge <= 1 and 0 < eta_discharge <= 1):
        raise ValueError("ESS efficiencies must be in (0, 1]")
    soc_init = float(ess_cfg["initial_soc_fraction"]) * ess_capacity

    econ = cfg["economics"]
    mean_grid = float(np.mean(list(p_da.values())))
    if mean_grid <= 0:
        raise ValueError("Mean day-ahead price must be positive for reference-price construction")
    cpu_energy_cost = mean_grid * (E_idle / C[CPU] + E_cpu)
    kappa_od = float(econ["kappa_on_demand"])
    gamma_spot = float(econ["spot_discount_ratio"])
    pi_spot = {j: gamma_spot * kappa_od * q_spot[(j, CPU)] * cpu_energy_cost for j in J}
    lambda_exc = float(econ["kappa_sla"]) * kappa_od * cpu_energy_cost
    kappa_mig = float(econ["kappa_migration"])
    if kappa_mig < 0:
        raise ValueError("kappa_migration must be nonnegative")
    migration_energy_coefficient = kappa_mig * E_full / C[MEM]
    c_mig = {i: migration_energy_coefficient for i in I}

    peak_cpu = max(
        sum(load_od.get((i, CPU, t, xi), 0.0) for i in I)
        + sum(load_spot.get((j, CPU, t, xi), 0.0) for j in J)
        for xi in scenario_ids
        for t in range(horizon)
    )
    peak_mem = max(
        sum(load_od.get((i, MEM, t, xi), 0.0) for i in I)
        + sum(load_spot.get((j, MEM, t, xi), 0.0) for j in J)
        for xi in scenario_ids
        for t in range(horizon)
    )
    lower_bound_servers = math.ceil(max(peak_cpu / C[CPU], peak_mem / C[MEM]) - 1e-12)
    if lower_bound_servers > len(S):
        raise ValueError(f"Service-load lower bound needs {lower_bound_servers} servers, only {len(S)} configured")

    risk = cfg["risk"]
    alpha = float(risk["alpha"])
    epsilon = float(risk["epsilon"])
    if not (0 < alpha < 1) or epsilon < 0:
        raise ValueError("CVaR alpha must be in (0,1) and epsilon must be nonnegative")
    instance = InstanceData(
        I=I,
        J=J,
        K=K,
        S=S,
        T=list(range(horizon)),
        Xi=scenario_ids,
        p=p,
        active_od={i: active[i] for i in I},
        active_spot={j: active[j] for j in J},
        q_od=q_od,
        q_spot=q_spot,
        load_od=load_od,
        load_spot=load_spot,
        spot_available=spot_available,
        q_batch=q_batch,
        rho_batch=rho_batch,
        W=W,
        batch_base=batch_base,
        batch_startup=batch_startup,
        C=C,
        E_idle=E_idle,
        E_cpu=E_cpu,
        min_on=min_on,
        min_off=min_off,
        p_da=p_da,
        p_rt=p_rt,
        p_sell=p_sell,
        renewable=renewable,
        solar_renewable=solar_renewable,
        wind_renewable=wind_renewable,
        ess_capacity=ess_capacity,
        ess_charge_max=ess_charge_max,
        ess_discharge_max=ess_discharge_max,
        eta_charge=eta_charge,
        eta_discharge=eta_discharge,
        soc_init=soc_init,
        pi_spot=pi_spot,
        c_mig=c_mig,
        lambda_exc=lambda_exc,
        alpha=alpha,
        epsilon=epsilon,
        source_files=[str(x) for x in paths.values()],
        metadata={
            "sampling": {
                "source_class_counts": source_counts,
                "target_class_counts": target_counts,
                "selected_class_counts": {
                    "on_demand": len(I),
                    "spot": len(J),
                    "batch_candidate": target_counts["batch_candidate"],
                },
                "eligible_complete_hourly_counts": {k: len(v) for k, v in eligible_by_class.items()},
                "ineligible_missing_pattern_examples": missing_pattern_by_class,
                "batch_workload_scale": batch_scale,
                "seed": seed,
            },
            "energy_alignment": {
                "timezone": energy_cfg["timezone"],
                "nyiso_zone": energy_cfg["nyiso_zone"],
                "nrel_site": energy_cfg["nrel_site"],
                "scenario_date_map": {str(xi): dates[n] for n, xi in enumerate(scenario_ids)},
                "day_ahead_mode": "scenario_mean",
                "day_ahead_source_stats_usd_per_kwh": {
                    "min": float(da_source.min()),
                    "mean": float(da_source.mean()),
                    "max": float(da_source.max()),
                    "negative_count": int((da_source < 0).sum()),
                    "extreme_abs_gt_1_count": int((np.abs(da_source) > 1.0).sum()),
                },
                "real_time_source_stats_usd_per_kwh": {
                    "min": float(rt_source.min()),
                    "mean": float(rt_source.mean()),
                    "max": float(rt_source.max()),
                    "negative_count": int((rt_source < 0).sum()),
                    "extreme_abs_gt_1_count": int((np.abs(rt_source) > 1.0).sum()),
                },
                "electricity_price_floor_usd_per_kwh": price_floor,
                "price_floor_adjustment": {
                    "day_ahead_adjusted_slot_count": da_adjusted_count,
                    "real_time_adjusted_slot_count": rt_adjusted_count,
                    "day_ahead_applied_stats_usd_per_kwh": _numeric_summary(da_matrix.ravel()),
                    "real_time_applied_stats_usd_per_kwh": _numeric_summary(rt_matrix.ravel()),
                },
                "renewable_scale": renewable_scale,
                "renewable_target_ratio": target_ratio,
                "solar_capacity_multiplier": solar_multiplier,
                "wind_capacity_multiplier": wind_multiplier,
                "reference_mean_energy_kwh": float(ref_energy.mean()),
                "raw_nrel_mean_energy_kwh": raw_mean,
                "raw_nrel_energy_stats_kwh": {
                    "min": float(renewable_raw.min()),
                    "mean": raw_mean,
                    "max": float(renewable_raw.max()),
                },
                "scaled_renewable_energy_stats_kwh": {
                    "min": float(min(renewable.values())),
                    "mean": float(np.mean(list(renewable.values()))),
                    "max": float(max(renewable.values())),
                },
                "scaled_solar_energy_stats_kwh": _numeric_summary(solar_renewable.values()),
                "scaled_wind_energy_stats_kwh": _numeric_summary(wind_renewable.values()),
                "effective_renewable_to_reference_demand_ratio": (
                    float(np.mean(list(renewable.values()))) / float(ref_energy.mean())
                ),
                "nyiso_daily_audit": price_daily_audit,
                "nrel_daily_audit": renewable_daily_audit,
            },
            "feasibility_diagnostics": {
                "peak_service_cpu": peak_cpu,
                "peak_service_mem": peak_mem,
                "service_load_server_lower_bound": lower_bound_servers,
                "configured_servers": len(S),
                "total_batch_cpu_volume": float(sum(rho_batch[(k, CPU)] * W[k] for k in K)),
                "total_batch_mem_volume": float(sum(rho_batch[(k, MEM)] * W[k] for k in K)),
            },
            "data_quality": {
                "selected_usage_rows": int(len(usage)),
                "selected_usage_duplicate_key_count": duplicate_key_count,
                "active_hour_coverage_summary": {
                    "selected_vm_scenario_pairs": len(selected_coverage),
                    "missing_active_hour_count": int(missing_active_hour_count),
                    "duplicate_active_hour_count": int(duplicate_active_hour_count),
                    "min_observed_active_hour_count": int(
                        min((row["observed_active_hour_count"] for row in selected_coverage), default=0)
                    ),
                    "max_observed_active_hour_count": int(
                        max((row["observed_active_hour_count"] for row in selected_coverage), default=0)
                    ),
                },
                "active_hour_coverage_by_vm_scenario": selected_coverage,
                "selected_cpu_min": float(usage["cpu_usage"].min()),
                "selected_cpu_max": float(usage["cpu_usage"].max()),
                "selected_mem_min": float(usage["mem_usage"].min()),
                "selected_mem_max": float(usage["mem_usage"].max()),
                "selected_spot_preemption_rows": int(len(preempt)),
                "scenario_probability_sum": float(sum(p.values())),
            },
            "batch_scaling": [
                {
                    "family_id": r.family_id,
                    "source_W_k": float(r.W_k),
                    "applied_scale": float(batch_scale),
                    "scaled_W_k": float(r.W_k) * batch_scale,
                }
                for r in batch.itertuples()
            ],
            "batch_parameter_canonicalization": batch_parameter_canonicalization,
            "unit_conversions": {
                "price_usd_per_mwh_to_usd_per_kwh": 0.001,
                "electricity_purchase_price_floor": (
                    "hourly DA and RT prices are floored after USD/MWh-to-USD/kWh conversion; no upper cap is applied"
                ),
                "renewable_mwh_to_kwh": 1000.0,
                "day_ahead_first_stage": "hour-wise mean across selected historical scenario dates",
                "real_time_and_renewable": "one selected historical local date mapped to one workload scenario in order",
                "sell_price": "model assumption: max(0, sell_price_ratio * day_ahead_price), not an NYISO observation",
                "renewable_global_capacity_scaling": "model assumption applied after MWh-to-kWh conversion, not an observed capacity",
                "renewable_component_scaling": "solar and wind capacity multipliers are applied after one common reference-demand scale",
            },
            "parameter_values": {
                "ess": {
                    "capacity": ess_capacity,
                    "charge_max": ess_charge_max,
                    "discharge_max": ess_discharge_max,
                    "charge_efficiency": eta_charge,
                    "discharge_efficiency": eta_discharge,
                    "initial_soc": soc_init,
                },
                "sell_price_usd_per_kwh": _numeric_summary(p_sell.values()),
                "spot_revenue": _numeric_summary(pi_spot.values()),
                "excess_penalty_lambda": lambda_exc,
                "migration_coefficient": _numeric_summary(c_mig.values()),
                "migration_coefficient_unit": "kWh per normalized memory unit migrated",
                "migration_energy_ratio_to_full_server_hour": kappa_mig,
            },
            "model_fidelity_warnings": {
                "migration_coefficient_units": (
                    "Resolved: c_mig=kappa_migration*(E_idle+E_cpu*C_cpu)/C_mem has units of kWh per "
                    "normalized memory unit migrated."
                ),
                "spot_preemption_path_usage": "Loaded and preserved for post-solve comparison; not imposed on the model.",
                "ess_exclusivity": "Not added; simultaneous charge/discharge is measured after solve.",
                "batch_startup_exactness": "Not added; false-positive startup indicators are measured after solve.",
            },
            "selected_vm_ids": {
                "on_demand": I,
                "spot": J,
            },
            "input_manifest": _file_manifest(paths.values()),
            "source_file_hashes": _file_manifest(paths.values()),
        },
    )
    return instance


def write_instance_artifacts(instance: InstanceData, run_dir: str | Path) -> None:
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "instance_summary.json").write_text(
        json.dumps(instance.summary(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "input_manifest.json").write_text(
        json.dumps(instance.metadata["input_manifest"], indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_prepared_data(instance: InstanceData, output_dir: str | Path, *, overwrite: bool = False) -> Path:
    """Write immutable, canonical model inputs atomically without altering source data."""
    destination = Path(output_dir)
    backup: Path | None = None
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Prepared-data directory already exists: {destination}")
        backup = destination.with_name(f".{destination.name}.backup-{os.getpid()}")
        if backup.exists():
            shutil.rmtree(backup)
        os.replace(destination, backup)
    temp = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True, exist_ok=False)
    try:
        manifest_rows = []
        for cls, ids in (("on_demand", instance.I), ("spot", instance.J)):
            for vm_id in ids:
                manifest_rows.append(
                    {
                        "vm_id": vm_id,
                        "class": cls,
                        "q_cpu": instance.q_od[(vm_id, CPU)] if cls == "on_demand" else instance.q_spot[(vm_id, CPU)],
                        "q_mem": instance.q_od[(vm_id, MEM)] if cls == "on_demand" else instance.q_spot[(vm_id, MEM)],
                        "arrival_hour": (instance.active_od if cls == "on_demand" else instance.active_spot)[vm_id][0],
                        "departure_hour_exclusive": (instance.active_od if cls == "on_demand" else instance.active_spot)[vm_id][-1] + 1,
                    }
                )
        pd.DataFrame(manifest_rows).to_csv(temp / "selected_vm_manifest.csv", index=False)

        for cls, ids, active, loads, filename in (
            ("on_demand", instance.I, instance.active_od, instance.load_od, "on_demand_usage.csv"),
            ("spot", instance.J, instance.active_spot, instance.load_spot, "spot_usage.csv"),
        ):
            rows = [
                {
                    "scenario_id": xi,
                    "vm_id": vm_id,
                    "t": t,
                    "cpu_usage": loads[(vm_id, CPU, t, xi)],
                    "mem_usage": loads[(vm_id, MEM, t, xi)],
                }
                for xi in instance.Xi
                for vm_id in ids
                for t in active[vm_id]
            ]
            pd.DataFrame(rows).to_csv(temp / filename, index=False)

        pd.DataFrame(
            [
                {
                    "scenario_id": xi,
                    "vm_id": j,
                    "t": t,
                    "observed_available": instance.spot_available[(j, t, xi)],
                }
                for xi in instance.Xi
                for j in instance.J
                for t in instance.active_spot[j]
            ]
        ).to_csv(temp / "spot_observed_availability.csv", index=False)

        pd.DataFrame(
            [
                {
                    "family_id": k,
                    "q_cpu_B": instance.q_batch[(k, CPU)],
                    "q_mem_B": instance.q_batch[(k, MEM)],
                    "W_k": instance.W[k],
                    "rho_cpu_B": instance.rho_batch[(k, CPU)],
                    "rho_mem_B": instance.rho_batch[(k, MEM)],
                    "base_cpu": instance.batch_base[CPU],
                    "base_mem": instance.batch_base[MEM],
                    "startup_cpu": instance.batch_startup[CPU],
                    "startup_mem": instance.batch_startup[MEM],
                }
                for k in instance.K
            ]
        ).to_csv(temp / "batch_parameters.csv", index=False)

        pd.DataFrame(
            [
                {
                    "server_id": s,
                    "C_cpu": instance.C[CPU],
                    "C_mem": instance.C[MEM],
                    "E_idle": instance.E_idle,
                    "E_cpu": instance.E_cpu,
                    "min_on_time": instance.min_on[s],
                    "min_off_time": instance.min_off[s],
                }
                for s in instance.S
            ]
        ).to_csv(temp / "server_parameters.csv", index=False)

        date_map = instance.metadata["energy_alignment"]["scenario_date_map"]
        pd.DataFrame(
            [
                {
                    "scenario_id": xi,
                    "source_local_date": date_map[str(xi)],
                    "t": t,
                    "day_ahead_price_usd_per_kwh": instance.p_da[t],
                    "real_time_price_usd_per_kwh": instance.p_rt[(t, xi)],
                    "sell_price_usd_per_kwh": instance.p_sell[t],
                    "solar_generation_kwh": instance.solar_renewable[(t, xi)],
                    "wind_generation_kwh": instance.wind_renewable[(t, xi)],
                    "renewable_generation_kwh": instance.renewable[(t, xi)],
                }
                for xi in instance.Xi
                for t in instance.T
            ]
        ).to_csv(temp / "energy_scenarios.csv", index=False)
        pd.DataFrame(
            [{"scenario_id": xi, "probability": instance.p[xi]} for xi in instance.Xi]
        ).to_csv(temp / "scenario_probabilities.csv", index=False)

        scalar = {
            "alpha": instance.alpha,
            "epsilon": instance.epsilon,
            "lambda_exc": instance.lambda_exc,
            "ess_capacity": instance.ess_capacity,
            "ess_charge_max": instance.ess_charge_max,
            "ess_discharge_max": instance.ess_discharge_max,
            "ess_charge_efficiency": instance.eta_charge,
            "ess_discharge_efficiency": instance.eta_discharge,
            "soc_init": instance.soc_init,
            "spot_revenue": instance.pi_spot,
            "migration_coefficient": instance.c_mig,
            "migration_coefficient_unit": "kWh per normalized memory unit migrated",
        }
        (temp / "scalar_parameters.json").write_text(json.dumps(scalar, indent=2), encoding="utf-8")
        canonical_files = sorted(
            p for p in temp.iterdir() if p.is_file() and p.name not in {"data_audit.json", "checksums.sha256"}
        )
        audit = instance.summary()
        audit["canonicalization_rules"] = {
            "workload_time": "Google 5-minute CPU mean and memory max, as already aggregated in vm_usage_hourly_scenarios.csv",
            "spot_hour_availability": "minimum of 5-minute active flags within each hour; diagnostic only",
            "price_time": "mean of two 30-minute LBMP rows; USD/MWh divided by 1000",
            "renewable_time": (
                "sum of two 30-minute component-energy rows; MWh multiplied by 1000, one common reference-demand "
                "scale, then independent solar/wind capacity multipliers"
            ),
            "electricity_purchase_prices": (
                "raw values are preserved in the source audit; hourly DA/RT model inputs are floored at the configured "
                "USD/kWh minimum with no upper cap"
            ),
            "day_ahead_first_stage": "hour-wise arithmetic mean across the same 10 historical scenario dates",
            "batch_scale": "same multiplicative scale for every W_k; q/rho/base/startup unchanged",
        }
        audit["canonical_file_hashes"] = _file_manifest(canonical_files)
        audit["canonical_checksum_note"] = (
            "checksums.sha256 is written after data_audit.json and contains every prepared file, "
            "including data_audit.json itself."
        )
        (temp / "data_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

        output_files = sorted(p for p in temp.iterdir() if p.is_file())
        checksum_lines = [f"{_sha256(path)}  {path.name}" for path in output_files]
        (temp / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp, destination)
        if backup is not None:
            shutil.rmtree(backup)
        return destination
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        if backup is not None and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
