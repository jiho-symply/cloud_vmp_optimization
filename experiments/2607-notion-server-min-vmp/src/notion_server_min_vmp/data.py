from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


CPU = "CPU"
MEM = "MEM"
RESOURCES = (CPU, MEM)
T5_US = 5 * 60 * 1_000_000
DEFAULT_SYNTHETIC_RNG_MODE = "joint_stream"
SYNTHETIC_RNG_MODES = frozenset({DEFAULT_SYNTHETIC_RNG_MODE, "per_vm_stable"})
PER_VM_SYNTHETIC_RNG_DOMAIN = (
    "notion_server_min_vmp.service_lognormal.per_vm.v1"
)


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
    q_batch: dict[tuple[str, str], float]
    rho_batch: dict[tuple[str, str], float]
    W: dict[str, float]
    batch_base_cpu: float
    batch_base_mem: float
    batch_startup_cpu: float
    batch_startup_mem: float
    C_cpu: float
    C_mem: float
    E_idle: float
    E_cpu: float
    min_on: int
    min_off: int
    p_rt: dict[tuple[int, int], float]
    pi_od: dict[str, float]
    pi_spot: dict[str, float]
    pi_batch: dict[str, float]
    c_mig: float
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
            "scenario_probability_sum": float(sum(self.p.values())),
            "capacity": {CPU: self.C_cpu, MEM: self.C_mem},
            "server_energy_kwh_per_slot": {
                "idle": self.E_idle,
                "per_served_cpu_unit": self.E_cpu,
            },
            "minimum_server_state_slots": {"on": self.min_on, "off": self.min_off},
            "risk": {"alpha": self.alpha, "epsilon": self.epsilon},
            "mean_rt_price_usd_per_kwh": float(np.mean(list(self.p_rt.values()))),
            "migration": {
                "coefficient": self.c_mig,
                "coefficient_unit": self.metadata.get("economics", {}).get(
                    "migration_coefficient_unit"
                ),
            },
        }


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    return config


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _required_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _stable_order(values: Iterable[str], seed: int) -> list[str]:
    def key(value: str) -> str:
        return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()

    return sorted((str(value) for value in values), key=key)


def _validated_synthetic_rng_mode(value: Any) -> str:
    if not isinstance(value, str) or value not in SYNTHETIC_RNG_MODES:
        raise ValueError(
            "workload_scenarios.synthetic_rng_mode must be one of "
            f"{sorted(SYNTHETIC_RNG_MODES)}"
        )
    return value


def _per_vm_stable_lognormal_multipliers(
    base: pd.DataFrame,
    *,
    seed: int,
    resource: str,
    scenario_id: int,
    sigma: float,
) -> np.ndarray:
    """Return draws whose stream is unaffected by other selected VMs or row order."""

    multipliers = np.empty(len(base), dtype=float)
    vm_ids = base["vm_id"].astype(str).to_numpy()
    t5_values = base["t5_day"].to_numpy(np.int64)
    for vm_id in sorted(set(vm_ids)):
        positions = np.flatnonzero(vm_ids == vm_id)
        unique_t5 = np.unique(t5_values[positions])
        key = (
            f"{PER_VM_SYNTHETIC_RNG_DOMAIN}\0{seed}\0{vm_id}\0"
            f"{resource}\0{scenario_id}"
        ).encode("utf-8")
        stable_seed = int.from_bytes(hashlib.sha256(key).digest(), "big")
        rng = np.random.Generator(np.random.PCG64(stable_seed))
        draws = rng.lognormal(
            mean=-0.5 * sigma**2,
            sigma=sigma,
            size=len(unique_t5),
        )
        multipliers[positions] = draws[
            np.searchsorted(unique_t5, t5_values[positions])
        ]
    return multipliers


def _finite_numeric(series: pd.Series, label: str, *, nonnegative: bool = True) -> pd.Series:
    numeric = pd.to_numeric(series, errors="raise").astype(float)
    values = numeric.to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains non-finite values")
    if nonnegative and (values < 0).any():
        raise ValueError(f"{label} contains negative values")
    return numeric


def _validated_coverage_us(series: pd.Series, label: str) -> pd.Series:
    """Validate observed durations without silently rounding partial windows."""

    try:
        numeric = pd.to_numeric(series, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} must contain finite integer values in (0, {T5_US}]"
        ) from exc
    values = numeric.to_numpy(float)
    if (
        not np.isfinite(values).all()
        or not np.equal(values, np.floor(values)).all()
        or (values <= 0).any()
        or (values > T5_US).any()
    ):
        raise ValueError(
            f"{label} must contain finite integer values in (0, {T5_US}]"
        )
    return numeric.astype(np.int64)


def _usage_coverage_us(frame: pd.DataFrame, label: str) -> pd.Series:
    """Return validated coverage, retaining compatibility for direct helper calls.

    Production data always enters through ``_read_scenario_zero_usage``, where
    ``coverage_us`` is mandatory.  The full-window default only preserves the
    historical programmatic helper API used by tests and downstream notebooks.
    """

    if "coverage_us" not in frame.columns:
        return pd.Series(T5_US, index=frame.index, dtype=np.int64)
    return _validated_coverage_us(frame["coverage_us"], label)


def _validated_configured_resources(
    frame: pd.DataFrame,
    label: str = "vm_requests.csv",
) -> pd.DataFrame:
    required = {"q_cpu", "q_mem"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns {sorted(missing)}")

    validated = frame.copy()
    for column in ("q_cpu", "q_mem"):
        error_message = (
            f"{label} {column} must contain finite values in (0, 1]; "
            "VMs larger than one representative server must be removed during preprocessing"
        )
        try:
            values = _finite_numeric(
                validated[column],
                f"{label} {column}",
                nonnegative=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(error_message) from exc
        if bool(values.le(0).any() or values.gt(1.0 + 1e-12).any()):
            raise ValueError(error_message)
        validated[column] = values
    return validated


def _homogeneous_numeric(frame: pd.DataFrame, column: str, *, atol: float = 1e-12) -> float:
    values = _finite_numeric(frame[column], column).to_numpy(float)
    if values.size == 0:
        raise ValueError(f"No values found for {column}")
    if not np.allclose(values, values[0], atol=atol, rtol=0.0):
        raise ValueError(f"Homogeneous baseline expected one {column}, found multiple values")
    return float(values[0])


def _integer_config_value(value: Any, label: str) -> int:
    """Return an explicitly integer configuration value without truncation."""

    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def _numeric_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "mean": float(array.mean()),
        "max": float(array.max()),
    }


def _read_scenario_zero_usage(
    path: Path,
    vm_ids: set[str],
    *,
    chunksize: int,
    sorted_by_scenario: bool,
) -> pd.DataFrame:
    required_columns = {
        "scenario_id",
        "vm_id",
        "t5_day",
        "cpu_usage",
        "mem_usage",
        "coverage_us",
    }
    available_columns = set(pd.read_csv(path, nrows=0).columns)
    missing_columns = required_columns - available_columns
    if missing_columns:
        raise ValueError(
            "Scenario usage CSV is missing columns "
            f"{sorted(missing_columns)}; coverage_us is required for partial-window weighting"
        )
    columns = sorted(required_columns)
    if "coverage_ratio" in available_columns:
        columns.append("coverage_ratio")
    parts: list[pd.DataFrame] = []

    for chunk in pd.read_csv(path, usecols=columns, dtype={"vm_id": str}, chunksize=chunksize):
        scenario_id = pd.to_numeric(chunk["scenario_id"], errors="raise").astype(int)
        has_zero = bool((scenario_id == 0).any())
        if sorted_by_scenario and not has_zero and bool((scenario_id > 0).all()):
            break

        zero = chunk.loc[scenario_id == 0].copy()
        if not zero.empty:
            zero["vm_id"] = zero["vm_id"].astype(str)
            zero = zero.loc[zero["vm_id"].isin(vm_ids)]
            if not zero.empty:
                parts.append(zero)

        # The canonical pool is sorted by scenario_id. Once a mixed 0/positive
        # chunk is reached, no later chunk can contain scenario 0.
        if sorted_by_scenario and has_zero and bool((scenario_id > 0).any()):
            break

    if not parts:
        raise ValueError("No scenario-0 usage rows were found for the requested VM candidates")

    usage = pd.concat(parts, ignore_index=True)
    usage["t5_day"] = pd.to_numeric(usage["t5_day"], errors="raise").astype(int)
    usage["cpu_usage"] = _finite_numeric(usage["cpu_usage"], "scenario-0 CPU usage")
    usage["mem_usage"] = _finite_numeric(usage["mem_usage"], "scenario-0 memory usage")
    usage["coverage_us"] = _validated_coverage_us(
        usage["coverage_us"], "scenario-0 coverage_us"
    )
    if "coverage_ratio" in usage.columns:
        coverage_ratio = _finite_numeric(
            usage["coverage_ratio"],
            "scenario-0 coverage_ratio",
            nonnegative=False,
        )
        if bool(coverage_ratio.le(0).any() or coverage_ratio.gt(1).any()):
            raise ValueError("scenario-0 coverage_ratio must be in (0, 1]")
        expected_ratio = usage["coverage_us"].to_numpy(float) / T5_US
        if not np.allclose(
            coverage_ratio.to_numpy(float),
            expected_ratio,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("scenario-0 coverage_ratio is inconsistent with coverage_us")

    key_columns = ["vm_id", "t5_day"]
    key_sizes = usage.groupby(key_columns, sort=False).size()
    duplicate_sizes = key_sizes.loc[key_sizes > 1]
    source_rows = int(len(usage))
    exact_duplicate_rows = int(source_rows - len(usage.drop_duplicates()))
    usage = usage.drop_duplicates().copy()

    # A canonical preprocessing output has one row per key.  For defensive
    # compatibility, collapse duplicate keys deterministically: CPU and memory
    # remain duration-weighted means.  Since duplicate rows
    # identify only a bucket, not covered sub-interval locations, their unique
    # bucket coverage is the maximum reported coverage rather than a sum.
    usage["_cpu_coverage_volume"] = usage["cpu_usage"] * usage["coverage_us"]
    usage["_mem_coverage_volume"] = usage["mem_usage"] * usage["coverage_us"]
    usage = (
        usage.groupby(key_columns, as_index=False, sort=True)
        .agg(
            cpu_coverage_volume=("_cpu_coverage_volume", "sum"),
            cpu_weight_us=("coverage_us", "sum"),
            mem_coverage_volume=("_mem_coverage_volume", "sum"),
            mem_weight_us=("coverage_us", "sum"),
            coverage_us=("coverage_us", "max"),
        )
        .sort_values(key_columns)
        .reset_index(drop=True)
    )
    usage["cpu_usage"] = usage["cpu_coverage_volume"] / usage["cpu_weight_us"]
    usage["mem_usage"] = usage["mem_coverage_volume"] / usage["mem_weight_us"]
    usage = usage[["vm_id", "t5_day", "cpu_usage", "mem_usage", "coverage_us"]]
    usage.attrs["coverage_audit"] = {
        "source_scenario_zero_rows": source_rows,
        "canonical_scenario_zero_rows": int(len(usage)),
        "duplicate_key_count": int(len(duplicate_sizes)),
        "duplicate_extra_row_count": int((duplicate_sizes - 1).sum()),
        "exact_duplicate_row_count": exact_duplicate_rows,
        "duplicate_coverage_policy": (
            "CPU and memory coverage-weighted means, and max coverage_us per duplicate VM/bucket key"
        ),
    }
    return usage


def _active_slots(row: pd.Series, *, slots_per_day: int, t5_per_slot: int) -> list[int]:
    arrival = int(row["arrival_t5"])
    departure = int(row["departure_t5"])
    if not (0 <= arrival < departure <= slots_per_day * t5_per_slot):
        raise ValueError(f"Invalid 5-minute active interval [{arrival}, {departure})")
    return list(range(arrival // t5_per_slot, (departure - 1) // t5_per_slot + 1))


def _eligible_service_ids(
    requests: pd.DataFrame,
    usage: pd.DataFrame,
    service_class: str,
    *,
    slots_per_day: int,
    t5_per_slot: int,
) -> list[str]:
    observations = {
        str(vm_id): set(group["t5_day"].astype(int))
        for vm_id, group in usage.groupby("vm_id", sort=False)
    }
    eligible: list[str] = []
    for _, row in requests.loc[requests["class"] == service_class].iterrows():
        vm_id = str(row["vm_id"])
        observed_t5 = observations.get(vm_id, set())
        arrival = int(row["arrival_t5"])
        departure = int(row["departure_t5"])
        observed_in_every_active_slot = True
        for slot in _active_slots(
            row,
            slots_per_day=slots_per_day,
            t5_per_slot=t5_per_slot,
        ):
            active_t5 = range(
                max(arrival, slot * t5_per_slot),
                min(departure, (slot + 1) * t5_per_slot),
            )
            if not any(t5 in observed_t5 for t5 in active_t5):
                observed_in_every_active_slot = False
                break
        if observed_in_every_active_slot:
            eligible.append(vm_id)
    return eligible


def _sampling_metadata(
    *,
    seed: int,
    source_counts: Mapping[str, int],
    eligible_od: list[str],
    eligible_spot: list[str],
    selected_od: list[str],
    selected_spot: list[str],
    selected_batch: list[str],
) -> dict[str, Any]:
    return {
        "seed": seed,
        "source_class_counts": dict(source_counts),
        "eligible_service_counts": {
            "on_demand": len(eligible_od),
            "spot": len(eligible_spot),
        },
        "selected_class_counts": {
            "on_demand": len(selected_od),
            "spot": len(selected_spot),
            "batch_jobs": len(selected_batch),
        },
        "selected_ids": {
            "on_demand": selected_od,
            "spot": selected_spot,
            "batch_jobs": selected_batch,
        },
    }


def _build_service_scenarios(
    requests: pd.DataFrame,
    usage: pd.DataFrame,
    I: list[str],
    J: list[str],
    active: dict[str, list[int]],
    Xi: list[int],
    *,
    t5_per_slot: int,
    seed: int,
    cpu_sigma: float,
    mem_sigma: float,
    synthetic_rng_mode: str = DEFAULT_SYNTHETIC_RNG_MODE,
) -> tuple[
    dict[tuple[str, str], float],
    dict[tuple[str, str], float],
    dict[tuple[str, str, int, int], float],
    dict[tuple[str, str, int, int], float],
    dict[str, Any],
]:
    synthetic_rng_mode = _validated_synthetic_rng_mode(synthetic_rng_mode)
    service_ids = I + J
    coverage_audit = dict(usage.attrs.get("coverage_audit", {}))
    selected = _validated_configured_resources(
        requests.set_index("vm_id").loc[service_ids],
        "selected service vm_requests",
    )
    base = usage.loc[usage["vm_id"].isin(service_ids)].copy()
    arrivals = base["vm_id"].map(selected["arrival_t5"])
    departures = base["vm_id"].map(selected["departure_t5"])
    base = base.loc[
        base["t5_day"].ge(arrivals) & base["t5_day"].lt(departures)
    ].copy()
    base["coverage_us"] = _usage_coverage_us(
        base, "selected service scenario-0 coverage_us"
    )
    base["t_slot"] = (base["t5_day"] // t5_per_slot).astype(int)
    if synthetic_rng_mode == "per_vm_stable":
        # Canonical row order also makes coverage-weighted floating-point
        # aggregation bitwise stable when the source CSV is reordered.
        base = base.sort_values(["vm_id", "t5_day"]).reset_index(drop=True)

    q_cpu: dict[str, float] = {}
    q_mem: dict[str, float] = {}
    for vm_id in service_ids:
        row = selected.loc[vm_id]
        q_cpu[vm_id] = float(row["q_cpu"])
        q_mem[vm_id] = float(row["q_mem"])

    base["class"] = base["vm_id"].map(selected["class"])
    base["q_cpu"] = base["vm_id"].map(q_cpu)
    base["q_mem"] = base["vm_id"].map(q_mem)
    base_cpu = base["cpu_usage"].to_numpy(float)
    base_mem = base["mem_usage"].to_numpy(float)
    q_cpu_array = base["q_cpu"].to_numpy(float)
    q_mem_array = base["q_mem"].to_numpy(float)
    rng = np.random.default_rng(seed)
    load_od: dict[tuple[str, str, int, int], float] = {}
    load_spot: dict[tuple[str, str, int, int], float] = {}
    selected_od = set(I)

    for xi in Xi:
        cpu = base_cpu.copy()
        mem = base_mem.copy()
        if xi > 0:
            if synthetic_rng_mode == "joint_stream":
                cpu *= rng.lognormal(
                    mean=-0.5 * cpu_sigma**2,
                    sigma=cpu_sigma,
                    size=cpu.size,
                )
                mem *= rng.lognormal(
                    mean=-0.5 * mem_sigma**2,
                    sigma=mem_sigma,
                    size=mem.size,
                )
            else:
                cpu *= _per_vm_stable_lognormal_multipliers(
                    base,
                    seed=seed,
                    resource=CPU,
                    scenario_id=xi,
                    sigma=cpu_sigma,
                )
                mem *= _per_vm_stable_lognormal_multipliers(
                    base,
                    seed=seed,
                    resource=MEM,
                    scenario_id=xi,
                    sigma=mem_sigma,
                )
            # Every synthetic draw is a configured-resource-bounded usage
            # realization, independent of service class. Scenario zero remains
            # the unmodified observed trace.
            cpu = np.clip(cpu, 0.0, q_cpu_array)
            mem = np.clip(mem, 0.0, q_mem_array)

        scenario = pd.DataFrame(
            {
                "vm_id": base["vm_id"].to_numpy(),
                "t_slot": base["t_slot"].to_numpy(int),
                "cpu_usage": cpu,
                "mem_usage": mem,
                # Observed duration is trace evidence and therefore does not
                # change across synthetic workload scenarios.
                "coverage_us": base["coverage_us"].to_numpy(np.int64),
            }
        )
        scenario["cpu_coverage_volume"] = (
            scenario["cpu_usage"] * scenario["coverage_us"]
        )
        scenario["mem_coverage_volume"] = (
            scenario["mem_usage"] * scenario["coverage_us"]
        )
        aggregated = scenario.groupby(["vm_id", "t_slot"], as_index=False).agg(
            cpu_coverage_volume=("cpu_coverage_volume", "sum"),
            mem_coverage_volume=("mem_coverage_volume", "sum"),
            coverage_us=("coverage_us", "sum"),
        )
        aggregated["cpu_usage"] = (
            aggregated["cpu_coverage_volume"] / aggregated["coverage_us"]
        )
        aggregated["mem_usage"] = (
            aggregated["mem_coverage_volume"] / aggregated["coverage_us"]
        )
        for row in aggregated.itertuples(index=False):
            vm_id = str(row.vm_id)
            t = int(row.t_slot)
            if t not in active[vm_id]:
                continue
            destination = load_od if vm_id in selected_od else load_spot
            destination[(vm_id, CPU, t, xi)] = float(row.cpu_usage)
            destination[(vm_id, MEM, t, xi)] = float(row.mem_usage)

    q_od = {(vm_id, CPU): q_cpu[vm_id] for vm_id in I}
    q_od.update({(vm_id, MEM): q_mem[vm_id] for vm_id in I})
    q_spot = {(vm_id, CPU): q_cpu[vm_id] for vm_id in J}
    q_spot.update({(vm_id, MEM): q_mem[vm_id] for vm_id in J})
    for loads, configured in (
        (load_od, q_od),
        (load_spot, q_spot),
    ):
        for (vm_id, resource, _t, xi), value in loads.items():
            if xi > 0 and not (0.0 <= value <= configured[(vm_id, resource)] + 1e-12):
                raise ValueError(
                    f"Synthetic {resource} usage is outside [0, q] for {vm_id}, scenario {xi}"
                )
    expected_od = sum(len(active[vm_id]) for vm_id in I) * len(Xi) * len(RESOURCES)
    expected_spot = sum(len(active[vm_id]) for vm_id in J) * len(Xi) * len(RESOURCES)
    if len(load_od) != expected_od or len(load_spot) != expected_spot:
        raise ValueError(
            "Incomplete 30-minute service scenario construction: "
            f"OD {len(load_od)}/{expected_od}, spot {len(load_spot)}/{expected_spot}"
        )
    return q_od, q_spot, load_od, load_spot, {
        "scenario_zero_policy": (
            "observed 5-minute CPU and memory aggregated by coverage_us-weighted means"
        ),
        "scenario_zero_usage_loader": coverage_audit,
        "coverage_policy": (
            "coverage_us is observed duration in microseconds, is fixed across scenarios, "
            "and weights 30-minute CPU and memory means"
        ),
        "synthetic_scenarios": Xi[1:],
        "synthetic_method": "independent mean-preserving lognormal multiplier per 5-minute VM observation",
        "synthetic_rng_mode": synthetic_rng_mode,
        "synthetic_rng_provenance": (
            {
                "bit_generator": "PCG64",
                "domain": PER_VM_SYNTHETIC_RNG_DOMAIN,
                "stream_key_fields": [
                    "seed",
                    "vm_id",
                    "resource",
                    "scenario_id",
                ],
                "within_vm_draw_order": "t5_day ascending",
            }
            if synthetic_rng_mode == "per_vm_stable"
            else {
                "bit_generator": "PCG64 via numpy.default_rng",
                "domain": "legacy joint selected-service row stream",
                "stream_key_fields": ["seed"],
                "within_vm_draw_order": "source row order",
            }
        ),
        "cpu_lognormal_sigma": cpu_sigma,
        "memory_lognormal_sigma": mem_sigma,
        "synthetic_caps": (
            "all service CPU and memory lognormal draws capped to [0, q] in synthetic scenarios"
        ),
        "cpu_quantile_inverse_cdf_fidelity": (
            "Not available in the local extract: it contains only average_usage.cpus. "
            "Lognormal perturbation is an explicit approximation to the page's quantile-summary inverse-CDF rule."
        ),
    }


def _build_batch_families(
    requests: pd.DataFrame,
    usage: pd.DataFrame,
    selected_batch_ids: list[str],
    *,
    t5_per_slot: int,
    max_families: int,
    pair_round_digits: int,
) -> tuple[
    list[str],
    dict[tuple[str, str], float],
    dict[tuple[str, str], float],
    dict[str, float],
    dict[str, Any],
]:
    if max_families <= 0:
        raise ValueError("batch.max_families must be positive")
    if t5_per_slot <= 0:
        raise ValueError("t5_per_slot must be positive")
    slot_us = t5_per_slot * T5_US
    if not selected_batch_ids:
        return [], {}, {}, {}, {
            "selected_batch_jobs": 0,
            "selected_batch_ids": [],
            "exact_configured_pair_count": 0,
            "max_families": max_families,
            "constructed_family_count": 0,
            "pair_round_digits": pair_round_digits,
            "binning_method": "no batch jobs selected",
            "workload_definition": (
                "W=sum(coverage_us)/slot_us; CPU and memory volumes="
                "sum(resource_usage*coverage_us)/slot_us"
            ),
            "slot_us": slot_us,
            "total_workload_slot_units": 0.0,
            "total_cpu_resource_slot_volume": 0.0,
            "total_memory_resource_slot_volume": 0.0,
            "families": [],
        }
    selected = _validated_configured_resources(
        requests.set_index("vm_id").loc[selected_batch_ids],
        "selected batch vm_requests",
    )
    batch_usage = usage.loc[usage["vm_id"].isin(selected_batch_ids)].copy()
    batch_usage["coverage_us"] = _usage_coverage_us(
        batch_usage, "selected batch scenario-0 coverage_us"
    )
    batch_usage["cpu_usage"] = _finite_numeric(
        batch_usage["cpu_usage"], "selected batch scenario-0 CPU usage"
    )
    batch_usage["mem_usage"] = _finite_numeric(
        batch_usage["mem_usage"], "selected batch scenario-0 memory usage"
    )
    batch_usage["cpu_coverage_volume"] = (
        batch_usage["cpu_usage"] * batch_usage["coverage_us"]
    )
    batch_usage["mem_coverage_volume"] = (
        batch_usage["mem_usage"] * batch_usage["coverage_us"]
    )
    grouped = batch_usage.groupby("vm_id", as_index=True).agg(
        coverage_us=("coverage_us", "sum"),
        cpu_coverage_volume=("cpu_coverage_volume", "sum"),
        mem_coverage_volume=("mem_coverage_volume", "sum"),
    )
    missing = sorted(set(selected_batch_ids) - set(grouped.index.astype(str)))
    if missing:
        raise ValueError(f"Selected batch jobs have no scenario-0 usage: {missing[:10]}")

    job_rows: list[dict[str, Any]] = []
    for vm_id in selected_batch_ids:
        request = selected.loc[vm_id]
        observed = grouped.loc[vm_id]
        q_cpu = float(request["q_cpu"])
        q_mem = float(request["q_mem"])
        W_job = float(observed["coverage_us"]) / slot_us
        if W_job <= 0:
            raise ValueError(f"Invalid batch parameters for {vm_id}")
        job_rows.append(
            {
                "vm_id": vm_id,
                "q_cpu": q_cpu,
                "q_mem": q_mem,
                "W": W_job,
                "cpu_volume": float(observed["cpu_coverage_volume"]) / slot_us,
                "mem_volume": float(observed["mem_coverage_volume"]) / slot_us,
                "pair_cpu": round(q_cpu, pair_round_digits),
                "pair_mem": round(q_mem, pair_round_digits),
            }
        )
    jobs = pd.DataFrame(job_rows)
    pairs = jobs[["pair_cpu", "pair_mem"]].drop_duplicates().sort_values(["pair_cpu", "pair_mem"])
    exact_family_count = int(len(pairs))
    family_count = min(max_families, exact_family_count)
    pairs = pairs.reset_index(drop=True)
    if exact_family_count <= max_families:
        pairs["family_number"] = np.arange(exact_family_count, dtype=int)
        binning_method = "exact configured CPU/MEM pairs"
    else:
        pairs["family_number"] = np.floor(
            np.arange(exact_family_count) * max_families / exact_family_count
        ).astype(int)
        binning_method = (
            "sorted configured-pair rank binned into max_families; identical pairs remain together"
        )
    jobs = jobs.merge(pairs, on=["pair_cpu", "pair_mem"], how="left", validate="many_to_one")
    jobs["family_id"] = jobs["family_number"].map(lambda number: f"batch{int(number):03d}")

    family = jobs.groupby("family_id", as_index=False, sort=True).agg(
        q_cpu=("q_cpu", "max"),
        q_mem=("q_mem", "max"),
        W=("W", "sum"),
        cpu_volume=("cpu_volume", "sum"),
        mem_volume=("mem_volume", "sum"),
        jobs=("vm_id", "size"),
    )
    family["rho_cpu"] = family["cpu_volume"] / family["W"]
    family["rho_mem"] = family["mem_volume"] / family["W"]
    K = family["family_id"].astype(str).tolist()
    q_batch = {(row.family_id, CPU): float(row.q_cpu) for row in family.itertuples()}
    q_batch.update({(row.family_id, MEM): float(row.q_mem) for row in family.itertuples()})
    rho_batch = {(row.family_id, CPU): float(row.rho_cpu) for row in family.itertuples()}
    rho_batch.update({(row.family_id, MEM): float(row.rho_mem) for row in family.itertuples()})
    W = {str(row.family_id): float(row.W) for row in family.itertuples()}
    return K, q_batch, rho_batch, W, {
        "selected_batch_jobs": len(selected_batch_ids),
        "selected_batch_ids": selected_batch_ids,
        "exact_configured_pair_count": exact_family_count,
        "max_families": max_families,
        "constructed_family_count": family_count,
        "pair_round_digits": pair_round_digits,
        "binning_method": binning_method,
        "job_family_mapping": [
            {
                "vm_id": str(row.vm_id),
                "configured_q_cpu": float(row.q_cpu),
                "configured_q_mem": float(row.q_mem),
                "family_id": str(row.family_id),
                "workload_slot_units": float(row.W),
                "cpu_resource_slot_volume": float(row.cpu_volume),
                "memory_resource_slot_volume": float(row.mem_volume),
            }
            for row in jobs.sort_values("vm_id").itertuples(index=False)
        ],
        "workload_definition": (
            "W=sum(coverage_us)/slot_us; CPU and memory volumes="
            "sum(resource_usage*coverage_us)/slot_us; family rho=family volume/family W"
        ),
        "slot_us": slot_us,
        "total_workload_slot_units": float(family["W"].sum()),
        "total_cpu_resource_slot_volume": float(family["cpu_volume"].sum()),
        "total_memory_resource_slot_volume": float(family["mem_volume"].sum()),
        "families": family.to_dict(orient="records"),
    }


def _load_rt_price_scenarios(
    path: Path,
    Xi: list[int],
    T: list[int],
    *,
    timezone: str,
    site_id: str,
    start_date: str,
    slot_minutes: int,
    multiplier: float,
    floor_usd_per_kwh: float,
    known_interpolated_raw_slots: list[str],
) -> tuple[dict[tuple[int, int], float], dict[str, Any]]:
    if path.suffix.lower() in {".parquet", ".pq"}:
        prices = pd.read_parquet(path)
    else:
        prices = pd.read_csv(path)
    site_column = "site_id" if "site_id" in prices.columns else "nyiso_zone"
    required = {"timestamp_utc", site_column, "rt_lbmp_usd_per_mwh"}
    if not required.issubset(prices.columns):
        raise ValueError(f"NYISO price input is missing columns {sorted(required - set(prices.columns))}")
    prices = prices.loc[prices[site_column].astype(str) == site_id].copy()
    if prices.empty:
        raise ValueError(f"No NYISO rows found for {site_column}={site_id}")

    timestamp_utc = pd.to_datetime(prices["timestamp_utc"], utc=True, errors="raise")
    local = timestamp_utc.dt.tz_convert(timezone)
    if (local.dt.minute % slot_minutes != 0).any():
        raise ValueError("NYISO timestamps are not aligned to the configured placement interval")
    prices["local_date"] = local.dt.strftime("%Y-%m-%d")
    prices["t_slot"] = (local.dt.hour * (60 // slot_minutes) + local.dt.minute // slot_minutes).astype(int)
    dates = pd.date_range(start_date, periods=len(Xi), freq="D").strftime("%Y-%m-%d").tolist()
    selected_interpolated_slots = [
        value
        for value in known_interpolated_raw_slots
        if str(value)[:10] in set(dates)
    ]
    selected = prices.loc[prices["local_date"].isin(dates)].copy()
    if selected.duplicated(["local_date", "t_slot"]).any():
        raise ValueError("NYISO input has duplicate local-date/slot rows")
    selected["rt_lbmp_usd_per_mwh"] = _finite_numeric(
        selected["rt_lbmp_usd_per_mwh"],
        "NYISO RT LBMP",
        nonnegative=False,
    )

    source_matrix = np.zeros((len(Xi), len(T)), dtype=float)
    daily_rows: list[dict[str, Any]] = []
    for scenario_number, (xi, date) in enumerate(zip(Xi, dates, strict=True)):
        day = selected.loc[selected["local_date"] == date].sort_values("t_slot")
        if day["t_slot"].astype(int).tolist() != T:
            raise ValueError(f"NYISO local date {date} does not contain exactly slots {T[0]}..{T[-1]}")
        source_matrix[scenario_number] = day["rt_lbmp_usd_per_mwh"].to_numpy(float) / 1000.0
        daily_rows.append(
            {
                "scenario_id": int(xi),
                "local_date": date,
                "rows": int(len(day)),
                "source_rt_usd_per_kwh": _numeric_summary(source_matrix[scenario_number]),
            }
        )
    if multiplier < 0 or not math.isfinite(multiplier):
        raise ValueError("electricity price multiplier must be finite and nonnegative")
    if floor_usd_per_kwh < 0:
        raise ValueError("electricity price floor must be nonnegative")
    multiplied = source_matrix * multiplier
    adjusted_count = int((multiplied < floor_usd_per_kwh).sum())
    applied = np.maximum(multiplied, floor_usd_per_kwh)
    p_rt = {
        (t, xi): float(applied[scenario_number, t])
        for scenario_number, xi in enumerate(Xi)
        for t in T
    }
    return p_rt, {
        "timezone": timezone,
        "site_id": site_id,
        "scenario_date_map": {str(xi): date for xi, date in zip(Xi, dates, strict=True)},
        "source_interval_minutes": slot_minutes,
        "source_rows": int(len(selected)),
        "source_rt_usd_per_kwh": _numeric_summary(source_matrix.ravel()),
        "multiplier": multiplier,
        "multiplied_rt_before_floor_usd_per_kwh": _numeric_summary(multiplied.ravel()),
        "floor_usd_per_kwh": floor_usd_per_kwh,
        "floor_adjusted_slots": adjusted_count,
        "applied_rt_usd_per_kwh": _numeric_summary(applied.ravel()),
        "known_raw_interpolation": {
            "count": len(selected_interpolated_slots),
            "local_slots": selected_interpolated_slots,
            "policy": (
                "The canonical processed NYISO table linearly interpolated these missing raw 5-minute points "
                "before taking six-row 30-minute means."
            ),
        },
        "daily_audit": daily_rows,
    }


def build_instance(
    config: Mapping[str, Any] | str | Path,
    config_path: str | Path | None = None,
) -> InstanceData:
    if isinstance(config, (str, Path)):
        resolved_config_path = Path(config).expanduser().resolve()
        cfg = load_config(resolved_config_path)
    else:
        cfg = dict(config)
        resolved_config_path = (
            Path(config_path).expanduser().resolve() if config_path is not None else None
        )

    base_dir = resolved_config_path.parent if resolved_config_path is not None else Path.cwd()
    configured_root = Path(cfg.get("workspace_root", base_dir)).expanduser()
    root = configured_root.resolve() if configured_root.is_absolute() else (base_dir / configured_root).resolve()
    experiment = cfg["experiment"]
    data_cfg = cfg["data"]
    scenario_cfg = cfg["workload_scenarios"]
    synthetic_rng_mode = _validated_synthetic_rng_mode(
        scenario_cfg.get("synthetic_rng_mode", DEFAULT_SYNTHETIC_RNG_MODE)
    )
    batch_cfg = cfg["batch"]
    server_cfg = cfg["server"]
    price_cfg = cfg["electricity_price"]

    deprecated_trace_options = {
        "minimum_5min_coverage_ratio",
        "require_contiguous_5min",
    }.intersection(data_cfg)
    if deprecated_trace_options:
        raise ValueError(
            "Deprecated trace-coverage configuration is not supported: "
            f"{sorted(deprecated_trace_options)}. Every active 30-minute slot must "
            "have at least one observed 5-minute value."
        )

    seed = int(experiment["seed"])
    horizon_hours = int(experiment["horizon_hours"])
    slot_minutes = int(experiment["slot_minutes"])
    if horizon_hours != 24 or slot_minutes != 30:
        raise ValueError("This experiment requires a 24-hour horizon with 30-minute slots")
    if 60 % slot_minutes != 0 or slot_minutes % 5 != 0:
        raise ValueError("slot_minutes must divide one hour and be divisible by five minutes")
    slots_per_day = horizon_hours * 60 // slot_minutes
    t5_per_slot = slot_minutes // 5
    T = list(range(slots_per_day))
    Xi = list(range(int(experiment["num_scenarios"])))
    if not Xi:
        raise ValueError("experiment.num_scenarios must be positive")
    p = {xi: 1.0 / len(Xi) for xi in Xi}

    configured_counts = {
        key: _integer_config_value(value, f"experiment.class_counts.{key}")
        for key, value in experiment["class_counts"].items()
    }
    allowed_count_keys = {"on_demand", "spot", "batch_jobs", "batch_candidate"}
    unknown_count_keys = set(configured_counts) - allowed_count_keys
    if unknown_count_keys or not {"on_demand", "spot"}.issubset(configured_counts):
        raise ValueError(
            "experiment.class_counts requires on_demand, spot, and batch_jobs "
            f"(batch_candidate is accepted as an alias); unknown={sorted(unknown_count_keys)}"
        )
    if "batch_jobs" in configured_counts and "batch_candidate" in configured_counts:
        if configured_counts["batch_jobs"] != configured_counts["batch_candidate"]:
            raise ValueError("batch_jobs and batch_candidate aliases specify different counts")
    batch_job_count = configured_counts.get(
        "batch_jobs", configured_counts.get("batch_candidate")
    )
    if batch_job_count is None:
        raise ValueError("experiment.class_counts is missing batch_jobs")
    class_counts = {
        "on_demand": configured_counts["on_demand"],
        "spot": configured_counts["spot"],
        "batch_candidate": int(batch_job_count),
    }
    if class_counts["on_demand"] <= 0:
        raise ValueError("experiment.class_counts.on_demand must be positive")
    if class_counts["spot"] < 0 or class_counts["batch_candidate"] < 0:
        raise ValueError("Spot and batch class counts must be nonnegative")
    num_servers = _integer_config_value(
        experiment["num_servers"], "experiment.num_servers"
    )
    if num_servers <= 0:
        raise ValueError("experiment.num_servers must be positive")

    google_dir = _resolve(root, data_cfg["google_dir"])
    paths = {
        "requests": _required_file(google_dir / data_cfg["vm_requests"]),
        "usage": _required_file(google_dir / data_cfg["usage_5min_scenarios"]),
        "servers": _required_file(google_dir / data_cfg["servers"]),
        "prices": _required_file(_resolve(root, data_cfg["nyiso_prices"])),
    }
    requests = pd.read_csv(paths["requests"], dtype={"vm_id": str})
    required_request_columns = {
        "vm_id",
        "class",
        "arrival_t5",
        "departure_t5",
        "q_cpu",
        "q_mem",
        "resource_request_cpu",
        "resource_request_mem",
    }
    missing_request_columns = required_request_columns - set(requests.columns)
    if missing_request_columns:
        raise ValueError(f"vm_requests.csv is missing columns {sorted(missing_request_columns)}")
    if requests["vm_id"].duplicated().any():
        raise ValueError("vm_requests.csv contains duplicate vm_id values")
    requests["vm_id"] = requests["vm_id"].astype(str)
    requests["class"] = requests["class"].astype(str)
    requests = _validated_configured_resources(requests)
    source_counts = requests["class"].value_counts().astype(int).to_dict()
    for service_class, count in class_counts.items():
        if source_counts.get(service_class, 0) < count:
            raise ValueError(f"Requested {count} {service_class} units, found {source_counts.get(service_class, 0)}")

    selected_batch_ids = _stable_order(
        requests.loc[requests["class"] == "batch_candidate", "vm_id"],
        seed + 2,
    )[: class_counts["batch_candidate"]]
    service_candidate_ids = set(
        requests.loc[requests["class"].isin(["on_demand", "spot"]), "vm_id"].astype(str)
    )
    usage = _read_scenario_zero_usage(
        paths["usage"],
        service_candidate_ids | set(selected_batch_ids),
        chunksize=int(data_cfg.get("csv_chunksize", 400_000)),
        sorted_by_scenario=bool(data_cfg.get("usage_sorted_by_scenario", True)),
    )
    eligible_od = _eligible_service_ids(
        requests,
        usage,
        "on_demand",
        slots_per_day=slots_per_day,
        t5_per_slot=t5_per_slot,
    )
    eligible_spot = _eligible_service_ids(
        requests,
        usage,
        "spot",
        slots_per_day=slots_per_day,
        t5_per_slot=t5_per_slot,
    )
    if len(eligible_od) < class_counts["on_demand"] or len(eligible_spot) < class_counts["spot"]:
        raise ValueError(
            "Not enough service traces with an observation in every active slot: "
            f"OD {len(eligible_od)}/{class_counts['on_demand']}, "
            f"spot {len(eligible_spot)}/{class_counts['spot']}"
        )
    I = _stable_order(eligible_od, seed)[: class_counts["on_demand"]]
    J = _stable_order(eligible_spot, seed + 1)[: class_counts["spot"]]
    selected_request_index = requests.set_index("vm_id")
    active = {
        vm_id: _active_slots(
            selected_request_index.loc[vm_id],
            slots_per_day=slots_per_day,
            t5_per_slot=t5_per_slot,
        )
        for vm_id in I + J
    }

    q_od, q_spot, load_od, load_spot, workload_metadata = _build_service_scenarios(
        requests,
        usage,
        I,
        J,
        active,
        Xi,
        t5_per_slot=t5_per_slot,
        seed=seed,
        cpu_sigma=float(scenario_cfg["cpu_lognormal_sigma"]),
        mem_sigma=float(scenario_cfg["memory_lognormal_sigma"]),
        synthetic_rng_mode=synthetic_rng_mode,
    )
    K, q_batch, rho_batch, W, batch_metadata = _build_batch_families(
        requests,
        usage,
        selected_batch_ids,
        t5_per_slot=t5_per_slot,
        max_families=int(batch_cfg["max_families"]),
        pair_round_digits=int(batch_cfg.get("configured_pair_round_digits", 12)),
    )

    servers = pd.read_csv(paths["servers"]).head(num_servers).copy()
    if len(servers) != num_servers:
        raise ValueError("Not enough server rows in servers.csv")
    if "server_id" not in servers:
        raise ValueError("servers.csv is missing server_id")
    S = servers["server_id"].astype(str).tolist()
    C_cpu = _homogeneous_numeric(servers, "C_cpu")
    C_mem = _homogeneous_numeric(servers, "C_mem")
    energy_scale = slot_minutes / float(server_cfg["source_energy_interval_minutes"])
    E_idle = _homogeneous_numeric(servers, "E_idle") * energy_scale
    E_cpu = _homogeneous_numeric(servers, "E_cpu") * energy_scale
    min_on = int(server_cfg["min_on_slots"])
    min_off = int(server_cfg["min_off_slots"])

    p_rt, price_metadata = _load_rt_price_scenarios(
        paths["prices"],
        Xi,
        T,
        timezone=str(price_cfg["timezone"]),
        site_id=str(price_cfg["site_id"]),
        start_date=str(price_cfg["scenario_start_date"]),
        slot_minutes=slot_minutes,
        multiplier=float(price_cfg.get("multiplier", 1.0)),
        floor_usd_per_kwh=float(price_cfg["floor_usd_per_kwh"]),
        known_interpolated_raw_slots=[str(value) for value in price_cfg.get("known_interpolated_raw_slots", [])],
    )

    batch_base_cpu = float(batch_cfg["base_cpu"])
    batch_base_mem = float(batch_cfg["base_mem"])
    batch_startup_cpu = float(batch_cfg["startup_cpu"])
    batch_startup_mem = float(batch_cfg["startup_mem"])
    economics_cfg = cfg["economics"]
    mean_rt_price = sum(p[xi] * p_rt[(t, xi)] for xi in Xi for t in T) / len(T)
    reference_cpu_cost = mean_rt_price * (E_idle / C_cpu + E_cpu)
    kappa_od = float(economics_cfg["kappa_on_demand"])
    gamma_spot = float(economics_cfg["spot_discount_ratio"])
    gamma_batch = float(economics_cfg["batch_discount_ratio"])
    pi_od = {vm_id: kappa_od * reference_cpu_cost * q_od[(vm_id, CPU)] for vm_id in I}
    pi_spot = {
        vm_id: gamma_spot * kappa_od * reference_cpu_cost * q_spot[(vm_id, CPU)]
        for vm_id in J
    }
    pi_batch: dict[str, float] = {}
    batch_max_throughput: dict[str, float] = {}
    for family_id in K:
        ratios = [
            q_batch[(family_id, resource)] / rho_batch[(family_id, resource)]
            for resource in RESOURCES
            if rho_batch[(family_id, resource)] > 0
        ]
        if not ratios:
            raise ValueError(f"Batch family {family_id} has zero CPU and memory consumption")
        max_throughput = min(ratios)
        batch_max_throughput[family_id] = max_throughput
        pi_batch[family_id] = (
            gamma_batch * kappa_od * reference_cpu_cost * q_batch[(family_id, CPU)] / max_throughput
        )

    migration_cfg = cfg["migration"]
    c_mig = float(migration_cfg["coefficient"])
    if not math.isfinite(c_mig) or c_mig < 0:
        raise ValueError("migration.coefficient must be finite and nonnegative")
    migration_unit = "kWh per normalized memory unit migrated"

    risk_cfg = cfg["risk"]
    raw_lineage = [
        _resolve(root, value)
        for value in data_cfg.get("additional_source_files", [])
        if _resolve(root, value).is_file()
    ]
    source_files = [str(path) for path in [*paths.values(), *raw_lineage]]
    metadata = {
        "time": {
            "horizon_hours": horizon_hours,
            "slot_minutes": slot_minutes,
            "periods": slots_per_day,
            "five_minute_rows_per_slot": t5_per_slot,
        },
        "sampling": _sampling_metadata(
            seed=seed,
            source_counts=source_counts,
            eligible_od=eligible_od,
            eligible_spot=eligible_spot,
            selected_od=I,
            selected_spot=J,
            selected_batch=selected_batch_ids,
        ),
        "workload_scenarios": workload_metadata,
        "batch": batch_metadata,
        "server": {
            "source_energy_interval_minutes": float(server_cfg["source_energy_interval_minutes"]),
            "applied_energy_scale": energy_scale,
            "energy_unit": "kWh per 30-minute model slot",
            "batch_overheads": {
                "base_cpu": batch_base_cpu,
                "base_mem": batch_base_mem,
                "startup_cpu": batch_startup_cpu,
                "startup_mem": batch_startup_mem,
            },
        },
        "electricity_price": price_metadata,
        "economics": {
            "mean_rt_price_usd_per_kwh": mean_rt_price,
            "reference_cpu_cost_usd_per_slot": reference_cpu_cost,
            "kappa_on_demand": kappa_od,
            "spot_discount_ratio": gamma_spot,
            "batch_discount_ratio": gamma_batch,
            "batch_max_throughput": batch_max_throughput,
            "migration_coefficient": c_mig,
            "migration_coefficient_unit": migration_unit,
        },
        "fidelity_warnings": {
            "workload_class_proxy": (
                "Google priority/scheduler labels are proxies for cloud on-demand, spot, and batch classes."
            ),
            "cpu_inverse_cdf": workload_metadata["cpu_quantile_inverse_cdf_fidelity"],
            "synthetic_joint_scenarios": (
                "Scenario 0 is the only workload scenario and is paired with one historical NYISO date; "
                "it is not a jointly observed workload-price outcome."
                if len(Xi) == 1
                else (
                    f"Scenario 0 is one Google day; scenarios 1-{len(Xi) - 1} are synthetic. "
                    f"They are paired by index with {len(Xi)} historical NYISO dates and do not represent "
                    "jointly observed workload-price outcomes."
                )
            ),
            "nyiso_interpolation": price_metadata["known_raw_interpolation"],
        },
    }
    instance = InstanceData(
        I=I,
        J=J,
        K=K,
        S=S,
        T=T,
        Xi=Xi,
        p=p,
        active_od={vm_id: active[vm_id] for vm_id in I},
        active_spot={vm_id: active[vm_id] for vm_id in J},
        q_od=q_od,
        q_spot=q_spot,
        load_od=load_od,
        load_spot=load_spot,
        q_batch=q_batch,
        rho_batch=rho_batch,
        W=W,
        batch_base_cpu=batch_base_cpu,
        batch_base_mem=batch_base_mem,
        batch_startup_cpu=batch_startup_cpu,
        batch_startup_mem=batch_startup_mem,
        C_cpu=C_cpu,
        C_mem=C_mem,
        E_idle=E_idle,
        E_cpu=E_cpu,
        min_on=min_on,
        min_off=min_off,
        p_rt=p_rt,
        pi_od=pi_od,
        pi_spot=pi_spot,
        pi_batch=pi_batch,
        c_mig=c_mig,
        alpha=float(risk_cfg["alpha"]),
        epsilon=float(risk_cfg["epsilon"]),
        source_files=source_files,
        metadata=metadata,
    )
    validate_instance(instance)
    return instance


def validate_instance(data: InstanceData) -> None:
    for label, values in (
        ("I", data.I),
        ("J", data.J),
        ("K", data.K),
        ("S", data.S),
        ("T", data.T),
        ("Xi", data.Xi),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"{label} contains duplicate values")
    if not data.T or data.T != list(range(len(data.T))):
        raise ValueError("T must be a nonempty contiguous zero-based range")
    if not data.Xi or data.Xi != list(range(len(data.Xi))):
        raise ValueError("Xi must be a nonempty contiguous zero-based range")
    if set(data.p) != set(data.Xi) or any(value < 0 for value in data.p.values()):
        raise ValueError("Scenario probabilities do not match Xi")
    if not math.isclose(sum(data.p.values()), 1.0, abs_tol=1e-12, rel_tol=0.0):
        raise ValueError("Scenario probabilities must sum to one")

    for vm_ids, active_map, q, loads, label in (
        (data.I, data.active_od, data.q_od, data.load_od, "on-demand"),
        (data.J, data.active_spot, data.q_spot, data.load_spot, "spot"),
    ):
        if set(active_map) != set(vm_ids):
            raise ValueError(f"{label} active-period keys do not match its VM set")
        expected_load_keys: set[tuple[str, str, int, int]] = set()
        for vm_id in vm_ids:
            periods = active_map[vm_id]
            if not periods or periods != sorted(set(periods)) or not set(periods).issubset(data.T):
                raise ValueError(f"Invalid active periods for {label} VM {vm_id}")
            for resource in RESOURCES:
                q_value = q.get((vm_id, resource))
                if (
                    q_value is None
                    or not math.isfinite(q_value)
                    or q_value <= 0
                    or q_value > 1.0 + 1e-12
                ):
                    raise ValueError(f"Invalid q for {label} VM {vm_id}, resource {resource}")
                expected_load_keys.update(
                    (vm_id, resource, t, xi) for t in periods for xi in data.Xi
                )
        if set(loads) != expected_load_keys:
            missing = list(expected_load_keys - set(loads))[:5]
            extra = list(set(loads) - expected_load_keys)[:5]
            raise ValueError(f"{label} load keys are incomplete; missing={missing}, extra={extra}")
        for (vm_id, resource, _t, xi), value in loads.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {label} load value")
            if xi > 0 and value > q[(vm_id, resource)] + 1e-10:
                raise ValueError(
                    f"Synthetic {label} {resource} usage exceeds q for {vm_id}"
                )

    if set(data.W) != set(data.K):
        raise ValueError("Batch W keys do not match K")
    for family_id in data.K:
        if not math.isfinite(data.W[family_id]) or data.W[family_id] <= 0:
            raise ValueError(f"Invalid W for {family_id}")
        for resource in RESOURCES:
            q_value = data.q_batch.get((family_id, resource))
            rho_value = data.rho_batch.get((family_id, resource))
            if (
                q_value is None
                or not math.isfinite(q_value)
                or q_value <= 0
                or q_value > 1.0 + 1e-12
            ):
                raise ValueError(f"Invalid batch q for {family_id}, {resource}")
            if rho_value is None or not math.isfinite(rho_value) or rho_value < 0:
                raise ValueError(f"Invalid batch rho for {family_id}, {resource}")

    scalar_nonnegative = {
        "batch_base_cpu": data.batch_base_cpu,
        "batch_base_mem": data.batch_base_mem,
        "batch_startup_cpu": data.batch_startup_cpu,
        "batch_startup_mem": data.batch_startup_mem,
        "E_idle": data.E_idle,
        "E_cpu": data.E_cpu,
    }
    if any(not math.isfinite(value) or value < 0 for value in scalar_nonnegative.values()):
        raise ValueError(f"Invalid nonnegative scalar parameter: {scalar_nonnegative}")
    if not math.isclose(data.batch_base_cpu, 0.0, abs_tol=1e-12, rel_tol=0.0):
        raise ValueError("Batch base CPU overhead must be structurally zero")
    if not math.isclose(data.batch_startup_mem, 0.0, abs_tol=1e-12, rel_tol=0.0):
        raise ValueError("Batch startup memory overhead must be structurally zero")
    if data.C_cpu <= 0 or data.C_mem <= 0 or data.min_on <= 0 or data.min_off <= 0:
        raise ValueError("Server capacity and minimum on/off slots must be positive")
    expected_price_keys = {(t, xi) for t in data.T for xi in data.Xi}
    if set(data.p_rt) != expected_price_keys:
        raise ValueError("RT price keys are incomplete")
    if any(not math.isfinite(value) or value < 0 for value in data.p_rt.values()):
        raise ValueError("Applied RT prices must be finite and nonnegative")
    for mapping, expected, label in (
        (data.pi_od, set(data.I), "pi_od"),
        (data.pi_spot, set(data.J), "pi_spot"),
        (data.pi_batch, set(data.K), "pi_batch"),
    ):
        if set(mapping) != expected:
            raise ValueError(f"{label} keys do not match their set")
        if any(not math.isfinite(value) or value < 0 for value in mapping.values()):
            raise ValueError(f"{label} contains invalid values")
    if not math.isfinite(data.c_mig) or data.c_mig < 0:
        raise ValueError("c_mig must be finite and nonnegative")
    if not (0 < data.alpha < 1) or data.epsilon < 0:
        raise ValueError("Risk parameters require alpha in (0,1) and epsilon >= 0")
