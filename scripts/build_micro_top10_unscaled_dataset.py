#!/usr/bin/env python3
"""Build an unscaled top-average-usage Google 2019 micro data set.

The canonical processed data is treated as read-only.  Within each workload
class, eligible candidates are ranked by

    coverage-weighted mean CPU usage + coverage-weighted mean memory usage.

The top ``count_per_class`` rows are copied.  Configured resources, observed
scenario-0 usage, requests, lifecycle fields, and provenance columns are
preserved exactly: there is no target-q scaling, usage rescaling, memory
pressure transform, or minimum-server guarantee.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


T5_US = 5 * 60 * 1_000_000
T5_PER_SLOT = 6
SLOTS_PER_DAY = 48
DEFAULT_SOURCE_DIR = Path(
    "data/processed/notion_toy_google2019_v5_mode_capacity_bounded"
)
DEFAULT_OUTPUT_DIR = Path(
    "data/processed/notion_toy_google2019_micro_top10_unscaled_v1"
)
SERVICE_CLASSES = ("on_demand", "spot")
ALL_CLASSES = ("on_demand", "spot", "batch_candidate")
CLASS_ORDER = {name: position for position, name in enumerate(ALL_CLASSES)}


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path, *, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _finite_numeric(
    values: pd.Series,
    label: str,
    *,
    nonnegative: bool = True,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise").astype(float)
    array = numeric.to_numpy(float)
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values")
    if nonnegative and bool((array < 0).any()):
        raise ValueError(f"{label} contains negative values")
    return numeric


def _numeric_summary(values: pd.Series) -> dict[str, float | int]:
    numeric = _finite_numeric(values, values.name or "values")
    return {
        "count": int(len(numeric)),
        "minimum": float(numeric.min()),
        "p10": float(numeric.quantile(0.10)),
        "p25": float(numeric.quantile(0.25)),
        "median": float(numeric.median()),
        "p75": float(numeric.quantile(0.75)),
        "p90": float(numeric.quantile(0.90)),
        "maximum": float(numeric.max()),
        "mean": float(numeric.mean()),
    }


def _read_scenario_zero(path: Path, *, chunksize: int) -> pd.DataFrame:
    """Read only canonical scenario zero from a scenario-sorted CSV."""

    required = {
        "scenario_id",
        "vm_id",
        "t5_day",
        "cpu_usage",
        "mem_usage",
        "coverage_us",
    }
    columns = set(pd.read_csv(path, nrows=0).columns)
    missing = required - columns
    if missing:
        raise ValueError(f"{path.name} is missing columns {sorted(missing)}")

    parts: list[pd.DataFrame] = []
    last_scenario = -1
    for chunk in pd.read_csv(path, dtype={"vm_id": str}, chunksize=chunksize):
        scenario = pd.to_numeric(chunk["scenario_id"], errors="raise").astype(int)
        array = scenario.to_numpy(int)
        if bool((scenario < 0).any()):
            raise ValueError("scenario_id must be nonnegative")
        if array.size and (
            array[0] < last_scenario or bool((np.diff(array) < 0).any())
        ):
            raise ValueError(
                "Canonical usage must be sorted by scenario_id so scenario 0 "
                "can be read without loading later scenarios"
            )
        if array.size:
            last_scenario = int(array[-1])
        zero = chunk.loc[scenario.eq(0)].copy()
        if not zero.empty:
            parts.append(zero)
        if bool((scenario > 0).any()):
            break

    if not parts:
        raise ValueError("No scenario-0 usage rows found")
    usage = pd.concat(parts, ignore_index=True)
    usage["scenario_id"] = 0
    usage["vm_id"] = usage["vm_id"].astype(str)
    usage["t5_day"] = pd.to_numeric(usage["t5_day"], errors="raise").astype(int)
    if bool(usage["t5_day"].lt(0).any() or usage["t5_day"].ge(288).any()):
        raise ValueError("scenario-0 t5_day must be in [0, 288)")
    for resource in ("cpu_usage", "mem_usage"):
        usage[resource] = _finite_numeric(usage[resource], resource)
    coverage = pd.to_numeric(usage["coverage_us"], errors="raise").astype(float)
    if (
        not np.isfinite(coverage.to_numpy(float)).all()
        or bool(coverage.le(0).any())
        or bool(coverage.gt(T5_US).any())
        or not np.equal(coverage, np.floor(coverage)).all()
    ):
        raise ValueError(f"coverage_us must contain integers in (0, {T5_US}]")
    usage["coverage_us"] = coverage.astype(np.int64)
    if usage.duplicated(["vm_id", "t5_day"]).any():
        duplicates = int(
            usage.duplicated(["vm_id", "t5_day"], keep=False).sum()
        )
        raise ValueError(
            "Canonical scenario zero must have one row per VM/t5 bucket; "
            f"found {duplicates} duplicate rows"
        )
    return usage


def _coverage_weighted_summaries(usage: pd.DataFrame) -> pd.DataFrame:
    scored = usage[
        ["vm_id", "cpu_usage", "mem_usage", "coverage_us"]
    ].copy()
    scored["cpu_volume"] = scored["cpu_usage"] * scored["coverage_us"]
    scored["mem_volume"] = scored["mem_usage"] * scored["coverage_us"]
    grouped = scored.groupby("vm_id", as_index=False, sort=True).agg(
        observed_coverage_us=("coverage_us", "sum"),
        observed_5min_rows=("t5_day", "size")
        if "t5_day" in scored.columns
        else ("coverage_us", "size"),
        cpu_volume=("cpu_volume", "sum"),
        mem_volume=("mem_volume", "sum"),
    )
    grouped["coverage_weighted_avg_cpu_usage"] = (
        grouped["cpu_volume"] / grouped["observed_coverage_us"]
    )
    grouped["coverage_weighted_avg_mem_usage"] = (
        grouped["mem_volume"] / grouped["observed_coverage_us"]
    )
    return grouped.drop(columns=["cpu_volume", "mem_volume"])


def _active_slots(row: pd.Series) -> range:
    arrival = int(row["arrival_t5"])
    departure = int(row["departure_t5"])
    if not 0 <= arrival < departure <= 288:
        raise ValueError(
            f"Invalid active interval for {row['vm_id']}: [{arrival}, {departure})"
        )
    return range(arrival // T5_PER_SLOT, (departure - 1) // T5_PER_SLOT + 1)


def _eligible_service_ids(
    requests: pd.DataFrame,
    usage: pd.DataFrame,
    service_class: str,
) -> set[str]:
    observations = {
        str(vm_id): set(group["t5_day"].astype(int))
        for vm_id, group in usage.groupby("vm_id", sort=False)
    }
    eligible: set[str] = set()
    for _, row in requests.loc[
        requests["class"].eq(service_class)
    ].iterrows():
        vm_id = str(row["vm_id"])
        arrival = int(row["arrival_t5"])
        departure = int(row["departure_t5"])
        observed = observations.get(vm_id, set())
        if all(
            any(
                t5 in observed
                for t5 in range(
                    max(arrival, slot * T5_PER_SLOT),
                    min(departure, (slot + 1) * T5_PER_SLOT),
                )
            )
            for slot in _active_slots(row)
        ):
            eligible.add(vm_id)
    return eligible


def _select_top_by_average(
    requests: pd.DataFrame,
    summaries: pd.DataFrame,
    eligible: dict[str, set[str]],
    *,
    count_per_class: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    candidates = requests.merge(
        summaries, on="vm_id", how="inner", validate="one_to_one"
    )
    candidates["selection_score_avg_cpu_plus_mem"] = (
        candidates["coverage_weighted_avg_cpu_usage"]
        + candidates["coverage_weighted_avg_mem_usage"]
    )

    selected_parts: list[pd.DataFrame] = []
    candidate_parts: list[pd.DataFrame] = []
    selected_ids: dict[str, list[str]] = {}
    usage_ids = set(summaries["vm_id"].astype(str))

    for vm_class in ALL_CLASSES:
        frame = candidates.loc[candidates["class"].eq(vm_class)].copy()
        if vm_class in SERVICE_CLASSES:
            frame = frame.loc[frame["vm_id"].isin(eligible[vm_class])]
            eligibility_rule = (
                "at least one observed 5-minute row in every active "
                "30-minute slot"
            )
        else:
            frame = frame.loc[frame["vm_id"].isin(usage_ids)]
            eligibility_rule = "at least one positive-coverage scenario-0 row"
        frame = frame.sort_values(
            ["selection_score_avg_cpu_plus_mem", "vm_id"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        if len(frame) < count_per_class:
            raise ValueError(
                f"Need {count_per_class} eligible {vm_class} rows, "
                f"found {len(frame)}"
            )
        frame["eligible_rank_within_class"] = np.arange(1, len(frame) + 1)
        frame["eligible_candidate_count"] = len(frame)
        frame["eligibility_rule"] = eligibility_rule
        selected_class = frame.head(count_per_class).copy()
        selected_class["selection_rank_within_class"] = np.arange(
            1, count_per_class + 1
        )
        selected_ids[vm_class] = selected_class["vm_id"].astype(str).tolist()
        selected_parts.append(selected_class)
        candidate_parts.append(frame)

    selected = pd.concat(selected_parts, ignore_index=True)
    all_candidates = pd.concat(candidate_parts, ignore_index=True)
    return selected, all_candidates, selected_ids


def _aggregate_30min(
    usage: pd.DataFrame,
    vm_ids: Iterable[str],
) -> pd.DataFrame:
    selected = usage.loc[usage["vm_id"].isin(set(vm_ids))].copy()
    selected["t"] = (selected["t5_day"] // T5_PER_SLOT).astype(int)
    for resource in ("cpu", "mem"):
        selected[f"{resource}_volume"] = (
            selected[f"{resource}_usage"] * selected["coverage_us"]
        )
    grouped = selected.groupby(
        ["vm_id", "t"], as_index=False, sort=True
    ).agg(
        coverage_us=("coverage_us", "sum"),
        cpu_volume=("cpu_volume", "sum"),
        mem_volume=("mem_volume", "sum"),
    )
    grouped["cpu_usage"] = (
        grouped["cpu_volume"] / grouped["coverage_us"]
    )
    grouped["mem_usage"] = (
        grouped["mem_volume"] / grouped["coverage_us"]
    )
    return grouped


def _first_fit_decreasing_bins(
    items: list[tuple[float, float]],
    *,
    tolerance: float = 1e-12,
) -> int:
    ordered = sorted(
        items, key=lambda pair: (max(pair), sum(pair)), reverse=True
    )
    bins: list[list[float]] = []
    for cpu, mem in ordered:
        for load in bins:
            if (
                load[0] + cpu <= 1.0 + tolerance
                and load[1] + mem <= 1.0 + tolerance
            ):
                load[0] += cpu
                load[1] += mem
                break
        else:
            bins.append([cpu, mem])
    return len(bins)


def _fits_two_dimensional_bins(
    items: list[tuple[float, float]],
    num_bins: int,
    *,
    tolerance: float = 1e-12,
) -> bool:
    """Return whether a small CPU/MEM item set fits exactly in num_bins."""

    if not items:
        return True
    if num_bins <= 0:
        return False
    if any(
        cpu > 1.0 + tolerance or mem > 1.0 + tolerance
        for cpu, mem in items
    ):
        return False
    if sum(cpu for cpu, _ in items) > num_bins + tolerance:
        return False
    if sum(mem for _, mem in items) > num_bins + tolerance:
        return False

    ordered = sorted(
        items, key=lambda pair: (max(pair), sum(pair)), reverse=True
    )
    bins = [[0.0, 0.0] for _ in range(num_bins)]

    def place(position: int) -> bool:
        if position == len(ordered):
            return True
        cpu, mem = ordered[position]
        seen: set[tuple[float, float]] = set()
        for load in bins:
            signature = (round(load[0], 12), round(load[1], 12))
            if signature in seen:
                continue
            seen.add(signature)
            if (
                load[0] + cpu <= 1.0 + tolerance
                and load[1] + mem <= 1.0 + tolerance
            ):
                load[0] += cpu
                load[1] += mem
                if place(position + 1):
                    return True
                load[0] -= cpu
                load[1] -= mem
            if signature == (0.0, 0.0):
                break
        return False

    return place(0)


def _minimum_two_dimensional_bins(
    items: list[tuple[float, float]],
) -> int:
    if not items:
        return 0
    lower_bound = max(
        1,
        int(math.ceil(sum(cpu for cpu, _ in items) - 1e-12)),
        int(math.ceil(sum(mem for _, mem in items) - 1e-12)),
    )
    for num_bins in range(lower_bound, len(items) + 1):
        if _fits_two_dimensional_bins(items, num_bins):
            return num_bins
    raise AssertionError("Every item bounded by one server must fit alone")


def _service_packing_diagnostics(
    requests: pd.DataFrame,
    usage: pd.DataFrame,
) -> dict[str, Any]:
    service_ids = set(
        requests.loc[
            requests["class"].isin(SERVICE_CLASSES), "vm_id"
        ].astype(str)
    )
    aggregated = _aggregate_30min(usage, service_ids)
    rows: list[dict[str, Any]] = []
    for t in range(SLOTS_PER_DAY):
        frame = aggregated.loc[aggregated["t"].eq(t)]
        items = list(
            frame[["cpu_usage", "mem_usage"]].itertuples(
                index=False, name=None
            )
        )
        cpu_sum = float(frame["cpu_usage"].sum())
        mem_sum = float(frame["mem_usage"].sum())
        lower_bound = max(
            int(math.ceil(cpu_sum - 1e-12)),
            int(math.ceil(mem_sum - 1e-12)),
        )
        rows.append(
            {
                "t": t,
                "active_vms": int(frame["vm_id"].nunique()),
                "cpu_sum": cpu_sum,
                "mem_sum": mem_sum,
                "aggregate_lower_bound_servers": lower_bound,
                "first_fit_decreasing_servers": (
                    _first_fit_decreasing_bins(items) if items else 0
                ),
                "exact_minimum_servers": (
                    _minimum_two_dimensional_bins(items) if items else 0
                ),
            }
        )
    table = pd.DataFrame(rows)
    return {
        "method": (
            "scenario-0 coverage-weighted 30-minute actual usage; "
            "two-dimensional first-fit-decreasing is a diagnostic upper bound"
        ),
        "maximum_active_vms": int(table["active_vms"].max()),
        "mean_active_vms": float(table["active_vms"].mean()),
        "maximum_aggregate_cpu": float(table["cpu_sum"].max()),
        "maximum_aggregate_memory": float(table["mem_sum"].max()),
        "maximum_aggregate_lower_bound_servers": int(
            table["aggregate_lower_bound_servers"].max()
        ),
        "maximum_first_fit_decreasing_servers": int(
            table["first_fit_decreasing_servers"].max()
        ),
        "maximum_exact_minimum_servers": int(
            table["exact_minimum_servers"].max()
        ),
        "slots": table.to_dict(orient="records"),
    }


def _configured_q_packing_diagnostics(
    requests: pd.DataFrame,
) -> dict[str, Any]:
    def static(
        frame: pd.DataFrame,
        *,
        compute_exact: bool = True,
    ) -> dict[str, Any]:
        items = list(
            frame[["q_cpu", "q_mem"]].itertuples(index=False, name=None)
        )
        cpu_sum = float(frame["q_cpu"].sum())
        mem_sum = float(frame["q_mem"].sum())
        return {
            "units": int(len(frame)),
            "sum_q_cpu": cpu_sum,
            "sum_q_mem": mem_sum,
            "aggregate_lower_bound_servers": max(
                int(math.ceil(cpu_sum - 1e-12)),
                int(math.ceil(mem_sum - 1e-12)),
            ),
            "first_fit_decreasing_servers_if_all_simultaneous": (
                _first_fit_decreasing_bins(items)
            ),
            "exact_minimum_servers_if_all_simultaneous": (
                _minimum_two_dimensional_bins(items)
                if compute_exact
                else None
            ),
        }

    def active(frame: pd.DataFrame) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for t in range(SLOTS_PER_DAY):
            present = frame.loc[
                frame.apply(lambda row: t in _active_slots(row), axis=1)
            ]
            items = list(
                present[["q_cpu", "q_mem"]].itertuples(
                    index=False, name=None
                )
            )
            cpu_sum = float(present["q_cpu"].sum())
            mem_sum = float(present["q_mem"].sum())
            rows.append(
                {
                    "t": t,
                    "active_units": int(len(present)),
                    "sum_q_cpu": cpu_sum,
                    "sum_q_mem": mem_sum,
                    "aggregate_lower_bound_servers": max(
                        int(math.ceil(cpu_sum - 1e-12)),
                        int(math.ceil(mem_sum - 1e-12)),
                    ),
                    "exact_minimum_servers": (
                        _minimum_two_dimensional_bins(items)
                        if items
                        else 0
                    ),
                }
            )
        table = pd.DataFrame(rows)
        return {
            "maximum_active_units": int(table["active_units"].max()),
            "maximum_sum_q_cpu": float(table["sum_q_cpu"].max()),
            "maximum_sum_q_mem": float(table["sum_q_mem"].max()),
            "maximum_aggregate_lower_bound_servers": int(
                table["aggregate_lower_bound_servers"].max()
            ),
            "maximum_exact_minimum_servers": int(
                table["exact_minimum_servers"].max()
            ),
            "slots": rows,
        }

    od = requests.loc[requests["class"].eq("on_demand")]
    spot = requests.loc[requests["class"].eq("spot")]
    batch = requests.loc[requests["class"].eq("batch_candidate")]
    service = requests.loc[requests["class"].isin(SERVICE_CLASSES)]
    return {
        "interpretation": (
            "Configured q packing is a size-envelope diagnostic. The current "
            "service capacity constraints use scenario usage, not q reservation."
        ),
        "static_all_simultaneous": {
            "on_demand": static(od),
            "spot": static(spot),
            "batch_candidate": static(batch),
            # Twenty-item static packing ignores lifecycle and can be
            # combinatorially expensive. The lifecycle-aware service envelope
            # below retains the exact check on the actually active set.
            "service": static(service, compute_exact=False),
        },
        "active_service_envelope": {
            "on_demand": active(od),
            "spot": active(spot),
            "service": active(service),
        },
    }


def _batch_diagnostics(
    requests: pd.DataFrame,
    usage: pd.DataFrame,
) -> dict[str, Any]:
    batch_ids = set(
        requests.loc[
            requests["class"].eq("batch_candidate"), "vm_id"
        ].astype(str)
    )
    frame = usage.loc[usage["vm_id"].isin(batch_ids)].copy()
    slot_us = T5_PER_SLOT * T5_US
    workload = float(frame["coverage_us"].sum()) / slot_us
    cpu_volume = float(
        (frame["cpu_usage"] * frame["coverage_us"]).sum()
    ) / slot_us
    mem_volume = float(
        (frame["mem_usage"] * frame["coverage_us"]).sum()
    ) / slot_us
    return {
        "selected_source_episode_count": len(batch_ids),
        "workload_slot_units": workload,
        "cpu_resource_slot_volume": cpu_volume,
        "memory_resource_slot_volume": mem_volume,
        "coverage_policy": (
            "W=sum(coverage_us)/30min; resource volume="
            "sum(usage*coverage_us)/30min"
        ),
        "model_loader_family_policy": (
            "configured q_cpu/q_mem pairs are sorted and rank-binned into "
            "at most max_families; family q is the componentwise maximum "
            "and family rho is total resource volume divided by W"
        ),
    }


def _validate_output(
    source_requests: pd.DataFrame,
    source_usage: pd.DataFrame,
    output_requests: pd.DataFrame,
    output_usage: pd.DataFrame,
    manifest: pd.DataFrame,
    selected_ids: dict[str, list[str]],
    *,
    count_per_class: int,
) -> dict[str, Any]:
    expected_counts = {
        vm_class: count_per_class for vm_class in ALL_CLASSES
    }
    counts = (
        output_requests["class"].value_counts().astype(int).to_dict()
    )
    if counts != expected_counts:
        raise ValueError(f"Unexpected derived class counts: {counts}")
    if output_requests["vm_id"].duplicated().any():
        raise ValueError("Derived vm_requests.csv has duplicate vm_id")
    if output_usage["scenario_id"].astype(int).unique().tolist() != [0]:
        raise ValueError("Derived usage must contain scenario 0 only")
    if output_usage.duplicated(["vm_id", "t5_day"]).any():
        raise ValueError("Derived usage has duplicate VM/t5 keys")

    expected_ids = set().union(*map(set, selected_ids.values()))
    if set(output_requests["vm_id"].astype(str)) != expected_ids:
        raise ValueError("Derived request IDs differ from selected IDs")
    if set(output_usage["vm_id"].astype(str)) != expected_ids:
        raise ValueError("Derived usage IDs differ from selected IDs")

    source_request_subset = source_requests.loc[
        source_requests["vm_id"].isin(expected_ids)
    ].sort_values("vm_id").reset_index(drop=True)
    output_request_subset = output_requests.sort_values(
        "vm_id"
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        output_request_subset[source_request_subset.columns],
        source_request_subset,
        check_dtype=False,
        check_exact=True,
    )

    source_usage_subset = source_usage.loc[
        source_usage["vm_id"].isin(expected_ids)
    ].sort_values(["vm_id", "t5_day"]).reset_index(drop=True)
    output_usage_subset = output_usage.sort_values(
        ["vm_id", "t5_day"]
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        output_usage_subset[source_usage_subset.columns],
        source_usage_subset,
        check_dtype=False,
        check_exact=True,
    )

    for vm_class in ALL_CLASSES:
        ranks = (
            manifest.loc[
                manifest["class"].eq(vm_class),
                "eligible_rank_within_class",
            ]
            .astype(int)
            .sort_values()
            .tolist()
        )
        if ranks != list(range(1, count_per_class + 1)):
            raise ValueError(
                f"{vm_class} selection is not the eligible top "
                f"{count_per_class}: ranks={ranks}"
            )

    indexed = output_requests.set_index("vm_id")
    actual_over_q: dict[str, int] = {}
    for resource in ("cpu", "mem"):
        q = _finite_numeric(
            indexed[f"q_{resource}"], f"derived q_{resource}"
        )
        if bool(q.le(0).any() or q.gt(1).any()):
            raise ValueError(f"Derived q_{resource} must be in (0, 1]")
        actual = _finite_numeric(
            output_usage[f"{resource}_usage"],
            f"derived {resource}_usage",
        )
        row_q = output_usage["vm_id"].map(q)
        actual_over_q[resource] = int(
            actual.gt(row_q + 1e-12).sum()
        )
        if bool(actual.gt(1.0 + 1e-12).any()):
            raise ValueError(
                f"Derived {resource}_usage exceeds one representative server"
            )

    return {
        "class_counts": counts,
        "scenario_ids_in_usage_file": [0],
        "scenario_zero_usage_rows": int(len(output_usage)),
        "selection_manifest_rows": int(len(manifest)),
        "selected_ids": selected_ids,
        "actual_usage_above_q_row_count": actual_over_q,
        "service_packing": _service_packing_diagnostics(
            output_requests, output_usage
        ),
        "configured_q_packing": _configured_q_packing_diagnostics(
            output_requests
        ),
        "batch_source_workload": _batch_diagnostics(
            output_requests, output_usage
        ),
        "invariants": {
            "canonical_requests_copied_without_value_changes": True,
            "canonical_scenario_zero_usage_copied_without_value_changes": True,
            "no_resource_scaling": True,
            "no_memory_pressure_transform": True,
            "top_average_usage_selection": True,
            "selection_does_not_use_a_specific_time_slot": True,
            "exact_class_counts": True,
            "scenario_zero_only": True,
            "unique_usage_keys": True,
        },
    }


def build_micro_top10_unscaled_dataset(
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    count_per_class: int = 10,
    num_scenarios: int = 10,
    chunksize: int = 400_000,
    overwrite: bool = False,
) -> dict[str, Any]:
    source = Path(source_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    try:
        source_label = str(source.relative_to(Path.cwd().resolve()))
    except ValueError:
        source_label = str(source)
    if source == output:
        raise ValueError(
            "source_dir and output_dir must differ; canonical data is read-only"
        )
    if count_per_class <= 0:
        raise ValueError("count_per_class must be positive")
    if num_scenarios <= 0:
        raise ValueError("num_scenarios must be positive")

    source_paths = {
        name: source / name
        for name in (
            "vm_requests.csv",
            "vm_usage_5min_scenarios.csv",
            "servers.csv",
        )
    }
    missing = [
        str(path) for path in source_paths.values() if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing canonical source files: {missing}")
    source_metadata = source / "metadata.json"
    if source_metadata.is_file():
        source_paths["metadata.json"] = source_metadata
    hashes_before = {
        name: _sha256(path) for name, path in source_paths.items()
    }

    requests = pd.read_csv(
        source_paths["vm_requests.csv"], dtype={"vm_id": str}
    )
    required_requests = {
        "vm_id",
        "class",
        "arrival_t5",
        "departure_t5",
        "q_cpu",
        "q_mem",
        "resource_request_cpu",
        "resource_request_mem",
    }
    missing_columns = required_requests - set(requests.columns)
    if missing_columns:
        raise ValueError(
            f"vm_requests.csv is missing columns {sorted(missing_columns)}"
        )
    if requests["vm_id"].duplicated().any():
        raise ValueError("Canonical vm_requests.csv contains duplicate vm_id")
    requests["vm_id"] = requests["vm_id"].astype(str)
    requests["class"] = requests["class"].astype(str)
    unknown_classes = set(requests["class"]) - set(ALL_CLASSES)
    if unknown_classes:
        raise ValueError(
            f"Canonical vm_requests.csv has unknown classes {unknown_classes}"
        )
    for column in ("q_cpu", "q_mem"):
        requests[column] = _finite_numeric(
            requests[column], f"source {column}"
        )
        if bool(
            requests[column].le(0).any()
            or requests[column].gt(1).any()
        ):
            raise ValueError(f"source {column} must be in (0, 1]")

    usage = _read_scenario_zero(
        source_paths["vm_usage_5min_scenarios.csv"],
        chunksize=chunksize,
    )
    summaries = _coverage_weighted_summaries(usage)
    eligible = {
        vm_class: _eligible_service_ids(requests, usage, vm_class)
        for vm_class in SERVICE_CLASSES
    }
    selected, candidates, selected_ids = _select_top_by_average(
        requests,
        summaries,
        eligible,
        count_per_class=count_per_class,
    )
    selected_id_set = set().union(*map(set, selected_ids.values()))

    output_requests = requests.loc[
        requests["vm_id"].isin(selected_id_set)
    ].copy()
    output_requests["_class_order"] = output_requests["class"].map(
        CLASS_ORDER
    )
    selection_order = selected.set_index("vm_id")[
        "selection_rank_within_class"
    ]
    output_requests["_selection_order"] = output_requests["vm_id"].map(
        selection_order
    )
    output_requests = output_requests.sort_values(
        ["_class_order", "_selection_order"], kind="mergesort"
    ).drop(columns=["_class_order", "_selection_order"])
    output_requests = output_requests.reset_index(drop=True)

    output_usage = usage.loc[
        usage["vm_id"].isin(selected_id_set)
    ].copy()
    output_usage = output_usage.sort_values(
        ["scenario_id", "vm_id", "t5_day"], kind="mergesort"
    ).reset_index(drop=True)

    manifest_columns = [
        "vm_id",
        "class",
        "selection_rank_within_class",
        "eligible_rank_within_class",
        "eligible_candidate_count",
        "eligibility_rule",
        "q_cpu",
        "q_mem",
        "selection_score_avg_cpu_plus_mem",
        "coverage_weighted_avg_cpu_usage",
        "coverage_weighted_avg_mem_usage",
        "observed_coverage_us",
        "observed_5min_rows",
        "arrival_t5",
        "departure_t5",
        "lifetime_t5",
    ]
    manifest = selected[
        [column for column in manifest_columns if column in selected.columns]
    ].copy()
    manifest["resource_transform"] = "none; exact canonical values copied"
    manifest = manifest.sort_values(
        ["class", "selection_rank_within_class"], kind="mergesort"
    ).reset_index(drop=True)

    report = _validate_output(
        requests,
        usage,
        output_requests,
        output_usage,
        manifest,
        selected_ids,
        count_per_class=count_per_class,
    )

    candidate_distribution: dict[str, Any] = {}
    selected_distribution: dict[str, Any] = {}
    for vm_class in ALL_CLASSES:
        candidate_class = candidates.loc[
            candidates["class"].eq(vm_class)
        ].drop_duplicates("vm_id")
        selected_class = selected.loc[selected["class"].eq(vm_class)]
        candidate_distribution[vm_class] = {
            "eligible_count": int(len(candidate_class)),
            "selection_score_avg_cpu_plus_mem": _numeric_summary(
                candidate_class["selection_score_avg_cpu_plus_mem"].rename(
                    "selection_score_avg_cpu_plus_mem"
                )
            ),
            "q_cpu": _numeric_summary(
                candidate_class["q_cpu"].rename("q_cpu")
            ),
            "q_mem": _numeric_summary(
                candidate_class["q_mem"].rename("q_mem")
            ),
        }
        selected_distribution[vm_class] = {
            "selected_count": int(len(selected_class)),
            "selection_score_avg_cpu_plus_mem": _numeric_summary(
                selected_class["selection_score_avg_cpu_plus_mem"].rename(
                    "selection_score_avg_cpu_plus_mem"
                )
            ),
            "q_cpu": _numeric_summary(
                selected_class["q_cpu"].rename("q_cpu")
            ),
            "q_mem": _numeric_summary(
                selected_class["q_mem"].rename("q_mem")
            ),
            "coverage_weighted_avg_cpu_usage": _numeric_summary(
                selected_class["coverage_weighted_avg_cpu_usage"].rename(
                    "coverage_weighted_avg_cpu_usage"
                )
            ),
            "coverage_weighted_avg_mem_usage": _numeric_summary(
                selected_class["coverage_weighted_avg_mem_usage"].rename(
                    "coverage_weighted_avg_mem_usage"
                )
            ),
        }

    temp = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    try:
        output_requests.to_csv(temp / "vm_requests.csv", index=False)
        output_usage.to_csv(
            temp / "vm_usage_5min_scenarios.csv", index=False
        )
        manifest.to_csv(temp / "selection_manifest.csv", index=False)
        servers = pd.read_csv(source_paths["servers.csv"])
        servers.to_csv(temp / "servers.csv", index=False)
        pd.DataFrame(
            {
                "scenario_id": list(range(num_scenarios)),
                "probability": [1.0 / num_scenarios] * num_scenarios,
            }
        ).to_csv(temp / "scenario_probabilities.csv", index=False)

        hashes_after = {
            name: _sha256(path) for name, path in source_paths.items()
        }
        if hashes_after != hashes_before:
            raise RuntimeError(
                "Canonical source changed while building the derived data set"
            )
        report["canonical_source_integrity"] = {
            "unchanged": True,
            "sha256_before": hashes_before,
            "sha256_after": hashes_after,
        }
        report["candidate_distribution"] = candidate_distribution
        report["selected_distribution"] = selected_distribution
        _json_dump(temp / "validation_report.json", report)

        policy = {
            "artifact_kind": (
                "trace-derived unscaled top-average-usage micro fixture"
            ),
            "canonical_data_modified": False,
            "source_dir": source_label,
            "selection": {
                "count_per_class": count_per_class,
                "classes": list(ALL_CLASSES),
                "score": (
                    "coverage-weighted mean(cpu_usage) + "
                    "coverage-weighted mean(mem_usage)"
                ),
                "order": (
                    "score descending, vm_id ascending deterministic tie-break"
                ),
                "uses_specific_time_slot": False,
                "service_eligibility": (
                    "at least one observed 5-minute row in every active "
                    "30-minute slot"
                ),
                "batch_eligibility": (
                    "at least one positive-coverage scenario-0 usage row"
                ),
            },
            "transformation": {
                "q_cpu": "copied unchanged",
                "q_mem": "copied unchanged",
                "resource_requests": "copied unchanged",
                "actual_cpu_usage": "copied unchanged",
                "actual_mem_usage": "copied unchanged",
                "lifecycle_and_provenance": "copied unchanged",
                "memory_pressure_transform": None,
                "target_q": None,
            },
            "scenario_policy": {
                "stored_scenarios": [0],
                "model_num_scenarios": num_scenarios,
                "model_loader_behavior": (
                    "scenario 0 is observed; later workload scenarios are "
                    "generated by the experiment loader"
                ),
            },
            "validation_summary": report,
        }
        _json_dump(temp / "selection_policy.json", policy)

        metadata: dict[str, Any] = {}
        if source_metadata.is_file():
            with source_metadata.open("r", encoding="utf-8") as stream:
                loaded = json.load(stream)
            if isinstance(loaded, dict):
                metadata = loaded
        metadata["micro_top10_unscaled_derivation"] = policy
        _json_dump(temp / "metadata.json", metadata)
        (temp / "README.md").write_text(
            "# Google 2019 unscaled top-average-usage micro fixture\n\n"
            "This directory is derived from the canonical v5 pool without "
            "modifying it. The ten eligible rows in each class with the "
            "largest coverage-weighted mean CPU plus memory usage are selected. "
            "Observed scenario-0 usage, q, requests, lifecycle, and provenance "
            "values are copied exactly. There is no target-q scaling, usage "
            "rescaling, memory-pressure transform, or minimum-server "
            "guarantee.\n\n"
            "The `on_demand`, `spot`, and `batch_candidate` labels are Google "
            "trace proxy classes, not observed public-cloud purchase types. "
            "The optimization loader later converts selected batch candidates "
            "into at most three flexible workload families. See "
            "`selection_policy.json`, `selection_manifest.csv`, and "
            "`validation_report.json`.\n",
            encoding="utf-8",
        )

        if output.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Output already exists: {output}; pass --overwrite "
                    "intentionally"
                )
            shutil.rmtree(output)
        temp.rename(output)
    except BaseException:
        if temp.exists():
            shutil.rmtree(temp)
        raise
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count-per-class", type=int, default=10)
    parser.add_argument("--num-scenarios", type=int, default=10)
    parser.add_argument("--chunksize", type=int, default=400_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = build_micro_top10_unscaled_dataset(
        args.source_dir,
        args.output_dir,
        count_per_class=args.count_per_class,
        num_scenarios=args.num_scenarios,
        chunksize=args.chunksize,
        overwrite=args.overwrite,
    )
    packing = report["service_packing"]
    print(
        "Built unscaled top-average-usage micro data: "
        f"max active service VMs={packing['maximum_active_vms']}, "
        "max scenario-0 FFD servers="
        f"{packing['maximum_first_fit_decreasing_servers']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
