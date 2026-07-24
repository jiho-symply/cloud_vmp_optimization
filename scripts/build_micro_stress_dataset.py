#!/usr/bin/env python3
"""Build an auditable, small capacity-pressure data set from the canonical pool.

The canonical preprocessing output is read-only.  Selection is based on each
VM's scenario-0, coverage-weighted mean CPU+memory usage, never on a chosen
time slot.  The derived data set intentionally increases placement pressure so
that ten on-demand VMs require at least five homogeneous servers in scenario 0.

This is a stress-benchmark transform, not a claim about the population-level
Google trace distribution.
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
    "data/processed/notion_toy_google2019_micro_stress_v1"
)
SERVICE_CLASSES = ("on_demand", "spot")
ALL_CLASSES = ("on_demand", "spot", "batch_candidate")
DEFAULT_BATCH_TARGET_PAIRS = ((0.47, 0.47), (0.48, 0.48), (0.49, 0.49))


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


def _read_scenario_zero(path: Path, *, chunksize: int) -> pd.DataFrame:
    """Read scenario zero without loading later canonical scenarios."""

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
        if bool((scenario < 0).any()):
            raise ValueError("scenario_id must be nonnegative")
        array = scenario.to_numpy(int)
        if array.size and (
            array[0] < last_scenario or bool((np.diff(array) < 0).any())
        ):
            raise ValueError(
                "Canonical usage must be sorted by scenario_id so scenario 0 can be read safely"
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
        duplicates = int(usage.duplicated(["vm_id", "t5_day"], keep=False).sum())
        raise ValueError(
            "Canonical scenario zero must have one row per VM/t5 bucket; "
            f"found {duplicates} duplicate rows"
        )
    return usage


def _coverage_weighted_scores(usage: pd.DataFrame) -> pd.DataFrame:
    scored = usage[["vm_id", "cpu_usage", "mem_usage", "coverage_us"]].copy()
    scored["cpu_volume"] = scored["cpu_usage"] * scored["coverage_us"]
    scored["mem_volume"] = scored["mem_usage"] * scored["coverage_us"]
    grouped = scored.groupby("vm_id", as_index=False, sort=True).agg(
        observed_coverage_us=("coverage_us", "sum"),
        cpu_volume=("cpu_volume", "sum"),
        mem_volume=("mem_volume", "sum"),
    )
    grouped["source_avg_cpu_usage"] = (
        grouped["cpu_volume"] / grouped["observed_coverage_us"]
    )
    grouped["source_avg_mem_usage"] = (
        grouped["mem_volume"] / grouped["observed_coverage_us"]
    )
    grouped["selection_score_avg_cpu_plus_mem"] = (
        grouped["source_avg_cpu_usage"] + grouped["source_avg_mem_usage"]
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
    for _, row in requests.loc[requests["class"].eq(service_class)].iterrows():
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
    scores: pd.DataFrame,
    eligible: dict[str, set[str]],
    *,
    count_per_class: int,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    candidates = requests.merge(scores, on="vm_id", how="inner", validate="one_to_one")
    selected_ids: dict[str, list[str]] = {}
    selected_parts: list[pd.DataFrame] = []
    for vm_class in ALL_CLASSES:
        frame = candidates.loc[candidates["class"].eq(vm_class)].copy()
        if vm_class in SERVICE_CLASSES:
            frame = frame.loc[frame["vm_id"].isin(eligible[vm_class])]
        frame = frame.sort_values(
            ["selection_score_avg_cpu_plus_mem", "vm_id"],
            ascending=[False, True],
            kind="mergesort",
        )
        if len(frame) < count_per_class:
            raise ValueError(
                f"Need {count_per_class} eligible {vm_class} rows, found {len(frame)}"
            )
        frame = frame.head(count_per_class).copy()
        frame["selection_rank_within_class"] = np.arange(1, count_per_class + 1)
        selected_ids[vm_class] = frame["vm_id"].astype(str).tolist()
        selected_parts.append(frame)
    selected = pd.concat(selected_parts, ignore_index=True)
    return selected, selected_ids


def _parse_batch_target_pairs(value: str) -> tuple[tuple[float, float], ...]:
    pairs: list[tuple[float, float]] = []
    for token in value.split(","):
        fields = token.strip().split(":")
        if len(fields) != 2:
            raise argparse.ArgumentTypeError(
                "batch target pairs must use cpu:mem,cpu:mem syntax"
            )
        try:
            pair = (float(fields[0]), float(fields[1]))
        except ValueError as exc:
            raise argparse.ArgumentTypeError("batch target pairs must be numeric") from exc
        pairs.append(pair)
    if len(pairs) != 3:
        raise argparse.ArgumentTypeError("exactly three batch target pairs are required")
    return tuple(pairs)


def _target_table(
    selected: pd.DataFrame,
    *,
    service_target_q: float,
    batch_target_pairs: tuple[tuple[float, float], ...],
) -> pd.DataFrame:
    if not 0 < service_target_q < 0.5:
        raise ValueError("service_target_q must be in (0, 0.5) for two-per-server packing")
    if len(set(batch_target_pairs)) != 3:
        raise ValueError("batch_target_pairs must contain exactly three distinct joint pairs")
    for cpu, mem in batch_target_pairs:
        if not (0 < cpu < 0.5 and 0 < mem < 0.5):
            raise ValueError("every batch target q must be in (0, 0.5)")

    targets = selected[
        ["vm_id", "class", "selection_rank_within_class", "q_cpu", "q_mem"]
    ].copy()
    targets = targets.rename(columns={"q_cpu": "source_q_cpu", "q_mem": "source_q_mem"})
    targets["target_q_cpu"] = service_target_q
    targets["target_q_mem"] = service_target_q
    targets["stress_batch_group"] = pd.NA

    batch_index = targets.index[targets["class"].eq("batch_candidate")].tolist()
    groups = np.array_split(np.asarray(batch_index, dtype=int), 3)
    # Highest average-usage group receives the largest pair.  The supplied
    # defaults are ascending, but sorting also makes custom CLI input deterministic.
    descending_pairs = sorted(batch_target_pairs, reverse=True)
    for group_number, (indices, pair) in enumerate(zip(groups, descending_pairs, strict=True)):
        targets.loc[indices, "target_q_cpu"] = pair[0]
        targets.loc[indices, "target_q_mem"] = pair[1]
        targets.loc[indices, "stress_batch_group"] = f"stress_family_{group_number:03d}"

    # The canonical top-ten batch rows are smaller than all default targets.
    # Fail rather than silently shrinking a different source data set.
    batch = targets["class"].eq("batch_candidate")
    if bool(
        targets.loc[batch, "source_q_cpu"].gt(targets.loc[batch, "target_q_cpu"] + 1e-12).any()
        or targets.loc[batch, "source_q_mem"].gt(targets.loc[batch, "target_q_mem"] + 1e-12).any()
    ):
        raise ValueError(
            "A selected batch q exceeds its exact family target; choose larger target pairs"
        )
    return targets


def _bounded_ratio(
    values: pd.Series,
    old_q: pd.Series,
    label: str,
    *,
    allow_null: bool,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise").astype(float)
    old = pd.to_numeric(old_q, errors="raise").astype(float)
    if bool(old.le(0).any()) or not np.isfinite(old.to_numpy(float)).all():
        raise ValueError(f"{label} has invalid source q")
    nonnull = numeric.notna()
    if not allow_null and not bool(nonnull.all()):
        raise ValueError(f"{label} contains NULL values")
    finite = numeric.loc[nonnull].to_numpy(float)
    if not np.isfinite(finite).all() or bool((finite < 0).any()):
        raise ValueError(f"{label} contains invalid resource values")
    ratio = numeric / old
    if bool(ratio.loc[nonnull].gt(1.0 + 1e-9).any()):
        maximum = float(ratio.loc[nonnull].max())
        raise ValueError(f"{label} exceeds source q (maximum ratio {maximum})")
    # Only absorb floating-point noise at the configured-resource boundary.
    return ratio.clip(lower=0.0, upper=1.0)


def _weighted_quantile(values: pd.Series, weights: pd.Series, q: float) -> float:
    array = values.to_numpy(float)
    weight = weights.to_numpy(float)
    order = np.argsort(array, kind="mergesort")
    array = array[order]
    weight = weight[order]
    cutoff = q * float(weight.sum())
    index = int(np.searchsorted(np.cumsum(weight), cutoff, side="left"))
    return float(array[min(index, len(array) - 1)])


def _transform_selected(
    selected: pd.DataFrame,
    usage: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    pressure_exponent: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < pressure_exponent <= 1:
        raise ValueError("pressure_exponent must be in (0, 1]")

    target_index = targets.set_index("vm_id")
    selected_ids = set(target_index.index.astype(str))
    output_usage = usage.loc[usage["vm_id"].isin(selected_ids)].copy()
    if set(output_usage["vm_id"].astype(str)) != selected_ids:
        raise ValueError("One or more selected VMs have no scenario-0 usage")
    output_usage["_class"] = output_usage["vm_id"].map(target_index["class"])

    for resource in ("cpu", "mem"):
        old_q = output_usage["vm_id"].map(target_index[f"source_q_{resource}"])
        new_q = output_usage["vm_id"].map(target_index[f"target_q_{resource}"])
        actual_column = f"{resource}_usage"
        ratio = _bounded_ratio(
            output_usage[actual_column],
            old_q,
            f"selected {actual_column}",
            allow_null=False,
        )
        pressure = output_usage["_class"].eq("on_demand") & (resource == "mem")
        transformed = new_q * ratio
        if isinstance(pressure, pd.Series) and bool(pressure.any()):
            transformed.loc[pressure] = (
                new_q.loc[pressure] * ratio.loc[pressure].pow(pressure_exponent)
            )
        output_usage[actual_column] = transformed

        maximum_column = f"max_{resource}_usage"
        if maximum_column in output_usage.columns:
            max_ratio = _bounded_ratio(
                output_usage[maximum_column],
                old_q,
                f"selected {maximum_column}",
                allow_null=True,
            )
            transformed_max = new_q * max_ratio
            if isinstance(pressure, pd.Series) and bool(pressure.any()):
                transformed_max.loc[pressure] = (
                    new_q.loc[pressure]
                    * max_ratio.loc[pressure].pow(pressure_exponent)
                )
            output_usage[maximum_column] = transformed_max

    if "assigned_memory" in output_usage.columns:
        old_q = output_usage["vm_id"].map(target_index["source_q_mem"])
        new_q = output_usage["vm_id"].map(target_index["target_q_mem"])
        assigned = pd.to_numeric(output_usage["assigned_memory"], errors="raise").astype(float)
        if not np.isfinite(assigned.to_numpy(float)).all() or bool(assigned.lt(0).any()):
            raise ValueError("selected assigned_memory contains invalid values")
        # assigned_memory is not actual demand.  Preserve its configured-unit
        # ratio without using it in the pressure proof.
        output_usage["assigned_memory"] = assigned * new_q / old_q

    output_usage = output_usage.drop(columns=["_class"])
    output_usage = output_usage.sort_values(
        ["scenario_id", "vm_id", "t5_day"], kind="mergesort"
    ).reset_index(drop=True)

    output_requests = selected.copy()
    output_requests = output_requests.set_index("vm_id", drop=False)
    for resource in ("cpu", "mem"):
        old_q = target_index[f"source_q_{resource}"]
        new_q = target_index[f"target_q_{resource}"]
        request_column = f"resource_request_{resource}"
        request_ratio = _bounded_ratio(
            output_requests[request_column],
            old_q,
            f"selected {request_column}",
            allow_null=False,
        )
        output_requests[request_column] = request_ratio * new_q
        output_requests[f"q_{resource}"] = new_q

    # Recompute descriptive usage statistics from the transformed scenario-0
    # rows.  maximum_usage columns remain the source of truth for max_*.
    summaries: list[dict[str, Any]] = []
    for vm_id, group in output_usage.groupby("vm_id", sort=True):
        row: dict[str, Any] = {"vm_id": str(vm_id)}
        weights = group["coverage_us"].astype(float)
        for resource in ("cpu", "mem"):
            actual = group[f"{resource}_usage"].astype(float)
            row[f"avg_{resource}_usage"] = float(np.average(actual, weights=weights))
            row[f"p95_{resource}_usage"] = _weighted_quantile(actual, weights, 0.95)
            maximum_column = f"max_{resource}_usage"
            if maximum_column in group.columns and group[maximum_column].notna().any():
                row[maximum_column] = float(group[maximum_column].max(skipna=True))
            else:
                row[maximum_column] = float(actual.max())
        summaries.append(row)
    summary = pd.DataFrame(summaries).set_index("vm_id")
    for column in summary.columns:
        output_requests[column] = summary[column]

    output_requests = output_requests.reset_index(drop=True)
    keep_request_columns = [column for column in selected.columns if column in output_requests]
    output_requests = output_requests[keep_request_columns]
    output_requests = output_requests.sort_values(
        ["class", "selection_rank_within_class"], kind="mergesort"
    ).reset_index(drop=True)

    manifest = targets.merge(
        selected[
            [
                "vm_id",
                "source_avg_cpu_usage",
                "source_avg_mem_usage",
                "selection_score_avg_cpu_plus_mem",
                "observed_coverage_us",
            ]
        ],
        on="vm_id",
        how="left",
        validate="one_to_one",
    )
    manifest["cpu_transform"] = "target_q * (source_usage / source_q)"
    manifest["mem_transform"] = "target_q * (source_usage / source_q)"
    manifest.loc[manifest["class"].eq("on_demand"), "mem_transform"] = (
        f"target_q * (source_usage / source_q) ** {pressure_exponent}"
    )
    return output_requests, output_usage, manifest


def _aggregate_30min(
    usage: pd.DataFrame,
    vm_ids: Iterable[str],
) -> pd.DataFrame:
    selected = usage.loc[usage["vm_id"].isin(set(vm_ids))].copy()
    selected["t_slot"] = (selected["t5_day"] // T5_PER_SLOT).astype(int)
    for resource in ("cpu", "mem"):
        selected[f"{resource}_volume"] = (
            selected[f"{resource}_usage"] * selected["coverage_us"]
        )
    grouped = selected.groupby(["vm_id", "t_slot"], as_index=False, sort=True).agg(
        coverage_us=("coverage_us", "sum"),
        cpu_volume=("cpu_volume", "sum"),
        mem_volume=("mem_volume", "sum"),
    )
    grouped["cpu_usage"] = grouped["cpu_volume"] / grouped["coverage_us"]
    grouped["mem_usage"] = grouped["mem_volume"] / grouped["coverage_us"]
    return grouped


def _fits_two_dimensional_bins(
    items: list[tuple[float, float]],
    num_bins: int,
    *,
    tolerance: float = 1e-12,
) -> bool:
    """Exact small-instance CPU/MEM bin-packing feasibility check."""

    if not items:
        return True
    if num_bins <= 0:
        return False
    if any(cpu > 1.0 + tolerance or mem > 1.0 + tolerance for cpu, mem in items):
        return False
    if sum(cpu for cpu, _ in items) > num_bins + tolerance:
        return False
    if sum(mem for _, mem in items) > num_bins + tolerance:
        return False
    ordered = sorted(items, key=lambda pair: (max(pair), sum(pair)), reverse=True)
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
            if load[0] + cpu <= 1.0 + tolerance and load[1] + mem <= 1.0 + tolerance:
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


def _minimum_two_dimensional_bins(items: list[tuple[float, float]]) -> int:
    if not items:
        return 0
    lower = max(
        1,
        int(math.ceil(sum(cpu for cpu, _ in items) - 1e-12)),
        int(math.ceil(sum(mem for _, mem in items) - 1e-12)),
    )
    for num_bins in range(lower, len(items) + 1):
        if _fits_two_dimensional_bins(items, num_bins):
            return num_bins
    raise AssertionError("Every unit-sized item must fit in its own bin")


def _validate_output(
    requests: pd.DataFrame,
    usage: pd.DataFrame,
    manifest: pd.DataFrame,
    selected_ids: dict[str, list[str]],
    *,
    count_per_class: int,
    minimum_required_servers: int,
) -> dict[str, Any]:
    counts = requests["class"].value_counts().astype(int).to_dict()
    expected_counts = {vm_class: count_per_class for vm_class in ALL_CLASSES}
    if counts != expected_counts:
        raise ValueError(f"Unexpected derived class counts: {counts}")
    if requests["vm_id"].duplicated().any():
        raise ValueError("Derived vm_requests.csv has duplicate vm_id")
    if usage["scenario_id"].astype(int).unique().tolist() != [0]:
        raise ValueError("Derived usage must contain scenario 0 only")
    if usage.duplicated(["vm_id", "t5_day"]).any():
        raise ValueError("Derived usage has duplicate VM/t5 keys")

    indexed = requests.set_index("vm_id")
    for resource in ("cpu", "mem"):
        q = _finite_numeric(indexed[f"q_{resource}"], f"derived q_{resource}")
        if bool(q.le(0).any() or q.gt(1).any()):
            raise ValueError(f"Derived q_{resource} must be in (0, 1]")
        actual = _finite_numeric(usage[f"{resource}_usage"], f"derived {resource}_usage")
        row_q = usage["vm_id"].map(q)
        if bool(actual.gt(row_q + 1e-12).any()):
            raise ValueError(f"Derived {resource}_usage exceeds q_{resource}")
        maximum_column = f"max_{resource}_usage"
        if maximum_column in usage.columns:
            maximum = pd.to_numeric(usage[maximum_column], errors="raise").astype(float)
            valid = maximum.notna()
            if not np.isfinite(maximum.loc[valid].to_numpy(float)).all():
                raise ValueError(f"Derived {maximum_column} contains non-finite values")
            if bool(
                maximum.loc[valid].lt(0).any()
                or maximum.loc[valid].gt(row_q.loc[valid] + 1e-12).any()
            ):
                raise ValueError(f"Derived {maximum_column} is outside [0, q]")
        request = _finite_numeric(
            indexed[f"resource_request_{resource}"],
            f"derived resource_request_{resource}",
        )
        maximum_by_vm = usage.groupby("vm_id")[maximum_column].max().reindex(indexed.index)
        maximum_by_vm = maximum_by_vm.fillna(
            usage.groupby("vm_id")[f"{resource}_usage"].max().reindex(indexed.index)
        )
        if bool(request.gt(q + 1e-12).any() or maximum_by_vm.gt(q + 1e-12).any()):
            raise ValueError(f"Derived {resource} request/max invariant failed")

    batch_pairs = (
        requests.loc[requests["class"].eq("batch_candidate"), ["q_cpu", "q_mem"]]
        .drop_duplicates()
        .sort_values(["q_cpu", "q_mem"])
    )
    if len(batch_pairs) != 3:
        raise ValueError(f"Expected exactly three batch pairs, found {len(batch_pairs)}")

    od_30min = _aggregate_30min(usage, selected_ids["on_demand"])
    aggregate = od_30min.groupby("t_slot", as_index=True).agg(
        cpu_usage=("cpu_usage", "sum"),
        mem_usage=("mem_usage", "sum"),
        active_vms=("vm_id", "nunique"),
    )
    peak_mem = float(aggregate["mem_usage"].max())
    peak_slot = int(aggregate["mem_usage"].idxmax())
    lower_bound = int(math.ceil(peak_mem - 1e-12))
    max_individual_mem = float(od_30min["mem_usage"].max())
    if peak_mem <= minimum_required_servers - 1 + 1e-9:
        raise ValueError(
            f"OD memory peak {peak_mem} does not prove {minimum_required_servers} servers"
        )
    if lower_bound < minimum_required_servers:
        raise ValueError("Derived aggregate server lower bound is too small")
    if max_individual_mem >= 0.5 - 1e-12:
        raise ValueError(
            "An individual OD 30-minute memory load is not below 0.5; "
            "the six-server packing-safe design was not preserved"
        )

    minimum_bins_by_slot: dict[str, int] = {}
    for slot, frame in od_30min.groupby("t_slot", sort=True):
        items = list(
            frame[["cpu_usage", "mem_usage"]].itertuples(index=False, name=None)
        )
        minimum_bins_by_slot[str(int(slot))] = _minimum_two_dimensional_bins(items)
    max_exact_bins = max(minimum_bins_by_slot.values())
    if max_exact_bins != minimum_required_servers:
        raise ValueError(
            "Expected the scenario-0 OD two-dimensional packing maximum to be "
            f"exactly {minimum_required_servers}, found {max_exact_bins}"
        )

    od_resource_volume = {
        resource: float(od_30min[f"{resource}_usage"].sum())
        for resource in ("cpu", "mem")
    }
    batch_usage = usage.loc[
        usage["vm_id"].isin(selected_ids["batch_candidate"])
    ]
    batch_resource_volume = {
        resource: float(
            (
                batch_usage[f"{resource}_usage"] * batch_usage["coverage_us"]
            ).sum()
            / (T5_PER_SLOT * T5_US)
        )
        for resource in ("cpu", "mem")
    }

    expected_selected = set().union(*map(set, selected_ids.values()))
    if set(usage["vm_id"].astype(str)) != expected_selected:
        raise ValueError("Derived usage VM set differs from selection")
    return {
        "class_counts": counts,
        "scenario_ids_in_usage_file": [0],
        "scenario_zero_usage_rows": int(len(usage)),
        "unique_vm_t5_keys": int(len(usage)),
        "batch_joint_q_pairs": batch_pairs.to_dict(orient="records"),
        "batch_family_count_expected_by_loader": 3,
        "od_30min_memory": {
            "peak_aggregate_load": peak_mem,
            "peak_slot_for_validation_only": peak_slot,
            "aggregate_server_lower_bound": lower_bound,
            "minimum_required_servers_requested": minimum_required_servers,
            "max_individual_vm_load": max_individual_mem,
            "all_individual_vm_loads_below_half_server": True,
            "selection_used_peak_slot": False,
        },
        "scenario_zero_od_two_dimensional_packing": {
            "resources": ["cpu", "mem"],
            "bin_capacity": {"cpu": 1.0, "mem": 1.0},
            "minimum_bins_by_slot": minimum_bins_by_slot,
            "maximum_minimum_bins": max_exact_bins,
            "fits_five_bins_in_every_slot": max_exact_bins <= 5,
            "audit_scope": "observed scenario 0; synthetic scenarios are generated by the model loader",
        },
        "scenario_zero_resource_slot_volumes": {
            "on_demand": od_resource_volume,
            "batch": batch_resource_volume,
            "on_demand_plus_batch": {
                resource: od_resource_volume[resource] + batch_resource_volume[resource]
                for resource in ("cpu", "mem")
            },
        },
        "selection_manifest_rows": int(len(manifest)),
        "invariants": {
            "exact_class_counts": True,
            "scenario_zero_only": True,
            "unique_usage_keys": True,
            "finite_nonnegative_actual_usage": True,
            "actual_and_maximum_usage_at_most_q": True,
            "request_and_maximum_at_most_q": True,
            "exactly_three_batch_joint_q_pairs": True,
            "five_server_aggregate_lower_bound": True,
            "scenario_zero_od_fits_exactly_five_two_dimensional_bins": True,
        },
    }


def build_micro_stress_dataset(
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    count_per_class: int = 10,
    num_scenarios: int = 10,
    service_target_q: float = 0.49,
    pressure_exponent: float = 0.2,
    batch_target_pairs: tuple[tuple[float, float], ...] = DEFAULT_BATCH_TARGET_PAIRS,
    minimum_required_servers: int = 5,
    chunksize: int = 400_000,
    overwrite: bool = False,
) -> dict[str, Any]:
    source = Path(source_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if source == output:
        raise ValueError("source_dir and output_dir must differ; canonical data is read-only")
    if count_per_class != 10:
        raise ValueError("This micro benchmark requires exactly 10 units per class")
    if num_scenarios != 10:
        raise ValueError("This micro benchmark is defined for exactly 10 model scenarios")
    if minimum_required_servers != 5:
        raise ValueError("This micro benchmark requires a five-server lower-bound target")

    source_paths = {
        name: source / name
        for name in ("vm_requests.csv", "vm_usage_5min_scenarios.csv", "servers.csv")
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing canonical source files: {missing}")
    source_metadata = source / "metadata.json"
    if source_metadata.is_file():
        source_paths["metadata.json"] = source_metadata
    hashes_before = {name: _sha256(path) for name, path in source_paths.items()}

    requests = pd.read_csv(source_paths["vm_requests.csv"], dtype={"vm_id": str})
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
        raise ValueError(f"vm_requests.csv is missing columns {sorted(missing_columns)}")
    if requests["vm_id"].duplicated().any():
        raise ValueError("Canonical vm_requests.csv contains duplicate vm_id")
    requests["vm_id"] = requests["vm_id"].astype(str)
    requests["class"] = requests["class"].astype(str)
    for column in ("q_cpu", "q_mem"):
        requests[column] = _finite_numeric(requests[column], f"source {column}")
        if bool(requests[column].le(0).any() or requests[column].gt(1).any()):
            raise ValueError(f"source {column} must be in (0, 1]")

    usage = _read_scenario_zero(
        source_paths["vm_usage_5min_scenarios.csv"], chunksize=chunksize
    )
    scores = _coverage_weighted_scores(usage)
    eligible = {
        vm_class: _eligible_service_ids(requests, usage, vm_class)
        for vm_class in SERVICE_CLASSES
    }
    selected, selected_ids = _select_top_by_average(
        requests,
        scores,
        eligible,
        count_per_class=count_per_class,
    )
    targets = _target_table(
        selected,
        service_target_q=service_target_q,
        batch_target_pairs=batch_target_pairs,
    )
    output_requests, output_usage, manifest = _transform_selected(
        selected,
        usage,
        targets,
        pressure_exponent=pressure_exponent,
    )
    report = _validate_output(
        output_requests,
        output_usage,
        manifest,
        selected_ids,
        count_per_class=count_per_class,
        minimum_required_servers=minimum_required_servers,
    )

    temp = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    try:
        output_requests.to_csv(temp / "vm_requests.csv", index=False)
        output_usage.to_csv(temp / "vm_usage_5min_scenarios.csv", index=False)
        manifest.to_csv(temp / "selection_manifest.csv", index=False)
        servers = pd.read_csv(source_paths["servers.csv"])
        servers.to_csv(temp / "servers.csv", index=False)
        pd.DataFrame(
            {
                "scenario_id": list(range(num_scenarios)),
                "probability": [1.0 / num_scenarios] * num_scenarios,
            }
        ).to_csv(temp / "scenario_probabilities.csv", index=False)

        hashes_after = {name: _sha256(path) for name, path in source_paths.items()}
        if hashes_after != hashes_before:
            raise RuntimeError("Canonical source changed while building the derived data set")
        report["canonical_source_integrity"] = {
            "unchanged": True,
            "sha256_before": hashes_before,
            "sha256_after": hashes_after,
        }
        _json_dump(temp / "validation_report.json", report)

        transform = {
            "artifact_kind": "trace-derived capacity-pressure stress benchmark",
            "canonical_data_modified": False,
            "source_dir": str(source),
            "selection": {
                "count_per_class": count_per_class,
                "classes": list(ALL_CLASSES),
                "service_eligibility": (
                    "at least one observed 5-minute row in every active 30-minute slot"
                ),
                "score": (
                    "coverage-weighted mean(cpu_usage) + coverage-weighted mean(mem_usage)"
                ),
                "order": "score descending, vm_id ascending deterministic tie-break",
                "uses_specific_slot": False,
            },
            "transformation": {
                "service_target_q": service_target_q,
                "proportional_resources": (
                    "OD CPU, Spot CPU/MEM, and Batch CPU/MEM: "
                    "new_usage = target_q * source_usage / source_q"
                ),
                "od_memory_pressure": {
                    "formula": "new_usage = target_q * (source_usage / source_q) ** exponent",
                    "exponent": pressure_exponent,
                    "zero_preserved": True,
                    "temporal_order_preserved": True,
                    "reason": (
                        "A monotone utilization transform is required to establish the five-server "
                        "hard-memory lower bound while keeping each VM below half a server."
                    ),
                },
                "batch_target_pairs_ascending": [list(pair) for pair in sorted(batch_target_pairs)],
                "batch_grouping": (
                    "balanced contiguous average-usage rank groups; highest group gets largest pair"
                ),
                "batch_family_count": 3,
            },
            "scenario_policy": {
                "stored_scenarios": [0],
                "model_num_scenarios": num_scenarios,
                "model_loader_behavior": (
                    "scenario 0 is observed; scenarios 1..9 are generated by the experiment loader"
                ),
            },
            "validation_summary": report,
        }
        _json_dump(temp / "stress_transform.json", transform)

        metadata: dict[str, Any] = {}
        if source_metadata.is_file():
            with source_metadata.open("r", encoding="utf-8") as stream:
                loaded = json.load(stream)
            if isinstance(loaded, dict):
                metadata = loaded
        metadata["micro_stress_derivation"] = transform
        _json_dump(temp / "metadata.json", metadata)
        (temp / "README.md").write_text(
            "# Google 2019 micro capacity-pressure benchmark\n\n"
            "This directory is derived from the canonical v5 pool without modifying it. "
            "It contains the top ten eligible on-demand, spot, and batch units by "
            "coverage-weighted average CPU+memory usage.  The monotone OD-memory "
            "stress transform is intentionally synthetic and proves a five-server "
            "scenario-0 lower bound; it must not be described as a representative "
            "Google-trace sample.  See `stress_transform.json`, "
            "`selection_manifest.csv`, and `validation_report.json` for the audit trail.\n",
            encoding="utf-8",
        )

        if output.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Output already exists: {output}; pass --overwrite intentionally"
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
    parser.add_argument("--service-target-q", type=float, default=0.49)
    parser.add_argument("--pressure-exponent", type=float, default=0.2)
    parser.add_argument(
        "--batch-target-pairs",
        type=_parse_batch_target_pairs,
        default=DEFAULT_BATCH_TARGET_PAIRS,
        help="exactly three cpu:mem pairs, e.g. 0.47:0.47,0.48:0.48,0.49:0.49",
    )
    parser.add_argument("--minimum-required-servers", type=int, default=5)
    parser.add_argument("--chunksize", type=int, default=400_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = build_micro_stress_dataset(
        args.source_dir,
        args.output_dir,
        count_per_class=args.count_per_class,
        num_scenarios=args.num_scenarios,
        service_target_q=args.service_target_q,
        pressure_exponent=args.pressure_exponent,
        batch_target_pairs=args.batch_target_pairs,
        minimum_required_servers=args.minimum_required_servers,
        chunksize=args.chunksize,
        overwrite=args.overwrite,
    )
    summary = report["od_30min_memory"]
    print(
        "Built micro stress data: "
        f"peak OD memory={summary['peak_aggregate_load']:.6f}, "
        f"server lower bound={summary['aggregate_server_lower_bound']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
