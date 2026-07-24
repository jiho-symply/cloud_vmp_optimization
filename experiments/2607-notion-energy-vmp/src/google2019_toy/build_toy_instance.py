from __future__ import annotations

import argparse
import hashlib
import json
import math
from bisect import bisect_right
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

T5_SECONDS = 300
T5_US = T5_SECONDS * 1_000_000
DAY_US = 24 * 60 * 60 * 1_000_000
HORIZON_T5 = 288
HORIZON_HOURS = 24
DEFAULT_RAW_DIR = Path("data/raw/google2019_cell_a_day0_cpu_distribution")
DEFAULT_OUTPUT_DIR = Path("data/processed/notion_toy_google2019_v5_mode_capacity_bounded")
EPSILON = 1e-6
DEFAULT_RAW_RESOURCE_UNIT = (
    "Google ClusterData 2019 v3 normalized CPU and memory units; CPU and memory are "
    "independently normalized against trace-wide maximum machine capacities. "
    "No utilization-target scaling is applied during extraction."
)
CPU_P100_AUDIT_STATUS = "cpu_p100_with_representative_capacity_vm_drop_no_clipping"

VM_REQUEST_COLUMNS = [
    "vm_id",
    "stable_episode_key",
    "collection_id",
    "instance_index",
    "episode_index",
    "episode_start_time_us",
    "episode_end_time_us",
    "termination_time_us",
    "termination_reason",
    "horizon_censored",
    "termination_time_uncertain",
    "episode_start_inferred_from_update",
    "episode_start_inferred_from_usage",
    "class",
    "class_rule",
    "scheduler",
    "arrival_t5",
    "departure_t5",
    "lifetime_t5",
    "arrival_time_us",
    "q_cpu",
    "q_mem",
    "resource_request_cpu",
    "resource_request_mem",
    "p95_cpu_usage",
    "p95_mem_usage",
    "avg_cpu_usage",
    "avg_mem_usage",
    "max_cpu_usage",
    "max_mem_usage",
    "priority",
    "scheduling_class",
    "arrival_machine_id",
    "arrival_state_ambiguous",
    "arrival_state_ambiguity_reason",
    "arrival_state_candidate_count",
    "scheduler_state_ambiguous",
    "scheduler_state_ambiguity_reason",
    "scheduler_state_candidate_count",
    "arrival_state_used_post_arrival",
    "scheduler_state_used_post_arrival",
    "resource_request_cpu_fallback_zero",
    "resource_request_mem_fallback_zero",
    "max_cpu_usage_fallback_from_average",
    "max_mem_usage_fallback_from_average",
    "cpu_p100_gt_one",
    "spot_revenue",
    "migration_energy_coeff",
]

USAGE_SCENARIO_COLUMNS = ["scenario_id", "vm_id", "t5_day", "t_hour", "cpu_usage", "mem_usage"]
USAGE_BUCKET_AUDIT_COLUMNS = [
    "coverage_us",
    "coverage_ratio",
    "max_cpu_usage",
    "max_mem_usage",
    "assigned_memory",
    "overlap_source_row_count",
    "overlap_conflict_flag",
    "duplicated_timeline_us",
]
HOURLY_USAGE_SCENARIO_COLUMNS = ["scenario_id", "vm_id", "t_hour", "cpu_usage", "mem_usage"]
SPOT_PREEMPTION_COLUMNS = ["scenario_id", "vm_id", "t5_day", "t_hour", "active", "preempted"]
BATCH_FAMILY_COLUMNS = [
    "family_id",
    "q_cpu_B",
    "q_mem_B",
    "W_k",
    "rho_cpu_B",
    "rho_mem_B",
    "base_cpu",
    "base_mem",
    "startup_cpu",
    "startup_mem",
]
BATCH_WORKLOAD_COLUMNS = ["scenario_id", "family_id", "t_hour", "workload_volume", "cpu_workload", "mem_workload"]
ENERGY_COLUMNS = [
    "scenario_id",
    "t_hour",
    "day_ahead_price",
    "real_time_price",
    "sell_price",
    "renewable_generation",
    "ess_capacity",
    "ess_charge_max",
    "ess_discharge_max",
    "ess_charge_efficiency",
    "ess_discharge_efficiency",
]
SCENARIO_PROBABILITY_COLUMNS = ["scenario_id", "probability"]


# Google ClusterData 2019 v3 ``instance_events.type`` is zero based.  The
# lifecycle order below is deliberately semantic rather than numeric: terminal
# transitions close the old execution first, then pending/running transitions
# establish the new execution.  In particular, numeric sorting would put
# UPDATE_* after terminal transitions but SUBMIT before them, which is wrong for
# a same-microsecond retry chain.
INSTANCE_EVENT_TYPE_BY_CODE = {
    0: "SUBMIT",
    1: "QUEUE",
    2: "ENABLE",
    3: "SCHEDULE",
    4: "EVICT",
    5: "FAIL",
    6: "FINISH",
    7: "KILL",
    8: "LOST",
    9: "UPDATE_PENDING",
    10: "UPDATE_RUNNING",
}
INSTANCE_EVENT_LIFECYCLE_ORDER = {
    "EVICT": 0,
    "FAIL": 1,
    "FINISH": 2,
    "KILL": 3,
    "LOST": 4,
    "SUBMIT": 5,
    "QUEUE": 6,
    "ENABLE": 7,
    "UPDATE_PENDING": 8,
    "SCHEDULE": 9,
    "UPDATE_RUNNING": 10,
}

TERMINAL_INSTANCE_EVENTS = {"EVICT", "FAIL", "FINISH", "KILL", "LOST"}
RUNNING_START_INSTANCE_EVENTS = {"SCHEDULE", "UPDATE_RUNNING"}
HORIZON_CENSORED = "HORIZON_CENSORED"

EPISODE_KEY_COLUMNS = ["collection_id", "instance_index", "episode_index"]
EPISODE_PROVENANCE_COLUMNS = [
    "episode_index",
    "episode_start_time_us",
    "episode_end_time_us",
    "termination_time_us",
    "termination_reason",
    "horizon_censored",
    "termination_time_uncertain",
    "episode_start_inferred_from_update",
    "episode_start_inferred_from_usage",
]


def _first_existing(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _require_column(df: pd.DataFrame, names: Iterable[str], label: str) -> str:
    column = _first_existing(df, names)
    if column is None:
        raise ValueError(f"missing required {label} column; expected one of {list(names)}")
    return column


def _stable_key_hash(
    collection_id: object,
    instance_index: object,
    seed: int,
    episode_index: object | None = None,
) -> int:
    suffix = "" if episode_index is None else f":{episode_index}"
    payload = f"{seed}:{collection_id}:{instance_index}{suffix}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), byteorder="big", signed=False)


def _unit_key_columns(frame: pd.DataFrame) -> list[str]:
    return EPISODE_KEY_COLUMNS if "episode_index" in frame.columns else [
        "collection_id",
        "instance_index",
    ]


def _stable_episode_key(
    collection_id: object,
    instance_index: object,
    episode_index: object,
) -> str:
    return (
        f"collection={_deterministic_value_key(collection_id)}|"
        f"instance={_deterministic_value_key(instance_index)}|"
        f"episode={int(episode_index)}"
    )


def _is_evict_event(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (int, np.integer, float, np.floating)):
        return int(value) == 4
    text = str(value).strip().upper()
    return "EVICT" in text or text == "4"


def _instance_event_name(value: object) -> str:
    """Return the documented v3 enum label without using enum magnitude."""

    if pd.isna(value):
        return "UNKNOWN"
    if isinstance(value, (int, np.integer, float, np.floating)):
        return INSTANCE_EVENT_TYPE_BY_CODE.get(int(value), f"UNKNOWN_{int(value)}")
    text = str(value).strip().upper()
    if text.lstrip("+-").isdigit():
        code = int(text)
        return INSTANCE_EVENT_TYPE_BY_CODE.get(code, f"UNKNOWN_{code}")
    for prefix in ("INSTANCE_EVENT_TYPE_", "EVENT_TYPE_"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    return text or "UNKNOWN"


def _instance_event_precedence(value: object) -> int:
    # Unknown legacy rows remain usable as sparse state observations, but they
    # precede every documented lifecycle transition at the same timestamp.
    return INSTANCE_EVENT_LIFECYCLE_ORDER.get(_instance_event_name(value), -1)


def _normal_missing_row_rank(value: object) -> int:
    """Rank a normal/raw row after synthesized missing-data rows."""

    if pd.isna(value):
        return 1
    if isinstance(value, (int, np.integer, float, np.floating)):
        return int(int(value) == 0)
    text = str(value).strip().upper()
    return int(text in {"", "0", "NONE", "NORMAL", "NOT_MISSING", "MISSING_TYPE_NONE"})


def _deterministic_value_key(value: object) -> str:
    if pd.isna(value):
        return "0:<NULL>"
    if isinstance(value, (bool, np.bool_)):
        return f"1:{int(value)}"
    if isinstance(value, (int, np.integer, float, np.floating)):
        return f"2:{float(value):.17g}"
    return f"3:{str(value)}"


def _json_scalar(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _hashable_repeated_value(value: object) -> object:
    """Canonicalize Arrow repeated values solely for exact-row deduplication."""

    if isinstance(value, pd.Series):
        value = value.tolist()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return tuple(_hashable_repeated_value(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _exact_usage_duplicate_mask(usage: pd.DataFrame) -> pd.Series:
    """Return an exact duplicate mask that handles native repeated ARRAY values."""

    comparison = usage
    if "cpu_usage_distribution" in usage.columns:
        comparison = usage.copy()
        comparison["cpu_usage_distribution"] = comparison[
            "cpu_usage_distribution"
        ].map(_hashable_repeated_value)
    return comparison.duplicated(keep="first")


def _infer_day_start_us(
    usage: pd.DataFrame,
    start_col: str,
    clipped_start_col: str | None,
    t5_day_col: str | None,
) -> int:
    """Infer a bucket-aligned origin only for legacy in-memory callers.

    Production builds obtain the exact day bounds from the extractor metadata.
    The inference path keeps the public helpers usable with the small legacy test
    frames that predate metadata and clipped interval columns.
    """

    starts = pd.to_numeric(
        usage[clipped_start_col] if clipped_start_col else usage[start_col],
        errors="coerce",
    )
    if t5_day_col:
        t5_day = pd.to_numeric(usage[t5_day_col], errors="coerce")
        candidates = starts - t5_day * T5_US
        candidates = candidates.dropna()
        if not candidates.empty:
            aligned = (candidates.astype("int64") // T5_US) * T5_US
            return int(aligned.mode().sort_values().iloc[0])
    starts = starts.dropna()
    if starts.empty:
        raise ValueError("cannot infer day start from usage timestamps")
    return int(int(starts.min()) // T5_US * T5_US)


def _normalize_usage(
    usage: pd.DataFrame,
    day_start_us: int | None = None,
    day_end_us: int | None = None,
) -> pd.DataFrame:
    """Clip usage intervals and reconstruct duration-weighted 5-minute buckets.

    Average CPU and memory use independent valid-duration denominators, so a
    NULL resource never becomes a zero or dilutes another valid observation.
    Source maximum-usage fields remain separate from average usage and are
    aggregated with ``max`` over every source interval touching the bucket.
    """

    if usage.empty:
        raise ValueError("instance usage input is empty")

    collection_col = _require_column(usage, ["collection_id", "collection"], "collection_id")
    instance_col = _require_column(usage, ["instance_index", "instance"], "instance_index")
    start_col = _require_column(usage, ["start_time", "start_time_us"], "start_time")
    end_col = _require_column(usage, ["end_time", "end_time_us"], "end_time")
    cpu_col = _require_column(
        usage,
        ["cpu_usage", "average_usage_cpus", "average_usage.cpu", "average_usage.cpus", "avg_cpu_usage"],
        "CPU usage",
    )
    mem_col = _require_column(
        usage,
        ["mem_usage", "memory_usage", "average_usage_memory", "average_usage.memory", "avg_mem_usage"],
        "memory usage",
    )
    max_cpu_col = _first_existing(
        usage,
        ["max_cpu_usage", "maximum_usage_cpus", "maximum_usage.cpu", "maximum_usage.cpus"],
    )
    max_mem_col = _first_existing(
        usage,
        ["max_mem_usage", "maximum_usage_memory", "maximum_usage.memory"],
    )
    assigned_memory_col = _first_existing(usage, ["assigned_memory", "memory_limit"])
    machine_id_col = _first_existing(usage, ["machine_id"])
    clipped_start_col = _first_existing(usage, ["clipped_start_time", "clipped_start_time_us"])
    clipped_end_col = _first_existing(usage, ["clipped_end_time", "clipped_end_time_us"])
    overlap_col = _first_existing(usage, ["overlap_us"])
    t5_day_col = _first_existing(usage, ["t5_day", "period_5min_day"])

    if day_start_us is None:
        day_start_us = _infer_day_start_us(usage, start_col, clipped_start_col, t5_day_col)
    day_start_us = int(day_start_us)
    day_end_us = int(day_end_us) if day_end_us is not None else day_start_us + DAY_US
    if day_end_us <= day_start_us:
        raise ValueError("day_end_us must be greater than day_start_us")

    # Remove rows that are identical in every raw field before any interval
    # arithmetic.  This is independent of BigQuery return order and prevents a
    # repeated physical row from influencing either means or overlap audit.
    exact_duplicate_mask = _exact_usage_duplicate_mask(usage)
    exact_duplicate_row_count = int(exact_duplicate_mask.sum())
    episode_columns = [column for column in EPISODE_PROVENANCE_COLUMNS if column in usage.columns]
    key_columns = ["collection_id", "instance_index"]
    if "episode_index" in episode_columns:
        key_columns.append("episode_index")
    raw_interval_keys = pd.DataFrame(
        {
            "collection_id": usage[collection_col],
            "instance_index": usage[instance_col],
            "start_time": pd.to_numeric(usage[start_col], errors="coerce"),
            "end_time": pd.to_numeric(usage[end_col], errors="coerce"),
            **{column: usage[column] for column in episode_columns},
        }
    )
    duplicate_interval_group_count = int(
        raw_interval_keys.groupby(
            key_columns + ["start_time", "end_time"],
            dropna=False,
        )
        .size()
        .gt(1)
        .sum()
    )
    usage_input = usage.loc[~exact_duplicate_mask].copy()

    source = pd.DataFrame(
        {
            "collection_id": usage_input[collection_col],
            "instance_index": usage_input[instance_col],
            "machine_id": usage_input[machine_id_col] if machine_id_col else np.nan,
            "start_time": pd.to_numeric(usage_input[start_col], errors="coerce"),
            "end_time": pd.to_numeric(usage_input[end_col], errors="coerce"),
            "cpu_usage": pd.to_numeric(usage_input[cpu_col], errors="coerce"),
            "mem_usage": pd.to_numeric(usage_input[mem_col], errors="coerce"),
            "max_cpu_usage": (
                pd.to_numeric(usage_input[max_cpu_col], errors="coerce") if max_cpu_col else np.nan
            ),
            "max_mem_usage": (
                pd.to_numeric(usage_input[max_mem_col], errors="coerce") if max_mem_col else np.nan
            ),
            "assigned_memory": (
                pd.to_numeric(usage_input[assigned_memory_col], errors="coerce")
                if assigned_memory_col
                else np.nan
            ),
            **{column: usage_input[column] for column in episode_columns},
        }
    )
    if clipped_start_col:
        source["clipped_start_time"] = pd.to_numeric(
            usage_input[clipped_start_col], errors="coerce"
        )
    else:
        source["clipped_start_time"] = source["start_time"].clip(lower=day_start_us)
    if clipped_end_col:
        source["clipped_end_time"] = pd.to_numeric(
            usage_input[clipped_end_col], errors="coerce"
        )
    else:
        source["clipped_end_time"] = source["end_time"].clip(upper=day_end_us)

    required_null = source[["collection_id", "instance_index", "start_time", "end_time"]].isna().any(axis=1)
    if required_null.any():
        raise ValueError(f"usage contains {int(required_null.sum())} rows with NULL identity or interval endpoints")
    invalid_source_interval = source["end_time"] <= source["start_time"]
    if invalid_source_interval.any():
        raise ValueError(f"usage contains {int(invalid_source_interval.sum())} non-positive source intervals")

    # Clamp even extractor-provided clipped endpoints to the metadata horizon;
    # this also protects legacy files from leaking an adjacent trace day.
    source["clipped_start_time"] = source[["clipped_start_time", "start_time"]].max(axis=1).clip(
        lower=day_start_us
    )
    source["clipped_end_time"] = source[["clipped_end_time", "end_time"]].min(axis=1).clip(
        upper=day_end_us
    )
    positive_overlap = source["clipped_end_time"] > source["clipped_start_time"]
    source = source.loc[positive_overlap].copy()
    if source.empty:
        raise ValueError("no usage intervals overlap the requested day")

    source["clipped_start_time"] = source["clipped_start_time"].astype("int64")
    source["clipped_end_time"] = source["clipped_end_time"].astype("int64")
    if overlap_col:
        supplied_overlap = pd.to_numeric(
            usage_input.loc[source.index, overlap_col], errors="coerce"
        )
        expected_overlap = source["clipped_end_time"] - source["clipped_start_time"]
        mismatch = supplied_overlap.isna() | supplied_overlap.ne(expected_overlap)
        if mismatch.any():
            raise ValueError(f"usage contains {int(mismatch.sum())} rows with inconsistent overlap_us")

    for column in ["cpu_usage", "mem_usage", "max_cpu_usage", "max_mem_usage", "assigned_memory"]:
        negative = source[column].notna() & source[column].lt(0)
        if negative.any():
            raise ValueError(f"usage contains {int(negative.sum())} negative {column} values")

    source = source.reset_index(drop=True)
    source["source_row_id"] = np.arange(len(source), dtype=np.int64)

    first_bucket = ((source["clipped_start_time"] - day_start_us) // T5_US).astype(int)
    last_bucket = ((source["clipped_end_time"] - 1 - day_start_us) // T5_US).astype(int)
    span = last_bucket - first_bucket + 1
    if (span <= 0).any():
        raise ValueError("usage interval splitting produced a non-positive bucket span")

    segment_frames: list[pd.DataFrame] = []
    max_span = int(span.max())
    for offset in range(max_span):
        positions = np.flatnonzero(span.to_numpy() > offset)
        if len(positions) == 0:
            continue
        part = source.iloc[positions][
            [
                "collection_id",
                "instance_index",
                *episode_columns,
                "machine_id",
                "source_row_id",
                "clipped_start_time",
                "clipped_end_time",
                "cpu_usage",
                "mem_usage",
                "max_cpu_usage",
                "max_mem_usage",
                "assigned_memory",
            ]
        ].copy()
        part["t5_day"] = first_bucket.iloc[positions].to_numpy() + offset
        bucket_start = day_start_us + part["t5_day"].to_numpy(dtype=np.int64) * T5_US
        part["segment_start"] = np.maximum(part["clipped_start_time"].to_numpy(dtype=np.int64), bucket_start)
        part["segment_end"] = np.minimum(part["clipped_end_time"].to_numpy(dtype=np.int64), bucket_start + T5_US)
        part["duration_us"] = part["segment_end"] - part["segment_start"]
        segment_frames.append(part)
    segments = pd.concat(segment_frames, ignore_index=True)
    segments = segments.loc[
        segments["t5_day"].between(0, HORIZON_T5 - 1) & segments["duration_us"].gt(0)
    ].copy()
    if segments.empty:
        raise ValueError("no positive usage overlap remains inside 5-minute day buckets")

    bucket_key_columns = key_columns + ["t5_day"]
    segments = segments.sort_values(
        bucket_key_columns + ["segment_start", "segment_end", "source_row_id"], kind="stable"
    )
    running_end = segments.groupby(bucket_key_columns, sort=False)["segment_end"].cummax()
    previous_end = running_end.groupby(
        [segments[column] for column in bucket_key_columns], sort=False
    ).shift()
    union_start = np.maximum(
        segments["segment_start"].to_numpy(dtype=np.int64),
        previous_end.fillna(segments["segment_start"]).to_numpy(dtype=np.int64),
    )
    segments["coverage_contribution_us"] = np.maximum(
        0, segments["segment_end"].to_numpy(dtype=np.int64) - union_start
    )
    segments["overlaps_previous"] = (
        previous_end.notna()
        & segments["segment_start"].lt(previous_end)
    )
    weighted_columns = {
        "cpu": "cpu_usage",
        "mem": "mem_usage",
        "assigned_memory": "assigned_memory",
    }
    for resource, column in weighted_columns.items():
        valid = segments[column].notna()
        segments[f"{resource}_weighted"] = np.where(
            valid,
            segments[column] * segments["duration_us"],
            0.0,
        )
        segments[f"{resource}_valid_us"] = np.where(valid, segments["duration_us"], 0)

    buckets = (
        segments.groupby(bucket_key_columns, as_index=False, sort=False)
        .agg(
            coverage_us=("coverage_contribution_us", "sum"),
            cpu_weighted=("cpu_weighted", "sum"),
            cpu_valid_us=("cpu_valid_us", "sum"),
            mem_weighted=("mem_weighted", "sum"),
            mem_valid_us=("mem_valid_us", "sum"),
            assigned_memory_weighted=("assigned_memory_weighted", "sum"),
            assigned_memory_valid_us=("assigned_memory_valid_us", "sum"),
            max_cpu_usage=("max_cpu_usage", "max"),
            max_mem_usage=("max_mem_usage", "max"),
            first_observed_us=("segment_start", "min"),
            last_observed_us=("segment_end", "max"),
            overlap_source_row_count=("source_row_id", "nunique"),
        )
    )

    affected_keys = segments.loc[
        segments["overlaps_previous"], bucket_key_columns
    ].drop_duplicates()
    affected_key_set = {
        tuple(row)
        for row in affected_keys.itertuples(index=False, name=None)
    }
    overlap_source_ids: set[int] = set()
    duplicated_timeline_us = 0
    max_concurrent_source_rows = 1
    overlap_records: list[dict[str, object]] = []
    if affected_key_set:
        affected_segments = segments.merge(
            affected_keys.assign(_overlap_bucket=True),
            on=bucket_key_columns,
            how="inner",
            validate="many_to_one",
        )
        for key, group in affected_segments.groupby(
            bucket_key_columns, sort=False, dropna=False
        ):
            boundaries = np.unique(
                np.concatenate(
                    [
                        group["segment_start"].to_numpy(dtype=np.int64),
                        group["segment_end"].to_numpy(dtype=np.int64),
                    ]
                )
            )
            accum = {
                "cpu": [0.0, 0],
                "mem": [0.0, 0],
                "assigned_memory": [0.0, 0],
            }
            coverage_us = 0
            bucket_duplicated_us = 0
            conflict = False
            for left, right in zip(boundaries[:-1], boundaries[1:]):
                duration_us = int(right - left)
                if duration_us <= 0:
                    continue
                active = group.loc[
                    group["segment_start"].lt(int(right))
                    & group["segment_end"].gt(int(left))
                ]
                concurrency = int(len(active))
                if concurrency == 0:
                    continue
                coverage_us += duration_us
                max_concurrent_source_rows = max(max_concurrent_source_rows, concurrency)
                if concurrency > 1:
                    excess = duration_us * (concurrency - 1)
                    bucket_duplicated_us += excess
                    duplicated_timeline_us += excess
                    overlap_source_ids.update(
                        int(value) for value in active["source_row_id"].tolist()
                    )
                for resource, column in (
                    ("cpu", "cpu_usage"),
                    ("mem", "mem_usage"),
                    ("assigned_memory", "assigned_memory"),
                ):
                    valid_values = pd.to_numeric(active[column], errors="coerce").dropna()
                    if not valid_values.empty:
                        # The simultaneous observations describe one logical VM,
                        # so average them within the elementary interval and
                        # count its unique duration exactly once.
                        accum[resource][0] += float(valid_values.mean()) * duration_us
                        accum[resource][1] += duration_us
                        if concurrency > 1 and valid_values.nunique(dropna=True) > 1:
                            conflict = True
            identity = dict(zip(bucket_key_columns, key))
            overlap_records.append(
                {
                    **identity,
                    "coverage_us": int(coverage_us),
                    "cpu_weighted": float(accum["cpu"][0]),
                    "cpu_valid_us": int(accum["cpu"][1]),
                    "mem_weighted": float(accum["mem"][0]),
                    "mem_valid_us": int(accum["mem"][1]),
                    "assigned_memory_weighted": float(accum["assigned_memory"][0]),
                    "assigned_memory_valid_us": int(accum["assigned_memory"][1]),
                    "max_cpu_usage": group["max_cpu_usage"].max(),
                    "max_mem_usage": group["max_mem_usage"].max(),
                    "first_observed_us": int(group["segment_start"].min()),
                    "last_observed_us": int(group["segment_end"].max()),
                    "overlap_source_row_count": int(group["source_row_id"].nunique()),
                    "overlap_conflict_flag": bool(conflict),
                    "duplicated_timeline_us": int(bucket_duplicated_us),
                }
            )

        overlap_buckets = pd.DataFrame(overlap_records)
        bucket_index = pd.MultiIndex.from_frame(buckets[bucket_key_columns])
        overlap_index = pd.MultiIndex.from_frame(overlap_buckets[bucket_key_columns])
        buckets = buckets.loc[~bucket_index.isin(overlap_index)].copy()
        buckets["overlap_conflict_flag"] = False
        buckets["duplicated_timeline_us"] = 0
        buckets = pd.concat([buckets, overlap_buckets], ignore_index=True, sort=False)
    else:
        buckets["overlap_conflict_flag"] = False
        buckets["duplicated_timeline_us"] = 0

    overlap_participating_rows = segments.loc[
        segments["source_row_id"].isin(overlap_source_ids),
        key_columns + ["source_row_id"],
    ].drop_duplicates()
    overlap_entering_rows = segments.loc[
        segments["overlaps_previous"],
        key_columns + ["source_row_id"],
    ].drop_duplicates()
    overlap_diagnostics: dict[str, object] = {
        "exact_duplicate_row_count": exact_duplicate_row_count,
        "duplicate_interval_group_count": duplicate_interval_group_count,
        "source_interval_overlap_rows": int(len(overlap_entering_rows)),
        "source_interval_overlap_rows_definition": (
            "unique source rows whose clipped segment starts before the prior running maximum end "
            "within the same VM and 5-minute bucket"
        ),
        "source_interval_overlap_participating_rows": int(len(overlap_participating_rows)),
        "source_interval_overlap_participating_rows_definition": (
            "unique source rows active in any elementary interval with concurrency greater than one"
        ),
        "source_interval_overlap_vm_count": int(
            overlap_participating_rows[key_columns]
            .drop_duplicates()
            .shape[0]
        ),
        "overlap_affected_bucket_count": int(len(affected_key_set)),
        "duplicated_timeline_us": int(duplicated_timeline_us),
        "duplicated_timeline_us_definition": (
            "sum over elementary intervals of duration_us * (concurrent_source_rows - 1), "
            "equivalent to summed source duration minus interval-union duration"
        ),
        "max_concurrent_source_rows": int(max_concurrent_source_rows),
        "source_cpu_p100_gt_one_row_count": int(source["max_cpu_usage"].gt(1.0).sum()),
        "source_cpu_p100_gt_one_unit_count": int(
            source.loc[source["max_cpu_usage"].gt(1.0), key_columns]
            .drop_duplicates()
            .shape[0]
        ),
        "source_cpu_p100_gt_one_summary": _numeric_summary(
            source.loc[source["max_cpu_usage"].gt(1.0), "max_cpu_usage"]
        ),
    }
    # Keep the VM-level source maximum independent of whether a particular
    # bucket has both average CPU and memory observations.  A source row may
    # legitimately carry maximum_usage while one average_usage field is NULL;
    # such a row must still contribute to configured-resource q.
    vm_source_maxima = (
        segments.groupby(key_columns, as_index=False, sort=False)
        .agg(
            vm_source_max_cpu_usage=("max_cpu_usage", "max"),
            vm_source_max_mem_usage=("max_mem_usage", "max"),
        )
    )
    buckets = buckets.merge(
        vm_source_maxima,
        on=key_columns,
        how="left",
        validate="many_to_one",
    )
    vm_first_times = (
        segments.groupby(key_columns, as_index=False, sort=False)
        .agg(vm_first_observed_us=("segment_start", "min"))
    )
    first_segments = segments.merge(
        vm_first_times,
        on=key_columns,
        how="inner",
        validate="many_to_one",
    )
    first_segments = first_segments.loc[
        first_segments["segment_start"].eq(first_segments["vm_first_observed_us"])
    ]
    arrival_machine_rows: list[dict[str, object]] = []
    for key, group in first_segments.groupby(
        key_columns, sort=False, dropna=False
    ):
        machines = sorted(
            {
                _json_scalar(value)
                for value in group["machine_id"].tolist()
                if not pd.isna(value)
            },
            key=_deterministic_value_key,
        )
        if not isinstance(key, tuple):
            key = (key,)
        arrival_machine_rows.append(
            {**dict(zip(key_columns, key)), "arrival_machine_ids": tuple(machines)}
        )
    buckets = buckets.merge(
        pd.DataFrame(arrival_machine_rows),
        on=key_columns,
        how="left",
        validate="many_to_one",
    )
    episode_metadata_columns = [
        column for column in episode_columns if column != "episode_index"
    ]
    if episode_metadata_columns:
        episode_metadata = segments[
            key_columns + episode_metadata_columns
        ].drop_duplicates()
        if episode_metadata[key_columns].duplicated().any():
            raise ValueError("episode provenance is inconsistent within one episode key")
        buckets = buckets.merge(
            episode_metadata,
            on=key_columns,
            how="left",
            validate="many_to_one",
        )
    buckets["cpu_usage"] = buckets["cpu_weighted"] / buckets["cpu_valid_us"].replace(0, np.nan)
    buckets["mem_usage"] = buckets["mem_weighted"] / buckets["mem_valid_us"].replace(0, np.nan)
    buckets["assigned_memory"] = buckets["assigned_memory_weighted"] / buckets[
        "assigned_memory_valid_us"
    ].replace(0, np.nan)
    buckets["coverage_ratio"] = buckets["coverage_us"] / float(T5_US)
    buckets["t5"] = (day_start_us // T5_US + buckets["t5_day"]).astype(int)
    buckets["t_hour"] = (buckets["t5_day"] // 12).astype(int)

    missing_cpu = buckets["cpu_usage"].isna()
    missing_mem = buckets["mem_usage"].isna()
    emitted = buckets.loc[~(missing_cpu | missing_mem)].copy()
    if emitted.empty:
        raise ValueError("no reconstructed usage buckets contain both CPU and memory observations")
    numeric_actual = emitted[["cpu_usage", "mem_usage"]].to_numpy(dtype=float)
    if not np.isfinite(numeric_actual).all() or (numeric_actual < 0).any():
        raise ValueError("reconstructed usage contains non-finite or negative CPU/memory values")
    if not emitted["coverage_ratio"].between(0.0, 1.0, inclusive="right").all():
        raise ValueError("reconstructed usage coverage_ratio must be in (0, 1]")
    if not emitted["coverage_us"].between(1, T5_US, inclusive="both").all():
        raise ValueError("reconstructed usage coverage_us must be in (0, 300 seconds]")

    keep_columns = [
        "collection_id",
        "instance_index",
        *episode_columns,
        "t5",
        "t5_day",
        "t_hour",
        "cpu_usage",
        "mem_usage",
        "max_cpu_usage",
        "max_mem_usage",
        "vm_source_max_cpu_usage",
        "vm_source_max_mem_usage",
        "arrival_machine_ids",
        "assigned_memory",
        "coverage_us",
        "coverage_ratio",
        "overlap_source_row_count",
        "overlap_conflict_flag",
        "duplicated_timeline_us",
        "first_observed_us",
        "last_observed_us",
    ]
    emitted = emitted[keep_columns].sort_values(
        key_columns + ["t5_day"], kind="stable"
    ).reset_index(drop=True)
    emitted.attrs["day_start_us"] = day_start_us
    emitted.attrs["day_end_us"] = day_end_us
    emitted.attrs["usage_overlap_diagnostics"] = overlap_diagnostics
    return emitted


def _normalize_events(events: pd.DataFrame | None) -> pd.DataFrame:
    columns = [
        "collection_id",
        "instance_index",
        "time",
        "event_type",
        "event_name",
        "event_precedence",
        "missing_type",
        "normal_missing_row_rank",
        "machine_id",
        "priority",
        "scheduling_class",
        "resource_request_cpu",
        "resource_request_mem",
    ]
    if events is None or events.empty:
        return pd.DataFrame(columns=columns)

    collection_col = _first_existing(events, ["collection_id", "collection"])
    instance_col = _first_existing(events, ["instance_index", "instance"])
    if collection_col is None or instance_col is None:
        return pd.DataFrame(columns=columns)

    time_col = _first_existing(events, ["time", "event_time", "timestamp_us"])
    event_col = _first_existing(events, ["event_type", "type", "event"])
    missing_type_col = _first_existing(events, ["missing_type"])
    machine_id_col = _first_existing(events, ["machine_id"])
    priority_col = _first_existing(events, ["priority"])
    scheduling_class_col = _first_existing(events, ["scheduling_class"])
    cpu_col = _first_existing(
        events,
        ["resource_request_cpu", "resource_request_cpus", "resource_request.cpus", "requested_cpu"],
    )
    mem_col = _first_existing(
        events,
        ["resource_request_mem", "resource_request_memory", "resource_request.memory", "requested_mem"],
    )

    out = pd.DataFrame(
        {
            "collection_id": events[collection_col],
            "instance_index": events[instance_col],
            "time": pd.to_numeric(events[time_col], errors="coerce") if time_col else np.nan,
            "event_type": events[event_col] if event_col else "",
            "missing_type": events[missing_type_col] if missing_type_col else np.nan,
            "machine_id": events[machine_id_col] if machine_id_col else np.nan,
            "priority": pd.to_numeric(events[priority_col], errors="coerce") if priority_col else np.nan,
            "scheduling_class": (
                pd.to_numeric(events[scheduling_class_col], errors="coerce")
                if scheduling_class_col
                else np.nan
            ),
            "resource_request_cpu": pd.to_numeric(events[cpu_col], errors="coerce") if cpu_col else np.nan,
            "resource_request_mem": pd.to_numeric(events[mem_col], errors="coerce") if mem_col else np.nan,
        }
    )
    out = out.dropna(subset=["collection_id", "instance_index"]).drop_duplicates()
    out["event_name"] = out["event_type"].map(_instance_event_name)
    out["event_precedence"] = out["event_type"].map(_instance_event_precedence).astype(int)
    out["normal_missing_row_rank"] = out["missing_type"].map(_normal_missing_row_rank).astype(int)
    sort_columns = ["collection_id", "instance_index", "time", "event_precedence"]
    for column in [
        "normal_missing_row_rank",
        "machine_id",
        "priority",
        "scheduling_class",
        "resource_request_cpu",
        "resource_request_mem",
    ]:
        key_column = f"_sort_{column}"
        out[key_column] = out[column].map(_deterministic_value_key)
        sort_columns.append(key_column)
    out = out.sort_values(sort_columns, kind="stable", na_position="first")
    return out[columns].reset_index(drop=True)


def _normalize_collection_events(collection_events: pd.DataFrame | None) -> pd.DataFrame:
    columns = [
        "collection_id",
        "time",
        "event_type",
        "event_name",
        "event_precedence",
        "missing_type",
        "normal_missing_row_rank",
        "scheduler",
        "collection_type",
        "scheduling_class",
        "priority",
    ]
    if collection_events is None or collection_events.empty:
        return pd.DataFrame(columns=columns)

    collection_col = _first_existing(collection_events, ["collection_id", "collection"])
    if collection_col is None:
        return pd.DataFrame(columns=columns)

    time_col = _first_existing(collection_events, ["time", "event_time", "timestamp_us"])
    scheduler_col = _first_existing(collection_events, ["scheduler", "scheduler_name"])
    event_col = _first_existing(collection_events, ["event_type", "type", "event"])
    type_col = _first_existing(collection_events, ["collection_type"])
    missing_type_col = _first_existing(collection_events, ["missing_type"])
    scheduling_class_col = _first_existing(collection_events, ["scheduling_class"])
    priority_col = _first_existing(collection_events, ["priority"])
    if scheduler_col:
        scheduler = collection_events[scheduler_col].copy()
        scheduler = scheduler.where(scheduler.isna(), scheduler.map(_normalize_scheduler))
    else:
        scheduler = np.nan
    out = pd.DataFrame(
        {
            "collection_id": collection_events[collection_col],
            "time": pd.to_numeric(collection_events[time_col], errors="coerce") if time_col else np.nan,
            "event_type": collection_events[event_col] if event_col else "",
            "missing_type": (
                collection_events[missing_type_col] if missing_type_col else np.nan
            ),
            "scheduler": scheduler,
            "collection_type": collection_events[type_col] if type_col else np.nan,
            "scheduling_class": (
                pd.to_numeric(collection_events[scheduling_class_col], errors="coerce")
                if scheduling_class_col
                else np.nan
            ),
            "priority": (
                pd.to_numeric(collection_events[priority_col], errors="coerce") if priority_col else np.nan
            ),
        }
    )
    out = out.dropna(subset=["collection_id"]).drop_duplicates()
    out["event_name"] = out["event_type"].map(_instance_event_name)
    out["event_precedence"] = out["event_type"].map(_instance_event_precedence).astype(int)
    out["normal_missing_row_rank"] = out["missing_type"].map(_normal_missing_row_rank).astype(int)
    sort_columns = ["collection_id", "time", "event_precedence"]
    for column in [
        "normal_missing_row_rank",
        "scheduler",
        "collection_type",
        "scheduling_class",
        "priority",
    ]:
        key_column = f"_sort_{column}"
        out[key_column] = out[column].map(_deterministic_value_key)
        sort_columns.append(key_column)
    out = out.sort_values(sort_columns, kind="stable", na_position="first")
    return out[columns].reset_index(drop=True)


def _normalize_scheduler(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer, float, np.floating)):
        code = int(value)
        if code == 1:
            return "SCHEDULER_BATCH"
        if code == 0:
            return "SCHEDULER_DEFAULT"
        return f"SCHEDULER_{code}"
    text = str(value).strip().upper()
    if text == "1" or "BATCH" in text:
        return "SCHEDULER_BATCH"
    if text == "0" or "DEFAULT" in text:
        return "SCHEDULER_DEFAULT"
    return text


def _resolve_sparse_state_at_time(
    history: pd.DataFrame,
    arrival_time_us: int,
    state_columns: list[str],
    *,
    arrival_machine_ids: tuple[object, ...] = (),
    scheduler_conflict: bool = False,
) -> tuple[dict[str, object], bool, bool, str, int, object]:
    """Resolve a sparse event history with semantic same-time precedence.

    Input row position is never consulted.  A normal row wins over a
    synthesized/missing-data row within the same final lifecycle transition;
    for instance running transitions, a first-observed usage machine match is
    then preferred.  A still-conflicting coherent state is reported rather
    than reduced with per-field maxima.
    """

    empty_state = {column: np.nan for column in state_columns}
    if history.empty:
        return empty_state, False, False, "", 0, np.nan

    machine_keys = {_deterministic_value_key(value) for value in arrival_machine_ids}
    # This function is called once per selected VM/collection.  Constructing
    # and sorting a DataFrame for every timestamp cohort made the real 274k-row
    # event history prohibitively slow.  Normalize the few input arrays once,
    # then perform the same cohort ordering and sparse updates in Python.  The
    # histories produced by _normalize_*_events are already time-sorted; the
    # stable argsort is only a defensive path for direct/legacy callers.
    effective_times = pd.to_numeric(history["time"], errors="coerce").to_numpy(
        dtype=float, copy=True
    )
    effective_times[np.isnan(effective_times)] = -np.inf
    row_order = np.arange(len(history), dtype=np.int64)
    if len(row_order) > 1 and np.any(effective_times[1:] < effective_times[:-1]):
        row_order = np.argsort(effective_times, kind="stable")

    if "event_name" in history:
        event_names = history["event_name"].to_numpy(copy=False)
    else:
        event_names = history.get("event_type", pd.Series("", index=history.index)).map(
            _instance_event_name
        ).to_numpy(copy=False)
    if "event_precedence" in history:
        event_precedences = pd.to_numeric(
            history["event_precedence"], errors="coerce"
        ).fillna(-1).to_numpy(dtype=int, copy=False)
    else:
        event_precedences = np.asarray(
            [_instance_event_precedence(value) for value in event_names], dtype=int
        )
    if "normal_missing_row_rank" in history:
        normal_missing_ranks = pd.to_numeric(
            history["normal_missing_row_rank"], errors="coerce"
        ).fillna(0).to_numpy(dtype=int, copy=False)
    else:
        missing_values = history.get(
            "missing_type", pd.Series(np.nan, index=history.index)
        ).to_numpy(copy=False)
        normal_missing_ranks = np.asarray(
            [_normal_missing_row_rank(value) for value in missing_values], dtype=int
        )
    has_machine_id = "machine_id" in history
    machine_values = (
        history["machine_id"].to_numpy(copy=False)
        if has_machine_id
        else np.full(len(history), np.nan, dtype=object)
    )
    state_values = {
        column: history[column].to_numpy(copy=False)
        if column in history
        else np.full(len(history), np.nan, dtype=object)
        for column in state_columns
    }

    current_state = dict(empty_state)
    current_machine: object = np.nan
    unresolved_fields: set[str] = set()
    ambiguity_reason = ""
    ambiguity_candidate_count = 0
    latest_before: tuple[dict[str, object], bool, str, int, object] | None = None

    def result(
        snapshot: tuple[dict[str, object], bool, str, int, object],
        used_post_arrival: bool,
    ) -> tuple[dict[str, object], bool, bool, str, int, object]:
        state, ambiguous, reason, candidate_count, machine_id = snapshot
        return state, used_post_arrival, ambiguous, reason, candidate_count, machine_id

    position = 0
    while position < len(row_order):
        cohort_start = position
        effective_time = float(effective_times[row_order[position]])
        position += 1
        while (
            position < len(row_order)
            and effective_times[row_order[position]] == effective_time
        ):
            position += 1

        # Once a useful state at/before arrival exists, later events cannot
        # affect the requested snapshot.
        if effective_time > int(arrival_time_us) and latest_before is not None:
            return result(latest_before, False)

        records: list[
            tuple[
                int,
                str,
                int,
                int,
                object,
                str,
                int,
                tuple[object, ...],
                tuple[str, ...],
            ]
        ] = []
        for raw_index in row_order[cohort_start:position]:
            index = int(raw_index)
            event_name_value = event_names[index]
            event_name = str(event_name_value)
            machine_id = machine_values[index]
            machine_key = _deterministic_value_key(machine_id)
            machine_match_rank = int(
                has_machine_id
                and event_name in {"SCHEDULE", "UPDATE_RUNNING"}
                and machine_key in machine_keys
            )
            values = tuple(state_values[column][index] for column in state_columns)
            records.append(
                (
                    index,
                    event_name,
                    int(event_precedences[index]),
                    int(normal_missing_ranks[index]),
                    machine_id,
                    machine_key,
                    machine_match_rank,
                    values,
                    tuple(_deterministic_value_key(value) for value in values),
                )
            )

        # Candidate rows which could be the final transition after applying
        # observed-machine, lifecycle, and normal-row precedence.
        if any(record[6] for record in records):
            final_candidates = [record for record in records if record[6] == 1]
        else:
            final_candidates = records
        max_precedence = max(record[2] for record in final_candidates)
        final_candidates = [
            record for record in final_candidates if record[2] == max_precedence
        ]
        max_normal_rank = max(record[3] for record in final_candidates)
        final_candidates = [
            record for record in final_candidates if record[3] == max_normal_rank
        ]

        conflicting_fields: set[str] = set()
        conflicting_count = 0
        conflict_cohorts: dict[object, list[tuple[object, ...]]] = {}
        for record in final_candidates:
            conflict_key: object
            if scheduler_conflict:
                conflict_key = record[1]
            else:
                conflict_key = (record[1], record[5])
            conflict_cohorts.setdefault(conflict_key, []).append(record)
        for conflict_rows in conflict_cohorts.values():
            if len(conflict_rows) <= 1:
                continue
            if scheduler_conflict:
                scheduler_index = state_columns.index("scheduler")
                distinct = {
                    record[8][scheduler_index]
                    for record in conflict_rows
                    if not pd.isna(record[7][scheduler_index])
                }
                if len(distinct) > 1:
                    conflicting_fields.add("scheduler")
                    conflicting_count = max(conflicting_count, len(conflict_rows))
            else:
                cohort_conflicts = {
                    column
                    for field_index, column in enumerate(state_columns)
                    if len(
                        {
                            record[8][field_index]
                            for record in conflict_rows
                            if not pd.isna(record[7][field_index])
                        }
                    )
                    > 1
                }
                if cohort_conflicts:
                    conflicting_fields.update(cohort_conflicts)
                    conflicting_count = max(conflicting_count, len(conflict_rows))

        # Apply sparse updates in exactly the former DataFrame sort order.  A
        # later row supplies one coherent snapshot; NULL fields retain state.
        records.sort(
            key=lambda record: (
                record[6],
                record[2],
                record[3],
                _deterministic_value_key(record[1]),
                record[5],
                *record[8],
            )
        )
        overwritten_fields: set[str] = set()
        for record in records:
            for field_index, column in enumerate(state_columns):
                value = record[7][field_index]
                if not pd.isna(value):
                    current_state[column] = value
                    overwritten_fields.add(column)
            if has_machine_id and not pd.isna(record[4]):
                current_machine = record[4]

        unresolved_fields.difference_update(overwritten_fields)
        if conflicting_fields:
            unresolved_fields.update(conflicting_fields)
            ambiguity_reason = (
                "same_time_conflicting_scheduler_state"
                if scheduler_conflict
                else "same_time_conflicting_instance_state"
            )
            ambiguity_candidate_count = max(ambiguity_candidate_count, conflicting_count)
        elif not unresolved_fields:
            ambiguity_reason = ""
            ambiguity_candidate_count = 0

        if all(pd.isna(value) for value in current_state.values()):
            continue
        snapshot = (
            dict(current_state),
            bool(unresolved_fields),
            ambiguity_reason,
            int(ambiguity_candidate_count),
            current_machine,
        )
        if effective_time <= int(arrival_time_us):
            latest_before = snapshot
        else:
            return result(snapshot, True)

    if latest_before is not None:
        return result(latest_before, False)
    return empty_state, False, False, "", 0, np.nan


def _sparse_state_at_time(
    history: pd.DataFrame,
    arrival_time_us: int,
    state_columns: list[str],
) -> tuple[dict[str, object], bool]:
    """Compatibility wrapper for legacy callers."""

    state, used_post, _, _, _, _ = _resolve_sparse_state_at_time(
        history, arrival_time_us, state_columns
    )
    return state, used_post


def _instance_states_at_arrival(vm_rows: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    state_columns = ["priority", "scheduling_class", "resource_request_cpu", "resource_request_mem"]
    groups = {
        key: group
        for key, group in events.groupby(["collection_id", "instance_index"], sort=False, dropna=False)
    }
    rows: list[dict[str, object]] = []
    for vm in vm_rows.itertuples(index=False):
        key = (vm.collection_id, vm.instance_index)
        history = groups.get(key, pd.DataFrame(columns=events.columns))
        episode_end = getattr(vm, "episode_end_time_us", None)
        if episode_end is not None and not pd.isna(episode_end) and not history.empty:
            history = history.loc[
                pd.to_numeric(history["time"], errors="coerce").lt(int(episode_end))
                | history["time"].isna()
            ]
        arrival_machine_ids = getattr(vm, "arrival_machine_ids", ())
        if not isinstance(arrival_machine_ids, tuple):
            arrival_machine_ids = ()
        state, used_post_arrival, ambiguous, reason, candidate_count, event_machine_id = (
            _resolve_sparse_state_at_time(
                history,
                int(vm.arrival_time_us),
                state_columns,
                arrival_machine_ids=arrival_machine_ids,
            )
        )
        rows.append(
            {
                "collection_id": vm.collection_id,
                "instance_index": vm.instance_index,
                **(
                    {"episode_index": int(vm.episode_index)}
                    if hasattr(vm, "episode_index")
                    else {}
                ),
                **state,
                "arrival_machine_id": event_machine_id,
                "arrival_state_used_post_arrival": used_post_arrival,
                "arrival_state_ambiguous": ambiguous,
                "arrival_state_ambiguity_reason": reason,
                "arrival_state_candidate_count": candidate_count,
            }
        )
    return pd.DataFrame(rows)


def _collection_states_at_arrival(vm_rows: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    state_columns = ["scheduler", "collection_type", "priority", "scheduling_class"]
    groups = {key: group for key, group in events.groupby("collection_id", sort=False, dropna=False)}
    rows: list[dict[str, object]] = []
    for vm in vm_rows.itertuples(index=False):
        state, used_post_arrival, ambiguous, reason, candidate_count, _ = (
            _resolve_sparse_state_at_time(
                groups.get(vm.collection_id, pd.DataFrame(columns=events.columns)),
                int(vm.arrival_time_us),
                state_columns,
                scheduler_conflict=True,
            )
        )
        rows.append(
            {
                "collection_id": vm.collection_id,
                "instance_index": vm.instance_index,
                **(
                    {"episode_index": int(vm.episode_index)}
                    if hasattr(vm, "episode_index")
                    else {}
                ),
                "scheduler": state["scheduler"],
                "collection_type": state["collection_type"],
                "collection_priority": state["priority"],
                "scheduler_state_used_post_arrival": used_post_arrival,
                "scheduler_state_ambiguous": ambiguous,
                "scheduler_state_ambiguity_reason": reason,
                "scheduler_state_candidate_count": candidate_count,
            }
        )
    return pd.DataFrame(rows)


def build_vm_requests(
    usage: pd.DataFrame,
    events: pd.DataFrame | None = None,
    collection_events: pd.DataFrame | None = None,
    machine_events: pd.DataFrame | None = None,
    max_instances: int = 5_000,
    seed: int = 42,
    min_usage_rows: int = 12,
    day_start_us: int | None = None,
    day_end_us: int | None = None,
    strict_event_state: bool = True,
    episode_unit: bool | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select VM-like lifecycle episodes and compute nominal resource requests.

    Production callers pass ``machine_events`` and therefore use strict
    episode-level preprocessing by default.  The ``None`` auto-mode preserves
    the legacy helper behavior for small callers that have no machine lifecycle
    fixture; it can be overridden explicitly for tests or reproduction.
    """

    if max_instances <= 0:
        raise ValueError("max_instances must be positive")
    if min_usage_rows <= 0:
        raise ValueError("min_usage_rows must be positive")
    events_norm = _normalize_events(events)
    collection_norm = _normalize_collection_events(collection_events)
    episode_unit = bool(machine_events is not None) if episode_unit is None else bool(episode_unit)
    episode_quality_diagnostics: dict[str, object] = {}
    usage_input = usage
    if episode_unit:
        if machine_events is None or machine_events.empty:
            raise ValueError("machine_events is required for strict episode preprocessing")
        if day_start_us is None or day_end_us is None:
            raise ValueError("explicit day_start_us/day_end_us are required for episode preprocessing")
        usage_input, episode_quality_diagnostics = _prepare_strict_episode_usage(
            usage,
            events_norm,
            machine_events,
            day_start_us=int(day_start_us),
            day_end_us=int(day_end_us),
        )
    usage_norm = _normalize_usage(
        usage_input, day_start_us=day_start_us, day_end_us=day_end_us
    )
    usage_overlap_diagnostics = dict(usage_norm.attrs.get("usage_overlap_diagnostics", {}))
    if episode_quality_diagnostics:
        usage_overlap_diagnostics["exact_duplicate_row_count"] = int(
            episode_quality_diagnostics["exact_duplicate_source_row_count_removed"]
        )
    resolved_day_start_us = int(usage_norm.attrs["day_start_us"])
    resolved_day_end_us = int(usage_norm.attrs["day_end_us"])
    key_columns = _unit_key_columns(usage_norm)

    counts = (
        usage_norm.groupby(key_columns, dropna=False)
        .size()
        .reset_index(name="usage_rows")
    )
    candidates = counts.loc[counts["usage_rows"] >= min_usage_rows].copy()
    if candidates.empty:
        raise ValueError(f"no VM-like candidates have at least {min_usage_rows} reconstructed 5-minute buckets")

    arrival_rows = (
        usage_norm.groupby(key_columns, as_index=False, dropna=False)
        .agg(
            arrival_time_us=("first_observed_us", "min"),
            arrival_machine_ids=("arrival_machine_ids", "first"),
            **(
                {
                    "episode_start_time_us": ("episode_start_time_us", "first"),
                    "episode_end_time_us": ("episode_end_time_us", "first"),
                    "termination_time_us": ("termination_time_us", "first"),
                    "termination_reason": ("termination_reason", "first"),
                    "horizon_censored": ("horizon_censored", "first"),
                    "termination_time_uncertain": ("termination_time_uncertain", "first"),
                    "episode_start_inferred_from_update": (
                        "episode_start_inferred_from_update",
                        "first",
                    ),
                    "episode_start_inferred_from_usage": (
                        "episode_start_inferred_from_usage",
                        "first",
                    ),
                }
                if episode_unit
                else {}
            ),
        )
    )
    if episode_unit:
        arrival_rows["arrival_time_us"] = arrival_rows["episode_start_time_us"].astype(
            "int64"
        )
    candidates = candidates.merge(
        arrival_rows,
        on=key_columns,
        how="left",
        validate="one_to_one",
    )
    if episode_unit:
        episode_quality_diagnostics["after_min_usage_rows"] = _episode_stage_summary(
            candidates
        )
        episode_quality_diagnostics["minimum_reconstructed_usage_buckets"] = int(
            min_usage_rows
        )
    instance_state = _instance_states_at_arrival(candidates, events_norm)
    candidates = candidates.merge(
        instance_state,
        on=key_columns,
        how="left",
        validate="one_to_one",
    )
    collection_state = _collection_states_at_arrival(candidates, collection_norm)
    candidates = candidates.merge(
        collection_state,
        on=key_columns,
        how="left",
        validate="one_to_one",
    )
    for flag in ["arrival_state_ambiguous", "scheduler_state_ambiguous"]:
        candidates[flag] = candidates[flag].fillna(False).astype(bool)

    provisional_scheduler = candidates["scheduler"].fillna("").map(_normalize_scheduler)
    provisional_priority = pd.to_numeric(candidates["priority"], errors="coerce")
    candidates["audit_class"] = "spot"
    candidates.loc[provisional_priority.ge(120), "audit_class"] = "on_demand"
    candidates.loc[provisional_scheduler.eq("SCHEDULER_BATCH"), "audit_class"] = (
        "batch_candidate"
    )
    ambiguous_mask = (
        candidates["arrival_state_ambiguous"] | candidates["scheduler_state_ambiguous"]
    )
    ambiguous_rows = candidates.loc[ambiguous_mask].copy()
    ambiguity_diagnostics: dict[str, object] = {
        "strict_event_state": bool(strict_event_state),
        "eligible_candidate_count_before_ambiguity_filter": int(len(candidates)),
        "ambiguous_candidate_count": int(ambiguous_mask.sum()),
        "excluded_ambiguous_candidate_count": int(ambiguous_mask.sum())
        if strict_event_state
        else 0,
        "ambiguous_candidate_count_by_class": {
            str(key): int(value)
            for key, value in ambiguous_rows["audit_class"].value_counts().sort_index().items()
        },
        "arrival_state_ambiguous_count": int(candidates["arrival_state_ambiguous"].sum()),
        "scheduler_state_ambiguous_count": int(candidates["scheduler_state_ambiguous"].sum()),
        "arrival_ambiguity_reason_counts": {
            str(key): int(value)
            for key, value in ambiguous_rows.loc[
                ambiguous_rows["arrival_state_ambiguous"],
                "arrival_state_ambiguity_reason",
            ]
            .value_counts()
            .sort_index()
            .items()
        },
        "scheduler_ambiguity_reason_counts": {
            str(key): int(value)
            for key, value in ambiguous_rows.loc[
                ambiguous_rows["scheduler_state_ambiguous"],
                "scheduler_state_ambiguity_reason",
            ]
            .value_counts()
            .sort_index()
            .items()
        },
        "ambiguous_candidates": [
            {
                "collection_id": _json_scalar(row.collection_id),
                "instance_index": _json_scalar(row.instance_index),
                "provisional_class": str(row.audit_class),
                "arrival_state_ambiguous": bool(row.arrival_state_ambiguous),
                "arrival_reason": str(row.arrival_state_ambiguity_reason or ""),
                "arrival_candidate_count": int(row.arrival_state_candidate_count or 0),
                "scheduler_state_ambiguous": bool(row.scheduler_state_ambiguous),
                "scheduler_reason": str(row.scheduler_state_ambiguity_reason or ""),
                "scheduler_candidate_count": int(row.scheduler_state_candidate_count or 0),
            }
            for row in ambiguous_rows.itertuples(index=False)
        ],
    }
    if strict_event_state:
        candidates = candidates.loc[~ambiguous_mask].copy()
    if candidates.empty:
        raise ValueError("no valid VM-like candidates remain after strict event-state ambiguity filtering")
    if episode_unit:
        episode_quality_diagnostics[
            "after_event_state_quality_filter_before_sampling"
        ] = _episode_stage_summary(candidates)

    candidates["sample_hash"] = [
        _stable_key_hash(
            row.collection_id,
            row.instance_index,
            seed,
            getattr(row, "episode_index", None),
        )
        for row in candidates.itertuples(index=False)
    ]
    candidates = candidates.sort_values(
        ["sample_hash", *key_columns], kind="stable"
    ).head(max_instances)
    if episode_unit:
        episode_quality_diagnostics["selected_after_deterministic_hash_head"] = (
            _episode_stage_summary(candidates)
        )
        episode_quality_diagnostics["max_instances"] = int(max_instances)

    selected_usage = usage_norm.merge(candidates[key_columns], on=key_columns)
    selected_usage = selected_usage.sort_values(
        key_columns + ["t5_day"], kind="stable"
    ).reset_index(drop=True)

    grouped = selected_usage.groupby(key_columns, dropna=False)
    vm_requests = grouped.agg(
        arrival_t5=("t5_day", "min"),
        departure_t5=("t5_day", lambda s: int(s.max()) + 1),
        first_observed_us=("first_observed_us", "min"),
        last_observed_us=("last_observed_us", "max"),
        p95_cpu_usage=("cpu_usage", lambda s: float(s.quantile(0.95))),
        p95_mem_usage=("mem_usage", lambda s: float(s.quantile(0.95))),
        avg_cpu_usage=("cpu_usage", "mean"),
        avg_mem_usage=("mem_usage", "mean"),
        max_average_cpu_usage=("cpu_usage", "max"),
        max_average_mem_usage=("mem_usage", "max"),
        source_max_cpu_usage=("vm_source_max_cpu_usage", "max"),
        source_max_mem_usage=("vm_source_max_mem_usage", "max"),
        **(
            {
                "episode_start_time_us": ("episode_start_time_us", "first"),
                "episode_end_time_us": ("episode_end_time_us", "first"),
                "termination_time_us": ("termination_time_us", "first"),
                "termination_reason": ("termination_reason", "first"),
                "horizon_censored": ("horizon_censored", "first"),
                "termination_time_uncertain": ("termination_time_uncertain", "first"),
                "episode_start_inferred_from_update": (
                    "episode_start_inferred_from_update",
                    "first",
                ),
                "episode_start_inferred_from_usage": (
                    "episode_start_inferred_from_usage",
                    "first",
                ),
            }
            if episode_unit
            else {}
        ),
    ).reset_index()
    if episode_unit:
        vm_requests["arrival_time_us"] = vm_requests["episode_start_time_us"].astype(
            "int64"
        )
        vm_requests["arrival_t5"] = np.floor(
            (vm_requests["episode_start_time_us"] - resolved_day_start_us) / T5_US
        ).astype(int)
        vm_requests["departure_t5"] = np.ceil(
            (vm_requests["episode_end_time_us"] - resolved_day_start_us) / T5_US
        ).astype(int)
        vm_requests["arrival_t5"] = vm_requests["arrival_t5"].clip(0, HORIZON_T5 - 1)
        vm_requests["departure_t5"] = vm_requests["departure_t5"].clip(1, HORIZON_T5)
    else:
        vm_requests["arrival_time_us"] = vm_requests["first_observed_us"].astype("int64")
    candidate_state_columns = [
        "collection_id",
        "instance_index",
        *(["episode_index"] if episode_unit else []),
        "sample_hash",
        "arrival_machine_ids",
        "priority",
        "scheduling_class",
        "resource_request_cpu",
        "resource_request_mem",
        "arrival_machine_id",
        "arrival_state_used_post_arrival",
        "arrival_state_ambiguous",
        "arrival_state_ambiguity_reason",
        "arrival_state_candidate_count",
        "scheduler",
        "collection_type",
        "collection_priority",
        "scheduler_state_used_post_arrival",
        "scheduler_state_ambiguous",
        "scheduler_state_ambiguity_reason",
        "scheduler_state_candidate_count",
    ]
    vm_requests = vm_requests.merge(
        candidates[candidate_state_columns],
        on=key_columns,
        how="left",
        validate="one_to_one",
    )

    for resource in ["cpu", "mem"]:
        request_column = f"resource_request_{resource}"
        source_max_column = f"source_max_{resource}_usage"
        average_max_column = f"max_average_{resource}_usage"
        fallback_column = f"max_{resource}_usage_fallback_from_average"
        request_fallback_column = f"{request_column}_fallback_zero"

        vm_requests[request_column] = pd.to_numeric(vm_requests[request_column], errors="coerce")
        vm_requests[request_fallback_column] = vm_requests[request_column].isna()
        vm_requests[request_column] = vm_requests[request_column].fillna(0.0)
        if (vm_requests[request_column] < 0).any():
            raise ValueError(f"arrival-state {request_column} contains negative values")

        vm_requests[source_max_column] = pd.to_numeric(vm_requests[source_max_column], errors="coerce")
        vm_requests[fallback_column] = vm_requests[source_max_column].isna()
        if episode_unit and resource == "cpu" and vm_requests[fallback_column].any():
            raise ValueError(
                "strict episode preprocessing retained an episode without cpu_usage_distribution p100"
            )
        vm_requests[f"max_{resource}_usage"] = vm_requests[source_max_column].where(
            ~vm_requests[fallback_column], vm_requests[average_max_column]
        )
        q_values = np.maximum(
            vm_requests[request_column].to_numpy(dtype=float),
            vm_requests[f"max_{resource}_usage"].to_numpy(dtype=float),
        )
        vm_requests[f"q_{resource}"] = np.maximum(q_values, EPSILON)

    # p100 is preserved without a capacity cap.  Values above one trace-wide
    # normalized unit remain an audit flag, not a scaling rule.
    vm_requests["cpu_p100_gt_one"] = pd.to_numeric(
        vm_requests["source_max_cpu_usage"], errors="coerce"
    ).gt(1.0)

    vm_requests["lifetime_t5"] = vm_requests["departure_t5"] - vm_requests["arrival_t5"]

    vm_requests["scheduler"] = vm_requests["scheduler"].fillna("").map(_normalize_scheduler)
    priority = pd.to_numeric(vm_requests["priority"], errors="coerce")

    vm_requests["class"] = "spot"
    vm_requests["class_rule"] = (
        "spot: proxy label for priority < 120 or missing priority; Google 2019 tiers are "
        "0-99 free/best-effort, 100-115 BE/BEB, 116-119 mid; not an observed cloud purchase class"
    )

    high_priority = priority >= 120
    vm_requests.loc[high_priority, "class"] = "on_demand"
    vm_requests.loc[high_priority, "class_rule"] = (
        "on_demand: proxy label for priority >= 120 production-tier Google 2019 workload; "
        "not an observed cloud purchase class"
    )

    scheduler_batch = vm_requests["scheduler"].eq("SCHEDULER_BATCH")
    vm_requests.loc[scheduler_batch, "class"] = "batch_candidate"
    vm_requests.loc[scheduler_batch, "class_rule"] = (
        "batch_candidate: proxy label because collection scheduler == SCHEDULER_BATCH; "
        "not an observed cloud purchase class"
    )

    vm_requests["spot_revenue"] = np.where(vm_requests["class"].eq("spot"), 0.05 * vm_requests["q_cpu"], 0.0)
    vm_requests["migration_energy_coeff"] = np.where(
        vm_requests["class"].eq("on_demand"),
        0.01 * vm_requests["q_cpu"] + 0.005 * vm_requests["q_mem"],
        0.0,
    )

    vm_requests = vm_requests.sort_values(["sample_hash", *key_columns])
    vm_requests = vm_requests.reset_index(drop=True)
    if episode_unit:
        vm_requests["stable_episode_key"] = [
            _stable_episode_key(row.collection_id, row.instance_index, row.episode_index)
            for row in vm_requests.itertuples(index=False)
        ]
        vm_requests["vm_id"] = [
            "ep_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
            for key in vm_requests["stable_episode_key"]
        ]
        if vm_requests["vm_id"].duplicated().any():
            raise ValueError("stable episode vm_id hash collision")
    else:
        vm_requests["stable_episode_key"] = [
            f"legacy:{_deterministic_value_key(row.collection_id)}:"
            f"{_deterministic_value_key(row.instance_index)}"
            for row in vm_requests.itertuples(index=False)
        ]
        vm_requests["vm_id"] = [f"vm{i:05d}" for i in range(len(vm_requests))]
        for column, default in {
            "episode_index": 0,
            "episode_start_time_us": vm_requests["arrival_time_us"],
            "episode_end_time_us": resolved_day_start_us
            + vm_requests["departure_t5"] * T5_US,
            "termination_time_us": np.nan,
            "termination_reason": HORIZON_CENSORED,
            "horizon_censored": True,
            "termination_time_uncertain": False,
            "episode_start_inferred_from_update": False,
            "episode_start_inferred_from_usage": True,
        }.items():
            vm_requests[column] = default

    for resource in ["cpu", "mem"]:
        q = vm_requests[f"q_{resource}"].to_numpy(dtype=float)
        request = vm_requests[f"resource_request_{resource}"].to_numpy(dtype=float)
        observed_max = vm_requests[f"max_{resource}_usage"].to_numpy(dtype=float)
        if not np.isfinite(q).all() or (q <= 0).any():
            raise ValueError(f"q_{resource} must be finite and positive")
        if (q + 1e-12 < request).any() or (q + 1e-12 < observed_max).any():
            raise ValueError(f"q_{resource} invariant violated: q must cover arrival request and observed maximum")

    key_to_vm = vm_requests[key_columns + ["vm_id"]]
    observed = selected_usage.merge(key_to_vm, on=key_columns, how="inner")
    observed = observed[
        [
            "vm_id",
            "collection_id",
            "instance_index",
            *(["episode_index"] if "episode_index" in observed.columns else []),
            "t5",
            "t5_day",
            "t_hour",
            "cpu_usage",
            "mem_usage",
            "max_cpu_usage",
            "max_mem_usage",
            "assigned_memory",
            "coverage_us",
            "coverage_ratio",
            "overlap_source_row_count",
            "overlap_conflict_flag",
            "duplicated_timeline_us",
        ]
    ]
    observed = observed.sort_values(["vm_id", "t5_day"]).reset_index(drop=True)

    output = vm_requests[VM_REQUEST_COLUMNS].copy()
    fallback_details: dict[str, object] = {}
    for flag in [
        "resource_request_cpu_fallback_zero",
        "resource_request_mem_fallback_zero",
        "max_cpu_usage_fallback_from_average",
        "max_mem_usage_fallback_from_average",
        "arrival_state_used_post_arrival",
        "scheduler_state_used_post_arrival",
        "arrival_state_ambiguous",
        "scheduler_state_ambiguous",
        "cpu_p100_gt_one",
    ]:
        mask = vm_requests[flag].fillna(False).astype(bool)
        fallback_details[f"{flag}_count"] = int(mask.sum())
    derivation_diagnostics = {
        "sampling_key": key_columns,
        "sampling_hash": (
            "blake2b(seed:collection_id:instance_index:episode_index, digest_size=8)"
            if episode_unit
            else "blake2b(seed:collection_id:instance_index, digest_size=8)"
        ),
        "seed": int(seed),
        "minimum_reconstructed_usage_buckets": int(min_usage_rows),
        "request_state_policy": (
            "select one sparse instance state row forward-filled through the exact first valid "
            "overlap time; if no row exists at or before arrival, select the earliest forward-filled "
            "post-arrival row; missing resource requests then fall back to zero"
        ),
        "maximum_usage_policy": (
            "CPU maximum is the episode maximum of cpu_usage_distribution[10] (p100), with no "
            "fallback; memory remains the source maximum_usage.memory with average fallback only "
            "when memory maximum is unavailable"
            if episode_unit
            else "legacy per-resource source maximum with average fallback"
        ),
        "same_time_event_policy": (
            "explicit Google v3 enum-to-lifecycle precedence; normal missing_type rows and first-observed "
            "usage machine matches are preferred; unresolved coherent-state conflicts are excluded before "
            "the unchanged deterministic hash head when strict_event_state is true"
        ),
        "event_state_ambiguity": ambiguity_diagnostics,
        "usage_interval_overlap": usage_overlap_diagnostics,
        "cpu_p100_gt_one_count": int(
            vm_requests["cpu_p100_gt_one"].sum()
        ),
        "cpu_p100_gt_one_raw_summary": _numeric_summary(
            vm_requests.loc[vm_requests["cpu_p100_gt_one"], "source_max_cpu_usage"]
        ),
        "episode_quality_filter": episode_quality_diagnostics,
        **fallback_details,
    }
    output.attrs["vm_request_derivation"] = derivation_diagnostics
    output.attrs["usage_overlap_diagnostics"] = usage_overlap_diagnostics
    output.attrs["event_state_ambiguity"] = ambiguity_diagnostics
    output.attrs["episode_quality_filter"] = episode_quality_diagnostics
    output.attrs["day_start_us"] = resolved_day_start_us
    output.attrs["day_end_us"] = resolved_day_end_us
    observed.attrs.update(output.attrs)
    return output, observed


def build_usage_scenarios(
    observed_usage: pd.DataFrame,
    vm_requests: pd.DataFrame,
    num_scenarios: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    observed_attrs = dict(observed_usage.attrs)
    rng = np.random.default_rng(seed)
    base = observed_usage.merge(vm_requests[["vm_id", "class", "q_cpu", "q_mem"]], on="vm_id", how="left")
    if base[["class", "q_cpu", "q_mem"]].isna().any().any():
        raise ValueError("observed usage contains VM IDs missing from vm_requests")

    frames = []
    for scenario_id in range(num_scenarios):
        scenario = base.copy()
        scenario["scenario_id"] = scenario_id
        if scenario_id > 0:
            cpu_sigma = 0.22
            mem_sigma = 0.08
            cpu_noise = rng.lognormal(mean=-0.5 * cpu_sigma**2, sigma=cpu_sigma, size=len(scenario))
            mem_noise = rng.lognormal(mean=-0.5 * mem_sigma**2, sigma=mem_sigma, size=len(scenario))
            scenario["cpu_usage"] = scenario["cpu_usage"] * cpu_noise
            scenario["mem_usage"] = scenario["mem_usage"] * mem_noise

            # Every synthetic demand draw is a configured-resource-bounded
            # usage sample, regardless of VM class.  Scenario 0 remains the
            # unmodified observed trace.
            scenario["cpu_usage"] = np.clip(
                scenario["cpu_usage"].to_numpy(dtype=float),
                0.0,
                scenario["q_cpu"].to_numpy(dtype=float),
            )
            scenario["mem_usage"] = np.clip(
                scenario["mem_usage"].to_numpy(dtype=float),
                0.0,
                scenario["q_mem"].to_numpy(dtype=float),
            )

        scenario["cpu_usage"] = scenario["cpu_usage"].clip(lower=0.0)
        scenario["mem_usage"] = scenario["mem_usage"].clip(lower=0.0)
        audit_columns = [column for column in USAGE_BUCKET_AUDIT_COLUMNS if column in scenario.columns]
        frames.append(scenario[USAGE_SCENARIO_COLUMNS + audit_columns])

    output = pd.concat(frames, ignore_index=True).sort_values(
        ["scenario_id", "vm_id", "t5_day"]
    ).reset_index(drop=True)
    output.attrs.update(observed_attrs)
    return output


def build_hourly_usage_scenarios(usage_5min_scenarios: pd.DataFrame) -> pd.DataFrame:
    if usage_5min_scenarios.empty:
        return pd.DataFrame(columns=HOURLY_USAGE_SCENARIO_COLUMNS)
    missing = set(USAGE_SCENARIO_COLUMNS) - set(usage_5min_scenarios.columns)
    if missing:
        raise ValueError(f"5-minute usage scenarios missing columns: {sorted(missing)}")
    if "coverage_us" not in usage_5min_scenarios:
        raise ValueError("5-minute usage scenarios missing coverage_us for duration-weighted CPU")
    work = usage_5min_scenarios.copy()
    work["coverage_us"] = pd.to_numeric(work["coverage_us"], errors="coerce")
    coverage = work["coverage_us"].to_numpy(dtype=float)
    if (
        not np.isfinite(coverage).all()
        or (coverage <= 0).any()
        or (coverage > T5_US).any()
        or not np.equal(coverage, np.floor(coverage)).all()
    ):
        raise ValueError("coverage_us must be a finite integer in (0, 300 seconds]")
    work["cpu_coverage_weighted"] = work["cpu_usage"] * work["coverage_us"]
    work["mem_coverage_weighted"] = work["mem_usage"] * work["coverage_us"]
    hourly = (
        work.groupby(["scenario_id", "vm_id", "t_hour"], as_index=False)
        .agg(
            cpu_coverage_weighted=("cpu_coverage_weighted", "sum"),
            mem_coverage_weighted=("mem_coverage_weighted", "sum"),
            coverage_us=("coverage_us", "sum"),
        )
        .sort_values(["scenario_id", "vm_id", "t_hour"])
        .reset_index(drop=True)
    )
    hourly["cpu_usage"] = hourly["cpu_coverage_weighted"] / hourly["coverage_us"]
    hourly["mem_usage"] = hourly["mem_coverage_weighted"] / hourly["coverage_us"]
    return hourly[HOURLY_USAGE_SCENARIO_COLUMNS]


def build_scenario_probabilities(num_scenarios: int = 5) -> pd.DataFrame:
    if num_scenarios <= 0:
        raise ValueError("num_scenarios must be positive")
    probability = 1.0 / num_scenarios
    return pd.DataFrame(
        [{"scenario_id": scenario_id, "probability": probability} for scenario_id in range(num_scenarios)],
        columns=SCENARIO_PROBABILITY_COLUMNS,
    )


def build_model_params(
    alpha: float = 0.95,
    epsilon: float = 0.05,
    soc_init: float = 0.50,
    ess_charge_efficiency: float = 0.92,
    ess_discharge_efficiency: float = 0.92,
) -> dict[str, float]:
    return {
        "alpha": float(alpha),
        "epsilon": float(epsilon),
        "soc_init": float(soc_init),
        "ess_charge_efficiency": float(ess_charge_efficiency),
        "ess_discharge_efficiency": float(ess_discharge_efficiency),
    }


def _eviction_t5_by_vm(
    vm_requests: pd.DataFrame,
    events: pd.DataFrame | None,
    min_t5: int,
    day_start_us: int | None = None,
) -> dict[str, int]:
    if (
        {"termination_reason", "termination_time_us"}.issubset(vm_requests.columns)
        and vm_requests["vm_id"].astype(str).str.startswith("ep_").all()
    ):
        evicted = vm_requests.loc[
            vm_requests["termination_reason"].astype(str).eq("EVICT")
            & pd.to_numeric(vm_requests["termination_time_us"], errors="coerce").notna()
        ].copy()
        if evicted.empty:
            return {}
        origin = (
            int(day_start_us)
            if day_start_us is not None
            else int(min_t5) * T5_US
        )
        evicted["t5_day"] = np.floor(
            (pd.to_numeric(evicted["termination_time_us"], errors="coerce") - origin)
            / T5_US
        ).astype(int)
        evicted = evicted.loc[evicted["t5_day"].between(0, HORIZON_T5 - 1)]
        return evicted.set_index("vm_id")["t5_day"].astype(int).to_dict()

    events_norm = _normalize_events(events)
    if events_norm.empty:
        return {}

    vm_keys = vm_requests[["vm_id", "collection_id", "instance_index"]]
    joined = events_norm.merge(vm_keys, on=["collection_id", "instance_index"], how="inner")
    joined = joined.loc[joined["event_type"].map(_is_evict_event)].copy()
    joined = joined.dropna(subset=["time"])
    if joined.empty:
        return {}

    if day_start_us is None:
        joined["t5_day"] = np.floor(joined["time"] / T5_US).astype(int) - min_t5
    else:
        joined["t5_day"] = np.floor((joined["time"] - int(day_start_us)) / T5_US).astype(int)
    joined = joined.loc[joined["t5_day"].between(0, HORIZON_T5 - 1)]
    return joined.groupby("vm_id")["t5_day"].min().astype(int).to_dict()


def build_spot_preemption_scenarios(
    vm_requests: pd.DataFrame,
    observed_usage: pd.DataFrame,
    events: pd.DataFrame | None = None,
    num_scenarios: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    spot_vms = vm_requests.loc[vm_requests["class"] == "spot"].copy()
    if spot_vms.empty:
        return pd.DataFrame(columns=SPOT_PREEMPTION_COLUMNS)

    rng = np.random.default_rng(seed + 17)
    min_t5 = int(observed_usage["t5"].min()) if "t5" in observed_usage and not observed_usage.empty else 0
    day_start_us = observed_usage.attrs.get("day_start_us")
    actual_evict = _eviction_t5_by_vm(
        vm_requests, events, min_t5, day_start_us=int(day_start_us) if day_start_us is not None else None
    )
    rows: list[dict[str, int | str]] = []

    for vm in spot_vms.itertuples(index=False):
        observed_times = observed_usage.loc[observed_usage["vm_id"] == vm.vm_id, "t5_day"]
        if observed_times.empty:
            t5_values = range(int(vm.arrival_t5), int(vm.departure_t5))
        else:
            t5_values = sorted(int(t) for t in observed_times.unique())

        for scenario_id in range(num_scenarios):
            if vm.vm_id in actual_evict:
                preempt_t5 = actual_evict[vm.vm_id]
            else:
                priority = 0.0 if pd.isna(vm.priority) else float(vm.priority)
                hazard = min(0.08, 0.003 + max(0.0, 100.0 - priority) / 100.0 * 0.025)
                preempt_t5 = None
                for t5 in t5_values:
                    if scenario_id > 0 and rng.random() < hazard:
                        preempt_t5 = t5
                        break

            for t5 in t5_values:
                active = int(preempt_t5 is None or t5 < int(preempt_t5))
                rows.append(
                    {
                        "scenario_id": scenario_id,
                        "vm_id": vm.vm_id,
                        "t5_day": int(t5),
                        "t_hour": int(t5) // 12,
                        "active": active,
                        "preempted": 1 - active,
                    }
                )

    return pd.DataFrame(rows, columns=SPOT_PREEMPTION_COLUMNS).sort_values(
        ["scenario_id", "vm_id", "t5_day"]
    ).reset_index(drop=True)


def build_batch_outputs(
    vm_requests: pd.DataFrame,
    observed_usage: pd.DataFrame,
    max_families: int = 10,
    num_scenarios: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    batch = vm_requests.loc[vm_requests["class"] == "batch_candidate"].copy()
    if batch.empty:
        return pd.DataFrame(columns=BATCH_FAMILY_COLUMNS), pd.DataFrame(columns=BATCH_WORKLOAD_COLUMNS)

    rounded = batch.assign(cpu_bin=batch["q_cpu"].round(2), mem_bin=batch["q_mem"].round(2))
    group_count = rounded[["cpu_bin", "mem_bin"]].drop_duplicates().shape[0]
    if group_count <= max_families:
        family_keys = rounded[["cpu_bin", "mem_bin"]].drop_duplicates().sort_values(["cpu_bin", "mem_bin"])
        family_keys["family_id"] = [f"batch{k:02d}" for k in range(len(family_keys))]
        batch = rounded.merge(family_keys, on=["cpu_bin", "mem_bin"], how="left")
    else:
        batch = batch.sort_values(["q_cpu", "q_mem", "vm_id"]).reset_index(drop=True)
        batch["family_bucket"] = np.floor(np.arange(len(batch)) * max_families / len(batch)).astype(int)
        batch["family_id"] = batch["family_bucket"].map(lambda i: f"batch{i:02d}")

    family_stats = []
    for family_id, group in batch.groupby("family_id", sort=True):
        q_cpu = float(group["q_cpu"].max())
        q_mem = float(group["q_mem"].max())
        avg_cpu = float(group["avg_cpu_usage"].mean())
        avg_mem = float(group["avg_mem_usage"].mean())
        family_stats.append(
            {
                "family_id": family_id,
                "q_cpu_B": q_cpu,
                "q_mem_B": q_mem,
                "W_k": int(len(group)),
                "rho_cpu_B": min(1.0, avg_cpu / max(q_cpu, EPSILON)),
                "rho_mem_B": min(1.0, avg_mem / max(q_mem, EPSILON)),
                "base_cpu": avg_cpu,
                "base_mem": avg_mem,
                "startup_cpu": 0.10 * q_cpu,
                "startup_mem": 0.05 * q_mem,
            }
        )
    families = pd.DataFrame(family_stats, columns=BATCH_FAMILY_COLUMNS)

    observed_batch = observed_usage.merge(batch[["vm_id", "family_id"]], on="vm_id", how="inner")
    hourly = (
        observed_batch.groupby(["family_id", "t_hour"], as_index=False)
        .agg(workload_volume=("vm_id", "nunique"), cpu_workload=("cpu_usage", "sum"), mem_workload=("mem_usage", "sum"))
    )

    rng = np.random.default_rng(seed + 29)
    family_q_cpu = families.set_index("family_id")["q_cpu_B"].to_dict()
    family_q_mem = families.set_index("family_id")["q_mem_B"].to_dict()
    frames = []
    for scenario_id in range(num_scenarios):
        scenario = hourly.copy()
        scenario["scenario_id"] = scenario_id
        if scenario_id > 0 and not scenario.empty:
            scale = rng.lognormal(mean=-0.5 * 0.12**2, sigma=0.12, size=len(scenario))
            scenario["workload_volume"] = scenario["workload_volume"] * scale
            scenario["cpu_workload"] = scenario["cpu_workload"] * scale
            scenario["mem_workload"] = scenario["mem_workload"] * rng.lognormal(
                mean=-0.5 * 0.06**2, sigma=0.06, size=len(scenario)
            )
            scenario["workload_volume"] = scenario["workload_volume"].clip(lower=0.0)
            cpu_upper = (
                scenario["family_id"].map(family_q_cpu).astype(float)
                * scenario["workload_volume"]
            )
            mem_upper = (
                scenario["family_id"].map(family_q_mem).astype(float)
                * scenario["workload_volume"]
            )
            scenario["cpu_workload"] = np.clip(
                scenario["cpu_workload"].to_numpy(dtype=float),
                0.0,
                cpu_upper.to_numpy(dtype=float),
            )
            scenario["mem_workload"] = np.clip(
                scenario["mem_workload"].to_numpy(dtype=float),
                0.0,
                mem_upper.to_numpy(dtype=float),
            )
        frames.append(scenario[BATCH_WORKLOAD_COLUMNS])
    workload = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=BATCH_WORKLOAD_COLUMNS)
    workload = workload.sort_values(["scenario_id", "family_id", "t_hour"]).reset_index(drop=True)
    return families, workload


def _machine_event_code(value: object) -> int:
    if pd.isna(value):
        return 0
    if isinstance(value, (int, np.integer, float, np.floating)):
        return int(value)
    text = str(value).strip().upper()
    if text in {"1", "ADD", "MACHINE_ADD"} or text.endswith(".ADD"):
        return 1
    if text in {"2", "REMOVE", "MACHINE_REMOVE"} or text.endswith(".REMOVE"):
        return 2
    if text in {"3", "UPDATE", "MACHINE_UPDATE"} or text.endswith(".UPDATE"):
        return 3
    if text in {"0", "EVENT_TYPE_UNKNOWN", "UNKNOWN", ""}:
        return 0
    raise ValueError(f"unsupported machine event type: {value!r}")


def _collection_instance_kind(value: object) -> str:
    """Map the documented v3 collection type to the audited instance kind."""

    if pd.isna(value):
        return "unknown"
    if isinstance(value, (int, np.integer, float, np.floating)):
        code = int(value)
        if code == 0:
            return "task"
        if code == 1:
            return "alloc"
        return "unknown"
    text = str(value).strip().upper()
    for prefix in ("COLLECTION_TYPE_", "COLLECTIONTYPE."):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    if text in {"0", "JOB", "TASK"}:
        return "task"
    if text in {"1", "ALLOC", "ALLOC_SET", "ALLOCSET"}:
        return "alloc"
    return "unknown"


def _cpu_distribution_max(value: object) -> tuple[float | None, int]:
    """Return the p100 (last) CPU percentile and vector length.

    BigQuery/Arrow normally yields a list or ``numpy.ndarray`` for the repeated
    field.  JSON strings are accepted for older exported parquet/CSV fixtures.
    The v3 field has exactly 11 entries (p0, p10, ..., p100).  NULL and empty
    distributions are unavailable, never zero.  A nonempty malformed vector is
    rejected instead of silently treating another percentile as p100.
    """

    if value is None:
        return None, 0
    if not isinstance(value, (str, list, tuple, np.ndarray, pd.Series)) and pd.isna(value):
        return None, 0
    parsed = value
    if isinstance(parsed, str):
        text = parsed.strip()
        if not text:
            return None, 0
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid JSON cpu_usage_distribution") from exc
    if isinstance(parsed, pd.Series):
        parsed = parsed.tolist()
    if isinstance(parsed, np.ndarray):
        parsed = parsed.tolist()
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(
            "cpu_usage_distribution must be an ARRAY/list (or a JSON array string)"
        )
    if len(parsed) == 0:
        return None, 0
    if len(parsed) != 11:
        raise ValueError(
            f"cpu_usage_distribution must contain exactly 11 percentiles, got {len(parsed)}"
        )
    values: list[float] = []
    for item in parsed:
        if item is None or pd.isna(item):
            return None, len(parsed)
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError("cpu_usage_distribution contains a non-numeric value") from exc
        if not np.isfinite(number) or number < 0:
            raise ValueError(
                "cpu_usage_distribution contains a non-finite or negative value"
            )
        values.append(number)
    return values[-1], len(parsed)


def _machine_cpu_capacity_intervals(
    machine_events: pd.DataFrame,
    day_start_us: int,
    day_end_us: int,
) -> dict[object, list[tuple[int, int, float | None]]]:
    """Reconstruct active machine CPU-capacity intervals inside the horizon."""

    if machine_events is None or machine_events.empty:
        raise ValueError("machine_events is required and must not be empty")
    time_col = _require_column(
        machine_events, ["time", "event_time", "timestamp_us"], "machine event time"
    )
    machine_col = _require_column(machine_events, ["machine_id"], "machine_id")
    event_col = _require_column(machine_events, ["event_type", "type"], "machine event type")
    cpu_col = _require_column(
        machine_events, ["capacity_cpu", "capacity.cpus"], "machine CPU capacity"
    )
    events = pd.DataFrame(
        {
            "time": pd.to_numeric(machine_events[time_col], errors="coerce"),
            "machine_id": machine_events[machine_col],
            "event_code": machine_events[event_col].map(_machine_event_code),
            "capacity_cpu": pd.to_numeric(machine_events[cpu_col], errors="coerce"),
            "row_order": np.arange(len(machine_events), dtype=np.int64),
        }
    )
    invalid_identity = events[["time", "machine_id"]].isna().any(axis=1)
    if invalid_identity.any():
        raise ValueError(
            "machine_events contains "
            f"{int(invalid_identity.sum())} rows with NULL time or machine_id"
        )
    invalid_capacity = events["capacity_cpu"].notna() & (
        ~np.isfinite(events["capacity_cpu"]) | events["capacity_cpu"].lt(0)
    )
    if invalid_capacity.any():
        raise ValueError("machine_events contains non-finite or negative capacity_cpu")
    events = events.loc[events["time"].lt(int(day_end_us))].copy()
    events["time"] = events["time"].astype("int64")
    events = events.sort_values(["machine_id", "time", "row_order"], kind="stable")

    def apply_event(
        row: object,
        active: bool,
        capacity_cpu: float | None,
    ) -> tuple[bool, float | None]:
        code = int(row.event_code)
        row_cpu = None if pd.isna(row.capacity_cpu) else float(row.capacity_cpu)
        if code in {1, 3} and row_cpu is not None:
            capacity_cpu = row_cpu
        if code == 1:
            active = True
        elif code == 2:
            active = False
            capacity_cpu = None
        return active, capacity_cpu

    intervals: dict[object, list[tuple[int, int, float | None]]] = {}
    for machine_id, history in events.groupby("machine_id", sort=False, dropna=False):
        active = False
        capacity_cpu: float | None = None
        rows = list(history.itertuples(index=False))
        position = 0
        while position < len(rows) and int(rows[position].time) < int(day_start_us):
            active, capacity_cpu = apply_event(rows[position], active, capacity_cpu)
            position += 1
        cursor = int(day_start_us)
        machine_intervals: list[tuple[int, int, float | None]] = []
        while position < len(rows):
            row = rows[position]
            event_time = max(int(day_start_us), min(int(day_end_us), int(row.time)))
            if event_time > cursor and active:
                machine_intervals.append((cursor, event_time, capacity_cpu))
            active, capacity_cpu = apply_event(row, active, capacity_cpu)
            cursor = event_time
            position += 1
        if cursor < int(day_end_us) and active:
            machine_intervals.append((cursor, int(day_end_us), capacity_cpu))
        if machine_intervals:
            intervals[_json_scalar(machine_id)] = machine_intervals
    return intervals


def _build_lifecycle_episodes(
    events: pd.DataFrame,
    usage_keys: pd.DataFrame,
    day_start_us: int,
    day_end_us: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Reconstruct nonoverlapping running episodes from instance lifecycle events.

    ``SCHEDULE`` starts a running episode.  ``UPDATE_RUNNING`` is accepted as a
    defensive start only when a SCHEDULE row is missing.  A terminal event
    closes the current episode before a same-time running transition opens the
    next one.  Open executions are right-censored at the planning-horizon end.
    Episode indices count the complete extracted event history, including
    executions that ended before this horizon, so an execution has a stable
    identity across adjacent day extracts.
    """

    if int(day_end_us) <= int(day_start_us):
        raise ValueError("day_end_us must be greater than day_start_us")
    key_columns = ["collection_id", "instance_index"]
    histories = {
        key: group
        for key, group in events.groupby(key_columns, sort=False, dropna=False)
    }
    rows: list[dict[str, object]] = []
    orphan_terminal_count = 0
    repeated_running_start_count = 0
    terminal_reason_conflict_count = 0
    inferred_usage_start_count = 0

    def append_episode(
        key: tuple[object, object],
        episode_index: int,
        lifecycle_start: int,
        lifecycle_end: int,
        termination_reason: str,
        inferred_update: bool,
        inferred_usage: bool,
    ) -> None:
        clipped_start = max(int(day_start_us), int(lifecycle_start))
        clipped_end = min(int(day_end_us), int(lifecycle_end))
        if clipped_end <= clipped_start:
            return
        censored = termination_reason == HORIZON_CENSORED
        rows.append(
            {
                "collection_id": key[0],
                "instance_index": key[1],
                "episode_index": int(episode_index),
                "lifecycle_start_time_us": int(lifecycle_start),
                "episode_start_time_us": int(clipped_start),
                "episode_end_time_us": int(clipped_end),
                "termination_time_us": np.nan if censored else int(lifecycle_end),
                "termination_reason": termination_reason,
                "horizon_censored": bool(censored),
                "termination_time_uncertain": termination_reason == "LOST",
                "episode_start_inferred_from_update": bool(inferred_update),
                "episode_start_inferred_from_usage": bool(inferred_usage),
            }
        )

    for key_row in usage_keys[key_columns].drop_duplicates().itertuples(index=False):
        key = (key_row.collection_id, key_row.instance_index)
        emitted_before_key = len(rows)
        history = histories.get(key, pd.DataFrame(columns=events.columns))
        running = False
        running_start = 0
        running_inferred_update = False
        episode_index = -1

        if not history.empty:
            work = history.dropna(subset=["time"]).copy()
            work = work.loc[pd.to_numeric(work["time"], errors="coerce").lt(int(day_end_us))]
            for event_time, cohort in work.groupby("time", sort=True, dropna=False):
                event_time = int(event_time)
                names = set(cohort["event_name"].astype(str))
                terminal_names = sorted(
                    names & TERMINAL_INSTANCE_EVENTS,
                    key=lambda name: INSTANCE_EVENT_LIFECYCLE_ORDER[name],
                )
                if len(terminal_names) > 1:
                    terminal_reason_conflict_count += 1
                if terminal_names:
                    # The highest explicit lifecycle precedence is the
                    # deterministic final terminal label for the cohort.
                    reason = terminal_names[-1]
                    if running and event_time > running_start:
                        append_episode(
                            key,
                            episode_index,
                            running_start,
                            event_time,
                            reason,
                            running_inferred_update,
                            False,
                        )
                    elif not running:
                        orphan_terminal_count += 1
                    running = False

                start_name: str | None = None
                if "SCHEDULE" in names:
                    start_name = "SCHEDULE"
                elif "UPDATE_RUNNING" in names:
                    start_name = "UPDATE_RUNNING"
                if start_name is not None:
                    if running:
                        repeated_running_start_count += 1
                    else:
                        episode_index += 1
                        running = True
                        running_start = event_time
                        running_inferred_update = start_name == "UPDATE_RUNNING"

        if running and running_start < int(day_end_us):
            append_episode(
                key,
                episode_index,
                running_start,
                int(day_end_us),
                HORIZON_CENSORED,
                running_inferred_update,
                False,
            )
        if len(rows) == emitted_before_key:
            # A positive-overlap usage key with no usable running transition is
            # retained as one explicitly usage-inferred, horizon-censored
            # episode.  This is auditable and avoids silently reverting to the
            # old first/last-observation VM identity.
            inferred_usage_start_count += 1
            append_episode(
                key,
                0,
                int(day_start_us),
                int(day_end_us),
                HORIZON_CENSORED,
                False,
                True,
            )

    episodes = pd.DataFrame(rows)
    if episodes.empty:
        raise ValueError("no lifecycle episodes overlap the requested horizon")
    episodes = episodes.sort_values(EPISODE_KEY_COLUMNS, kind="stable").reset_index(drop=True)
    if episodes[EPISODE_KEY_COLUMNS].duplicated().any():
        raise ValueError("lifecycle reconstruction produced duplicate episode keys")
    if not (
        episodes["episode_start_time_us"].lt(episodes["episode_end_time_us"])
    ).all():
        raise ValueError("lifecycle reconstruction produced a non-positive episode")

    diagnostics = {
        "lifecycle_episode_count_overlapping_horizon": int(len(episodes)),
        "parent_instance_count": int(usage_keys[key_columns].drop_duplicates().shape[0]),
        "episode_count_by_termination_reason": {
            str(key): int(value)
            for key, value in episodes["termination_reason"].value_counts().sort_index().items()
        },
        "orphan_terminal_event_cohort_count": int(orphan_terminal_count),
        "repeated_running_start_cohort_count": int(repeated_running_start_count),
        "terminal_reason_conflict_cohort_count": int(terminal_reason_conflict_count),
        "usage_inferred_episode_count": int(inferred_usage_start_count),
        "right_censor_policy": (
            "an execution without a terminal event is active from its evidenced start through "
            "day_end_us; only an execution already running at day_start_us spans the full horizon"
        ),
    }
    return episodes, diagnostics


def _prepare_strict_episode_usage(
    usage: pd.DataFrame,
    events: pd.DataFrame,
    machine_events: pd.DataFrame,
    day_start_us: int,
    day_end_us: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Split raw usage by lifecycle episode and apply strict episode quality gates."""

    if usage is None or usage.empty:
        raise ValueError("instance usage input is empty")
    distribution_col = _require_column(
        usage, ["cpu_usage_distribution"], "cpu_usage_distribution"
    )
    collection_col = _require_column(usage, ["collection_id", "collection"], "collection_id")
    instance_col = _require_column(usage, ["instance_index", "instance"], "instance_index")
    machine_col = _require_column(usage, ["machine_id"], "usage machine_id")
    start_col = _require_column(usage, ["start_time", "start_time_us"], "start_time")
    end_col = _require_column(usage, ["end_time", "end_time_us"], "end_time")
    clipped_start_col = _first_existing(
        usage, ["clipped_start_time", "clipped_start_time_us"]
    )
    clipped_end_col = _first_existing(
        usage, ["clipped_end_time", "clipped_end_time_us"]
    )
    raw_max_cpu_col = _first_existing(
        usage,
        ["max_cpu_usage", "maximum_usage_cpus", "maximum_usage.cpu", "maximum_usage.cpus"],
    )

    exact_duplicate_mask = _exact_usage_duplicate_mask(usage)
    exact_duplicate_row_count = int(exact_duplicate_mask.sum())
    source = usage.loc[~exact_duplicate_mask].copy().reset_index(drop=True)
    source["_source_row_id"] = np.arange(len(source), dtype=np.int64)
    source_start = pd.to_numeric(source[start_col], errors="coerce")
    source_end = pd.to_numeric(source[end_col], errors="coerce")
    clipped_start = (
        pd.to_numeric(source[clipped_start_col], errors="coerce")
        if clipped_start_col
        else source_start
    )
    clipped_end = (
        pd.to_numeric(source[clipped_end_col], errors="coerce")
        if clipped_end_col
        else source_end
    )
    invalid_identity = pd.DataFrame(
        {
            "collection_id": source[collection_col],
            "instance_index": source[instance_col],
            "start": source_start,
            "end": source_end,
        }
    ).isna().any(axis=1)
    if invalid_identity.any():
        raise ValueError(
            f"usage contains {int(invalid_identity.sum())} rows with NULL identity or interval endpoints"
        )
    if source_end.le(source_start).any():
        raise ValueError("usage contains non-positive source intervals")
    clipped_start = pd.concat([clipped_start, source_start], axis=1).max(axis=1).clip(
        lower=int(day_start_us)
    )
    clipped_end = pd.concat([clipped_end, source_end], axis=1).min(axis=1).clip(
        upper=int(day_end_us)
    )
    positive = clipped_end.gt(clipped_start)
    source = source.loc[positive].copy().reset_index(drop=True)
    clipped_start = clipped_start.loc[positive].astype("int64").reset_index(drop=True)
    clipped_end = clipped_end.loc[positive].astype("int64").reset_index(drop=True)
    if source.empty:
        raise ValueError("no usage intervals overlap the requested horizon")
    source["_horizon_start"] = clipped_start
    source["_horizon_end"] = clipped_end

    p100_values: list[float | None] = []
    distribution_lengths: list[int] = []
    for row_index, value in source[distribution_col].items():
        try:
            p100, length = _cpu_distribution_max(value)
        except ValueError as exc:
            raise ValueError(
                f"invalid cpu_usage_distribution at deduplicated source row {row_index}: {exc}"
            ) from exc
        p100_values.append(p100)
        distribution_lengths.append(length)
    source["_cpu_p100"] = pd.to_numeric(pd.Series(p100_values), errors="coerce")
    source["_cpu_distribution_length"] = distribution_lengths
    source["_empty_cpu_distribution"] = source["_cpu_p100"].isna()

    usage_keys = pd.DataFrame(
        {
            "collection_id": source[collection_col],
            "instance_index": source[instance_col],
        }
    )
    episodes, lifecycle_diagnostics = _build_lifecycle_episodes(
        events,
        usage_keys,
        day_start_us=int(day_start_us),
        day_end_us=int(day_end_us),
    )
    episode_groups = {
        key: group.sort_values("episode_start_time_us", kind="stable")
        for key, group in episodes.groupby(
            ["collection_id", "instance_index"], sort=False, dropna=False
        )
    }

    source_positions: list[int] = []
    episode_records: list[tuple[object, ...]] = []
    fragment_starts: list[int] = []
    fragment_ends: list[int] = []
    unassigned_source_rows = 0
    boundary_split_source_rows = 0
    source_groups = source.groupby([collection_col, instance_col], sort=False, dropna=False).indices
    episode_columns = EPISODE_KEY_COLUMNS + [
        "episode_start_time_us",
        "episode_end_time_us",
        "termination_time_us",
        "termination_reason",
        "horizon_censored",
        "termination_time_uncertain",
        "episode_start_inferred_from_update",
        "episode_start_inferred_from_usage",
    ]
    for key, positions in source_groups.items():
        if not isinstance(key, tuple):
            key = (key,)
        group_episodes = episode_groups.get(key)
        if group_episodes is None or group_episodes.empty:
            unassigned_source_rows += len(positions)
            continue
        starts = group_episodes["episode_start_time_us"].to_numpy(dtype=np.int64)
        ends = group_episodes["episode_end_time_us"].to_numpy(dtype=np.int64)
        episode_tuples = list(group_episodes[episode_columns].itertuples(index=False, name=None))
        for position in positions:
            left = int(source.at[int(position), "_horizon_start"])
            right = int(source.at[int(position), "_horizon_end"])
            episode_position = int(np.searchsorted(ends, left, side="right"))
            overlap_count = 0
            while episode_position < len(starts) and int(starts[episode_position]) < right:
                overlap_start = max(left, int(starts[episode_position]))
                overlap_end = min(right, int(ends[episode_position]))
                if overlap_end > overlap_start:
                    source_positions.append(int(position))
                    episode_records.append(episode_tuples[episode_position])
                    fragment_starts.append(overlap_start)
                    fragment_ends.append(overlap_end)
                    overlap_count += 1
                episode_position += 1
            if overlap_count == 0:
                unassigned_source_rows += 1
            elif overlap_count > 1:
                boundary_split_source_rows += 1

    if not source_positions:
        raise ValueError("no usage interval overlaps a reconstructed lifecycle episode")
    fragments = source.iloc[source_positions].copy().reset_index(drop=True)
    episode_frame = pd.DataFrame(episode_records, columns=episode_columns)
    # Use the normalized identity names from this point onward.
    fragments["collection_id"] = episode_frame["collection_id"]
    fragments["instance_index"] = episode_frame["instance_index"]
    for column in episode_columns[2:]:
        fragments[column] = episode_frame[column]
    fragments["clipped_start_time"] = np.asarray(fragment_starts, dtype=np.int64)
    fragments["clipped_end_time"] = np.asarray(fragment_ends, dtype=np.int64)
    fragments["overlap_us"] = (
        fragments["clipped_end_time"] - fragments["clipped_start_time"]
    )
    if raw_max_cpu_col:
        fragments["maximum_usage_cpu_raw"] = pd.to_numeric(
            fragments[raw_max_cpu_col], errors="coerce"
        )
    fragments["max_cpu_usage"] = pd.to_numeric(fragments["_cpu_p100"], errors="coerce")

    capacity_intervals = _machine_cpu_capacity_intervals(
        machine_events,
        day_start_us=int(day_start_us),
        day_end_us=int(day_end_us),
    )
    capacity_interval_ends = {
        machine_id: [item[1] for item in history]
        for machine_id, history in capacity_intervals.items()
    }
    machine_history_missing: list[bool] = []
    capacity_incomplete: list[bool] = []
    p100_over_capacity: list[bool] = []
    for row in fragments.itertuples(index=False):
        start = int(row.clipped_start_time)
        end = int(row.clipped_end_time)
        machine_id_value = getattr(row, machine_col)
        machine_id = _json_scalar(machine_id_value)
        history = [] if pd.isna(machine_id_value) else capacity_intervals.get(machine_id, [])
        missing_history = not bool(history)
        valid_duration = 0
        capacities: list[float] = []
        if history:
            history_position = bisect_right(capacity_interval_ends[machine_id], start)
            while history_position < len(history) and history[history_position][0] < end:
                cap_start, cap_end, capacity = history[history_position]
                overlap = min(end, int(cap_end)) - max(start, int(cap_start))
                if (
                    overlap > 0
                    and capacity is not None
                    and np.isfinite(capacity)
                    and float(capacity) > 0
                ):
                    valid_duration += int(overlap)
                    capacities.append(float(capacity))
                history_position += 1
        full = bool(capacities) and valid_duration == end - start
        p100 = row.max_cpu_usage
        machine_history_missing.append(missing_history)
        capacity_incomplete.append(not full)
        p100_over_capacity.append(
            bool(capacities)
            and not pd.isna(p100)
            and float(p100) > min(capacities) + 1e-12
        )
    fragments["_machine_history_missing"] = machine_history_missing
    fragments["_machine_capacity_incomplete"] = capacity_incomplete
    fragments["_cpu_p100_over_capacity"] = p100_over_capacity

    flags = (
        fragments.groupby(EPISODE_KEY_COLUMNS, as_index=False, sort=False)
        .agg(
            termination_reason=("termination_reason", "first"),
            episode_start_time_us=("episode_start_time_us", "first"),
            episode_end_time_us=("episode_end_time_us", "first"),
            empty_cpu_distribution=("_empty_cpu_distribution", "any"),
            machine_history_missing=("_machine_history_missing", "any"),
            machine_capacity_incomplete=("_machine_capacity_incomplete", "any"),
            cpu_p100_over_capacity=("_cpu_p100_over_capacity", "any"),
            source_fragment_count=("_source_row_id", "size"),
        )
    )
    flags["drop"] = (
        flags["empty_cpu_distribution"]
        | flags["machine_history_missing"]
        | flags["machine_capacity_incomplete"]
    )
    retained_keys = flags.loc[~flags["drop"], EPISODE_KEY_COLUMNS]
    fragments = fragments.merge(
        retained_keys.assign(_episode_retained=True),
        on=EPISODE_KEY_COLUMNS,
        how="inner",
        validate="many_to_one",
    ).drop(columns=["_episode_retained"])
    if fragments.empty:
        raise ValueError("no episodes remain after strict usage-distribution/machine-capacity filtering")

    def reason_counts(frame: pd.DataFrame) -> dict[str, int]:
        return {
            str(key): int(value)
            for key, value in frame["termination_reason"].value_counts().sort_index().items()
        }

    lifetime_hours = (
        flags["episode_end_time_us"] - flags["episode_start_time_us"]
    ) / 3_600_000_000.0
    retained_lifetime_hours = lifetime_hours.loc[~flags["drop"]]
    diagnostics: dict[str, object] = {
        **lifecycle_diagnostics,
        "unit_key": EPISODE_KEY_COLUMNS,
        "source_usage_row_count": int(len(usage)),
        "positive_horizon_source_row_count_after_exact_dedup": int(len(source)),
        "exact_duplicate_source_row_count_removed": int(exact_duplicate_row_count),
        "source_row_count_without_running_episode_overlap": int(unassigned_source_rows),
        "source_row_count_split_at_episode_boundary": int(boundary_split_source_rows),
        "episode_count_with_usage_before_quality_filter": int(len(flags)),
        "episode_count_retained_after_quality_filter": int((~flags["drop"]).sum()),
        "episode_count_dropped": int(flags["drop"].sum()),
        "drop_fraction": _safe_fraction(int(flags["drop"].sum()), int(len(flags))),
        "drop_reason_counts_nonexclusive": {
            "empty_cpu_usage_distribution": int(flags["empty_cpu_distribution"].sum()),
            "machine_history_missing": int(flags["machine_history_missing"].sum()),
            "machine_capacity_incomplete": int(flags["machine_capacity_incomplete"].sum()),
        },
        "cpu_p100_over_capacity_episode_count_retained_not_dropped": int(
            flags.loc[~flags["drop"], "cpu_p100_over_capacity"].sum()
        ),
        "episode_count_by_termination_reason_before_quality_filter": reason_counts(flags),
        "episode_count_by_termination_reason_retained": reason_counts(flags.loc[~flags["drop"]]),
        "episode_count_by_termination_reason_dropped": reason_counts(flags.loc[flags["drop"]]),
        "lifetime_hours_before_quality_filter": _numeric_summary(lifetime_hours),
        "lifetime_hours_retained_after_quality_filter": _numeric_summary(
            retained_lifetime_hours
        ),
        "cpu_maximum_policy": (
            "max_cpu_usage is cpu_usage_distribution[10] (p100); extracted maximum_usage.cpus "
            "is ignored for q and retained only as raw audit provenance"
        ),
        "quality_filter_policy": (
            "drop an episode before min_usage_rows and deterministic sampling if any assigned "
            "positive-duration source fragment has an empty/unavailable CPU distribution, no "
            "assigned-machine history, or incomplete positive CPU-capacity coverage"
        ),
    }
    fragments.attrs["episode_quality_filter"] = diagnostics
    return fragments, diagnostics


def _safe_fraction(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _episode_stage_summary(frame: pd.DataFrame) -> dict[str, object]:
    """Return counts and clipped lifetime quantiles for an episode-stage frame."""

    if frame.empty or not {
        "episode_start_time_us",
        "episode_end_time_us",
        "termination_reason",
    }.issubset(frame.columns):
        return {"episode_count": int(len(frame)), "by_termination_reason": {}}

    def lifetime_summary(values: pd.Series) -> dict[str, float | int | None]:
        numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
        if numeric.empty:
            return {
                "count": 0,
                "min": None,
                "p10": None,
                "p25": None,
                "p50": None,
                "mean": None,
                "p75": None,
                "p90": None,
                "p95": None,
                "p99": None,
                "max": None,
            }
        return {
            "count": int(len(numeric)),
            "min": float(numeric.min()),
            "p10": float(numeric.quantile(0.10)),
            "p25": float(numeric.quantile(0.25)),
            "p50": float(numeric.quantile(0.50)),
            "mean": float(numeric.mean()),
            "p75": float(numeric.quantile(0.75)),
            "p90": float(numeric.quantile(0.90)),
            "p95": float(numeric.quantile(0.95)),
            "p99": float(numeric.quantile(0.99)),
            "max": float(numeric.max()),
        }

    work = frame.copy()
    work["lifetime_hours"] = (
        pd.to_numeric(work["episode_end_time_us"], errors="coerce")
        - pd.to_numeric(work["episode_start_time_us"], errors="coerce")
    ) / 3_600_000_000.0
    by_reason: dict[str, object] = {}
    for reason, group in work.groupby("termination_reason", sort=True, dropna=False):
        by_reason[str(reason)] = lifetime_summary(group["lifetime_hours"])
    return {
        "episode_count": int(len(work)),
        "lifetime_hours": lifetime_summary(work["lifetime_hours"]),
        "by_termination_reason": by_reason,
    }


def audit_assigned_machine_cpu_capacity(
    usage: pd.DataFrame,
    machine_events: pd.DataFrame,
    day_start_us: int,
    day_end_us: int,
    selected_keys: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Audit CPU observations against the capacity of the assigned machine.

    The audit is policy-neutral: it neither caps nor drops a usage row.  Its
    primary denominator is the number of unique top-level instance keys
    ``(collection_id, instance_index)``.  In ClusterData v3, collection type 0
    denotes a task in a JOB and type 1 denotes an alloc instance in an
    ALLOC_SET.  A source interval is assessable when at least one positive
    machine-capacity state overlaps it.  Complete capacity coverage is reported
    separately; when capacity changes inside a window, the audit distinguishes
    exceeding any known state from exceeding every known state.
    """

    if usage is None or usage.empty:
        raise ValueError("instance usage input is empty")
    if int(day_end_us) <= int(day_start_us):
        raise ValueError("day_end_us must be greater than day_start_us")
    collection_col = _require_column(usage, ["collection_id", "collection"], "collection_id")
    instance_col = _require_column(usage, ["instance_index", "instance"], "instance_index")
    machine_col = _require_column(usage, ["machine_id"], "usage machine_id")
    start_col = _require_column(usage, ["start_time", "start_time_us"], "start_time")
    end_col = _require_column(usage, ["end_time", "end_time_us"], "end_time")
    clipped_start_col = _first_existing(usage, ["clipped_start_time", "clipped_start_time_us"])
    clipped_end_col = _first_existing(usage, ["clipped_end_time", "clipped_end_time_us"])
    collection_type_col = _first_existing(usage, ["collection_type"])
    max_cpu_col = _first_existing(
        usage,
        ["max_cpu_usage", "maximum_usage_cpus", "maximum_usage.cpu", "maximum_usage.cpus"],
    )
    distribution_col = _first_existing(usage, ["cpu_usage_distribution"])

    exact_duplicate_mask = _exact_usage_duplicate_mask(usage)
    exact_duplicate_row_count = int(exact_duplicate_mask.sum())
    usage_input = usage.loc[~exact_duplicate_mask].copy()

    rows = pd.DataFrame(
        {
            "collection_id": usage_input[collection_col],
            "instance_index": usage_input[instance_col],
            "machine_id": usage_input[machine_col],
            "collection_type": (
                usage_input[collection_type_col] if collection_type_col else np.nan
            ),
            "start_time": pd.to_numeric(usage_input[start_col], errors="coerce"),
            "end_time": pd.to_numeric(usage_input[end_col], errors="coerce"),
            "max_cpu_usage": (
                pd.to_numeric(usage_input[max_cpu_col], errors="coerce")
                if max_cpu_col
                else np.nan
            ),
        }
    )
    if clipped_start_col:
        rows["clipped_start_time"] = pd.to_numeric(
            usage_input[clipped_start_col], errors="coerce"
        )
    else:
        rows["clipped_start_time"] = rows["start_time"]
    if clipped_end_col:
        rows["clipped_end_time"] = pd.to_numeric(
            usage_input[clipped_end_col], errors="coerce"
        )
    else:
        rows["clipped_end_time"] = rows["end_time"]

    invalid_identity = rows[
        ["collection_id", "instance_index", "start_time", "end_time"]
    ].isna().any(axis=1)
    if invalid_identity.any():
        raise ValueError("usage capacity audit found NULL identity or interval endpoint")
    invalid_maximum = rows["max_cpu_usage"].notna() & (
        ~np.isfinite(rows["max_cpu_usage"]) | rows["max_cpu_usage"].lt(0)
    )
    if invalid_maximum.any():
        raise ValueError("usage capacity audit found non-finite or negative max_cpu_usage")
    rows["clipped_start_time"] = (
        rows[["clipped_start_time", "start_time"]].max(axis=1).clip(lower=int(day_start_us))
    )
    rows["clipped_end_time"] = (
        rows[["clipped_end_time", "end_time"]].min(axis=1).clip(upper=int(day_end_us))
    )
    rows = rows.loc[rows["clipped_end_time"].gt(rows["clipped_start_time"])].copy()
    if rows.empty:
        raise ValueError("no usage intervals overlap the requested day for capacity audit")
    rows["clipped_start_time"] = rows["clipped_start_time"].astype("int64")
    rows["clipped_end_time"] = rows["clipped_end_time"].astype("int64")
    rows["instance_kind_row"] = rows["collection_type"].map(_collection_instance_kind)

    distribution_lengths: list[int] = []
    distribution_maxima: list[float | None] = []
    if distribution_col:
        for row_index, value in usage_input.loc[rows.index, distribution_col].items():
            try:
                maximum, length = _cpu_distribution_max(value)
            except ValueError as exc:
                raise ValueError(
                    f"invalid cpu_usage_distribution at usage row {row_index}: {exc}"
                ) from exc
            distribution_maxima.append(maximum)
            distribution_lengths.append(length)
    else:
        distribution_maxima = [None] * len(rows)
        distribution_lengths = [0] * len(rows)
    rows["cpu_distribution_max"] = pd.to_numeric(
        pd.Series(distribution_maxima, index=rows.index), errors="coerce"
    )
    rows["cpu_distribution_length"] = distribution_lengths

    capacity_intervals = _machine_cpu_capacity_intervals(
        machine_events,
        day_start_us=int(day_start_us),
        day_end_us=int(day_end_us),
    )
    capacity_full: list[bool] = []
    capacity_minima: list[float] = []
    capacity_maxima: list[float] = []
    capacity_any: list[bool] = []
    capacity_state_changes: list[bool] = []
    machine_history_found: list[bool] = []
    for row in rows.itertuples(index=False):
        start = int(row.clipped_start_time)
        end = int(row.clipped_end_time)
        machine_id = _json_scalar(row.machine_id)
        history = [] if pd.isna(row.machine_id) else capacity_intervals.get(machine_id, [])
        machine_history_found.append(bool(history))
        valid_duration = 0
        capacities: list[float] = []
        if history:
            interval_ends = [item[1] for item in history]
            position = bisect_right(interval_ends, start)
            while position < len(history) and history[position][0] < end:
                left, right, capacity = history[position]
                overlap = min(end, int(right)) - max(start, int(left))
                if overlap > 0 and capacity is not None and np.isfinite(capacity) and capacity > 0:
                    valid_duration += int(overlap)
                    capacities.append(float(capacity))
                position += 1
        is_full = valid_duration == end - start and bool(capacities)
        capacity_any.append(bool(capacities))
        capacity_full.append(is_full)
        capacity_minima.append(min(capacities) if capacities else np.nan)
        capacity_maxima.append(max(capacities) if capacities else np.nan)
        capacity_state_changes.append(len(set(capacities)) > 1)
    rows["machine_history_found"] = machine_history_found
    rows["capacity_any_overlap"] = capacity_any
    rows["capacity_full_interval"] = capacity_full
    rows["capacity_cpu_min"] = capacity_minima
    rows["capacity_cpu_max"] = capacity_maxima
    rows["capacity_changed_inside_interval"] = capacity_state_changes

    tolerance = 1e-12
    for metric in ["maximum_usage", "cpu_usage_distribution"]:
        value_column = "max_cpu_usage" if metric == "maximum_usage" else "cpu_distribution_max"
        observed = rows[value_column].notna()
        matched = observed & rows["capacity_any_overlap"]
        complete = observed & rows["capacity_full_interval"]
        above_any_state = matched & rows[value_column].gt(
            rows["capacity_cpu_min"] + tolerance
        )
        above_all_states = matched & rows[value_column].gt(
            rows["capacity_cpu_max"] + tolerance
        )
        at_or_below_all_states = complete & rows[value_column].le(
            rows["capacity_cpu_min"] + tolerance
        )
        rows[f"{metric}_observed"] = observed
        rows[f"{metric}_assessed"] = matched
        rows[f"{metric}_over_capacity"] = above_any_state
        rows[f"{metric}_over_all_overlapping_capacities"] = above_all_states
        rows[f"{metric}_capacity_change_ambiguous"] = complete & ~(
            above_all_states | at_or_below_all_states
        )

    key_columns = ["collection_id", "instance_index"]
    kind_rows: list[dict[str, object]] = []
    for key, group in rows.groupby(key_columns, sort=False, dropna=False):
        kinds = sorted(set(group["instance_kind_row"]))
        instance_kind = kinds[0] if len(kinds) == 1 else "unknown"
        kind_rows.append(
            {
                "collection_id": key[0],
                "instance_index": key[1],
                "instance_kind": instance_kind,
            }
        )
    units = pd.DataFrame(kind_rows)
    unit_flags = (
        rows.groupby(key_columns, as_index=False, sort=False)
        .agg(
            machine_history_found=("machine_history_found", "any"),
            machine_history_unmatched=("machine_history_found", lambda values: (~values).any()),
            capacity_full_interval=("capacity_full_interval", "any"),
            capacity_incomplete_interval=(
                "capacity_full_interval",
                lambda values: (~values).any(),
            ),
            maximum_usage_observed=("maximum_usage_observed", "any"),
            maximum_usage_assessed=("maximum_usage_assessed", "any"),
            maximum_usage_over_capacity=("maximum_usage_over_capacity", "any"),
            maximum_usage_over_all_overlapping_capacities=(
                "maximum_usage_over_all_overlapping_capacities",
                "any",
            ),
            maximum_usage_capacity_change_ambiguous=(
                "maximum_usage_capacity_change_ambiguous",
                "any",
            ),
            cpu_usage_distribution_observed=("cpu_usage_distribution_observed", "any"),
            cpu_usage_distribution_assessed=("cpu_usage_distribution_assessed", "any"),
            cpu_usage_distribution_over_capacity=(
                "cpu_usage_distribution_over_capacity",
                "any",
            ),
            cpu_usage_distribution_over_all_overlapping_capacities=(
                "cpu_usage_distribution_over_all_overlapping_capacities",
                "any",
            ),
            cpu_usage_distribution_capacity_change_ambiguous=(
                "cpu_usage_distribution_capacity_change_ambiguous",
                "any",
            ),
        )
    )
    units = units.merge(unit_flags, on=key_columns, how="left", validate="one_to_one")
    units["either_metric_observed"] = (
        units["maximum_usage_observed"] | units["cpu_usage_distribution_observed"]
    )
    units["either_metric_assessed"] = (
        units["maximum_usage_assessed"] | units["cpu_usage_distribution_assessed"]
    )
    units["either_metric_over_capacity"] = (
        units["maximum_usage_over_capacity"]
        | units["cpu_usage_distribution_over_capacity"]
    )
    units["either_metric_over_all_overlapping_capacities"] = (
        units["maximum_usage_over_all_overlapping_capacities"]
        | units["cpu_usage_distribution_over_all_overlapping_capacities"]
    )

    rows["_key"] = list(zip(rows["collection_id"], rows["instance_index"]))
    units["_key"] = list(zip(units["collection_id"], units["instance_index"]))

    def metric_summary(unit_subset: pd.DataFrame, row_subset: pd.DataFrame, metric: str) -> dict[str, object]:
        observed_units = int(unit_subset[f"{metric}_observed"].sum())
        assessed_units = int(unit_subset[f"{metric}_assessed"].sum())
        exceeded_units = int(unit_subset[f"{metric}_over_capacity"].sum())
        exceeded_all_units = int(
            unit_subset[f"{metric}_over_all_overlapping_capacities"].sum()
        )
        return {
            "observed_source_row_count": int(row_subset[f"{metric}_observed"].sum()),
            "assessed_source_row_count": int(row_subset[f"{metric}_assessed"].sum()),
            "over_capacity_source_row_count": int(
                row_subset[f"{metric}_over_capacity"].sum()
            ),
            "over_all_overlapping_capacities_source_row_count": int(
                row_subset[f"{metric}_over_all_overlapping_capacities"].sum()
            ),
            "observed_instance_count": observed_units,
            "assessed_instance_count": assessed_units,
            "over_capacity_instance_count": exceeded_units,
            "over_all_overlapping_capacities_instance_count": exceeded_all_units,
            "fraction_of_all_instances": _safe_fraction(exceeded_units, len(unit_subset)),
            "fraction_of_observed_instances": _safe_fraction(exceeded_units, observed_units),
            "fraction_of_assessable_instances": _safe_fraction(exceeded_units, assessed_units),
            "capacity_change_ambiguous_source_row_count": int(
                row_subset[f"{metric}_capacity_change_ambiguous"].sum()
            ),
        }

    def scope_summary(scope_units: pd.DataFrame) -> dict[str, object]:
        keys = set(scope_units["_key"])
        scope_rows = rows.loc[rows["_key"].isin(keys)]
        total = int(len(scope_units))
        either_observed = int(scope_units["either_metric_observed"].sum())
        either_assessed = int(scope_units["either_metric_assessed"].sum())
        either_exceeded = int(scope_units["either_metric_over_capacity"].sum())
        either_exceeded_all = int(
            scope_units["either_metric_over_all_overlapping_capacities"].sum()
        )
        result: dict[str, object] = {
            "instance_count": total,
            "source_usage_row_count": int(len(scope_rows)),
            "machine_history_unmatched_source_row_count": int(
                (~scope_rows["machine_history_found"]).sum()
            ),
            "machine_history_unmatched_instance_count": int(
                scope_units["machine_history_unmatched"].sum()
            ),
            "machine_capacity_incomplete_source_row_count": int(
                (~scope_rows["capacity_full_interval"]).sum()
            ),
            "machine_capacity_incomplete_instance_count": int(
                scope_units["capacity_incomplete_interval"].sum()
            ),
            "maximum_usage": metric_summary(scope_units, scope_rows, "maximum_usage"),
            "cpu_usage_distribution": metric_summary(
                scope_units, scope_rows, "cpu_usage_distribution"
            ),
            "either_metric": {
                "observed_instance_count": either_observed,
                "assessed_instance_count": either_assessed,
                "over_capacity_instance_count": either_exceeded,
                "over_all_overlapping_capacities_instance_count": either_exceeded_all,
                "fraction_of_all_instances": _safe_fraction(either_exceeded, total),
                "fraction_of_observed_instances": _safe_fraction(
                    either_exceeded, either_observed
                ),
                "fraction_of_assessable_instances": _safe_fraction(
                    either_exceeded, either_assessed
                ),
            },
        }
        by_kind: dict[str, object] = {}
        for kind in ["task", "alloc", "unknown"]:
            kind_units = scope_units.loc[scope_units["instance_kind"].eq(kind)]
            kind_keys = set(kind_units["_key"])
            kind_rows = scope_rows.loc[scope_rows["_key"].isin(kind_keys)]
            kind_total = int(len(kind_units))
            kind_exceeded = int(kind_units["either_metric_over_capacity"].sum())
            kind_exceeded_all = int(
                kind_units["either_metric_over_all_overlapping_capacities"].sum()
            )
            kind_assessed = int(kind_units["either_metric_assessed"].sum())
            by_kind[kind] = {
                "instance_count": kind_total,
                "source_usage_row_count": int(len(kind_rows)),
                "machine_capacity_incomplete_instance_count": int(
                    kind_units["capacity_incomplete_interval"].sum()
                ),
                "maximum_usage": metric_summary(kind_units, kind_rows, "maximum_usage"),
                "cpu_usage_distribution": metric_summary(
                    kind_units, kind_rows, "cpu_usage_distribution"
                ),
                "either_metric": {
                    "over_capacity_instance_count": kind_exceeded,
                    "over_all_overlapping_capacities_instance_count": kind_exceeded_all,
                    "fraction_of_all_instances": _safe_fraction(kind_exceeded, kind_total),
                    "fraction_of_assessable_instances": _safe_fraction(
                        kind_exceeded, kind_assessed
                    ),
                },
            }
        result["by_instance_kind"] = by_kind
        return result

    raw_scope = units.copy()
    scopes: dict[str, object] = {
        "raw_extracted_top_level_instances": scope_summary(raw_scope),
    }
    if selected_keys is not None:
        selected = selected_keys[key_columns].drop_duplicates().copy()
        selected["_key"] = list(zip(selected["collection_id"], selected["instance_index"]))
        selected_scope = units.loc[units["_key"].isin(set(selected["_key"]))].copy()
        scopes["selected_model_input_instances"] = scope_summary(selected_scope)

    nonempty_lengths = rows.loc[
        rows["cpu_distribution_length"].gt(0), "cpu_distribution_length"
    ]
    return {
        "status": "audit_only_no_cap_drop_or_rescaling_applied",
        "resource": "cpu",
        "unit_key": ["collection_id", "instance_index"],
        "collection_type_mapping": {"0": "task (JOB instance)", "1": "alloc (ALLOC_SET instance)"},
        "denominator_definition": (
            "unique top-level (collection_id, instance_index) keys with a positive-overlap "
            "usage interval in the requested horizon"
        ),
        "comparison_policy": (
            "compare max_cpu_usage and max(cpu_usage_distribution) with the positive CPU "
            "capacity reconstructed from the assigned usage.machine_id; report both exceedance "
            "of any known overlapping capacity (metric > interval minimum) and exceedance of all "
            "known overlapping capacities (metric > interval maximum), while separately reporting "
            "incomplete capacity coverage and capacity changes"
        ),
        "strict_exceedance_tolerance": tolerance,
        "exact_duplicate_source_row_count_removed": exact_duplicate_row_count,
        "cpu_usage_distribution_column_present": distribution_col is not None,
        "cpu_usage_distribution_expected_length": 11,
        "cpu_usage_distribution_nonempty_row_count": int(len(nonempty_lengths)),
        "cpu_usage_distribution_unexpected_length_row_count": int(
            nonempty_lengths.ne(11).sum()
        ),
        "scopes": scopes,
    }


def derive_representative_machine_capacity(
    machine_events: pd.DataFrame,
    day_start_us: int,
    day_end_us: int,
) -> tuple[dict[str, float | int | str], dict[str, object]]:
    """Select the active-duration-weighted mode of joint CPU/MEM shapes."""

    if machine_events is None or machine_events.empty:
        raise ValueError("machine_events is required and must not be empty")
    if int(day_end_us) <= int(day_start_us):
        raise ValueError("day_end_us must be greater than day_start_us")
    time_col = _require_column(machine_events, ["time", "event_time", "timestamp_us"], "machine event time")
    machine_col = _require_column(machine_events, ["machine_id"], "machine_id")
    event_col = _require_column(machine_events, ["event_type", "type"], "machine event type")
    cpu_col = _require_column(machine_events, ["capacity_cpu", "capacity.cpus"], "machine CPU capacity")
    mem_col = _require_column(machine_events, ["capacity_mem", "capacity.memory"], "machine memory capacity")

    events = pd.DataFrame(
        {
            "time": pd.to_numeric(machine_events[time_col], errors="coerce"),
            "machine_id": machine_events[machine_col],
            "event_code": machine_events[event_col].map(_machine_event_code),
            "capacity_cpu": pd.to_numeric(machine_events[cpu_col], errors="coerce"),
            "capacity_mem": pd.to_numeric(machine_events[mem_col], errors="coerce"),
            "row_order": np.arange(len(machine_events), dtype=int),
        }
    )
    invalid_identity = events[["time", "machine_id"]].isna().any(axis=1)
    if invalid_identity.any():
        raise ValueError(
            f"machine_events contains {int(invalid_identity.sum())} rows with NULL time or machine_id"
        )
    for column in ["capacity_cpu", "capacity_mem"]:
        negative = events[column].notna() & events[column].lt(0)
        if negative.any():
            raise ValueError(f"machine_events contains negative {column}")
    events = events.loc[events["time"].lt(int(day_end_us))].copy()
    if events.empty:
        raise ValueError("machine_events contains no event before day_end_us")
    events["time"] = events["time"].astype("int64")
    events = events.sort_values(["machine_id", "time", "row_order"], kind="stable")

    shape_weights: dict[tuple[float, float], int] = {}
    total_active_us = 0
    active_without_positive_joint_capacity_us = 0
    unknown_event_rows = int(events["event_code"].eq(0).sum())

    def apply_event(row: object, active: bool, cpu: float | None, mem: float | None) -> tuple[bool, float | None, float | None]:
        code = int(row.event_code)
        row_cpu = None if pd.isna(row.capacity_cpu) else float(row.capacity_cpu)
        row_mem = None if pd.isna(row.capacity_mem) else float(row.capacity_mem)
        if code in {1, 3}:
            if row_cpu is not None:
                cpu = row_cpu
            if row_mem is not None:
                mem = row_mem
        if code == 1:
            active = True
        elif code == 2:
            active = False
            # REMOVE closes the machine lifecycle.  A later sparse ADD must
            # not inherit a CPU or memory capacity from the removed machine.
            cpu = None
            mem = None
        return active, cpu, mem

    for _, machine_history in events.groupby("machine_id", sort=False, dropna=False):
        active = False
        capacity_cpu: float | None = None
        capacity_mem: float | None = None
        rows = list(machine_history.itertuples(index=False))
        position = 0
        while position < len(rows) and int(rows[position].time) < int(day_start_us):
            active, capacity_cpu, capacity_mem = apply_event(
                rows[position], active, capacity_cpu, capacity_mem
            )
            position += 1
        cursor = int(day_start_us)
        while position < len(rows):
            row = rows[position]
            event_time = max(int(day_start_us), min(int(day_end_us), int(row.time)))
            duration = event_time - cursor
            if duration > 0 and active:
                total_active_us += duration
                if (
                    capacity_cpu is not None
                    and capacity_mem is not None
                    and capacity_cpu > 0
                    and capacity_mem > 0
                ):
                    pair = (float(capacity_cpu), float(capacity_mem))
                    shape_weights[pair] = shape_weights.get(pair, 0) + duration
                else:
                    active_without_positive_joint_capacity_us += duration
            active, capacity_cpu, capacity_mem = apply_event(row, active, capacity_cpu, capacity_mem)
            cursor = event_time
            position += 1
        duration = int(day_end_us) - cursor
        if duration > 0 and active:
            total_active_us += duration
            if (
                capacity_cpu is not None
                and capacity_mem is not None
                and capacity_cpu > 0
                and capacity_mem > 0
            ):
                pair = (float(capacity_cpu), float(capacity_mem))
                shape_weights[pair] = shape_weights.get(pair, 0) + duration
            else:
                active_without_positive_joint_capacity_us += duration

    if not shape_weights:
        raise ValueError(
            "machine event history yields no active interval with positive joint CPU/MEM capacity"
        )
    ranked = sorted(shape_weights.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
    (capacity_cpu, capacity_mem), selected_weight_us = ranked[0]
    positive_joint_weight_us = int(sum(shape_weights.values()))
    method = "active_duration_weighted_mode_of_joint_cpu_mem_machine_shape"
    representative: dict[str, float | int | str] = {
        "capacity_cpu": float(capacity_cpu),
        "capacity_mem": float(capacity_mem),
        "selected_weight_us": int(selected_weight_us),
        "method": method,
    }
    shape_rows = [
        {
            "capacity_cpu": float(pair[0]),
            "capacity_mem": float(pair[1]),
            "active_duration_us": int(weight),
            "share_of_positive_joint_duration": float(weight / positive_joint_weight_us),
        }
        for pair, weight in sorted(shape_weights.items(), key=lambda item: (item[0][0], item[0][1]))
    ]
    diagnostics: dict[str, object] = {
        "method": method,
        "tie_break": "smallest capacity_cpu, then smallest capacity_mem",
        "day_start_us": int(day_start_us),
        "day_end_us": int(day_end_us),
        "machine_event_rows": int(len(events)),
        "machine_count": int(events["machine_id"].nunique()),
        "unknown_event_rows": unknown_event_rows,
        "total_active_machine_duration_us": int(total_active_us),
        "positive_joint_capacity_duration_us": positive_joint_weight_us,
        "active_duration_without_positive_joint_capacity_us": int(
            active_without_positive_joint_capacity_us
        ),
        "joint_shape_count": int(len(shape_rows)),
        "representative": representative,
        "joint_shape_weights": shape_rows,
    }
    return representative, diagnostics


def convert_resource_units(
    vm_requests: pd.DataFrame,
    observed_usage: pd.DataFrame,
    representative_capacity: dict[str, float | int | str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Convert trace-normalized resources to representative-machine fractions."""

    cpu_divisor = float(representative_capacity["capacity_cpu"])
    mem_divisor = float(representative_capacity["capacity_mem"])
    if not np.isfinite([cpu_divisor, mem_divisor]).all() or cpu_divisor <= 0 or mem_divisor <= 0:
        raise ValueError("representative CPU and memory capacities must be finite and positive")
    vm_attrs = dict(vm_requests.attrs)
    observed_attrs = dict(observed_usage.attrs)
    vm_converted = vm_requests.copy()
    observed_converted = observed_usage.copy()
    cpu_columns = ["q_cpu", "resource_request_cpu", "p95_cpu_usage", "avg_cpu_usage", "max_cpu_usage"]
    mem_columns = ["q_mem", "resource_request_mem", "p95_mem_usage", "avg_mem_usage", "max_mem_usage"]
    for column in cpu_columns:
        if column in vm_converted:
            vm_converted[column] = pd.to_numeric(vm_converted[column], errors="coerce") / cpu_divisor
    for column in mem_columns:
        if column in vm_converted:
            vm_converted[column] = pd.to_numeric(vm_converted[column], errors="coerce") / mem_divisor
    for column in ["cpu_usage", "max_cpu_usage"]:
        if column in observed_converted:
            observed_converted[column] = pd.to_numeric(observed_converted[column], errors="coerce") / cpu_divisor
    for column in ["mem_usage", "max_mem_usage", "assigned_memory"]:
        if column in observed_converted:
            observed_converted[column] = pd.to_numeric(observed_converted[column], errors="coerce") / mem_divisor

    # This pipeline permits unit conversion only.  Assert directly that the
    # actual-demand rows are raw values divided by the representative joint
    # machine shape, with no hidden utilization calibration.
    for column, divisor in (("cpu_usage", cpu_divisor), ("mem_usage", mem_divisor)):
        raw_values = pd.to_numeric(observed_usage[column], errors="coerce").to_numpy(dtype=float)
        converted_values = pd.to_numeric(
            observed_converted[column], errors="coerce"
        ).to_numpy(dtype=float)
        if not np.allclose(
            converted_values,
            raw_values / divisor,
            rtol=0.0,
            atol=1e-15,
            equal_nan=True,
        ):
            raise ValueError(
                f"converted {column} is not exactly the raw value divided by representative capacity"
            )

    vm_converted["q_cpu"] = vm_converted["q_cpu"].clip(lower=EPSILON)
    vm_converted["q_mem"] = vm_converted["q_mem"].clip(lower=EPSILON)
    vm_converted["spot_revenue"] = np.where(
        vm_converted["class"].eq("spot"), 0.05 * vm_converted["q_cpu"], 0.0
    )
    vm_converted["migration_energy_coeff"] = np.where(
        vm_converted["class"].eq("on_demand"),
        0.01 * vm_converted["q_cpu"] + 0.005 * vm_converted["q_mem"],
        0.0,
    )
    for resource in ["cpu", "mem"]:
        values = observed_converted[f"{resource}_usage"].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"converted observed {resource} usage must be finite and nonnegative")
        q = vm_converted[f"q_{resource}"].to_numpy(dtype=float)
        request = vm_converted[f"resource_request_{resource}"].to_numpy(dtype=float)
        observed_max = vm_converted[f"max_{resource}_usage"].to_numpy(dtype=float)
        if not np.isfinite(q).all() or (q <= 0).any():
            raise ValueError(f"converted q_{resource} must be finite and positive")
        if (q + 1e-12 < request).any() or (q + 1e-12 < observed_max).any():
            raise ValueError(f"converted q_{resource} invariant violated")

    resource_scale: dict[str, object] = {
        "semantic": (
            "unit conversion only: divide every CPU quantity by the representative machine CPU "
            "capacity and every memory quantity by the representative machine memory capacity"
        ),
        "representative_capacity_cpu": cpu_divisor,
        "representative_capacity_mem": mem_divisor,
        "cpu_divisor": cpu_divisor,
        "mem_divisor": mem_divisor,
        "cpu_multiplier": float(1.0 / cpu_divisor),
        "mem_multiplier": float(1.0 / mem_divisor),
        "automatic_utilization_calibration_applied": False,
        "method": str(representative_capacity["method"]),
    }
    vm_converted.attrs.update(vm_attrs)
    observed_converted.attrs.update(observed_attrs)
    vm_converted.attrs["resource_scale"] = resource_scale
    observed_converted.attrs["resource_scale"] = resource_scale
    return vm_converted, observed_converted, resource_scale


def drop_vms_exceeding_representative_capacity(
    raw_vm_requests: pd.DataFrame,
    raw_observed_usage: pd.DataFrame,
    converted_vm_requests: pd.DataFrame,
    converted_observed_usage: pd.DataFrame,
    *,
    tolerance: float = 1e-12,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Drop sampled VMs whose configured demand exceeds one representative server.

    Deterministic candidate sampling happens before this quality gate.  The
    gate never clips configured resources and therefore may leave fewer than
    ``max_instances`` VMs in the processed pool.
    """

    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("representative-capacity filter tolerance must be finite and nonnegative")
    if converted_vm_requests.empty:
        raise ValueError("cannot apply representative-capacity filter to an empty VM table")
    if converted_vm_requests["vm_id"].duplicated().any():
        raise ValueError("converted vm_requests contains duplicate vm_id values")

    q_cpu = pd.to_numeric(converted_vm_requests["q_cpu"], errors="coerce")
    q_mem = pd.to_numeric(converted_vm_requests["q_mem"], errors="coerce")
    if not np.isfinite(q_cpu.to_numpy(dtype=float)).all() or not np.isfinite(
        q_mem.to_numpy(dtype=float)
    ).all():
        raise ValueError("representative-capacity filter requires finite q_cpu and q_mem")

    cpu_exceeds = q_cpu.gt(1.0 + tolerance)
    mem_exceeds = q_mem.gt(1.0 + tolerance)
    dropped = cpu_exceeds | mem_exceeds
    retained_ids = set(
        converted_vm_requests.loc[~dropped, "vm_id"].astype(str).tolist()
    )
    dropped_rows = converted_vm_requests.loc[dropped].copy()
    if not retained_ids:
        raise ValueError(
            "all sampled VMs exceed the representative server capacity in q_cpu or q_mem"
        )

    by_class: dict[str, dict[str, int]] = {}
    for vm_class, group in converted_vm_requests.groupby("class", sort=True, dropna=False):
        group_cpu = pd.to_numeric(group["q_cpu"], errors="coerce").gt(1.0 + tolerance)
        group_mem = pd.to_numeric(group["q_mem"], errors="coerce").gt(1.0 + tolerance)
        group_dropped = group_cpu | group_mem
        by_class[str(vm_class)] = {
            "source_vm_count": int(len(group)),
            "dropped_vm_count": int(group_dropped.sum()),
            "retained_vm_count": int((~group_dropped).sum()),
        }

    diagnostics: dict[str, object] = {
        "policy": (
            "after deterministic hash sampling and representative-machine unit conversion, "
            "drop (without clipping or replacement) every VM with q_cpu > 1 or q_mem > 1"
        ),
        "tolerance": float(tolerance),
        "source_vm_count": int(len(converted_vm_requests)),
        "retained_vm_count": int((~dropped).sum()),
        "dropped_vm_count": int(dropped.sum()),
        "dropped_cpu_exceeds_count": int(cpu_exceeds.sum()),
        "dropped_mem_exceeds_count": int(mem_exceeds.sum()),
        "dropped_both_exceed_count": int((cpu_exceeds & mem_exceeds).sum()),
        "by_class": by_class,
        "dropped_vm_ids": sorted(dropped_rows["vm_id"].astype(str).tolist()),
        "dropped_q_cpu_summary": _numeric_summary(dropped_rows["q_cpu"]),
        "dropped_q_mem_summary": _numeric_summary(dropped_rows["q_mem"]),
        "sampling_replacement_applied": False,
        "configured_resource_clipping_applied": False,
    }

    def filtered(frame: pd.DataFrame) -> pd.DataFrame:
        attrs = dict(frame.attrs)
        result = frame.loc[frame["vm_id"].astype(str).isin(retained_ids)].copy()
        result.attrs.update(attrs)
        result.attrs["representative_capacity_vm_filter"] = diagnostics
        return result

    raw_vm_filtered = filtered(raw_vm_requests)
    raw_usage_filtered = filtered(raw_observed_usage)
    converted_vm_filtered = filtered(converted_vm_requests)
    converted_usage_filtered = filtered(converted_observed_usage)

    if len(raw_vm_filtered) != len(converted_vm_filtered):
        raise ValueError("raw and converted VM tables disagree after representative-capacity filtering")
    if converted_vm_filtered[["q_cpu", "q_mem"]].gt(1.0 + tolerance).any().any():
        raise ValueError("representative-capacity filter retained a VM with q above one")
    return (
        raw_vm_filtered,
        raw_usage_filtered,
        converted_vm_filtered,
        converted_usage_filtered,
        diagnostics,
    )


def build_servers(num_servers: int = 6) -> pd.DataFrame:
    if num_servers <= 0:
        raise ValueError("num_servers must be positive")
    return pd.DataFrame(
        [
            {
                "server_id": f"s{i:03d}",
                "C_cpu": 1.0,
                "C_mem": 1.0,
                "E_idle": 0.35,
                "E_cpu": 0.65,
                "min_on_time": 1,
                "min_off_time": 1,
            }
            for i in range(num_servers)
        ]
    )


def build_energy_scenarios(num_scenarios: int = 5, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 43)
    rows = []
    hours = np.arange(HORIZON_HOURS)
    base_price = 45.0 + 8.0 * np.sin((hours - 7) / 24.0 * 2 * np.pi) + 10.0 * np.exp(
        -0.5 * ((hours - 18) / 3.0) ** 2
    )
    base_price = np.maximum(base_price, 20.0)
    solar_shape = np.exp(-0.5 * ((hours - 13) / 3.0) ** 2)
    solar_shape[(hours < 6) | (hours > 19)] = 0.0

    for scenario_id in range(num_scenarios):
        da_noise = rng.normal(0.0, 1.5) if scenario_id > 0 else 0.0
        rt_noise = rng.normal(0.0, 4.0, size=HORIZON_HOURS) if scenario_id > 0 else np.zeros(HORIZON_HOURS)
        renewable_noise = (
            rng.lognormal(mean=-0.5 * 0.18**2, sigma=0.18, size=HORIZON_HOURS)
            if scenario_id > 0
            else np.ones(HORIZON_HOURS)
        )
        day_ahead = np.maximum(base_price + da_noise, 0.0)
        real_time = np.maximum(day_ahead + rt_noise, -20.0)
        renewable = np.maximum(0.0, 0.75 * solar_shape * renewable_noise)
        sell_price = np.minimum(day_ahead * 0.85, real_time * 0.95)

        for hour in hours:
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "t_hour": int(hour),
                    "day_ahead_price": float(day_ahead[hour]),
                    "real_time_price": float(real_time[hour]),
                    "sell_price": float(sell_price[hour]),
                    "renewable_generation": float(renewable[hour]),
                    "ess_capacity": 1.0,
                    "ess_charge_max": 0.25,
                    "ess_discharge_max": 0.25,
                    "ess_charge_efficiency": 0.92,
                    "ess_discharge_efficiency": 0.92,
                }
            )
    return pd.DataFrame(rows, columns=ENERGY_COLUMNS)


def calibrate_resource_scale(
    vm_requests: pd.DataFrame,
    observed_usage: pd.DataFrame,
    num_servers: int,
    target_peak_utilization: float = 0.75,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Deprecated compatibility no-op; automatic utilization calibration is disabled."""

    del num_servers, target_peak_utilization
    vm_copy = vm_requests.copy()
    observed_copy = observed_usage.copy()
    vm_copy.attrs.update(vm_requests.attrs)
    observed_copy.attrs.update(observed_usage.attrs)
    return vm_copy, observed_copy, {
        "semantic": "compatibility no-op",
        "cpu_multiplier": 1.0,
        "mem_multiplier": 1.0,
        "automatic_utilization_calibration_applied": False,
    }


def _read_optional_table(raw_dir: Path, stems: Iterable[str]) -> pd.DataFrame | None:
    for stem in stems:
        parquet_path = raw_dir / f"{stem}.parquet"
        csv_path = raw_dir / f"{stem}.csv"
        csv_gz_path = raw_dir / f"{stem}.csv.gz"
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        if csv_path.exists():
            return pd.read_csv(csv_path)
        if csv_gz_path.exists():
            return pd.read_csv(csv_gz_path)
    return None


def _read_raw_day_start_us(raw_dir: Path) -> int | None:
    metadata_path = raw_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    value = metadata.get("day_start_us")
    return int(value) if value is not None else None


def _read_raw_metadata(raw_dir: Path) -> dict[str, object]:
    metadata_path = raw_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing required raw extraction metadata: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid raw extraction metadata JSON: {metadata_path}") from exc
    if not isinstance(metadata, dict):
        raise ValueError("raw extraction metadata must be a JSON object")
    return metadata


def _numeric_summary(values: pd.Series) -> dict[str, float | None]:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))]
    if numeric.empty:
        return {"min": None, "p50": None, "mean": None, "p95": None, "max": None}
    return {
        "min": float(numeric.min()),
        "p50": float(numeric.quantile(0.50)),
        "mean": float(numeric.mean()),
        "p95": float(numeric.quantile(0.95)),
        "max": float(numeric.max()),
    }


def _resource_summary(vm_requests: pd.DataFrame, observed_usage: pd.DataFrame) -> dict[str, object]:
    vm_columns = [
        "q_cpu",
        "q_mem",
        "resource_request_cpu",
        "resource_request_mem",
        "max_cpu_usage",
        "max_mem_usage",
    ]
    usage_columns = ["cpu_usage", "mem_usage", "max_cpu_usage", "max_mem_usage", "assigned_memory"]
    return {
        "vm_requests": {
            column: _numeric_summary(vm_requests[column])
            for column in vm_columns
            if column in vm_requests
        },
        "observed_usage": {
            column: _numeric_summary(observed_usage[column])
            for column in usage_columns
            if column in observed_usage
        },
    }


def _peak_30min_load(
    observed_usage: pd.DataFrame,
    cpu_capacity: float,
    mem_capacity: float,
) -> dict[str, object]:
    usage = observed_usage.copy()
    usage["t30"] = (usage["t5_day"].astype(int) // 6).astype(int)
    if "coverage_us" not in usage:
        raise ValueError("observed usage missing coverage_us for 30-minute load diagnostics")
    usage["cpu_coverage_weighted"] = usage["cpu_usage"] * usage["coverage_us"]
    usage["mem_coverage_weighted"] = usage["mem_usage"] * usage["coverage_us"]
    per_vm = (
        usage.groupby(["vm_id", "t30"], as_index=False)
        .agg(
            cpu_coverage_weighted=("cpu_coverage_weighted", "sum"),
            mem_coverage_weighted=("mem_coverage_weighted", "sum"),
            coverage_us=("coverage_us", "sum"),
        )
    )
    per_vm["cpu_load"] = per_vm["cpu_coverage_weighted"] / per_vm["coverage_us"]
    per_vm["mem_load"] = per_vm["mem_coverage_weighted"] / per_vm["coverage_us"]
    aggregate = (
        per_vm.groupby("t30", as_index=False)
        .agg(cpu_load=("cpu_load", "sum"), mem_load=("mem_load", "sum"))
        .sort_values("t30")
    )
    cpu_row = aggregate.loc[aggregate["cpu_load"].idxmax()]
    mem_row = aggregate.loc[aggregate["mem_load"].idxmax()]
    peak_cpu = float(cpu_row["cpu_load"])
    peak_mem = float(mem_row["mem_load"])
    return {
        "cpu": {
            "load": peak_cpu,
            "capacity": float(cpu_capacity),
            "utilization": float(peak_cpu / cpu_capacity),
            "t30": int(cpu_row["t30"]),
        },
        "memory": {
            "load": peak_mem,
            "capacity": float(mem_capacity),
            "utilization": float(peak_mem / mem_capacity),
            "t30": int(mem_row["t30"]),
        },
        "aggregation": (
            "per-VM CPU and memory coverage-duration-weighted means over available "
            "5-minute buckets, then sum"
        ),
    }


def _cpu_p100_audit(
    raw_vm_requests: pd.DataFrame,
    converted_vm_requests: pd.DataFrame,
) -> dict[str, object]:
    raw_outlier_mask = raw_vm_requests["cpu_p100_gt_one"].fillna(False).astype(bool)
    converted_q_outlier_mask = converted_vm_requests["q_cpu"].gt(1.0 + 1e-12)
    return {
        "status": CPU_P100_AUDIT_STATUS,
        "raw_threshold": 1.0,
        "selected_vm_count": int(raw_outlier_mask.sum()),
        "selected_vm_fraction": float(raw_outlier_mask.mean()) if len(raw_outlier_mask) else 0.0,
        "raw_cpu_p100_summary": _numeric_summary(
            raw_vm_requests.loc[raw_outlier_mask, "max_cpu_usage"]
        ),
        "raw_q_cpu_summary_for_flagged_vms": _numeric_summary(
            raw_vm_requests.loc[raw_outlier_mask, "q_cpu"]
        ),
        "converted_q_cpu_gt_one_count": int(converted_q_outlier_mask.sum()),
        "converted_q_cpu_gt_one_summary": _numeric_summary(
            converted_vm_requests.loc[converted_q_outlier_mask, "q_cpu"]
        ),
        "representative_capacity_vm_filter": converted_vm_requests.attrs.get(
            "representative_capacity_vm_filter", {}
        ),
        "machine_capacity_join_note": (
            "Episode quality filtering requires complete assigned-machine CPU-capacity coverage. "
            "After representative-shape conversion, q above one causes whole-VM removal, never clipping."
        ),
    }


def _write_preprocessing_diagnostics(
    output_dir: Path,
    raw_vm_requests: pd.DataFrame,
    converted_vm_requests: pd.DataFrame,
    assigned_machine_cpu_capacity_audit: dict[str, object],
) -> None:
    derivation = raw_vm_requests.attrs.get("vm_request_derivation", {})
    diagnostics = {
        "usage_interval_overlap_audit": raw_vm_requests.attrs.get(
            "usage_overlap_diagnostics",
            derivation.get("usage_interval_overlap", {}),
        ),
        "event_state_ambiguity_audit": raw_vm_requests.attrs.get(
            "event_state_ambiguity",
            derivation.get("event_state_ambiguity", {}),
        ),
        "cpu_p100_audit": _cpu_p100_audit(
            raw_vm_requests, converted_vm_requests
        ),
        "episode_quality_filter": raw_vm_requests.attrs.get("episode_quality_filter", {}),
        "representative_capacity_vm_filter": raw_vm_requests.attrs.get(
            "representative_capacity_vm_filter", {}
        ),
        "assigned_machine_cpu_capacity_audit": assigned_machine_cpu_capacity_audit,
        "policies": {
            "exact_duplicates": "drop rows identical across every raw input field before interval reconstruction",
            "overlapping_usage": (
                "elementary-interval union weighting; simultaneous non-NULL observations are averaged "
                "per resource and unique duration is counted once"
            ),
            "same_time_events": (
                "explicit lifecycle precedence with normal missing_type and first-observed machine "
                "preference; no BigQuery/input row-order tie-break"
            ),
            "ambiguous_event_state": (
                "strictly exclude unresolved candidates before the unchanged deterministic hash head"
            ),
            "cpu_maximum": (
                "derive from cpu_usage_distribution[10] p100; no maximum_usage.cpus fallback"
            ),
            "episode_quality_filter": (
                "drop before sampling on any empty p100, missing machine history, or incomplete "
                "positive CPU-capacity coverage"
            ),
            "assigned_machine_cpu_capacity": (
                "raw assigned-machine audit plus strict episode filter; after unit conversion, "
                "drop an entire VM when q_cpu or q_mem exceeds one representative server"
            ),
        },
    }
    (output_dir / "preprocessing_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, allow_nan=False), encoding="utf-8"
    )


def _write_scaling_diagnostics(
    output_dir: Path,
    raw_vm_requests: pd.DataFrame,
    raw_observed_usage: pd.DataFrame,
    converted_vm_requests: pd.DataFrame,
    converted_observed_usage: pd.DataFrame,
    representative_capacity: dict[str, float | int | str],
    machine_diagnostics: dict[str, object],
    resource_scale: dict[str, object],
    raw_resource_unit: str,
    num_servers: int,
    assigned_machine_cpu_capacity_audit: dict[str, object],
) -> None:
    cpu_capacity = float(representative_capacity["capacity_cpu"])
    mem_capacity = float(representative_capacity["capacity_mem"])
    fallback_counts = {
        key: value
        for key, value in converted_vm_requests.attrs.get("vm_request_derivation", {}).items()
        if str(key).endswith("_count")
    }
    raw_peak = _peak_30min_load(
        raw_observed_usage,
        cpu_capacity=float(num_servers) * cpu_capacity,
        mem_capacity=float(num_servers) * mem_capacity,
    )
    converted_peak = _peak_30min_load(
        converted_observed_usage,
        cpu_capacity=float(num_servers),
        mem_capacity=float(num_servers),
    )
    derivation = converted_vm_requests.attrs.get("vm_request_derivation", {})
    overlap_diagnostics = converted_vm_requests.attrs.get(
        "usage_overlap_diagnostics",
        derivation.get("usage_interval_overlap", {}),
    )
    event_ambiguity = converted_vm_requests.attrs.get(
        "event_state_ambiguity",
        derivation.get("event_state_ambiguity", {}),
    )
    cpu_p100_audit = _cpu_p100_audit(
        raw_vm_requests, converted_vm_requests
    )
    diagnostics = {
        "raw_resource_unit": raw_resource_unit,
        "representative_machine_capacity_raw": {
            "cpu": cpu_capacity,
            "mem": mem_capacity,
        },
        "representative_machine_selection_method": representative_capacity["method"],
        "unit_conversion_factors": {
            "cpu_divisor": cpu_capacity,
            "mem_divisor": mem_capacity,
            "cpu_multiplier": float(1.0 / cpu_capacity),
            "mem_multiplier": float(1.0 / mem_capacity),
        },
        "automatic_utilization_calibration_applied": False,
        "num_machine_capacity_pairs": int(machine_diagnostics["joint_shape_count"]),
        "selected_capacity_pair_weight": int(representative_capacity["selected_weight_us"]),
        "raw_request_cpu_summary": _numeric_summary(raw_vm_requests["resource_request_cpu"]),
        "raw_request_mem_summary": _numeric_summary(raw_vm_requests["resource_request_mem"]),
        "raw_usage_cpu_summary": _numeric_summary(raw_observed_usage["cpu_usage"]),
        "raw_usage_mem_summary": _numeric_summary(raw_observed_usage["mem_usage"]),
        "converted_request_cpu_summary": _numeric_summary(
            converted_vm_requests["resource_request_cpu"]
        ),
        "converted_request_mem_summary": _numeric_summary(
            converted_vm_requests["resource_request_mem"]
        ),
        "converted_usage_cpu_summary": _numeric_summary(converted_observed_usage["cpu_usage"]),
        "converted_usage_mem_summary": _numeric_summary(converted_observed_usage["mem_usage"]),
        "peak_30min_cpu_load": converted_peak["cpu"]["load"],
        "peak_30min_mem_load": converted_peak["memory"]["load"],
        "total_server_cpu_capacity": float(num_servers),
        "total_server_mem_capacity": float(num_servers),
        "peak_cpu_utilization": converted_peak["cpu"]["utilization"],
        "peak_mem_utilization": converted_peak["memory"]["utilization"],
        "representative_machine_capacity": representative_capacity,
        "machine_capacity_derivation": machine_diagnostics,
        "unit_conversion": resource_scale,
        "fallback_counts": fallback_counts,
        "usage_interval_overlap_audit": overlap_diagnostics,
        "event_state_ambiguity_audit": event_ambiguity,
        "cpu_p100_audit": cpu_p100_audit,
        "episode_quality_filter": converted_vm_requests.attrs.get(
            "episode_quality_filter", {}
        ),
        "representative_capacity_vm_filter": converted_vm_requests.attrs.get(
            "representative_capacity_vm_filter", {}
        ),
        "assigned_machine_cpu_capacity_audit": assigned_machine_cpu_capacity_audit,
        "raw_resource_summary": _resource_summary(raw_vm_requests, raw_observed_usage),
        "converted_resource_summary": _resource_summary(
            converted_vm_requests, converted_observed_usage
        ),
        "peak_30min": {
            "raw_trace_units": raw_peak,
            "representative_machine_units": converted_peak,
        },
    }
    (output_dir / "scaling_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, allow_nan=False), encoding="utf-8"
    )


def _write_metadata(
    output_dir: Path,
    raw_dir: Path,
    vm_requests: pd.DataFrame,
    usage_scenarios: pd.DataFrame,
    num_servers: int,
    seed: int,
    num_scenarios: int,
    resource_scale: dict[str, object],
    representative_capacity: dict[str, float | int | str],
    assigned_machine_cpu_capacity_audit: dict[str, object],
) -> None:
    fallback_counts = {
        key: value
        for key, value in vm_requests.attrs.get("vm_request_derivation", {}).items()
        if str(key).endswith("_count")
    }
    representative_capacity_raw = {
        "cpu": float(representative_capacity["capacity_cpu"]),
        "mem": float(representative_capacity["capacity_mem"]),
    }
    metadata_resource_scale = {
        "automatic_utilization_calibration_applied": False,
        "cpu_divisor": "representative_machine_capacity_raw.cpu",
        "mem_divisor": "representative_machine_capacity_raw.mem",
        "converted_server_capacity": {"cpu": 1.0, "mem": 1.0},
    }
    metadata = {
        "dataset_name": "notion_toy_google2019_v5_mode_capacity_bounded",
        "source": "Google ClusterData 2019 BigQuery public dataset",
        "source_note": "This is a VM-like toy dataset derived from Google ClusterData 2019 Borg instance traces, not a real public-cloud VM trace.",
        "raw_dir": str(raw_dir),
        "seed": seed,
        "num_scenarios": num_scenarios,
        "num_servers": num_servers,
        "num_vms": int(len(vm_requests)),
        "horizon_t5": HORIZON_T5,
        "horizon_hours": HORIZON_HOURS,
        "resource_scale": metadata_resource_scale,
        "representative_machine_capacity_raw": representative_capacity_raw,
        "representative_machine_selection_method": representative_capacity["method"],
        "unit_conversion": {
            "cpu_divisor": float(resource_scale["cpu_divisor"]),
            "mem_divisor": float(resource_scale["mem_divisor"]),
        },
        "automatic_utilization_calibration_applied": False,
        "fallback_counts": fallback_counts,
        "usage_interval_overlap_audit": vm_requests.attrs.get(
            "usage_overlap_diagnostics", {}
        ),
        "event_state_ambiguity_audit": vm_requests.attrs.get(
            "event_state_ambiguity", {}
        ),
        "cpu_p100_audit": {
            "status": CPU_P100_AUDIT_STATUS,
            "selected_vm_count": int(
                vm_requests["cpu_p100_gt_one"].fillna(False).astype(bool).sum()
            ),
            "converted_q_cpu_gt_one_count": int(
                vm_requests["q_cpu"].gt(1.0 + 1e-12).sum()
            ),
        },
        "episode_quality_filter": vm_requests.attrs.get("episode_quality_filter", {}),
        "representative_capacity_vm_filter": vm_requests.attrs.get(
            "representative_capacity_vm_filter", {}
        ),
        "assigned_machine_cpu_capacity_audit": {
            "status": assigned_machine_cpu_capacity_audit["status"],
            "unit_key": assigned_machine_cpu_capacity_audit["unit_key"],
            "denominator_definition": assigned_machine_cpu_capacity_audit[
                "denominator_definition"
            ],
            "scopes": assigned_machine_cpu_capacity_audit["scopes"],
        },
        "scaling_diagnostics_file": "scaling_diagnostics.json",
        "preprocessing_diagnostics_file": "preprocessing_diagnostics.json",
        "default_model_usage_file": "vm_usage_hourly_scenarios.csv",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def build_toy_instance(
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_instances: int = 5_000,
    seed: int = 42,
    num_servers: int = 6,
    num_scenarios: int = 5,
    strict_event_state: bool = True,
) -> dict[str, int | str]:
    raw_metadata = _read_raw_metadata(raw_dir)
    if "day_start_us" not in raw_metadata or "day_end_us" not in raw_metadata:
        raise ValueError("raw extraction metadata must contain day_start_us and day_end_us")
    day_start_us = int(raw_metadata["day_start_us"])
    day_end_us = int(raw_metadata["day_end_us"])
    raw_resource_unit = str(raw_metadata.get("resource_unit") or DEFAULT_RAW_RESOURCE_UNIT)
    if day_end_us <= day_start_us:
        raise ValueError("raw extraction metadata has invalid day bounds")
    usage = _read_optional_table(raw_dir, ["usage_5min", "instance_usage", "usage"])
    if usage is None:
        raise FileNotFoundError(f"could not find usage_5min/instance_usage table under {raw_dir}")
    events = _read_optional_table(raw_dir, ["instance_events", "events"])
    collection_events = _read_optional_table(raw_dir, ["collection_events"])
    machine_events = _read_optional_table(raw_dir, ["machine_events"])
    if machine_events is None or machine_events.empty:
        raise FileNotFoundError(
            f"machine_events.parquet is required and must not be empty under {raw_dir}"
        )
    representative_capacity, machine_diagnostics = derive_representative_machine_capacity(
        machine_events, day_start_us=day_start_us, day_end_us=day_end_us
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_vm_requests, raw_observed = build_vm_requests(
        usage,
        events=events,
        collection_events=collection_events,
        machine_events=machine_events,
        max_instances=max_instances,
        seed=seed,
        day_start_us=day_start_us,
        day_end_us=day_end_us,
        strict_event_state=strict_event_state,
        episode_unit=True,
    )
    vm_requests, observed, resource_scale = convert_resource_units(
        raw_vm_requests,
        raw_observed,
        representative_capacity=representative_capacity,
    )
    (
        raw_vm_requests,
        raw_observed,
        vm_requests,
        observed,
        _representative_capacity_filter,
    ) = drop_vms_exceeding_representative_capacity(
        raw_vm_requests,
        raw_observed,
        vm_requests,
        observed,
    )
    assigned_machine_cpu_capacity_audit = audit_assigned_machine_cpu_capacity(
        usage,
        machine_events,
        day_start_us=day_start_us,
        day_end_us=day_end_us,
        selected_keys=raw_vm_requests,
    )
    usage_scenarios = build_usage_scenarios(observed, vm_requests, num_scenarios=num_scenarios, seed=seed)
    scenario0 = usage_scenarios.loc[usage_scenarios["scenario_id"].eq(0)].sort_values(
        ["vm_id", "t5_day"], kind="stable"
    ).reset_index(drop=True)
    expected_scenario0 = observed.sort_values(["vm_id", "t5_day"], kind="stable").reset_index(
        drop=True
    )
    same_keys = len(scenario0) == len(expected_scenario0)
    if same_keys:
        same_keys = scenario0[["vm_id", "t5_day"]].equals(
            expected_scenario0[["vm_id", "t5_day"]]
        )
    same_values = same_keys and all(
        np.allclose(
            scenario0[column].to_numpy(dtype=float),
            expected_scenario0[column].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-15,
        )
        for column in ["cpu_usage", "mem_usage"]
    )
    if not same_values:
        raise ValueError(
            "scenario 0 must exactly preserve converted observed CPU and memory demand"
        )
    invalid_memory = scenario0["mem_usage"].gt(1.0 + 1e-12)
    if invalid_memory.any():
        invalid_vm_ids = sorted(scenario0.loc[invalid_memory, "vm_id"].astype(str).unique().tolist())
        raise ValueError(
            "scenario 0 contains per-VM memory usage above one representative machine capacity; "
            f"VMs: {invalid_vm_ids[:20]}"
        )
    hourly_usage_scenarios = build_hourly_usage_scenarios(usage_scenarios)
    spot_preemptions = build_spot_preemption_scenarios(
        vm_requests, observed, events=events, num_scenarios=num_scenarios, seed=seed
    )
    batch_families, batch_workload = build_batch_outputs(
        vm_requests, observed, max_families=10, num_scenarios=num_scenarios, seed=seed
    )
    servers = build_servers(num_servers=num_servers)
    if not servers[["C_cpu", "C_mem"]].eq(1.0).all().all():
        raise ValueError("all server CPU and memory capacities must equal 1.0 after unit conversion")
    energy = build_energy_scenarios(num_scenarios=num_scenarios, seed=seed)
    scenario_probabilities = build_scenario_probabilities(num_scenarios=num_scenarios)
    model_params = build_model_params()

    vm_requests.to_csv(output_dir / "vm_requests.csv", index=False)
    usage_scenarios.to_csv(output_dir / "vm_usage_5min_scenarios.csv", index=False)
    usage_scenarios.to_csv(output_dir / "vm_usage_scenarios.csv", index=False)
    hourly_usage_scenarios.to_csv(output_dir / "vm_usage_hourly_scenarios.csv", index=False)
    spot_preemptions.to_csv(output_dir / "spot_preemption_scenarios.csv", index=False)
    batch_families.to_csv(output_dir / "batch_families.csv", index=False)
    batch_workload.to_csv(output_dir / "batch_workload.csv", index=False)
    servers.to_csv(output_dir / "servers.csv", index=False)
    energy.to_csv(output_dir / "energy_scenarios.csv", index=False)
    scenario_probabilities.to_csv(output_dir / "scenario_probabilities.csv", index=False)
    (output_dir / "model_params.json").write_text(json.dumps(model_params, indent=2), encoding="utf-8")
    _write_scaling_diagnostics(
        output_dir,
        raw_vm_requests=raw_vm_requests,
        raw_observed_usage=raw_observed,
        converted_vm_requests=vm_requests,
        converted_observed_usage=observed,
        representative_capacity=representative_capacity,
        machine_diagnostics=machine_diagnostics,
        resource_scale=resource_scale,
        raw_resource_unit=raw_resource_unit,
        num_servers=num_servers,
        assigned_machine_cpu_capacity_audit=assigned_machine_cpu_capacity_audit,
    )
    _write_preprocessing_diagnostics(
        output_dir,
        raw_vm_requests=raw_vm_requests,
        converted_vm_requests=vm_requests,
        assigned_machine_cpu_capacity_audit=assigned_machine_cpu_capacity_audit,
    )
    _write_metadata(
        output_dir,
        raw_dir,
        vm_requests,
        usage_scenarios,
        num_servers,
        seed,
        num_scenarios,
        resource_scale,
        representative_capacity,
        assigned_machine_cpu_capacity_audit,
    )

    from google2019_toy.validate_outputs import validate_output_dir

    summary = validate_output_dir(output_dir, write_summary=True)
    return {
        "output_dir": str(output_dir),
        "num_vms": int(len(vm_requests)),
        "num_usage_rows": int(len(usage_scenarios)),
        "num_hourly_usage_rows": int(len(hourly_usage_scenarios)),
        "num_batch_families": int(len(batch_families)),
        "num_servers": int(len(servers)),
        "num_scenarios": int(summary["num_scenarios"]),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a toy stochastic VM placement / energy dataset.")
    parser.add_argument("--raw_dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_instances", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_servers", type=int, default=6)
    parser.add_argument("--num_scenarios", type=int, default=5)
    parser.add_argument(
        "--allow_ambiguous_event_state",
        action="store_true",
        help="Legacy reproduction only: retain unresolved same-time event-state conflicts.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_toy_instance(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        max_instances=args.max_instances,
        seed=args.seed,
        num_servers=args.num_servers,
        num_scenarios=args.num_scenarios,
        strict_event_state=not args.allow_ambiguous_event_state,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
