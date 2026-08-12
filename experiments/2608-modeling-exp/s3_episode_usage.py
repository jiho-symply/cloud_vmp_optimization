"""usage_5min과 s2 episode를 결합해 정규화 usage를 생성한다.
품질 플래그를 보존해 s4에서 필터링할 산출물을 저장한다.
검증: 2/867904, 23, 1760, 3, 9979126571, 27536, 7155, 24883.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from paths import S1_MACHINE_INTERVALS_PATH, S2_EPISODES_PATH, WORK_DIR
from raw_tables import load_usage


DAY_START_US = 600_000_000
DAY_END_US = 87_000_000_000
T5_US = 300_000_000
S3_EPISODE_USAGE_PATH = WORK_DIR / "s3_episode_usage.parquet"
EPISODE_KEYS = ["collection_id", "instance_index", "episode_index"]
DUPLICATE_KEYS = EPISODE_KEYS + ["start_time", "end_time"]
OVERLAP_ROW_COLUMNS = EPISODE_KEYS + ["_source_row_id"]
EPISODE_COLUMNS = [
    "collection_id", "instance_index", "episode_index",
    "lifecycle_start_time_us", "episode_start_time_us", "episode_end_time_us",
    "termination_time_us", "termination_reason", "horizon_censored",
    "termination_time_uncertain", "episode_start_inferred_from_update",
    "episode_start_inferred_from_usage",
]
CAPACITY_JOIN_COLUMNS = [
    "_source_row_id", "machine_id", "clipped_start_time", "clipped_end_time"
]
CAPACITY_INTERVAL_COLUMNS = ["machine_id", "start_time", "end_time", "capacity_cpu"]
SEGMENT_COLUMNS = [
    *EPISODE_COLUMNS, "machine_id", "_source_row_id", "clipped_start_time",
    "clipped_end_time", "cpu_usage", "mem_usage", "max_cpu_usage",
    "max_mem_usage", "assigned_memory", "cpu_complete", "capacity_complete",
]
METADATA_COLUMNS = EPISODE_COLUMNS[3:]
METADATA_AGGREGATIONS = {
    column: (column, "first") for column in METADATA_COLUMNS
}
OUTPUT_COLUMNS = [
    *EPISODE_COLUMNS, "t5", "t5_day", "t_hour", "cpu_usage", "mem_usage",
    "max_cpu_usage", "max_mem_usage", "vm_source_max_cpu_usage",
    "vm_source_max_mem_usage", "arrival_machine_ids", "assigned_memory",
    "cpu_complete", "capacity_complete", "coverage_us", "coverage_ratio",
    "overlap_source_row_count", "overlap_conflict_flag", "duplicated_timeline_us",
    "first_observed_us", "last_observed_us",
]
RESOURCE_COLUMNS = {
    "cpu": "cpu_usage", "mem": "mem_usage", "assigned_memory": "assigned_memory"
}
MAXIMA_AGGREGATIONS = {
    "vm_source_max_cpu_usage": ("max_cpu_usage", "max"),
    "vm_source_max_mem_usage": ("max_mem_usage", "max"),
}
QUALITY_AGGREGATIONS = {
    "cpu_complete": ("cpu_complete", "all"),
    "capacity_complete": ("capacity_complete", "all"),
}
USAGE_COLUMNS = ["cpu_usage", "mem_usage", "assigned_memory"]
WEIGHTED_COLUMNS = ["cpu_weighted", "mem_weighted", "assigned_memory_weighted"]
VALID_COLUMNS = ["cpu_valid_us", "mem_valid_us", "assigned_memory_valid_us"]
BUCKET_AGGREGATIONS = {
    "coverage_us": ("coverage_contribution_us", "sum"),
    "cpu_weighted": ("cpu_weighted", "sum"), "cpu_valid_us": ("cpu_valid_us", "sum"),
    "mem_weighted": ("mem_weighted", "sum"), "mem_valid_us": ("mem_valid_us", "sum"),
    "assigned_memory_weighted": ("assigned_memory_weighted", "sum"),
    "assigned_memory_valid_us": ("assigned_memory_valid_us", "sum"),
    "max_cpu_usage": ("max_cpu_usage", "max"), "max_mem_usage": ("max_mem_usage", "max"),
    "first_observed_us": ("segment_start", "min"),
    "last_observed_us": ("segment_end", "max"),
    "overlap_source_row_count": ("_source_row_id", "nunique"),
    "cpu_complete": ("cpu_complete", "all"),
    "capacity_complete": ("capacity_complete", "all"),
    **METADATA_AGGREGATIONS,
}


def _prepare_usage_source(usage: pd.DataFrame) -> pd.DataFrame:
    """중복 제거와 horizon 정규화를 마친 usage source를 반환한다."""
    # Trace 결함 1: ARRAY 필드가 있는 완전히 같은 usage 행을 한 번만 남긴다.
    comparison = usage.copy()
    comparison["cpu_usage_distribution"] = comparison[
        "cpu_usage_distribution"
    ].map(tuple)
    duplicate_mask = comparison.duplicated(keep="first")
    source = usage.loc[~duplicate_mask].copy().reset_index(drop=True)
    bounds = source[[
        "start_time", "end_time", "clipped_start_time", "clipped_end_time"
    ]].apply(pd.to_numeric, errors="coerce")
    source["clipped_start_time"] = bounds[
        ["start_time", "clipped_start_time"]
    ].max(axis=1).clip(lower=DAY_START_US)
    source["clipped_end_time"] = bounds[
        ["end_time", "clipped_end_time"]
    ].min(axis=1).clip(upper=DAY_END_US)
    distribution = source["cpu_usage_distribution"]
    source["max_cpu_usage"] = distribution.str[10].where(distribution.str.len().eq(11))
    valid = source["clipped_end_time"].gt(source["clipped_start_time"])
    source = source.loc[valid].reset_index(drop=True)
    source.attrs.update(
        exact_duplicate_rows_removed=int(duplicate_mask.sum()),
        rows_after_exact_dedup=len(source),
    )
    return source


def _split_usage_at_episode_boundaries(
    source: pd.DataFrame, episodes: pd.DataFrame,
) -> pd.DataFrame:
    """usage 행을 running episode와의 양의 겹침 조각으로 반환한다."""
    # Trace 결함 3: 5분 창이 episode 경계를 넘으면 episode별 조각으로 나눈다.
    source = _prepare_usage_source(source)
    episode_table = episodes.sort_values(
        ["collection_id", "instance_index", "episode_start_time_us"],
        kind="stable",
    )[EPISODE_COLUMNS]
    source["_source_position"] = np.arange(len(source), dtype=np.int64)
    joined = source.merge(
        episode_table, on=["collection_id", "instance_index"], how="left", sort=False
    )
    overlap_start = np.maximum(
        joined["clipped_start_time"], joined["episode_start_time_us"]
    )
    overlap_end = np.minimum(joined["clipped_end_time"], joined["episode_end_time_us"])
    matched = overlap_end.gt(overlap_start)
    matched_counts = matched.groupby(joined["_source_position"]).sum()
    all_positions = pd.Index(np.arange(len(source), dtype=np.int64))
    unassigned = int(matched_counts.reindex(all_positions, fill_value=0).eq(0).sum())
    boundary_split = int(matched_counts.gt(1).sum())
    result = joined.loc[matched].drop(columns="_source_position").reset_index(drop=True)
    result["clipped_start_time"] = overlap_start.loc[matched].to_numpy(dtype=np.int64)
    result["clipped_end_time"] = overlap_end.loc[matched].to_numpy(dtype=np.int64)
    result["overlap_us"] = result["clipped_end_time"] - result["clipped_start_time"]
    counts = result.groupby(EPISODE_KEYS)["max_cpu_usage"].transform("count")
    sizes = result.groupby(EPISODE_KEYS)["max_cpu_usage"].transform("size")
    result["cpu_complete"] = counts.eq(sizes)
    result.attrs.update(source.attrs)
    result.attrs.update(unassigned_usage_rows=unassigned,
                        boundary_split_usage_rows=boundary_split)
    result.attrs["usage_episode_count"] = len(result[EPISODE_KEYS].drop_duplicates())
    return result


# Trace 결함 3: 5분 창이 episode 경계를 넘으면 episode별 조각으로 나눈다.
def _make_t5_segments(
    source: pd.DataFrame, intervals: pd.DataFrame,
) -> pd.DataFrame:
    """usage 조각을 품질 플래그가 붙은 5분 bucket segment로 반환한다."""
    source = source.copy()
    source["_source_row_id"] = np.arange(len(source), dtype=np.int64)
    joined = source[CAPACITY_JOIN_COLUMNS].merge(
        intervals[CAPACITY_INTERVAL_COLUMNS], on="machine_id", how="left"
    )
    joined["capacity_overlap_us"] = (
        np.minimum(joined["clipped_end_time"], joined["end_time"])
        - np.maximum(joined["clipped_start_time"], joined["start_time"])
    ).clip(lower=0).where(joined["capacity_cpu"].gt(0), 0)
    covered = joined.groupby("_source_row_id")["capacity_overlap_us"].sum()
    duration = source["clipped_end_time"] - source["clipped_start_time"]
    source["capacity_complete"] = source["_source_row_id"].map(covered).fillna(0).eq(duration)
    duplicate_groups = source.groupby(DUPLICATE_KEYS, dropna=False).size().gt(1).sum()
    first_bucket = ((source["clipped_start_time"] - DAY_START_US) // T5_US).astype(int)
    last_bucket = ((source["clipped_end_time"] - 1 - DAY_START_US) // T5_US).astype(int)
    span = last_bucket - first_bucket + 1
    frames: list[pd.DataFrame] = []
    for offset in range(int(span.max())):
        positions = np.flatnonzero(span.to_numpy() > offset)
        if len(positions) == 0:
            continue
        part = source.iloc[positions][SEGMENT_COLUMNS].copy()
        part["t5_day"] = first_bucket.iloc[positions].to_numpy() + offset
        bucket_start = DAY_START_US + part["t5_day"].to_numpy() * T5_US
        part["segment_start"] = np.maximum(part["clipped_start_time"], bucket_start)
        part["segment_end"] = np.minimum(part["clipped_end_time"], bucket_start + T5_US)
        part["duration_us"] = part["segment_end"] - part["segment_start"]
        frames.append(part)
    segments = pd.concat(frames, ignore_index=True)
    valid = segments["t5_day"].between(0, 287) & segments["duration_us"].gt(0)
    segments = segments.loc[valid].copy()
    segments.attrs["duplicate_interval_groups"] = int(duplicate_groups)
    return segments


def _aggregate_overlap_bucket(
    group: pd.DataFrame, key: tuple[object, ...], bucket_keys: list[str],
) -> tuple[dict[str, object], int, int]:
    """한 overlap bucket의 elementary union 값과 overlap 집계를 반환한다."""
    boundaries = np.unique(np.concatenate([group["segment_start"], group["segment_end"]]))
    totals = {name: [0.0, 0] for name in RESOURCE_COLUMNS}
    record = dict(zip(bucket_keys, key))
    coverage = duplicated = max_concurrent = conflict = 0
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        duration = int(right - left)
        active = group.loc[group["segment_start"].lt(right) & group["segment_end"].gt(left)]
        concurrency = len(active)
        if duration > 0 and concurrency:
            coverage += duration
            duplicated += duration * max(concurrency - 1, 0)
            max_concurrent = max(max_concurrent, concurrency)
            for name, column in RESOURCE_COLUMNS.items():
                values = pd.to_numeric(active[column], errors="coerce").dropna()
                if not values.empty:
                    totals[name][0] += float(values.mean()) * duration
                    totals[name][1] += duration
                    conflict = conflict or concurrency > 1 and values.nunique() > 1
    cpu, mem, assigned = totals["cpu"], totals["mem"], totals["assigned_memory"]
    record.update(dict(
        coverage_us=coverage, cpu_weighted=cpu[0], cpu_valid_us=cpu[1],
        mem_weighted=mem[0], mem_valid_us=mem[1],
        assigned_memory_weighted=assigned[0], assigned_memory_valid_us=assigned[1],
        max_cpu_usage=group["max_cpu_usage"].max(),
        max_mem_usage=group["max_mem_usage"].max(),
        cpu_complete=bool(group["cpu_complete"].all()),
        capacity_complete=bool(group["capacity_complete"].all()),
        first_observed_us=int(group["segment_start"].min()),
        last_observed_us=int(group["segment_end"].max()),
        overlap_source_row_count=int(group["_source_row_id"].nunique()),
        overlap_conflict_flag=bool(conflict), duplicated_timeline_us=duplicated,
    ))
    record.update(group[METADATA_COLUMNS].iloc[0].to_dict())
    return record, max_concurrent, duplicated


def _elementary_interval_frame(segments: pd.DataFrame) -> pd.DataFrame:
    """segment에 elementary 경계와 resource weight를 붙인 DataFrame을 반환한다."""
    bucket_keys = EPISODE_KEYS + ["t5_day"]
    sort_columns = bucket_keys + ["segment_start", "segment_end", "_source_row_id"]
    segments = segments.sort_values(sort_columns, kind="stable").copy()
    running_end = segments.groupby(bucket_keys, sort=False)["segment_end"].cummax()
    groupers = [segments[column] for column in bucket_keys]
    previous_end = running_end.groupby(groupers, sort=False).shift()
    previous_end = previous_end.fillna(segments["segment_start"])
    union_start = np.maximum(segments["segment_start"], previous_end)
    segments["coverage_contribution_us"] = np.maximum(0, segments["segment_end"] - union_start)
    segments["overlaps_previous"] = segments["segment_start"].lt(previous_end)
    for name, column in RESOURCE_COLUMNS.items():
        valid = segments[column].notna()
        segments[f"{name}_weighted"], segments[f"{name}_valid_us"] = (
            np.where(valid, segments[column] * segments["duration_us"], 0.0),
            np.where(valid, segments["duration_us"], 0),
        )
    return segments


# Trace 결함 2: 같은 episode·t5_day의 겹친 구간을 elementary-interval
# union으로 합친다.
def _elementary_interval_union(segments: pd.DataFrame) -> pd.DataFrame:
    """overlap bucket을 union한 normalized aggregate를 반환한다."""
    bucket_keys = EPISODE_KEYS + ["t5_day"]
    segments = _elementary_interval_frame(segments)
    buckets = segments.groupby(bucket_keys, as_index=False, sort=False).agg(
        **BUCKET_AGGREGATIONS
    )
    affected = segments.loc[segments["overlaps_previous"], bucket_keys].drop_duplicates()
    rows = OVERLAP_ROW_COLUMNS
    overlap_rows = segments.loc[segments["overlaps_previous"], rows].drop_duplicates()
    buckets = buckets.assign(overlap_conflict_flag=False, duplicated_timeline_us=0)
    overlap_stats = {"max_concurrent_rows": 1, "duplicated_timeline_us": 0}
    if not affected.empty:
        affected["_affected"] = True
        affected_segments = segments.merge(affected, on=bucket_keys, how="inner")
        aggregate, groups = _aggregate_overlap_bucket, affected_segments.groupby(bucket_keys)
        aggregates = [aggregate(group, key, bucket_keys) for key, group in groups]
        records = [item[0] for item in aggregates]
        overlap_stats["max_concurrent_rows"] = max(item[1] for item in aggregates)
        overlap_stats["duplicated_timeline_us"] = sum(item[2] for item in aggregates)
        buckets = buckets.merge(affected, on=bucket_keys, how="left")
        buckets = buckets.loc[buckets["_affected"].isna()].drop(columns="_affected")
        buckets = pd.concat([buckets, pd.DataFrame(records)], ignore_index=True, sort=False)
    buckets.attrs["overlap_rows"] = int(len(overlap_rows))
    buckets.attrs.update(overlap_stats)
    return buckets


def _normalize_usage(usage: pd.DataFrame, episodes: pd.DataFrame,
                     intervals: pd.DataFrame) -> pd.DataFrame:
    """usage를 episode별 duration-weighted 5분 관측 bucket으로 반환한다."""
    fragments = _split_usage_at_episode_boundaries(usage, episodes)
    source_stats = dict(fragments.attrs)
    for column in ["cpu_usage", "mem_usage", "assigned_memory", "max_mem_usage"]:
        fragments[column] = pd.to_numeric(fragments[column], errors="coerce")
    split_stats = dict(fragments.attrs)
    segments = _make_t5_segments(fragments, intervals)
    buckets = _elementary_interval_union(segments)
    quality_flags = segments.groupby(EPISODE_KEYS).agg(**QUALITY_AGGREGATIONS)
    quality_keys = quality_flags.index[quality_flags.all(axis=1)]
    quality_fragments = fragments.set_index(EPISODE_KEYS).loc[quality_keys].reset_index()
    duplicate_groups = quality_fragments.groupby(DUPLICATE_KEYS, dropna=False).size().gt(1).sum()
    quality_segments = segments.set_index(EPISODE_KEYS).loc[quality_keys].reset_index()
    quality_buckets = _elementary_interval_union(quality_segments)
    quality_buckets.attrs["duplicate_interval_groups"] = int(duplicate_groups)
    maxima = segments.groupby(EPISODE_KEYS, as_index=False).agg(**MAXIMA_AGGREGATIONS)
    buckets = buckets.merge(maxima, on=EPISODE_KEYS, how="left")
    first_starts = segments.groupby(EPISODE_KEYS)["segment_start"].transform("min")
    first = segments.loc[segments["segment_start"].eq(first_starts)]
    machines = first.groupby(EPISODE_KEYS, sort=False)["machine_id"].agg(
        lambda values: tuple(sorted(values.dropna().astype(int).unique()))
    )
    buckets = buckets.merge(machines.rename("arrival_machine_ids"), on=EPISODE_KEYS, how="left")
    buckets[USAGE_COLUMNS] = buckets[WEIGHTED_COLUMNS].to_numpy() / buckets[
        VALID_COLUMNS
    ].replace(0, np.nan).to_numpy()
    buckets["coverage_ratio"] = buckets["coverage_us"] / T5_US
    buckets["t5"] = (DAY_START_US // T5_US + buckets["t5_day"]).astype(int)
    buckets["t_hour"] = (buckets["t5_day"] // 12).astype(int)
    valid = buckets[["cpu_usage", "mem_usage"]].notna().all(axis=1)
    result = buckets.loc[valid, OUTPUT_COLUMNS].sort_values(
        EPISODE_KEYS + ["t5_day"], kind="stable"
    ).reset_index(drop=True)
    result.attrs.update(source_stats | split_stats | quality_buckets.attrs)
    return result


def main() -> None:
    """s3 산출물을 parquet으로 저장하고 검증 수치를 출력한다."""
    usage = load_usage()
    episodes = pd.read_parquet(S2_EPISODES_PATH)
    intervals = pd.read_parquet(S1_MACHINE_INTERVALS_PATH)
    result = _normalize_usage(usage, episodes, intervals)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(S3_EPISODE_USAGE_PATH, index=False)
    print(
        "exact_duplicate_rows_removed="
        f"{result.attrs['exact_duplicate_rows_removed']}, "
        "rows_after_exact_dedup="
        f"{result.attrs['rows_after_exact_dedup']}"
    )
    for name in [
        "duplicate_interval_groups", "overlap_rows", "max_concurrent_rows",
        "duplicated_timeline_us", "unassigned_usage_rows",
        "boundary_split_usage_rows", "usage_episode_count",
    ]:
        print(f"{name}={result.attrs[name]}")


if __name__ == "__main__":
    main()
