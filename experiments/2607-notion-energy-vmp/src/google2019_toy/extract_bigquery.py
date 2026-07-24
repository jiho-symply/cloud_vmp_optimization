from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TRACE_START_US = 600 * 1_000_000
DAY_US = 24 * 60 * 60 * 1_000_000
STEP_US = 300 * 1_000_000
MIN_USAGE_ROWS = 12
DEFAULT_OUTPUT_DIR = Path("data/raw/google2019_cell_a_day0")
RESOURCE_UNIT = (
    "Google ClusterData 2019 v3 normalized CPU and memory units; "
    "CPU and memory are independently normalized against trace-wide maximum "
    "machine capacities. No utilization-target scaling is applied during extraction."
)
USAGE_INTERVAL_POLICY = (
    "Intervals overlapping the requested day are retained and clipped; "
    "partial windows are preserved."
)
USAGE_OUTPUT_COLUMNS = (
    "collection_id",
    "instance_index",
    "machine_id",
    "collection_type",
    "alloc_collection_id",
    "alloc_instance_index",
    "start_time",
    "end_time",
    "clipped_start_time",
    "clipped_end_time",
    "overlap_us",
    "cpu_usage",
    "cpu_usage_distribution",
    "mem_usage",
    "max_cpu_usage",
    "max_mem_usage",
    "assigned_memory",
    "t5",
    "t5_day",
)
INSTANCE_EVENT_OUTPUT_COLUMNS = (
    "collection_id",
    "instance_index",
    "time",
    "event_type",
    "missing_type",
    "collection_type",
    "alloc_collection_id",
    "alloc_instance_index",
    "scheduling_class",
    "priority",
    "machine_id",
    "resource_request_cpu",
    "resource_request_mem",
)
COLLECTION_EVENT_OUTPUT_COLUMNS = (
    "collection_id",
    "time",
    "event_type",
    "missing_type",
    "scheduler",
    "collection_type",
    "scheduling_class",
    "priority",
    "max_per_machine",
    "max_per_switch",
)
MACHINE_EVENT_OUTPUT_COLUMNS = (
    "time",
    "machine_id",
    "event_type",
    "capacity_cpu",
    "capacity_mem",
    "platform_id",
    "switch_id",
    "missing_data_reason",
)


def day_bounds_us(
    day_index: int,
    trace_start_offset_seconds: int = TRACE_START_US // 1_000_000,
) -> tuple[int, int]:
    if day_index < 0:
        raise ValueError("day_index must be non-negative")
    if trace_start_offset_seconds < 0:
        raise ValueError("trace_start_offset_seconds must be non-negative")
    day_start_us = trace_start_offset_seconds * 1_000_000 + day_index * DAY_US
    return day_start_us, day_start_us + DAY_US


def table_name(cell: str, table: str) -> str:
    safe_cell = "".join(ch for ch in cell.lower() if ch.isalnum())
    if not safe_cell:
        raise ValueError("cell must contain at least one alphanumeric character")
    return f"`google.com:google-cluster-data`.clusterdata_2019_{safe_cell}.{table}"


def build_selected_usage_cte(cell: str) -> str:
    usage_table = table_name(cell, "instance_usage")
    return f"""
filtered_usage AS (
  SELECT
    collection_id,
    instance_index,
    machine_id,
    collection_type,
    alloc_collection_id,
    alloc_instance_index,
    start_time,
    end_time,
    GREATEST(start_time, @day_start_us) AS clipped_start_time,
    LEAST(end_time, @day_end_us) AS clipped_end_time,
    LEAST(end_time, @day_end_us)
      - GREATEST(start_time, @day_start_us) AS overlap_us,
    SAFE_CAST(average_usage.cpus AS FLOAT64) AS cpu_usage,
    ARRAY(
      SELECT SAFE_CAST(cpu_value AS FLOAT64)
      FROM UNNEST(cpu_usage_distribution) AS cpu_value WITH OFFSET AS value_offset
      ORDER BY value_offset
    ) AS cpu_usage_distribution,
    SAFE_CAST(average_usage.memory AS FLOAT64) AS mem_usage,
    SAFE_CAST(maximum_usage.cpus AS FLOAT64) AS max_cpu_usage,
    SAFE_CAST(maximum_usage.memory AS FLOAT64) AS max_mem_usage,
    SAFE_CAST(assigned_memory AS FLOAT64) AS assigned_memory,
    DIV(start_time, {STEP_US}) AS t5,
    DIV(
      GREATEST(start_time, @day_start_us) - @day_start_us,
      300 * 1000000
    ) AS t5_day,
    FARM_FINGERPRINT(
      CONCAT(CAST(collection_id AS STRING), ':', CAST(instance_index AS STRING), ':', CAST(@seed AS STRING))
    ) AS sample_hash
  FROM {usage_table}
  WHERE start_time < @day_end_us
    AND end_time > @day_start_us
    AND end_time > start_time
    AND (alloc_collection_id IS NULL OR alloc_collection_id = 0)
),
candidates AS (
  SELECT
    collection_id,
    instance_index,
    ANY_VALUE(sample_hash) AS sample_hash,
    COUNT(*) AS usage_rows
  FROM filtered_usage
  GROUP BY collection_id, instance_index
  HAVING usage_rows >= {MIN_USAGE_ROWS}
  ORDER BY sample_hash, collection_id, instance_index
  LIMIT @max_instances
)
""".strip()


def build_usage_query(cell: str) -> str:
    return f"""
WITH {build_selected_usage_cte(cell)}
SELECT
  u.collection_id,
  u.instance_index,
  u.machine_id,
  u.collection_type,
  u.alloc_collection_id,
  u.alloc_instance_index,
  u.start_time,
  u.end_time,
  u.clipped_start_time,
  u.clipped_end_time,
  u.overlap_us,
  u.cpu_usage,
  u.cpu_usage_distribution,
  u.mem_usage,
  u.max_cpu_usage,
  u.max_mem_usage,
  u.assigned_memory,
  u.t5,
  u.t5_day
FROM filtered_usage AS u
JOIN candidates AS c
USING (collection_id, instance_index)
ORDER BY c.sample_hash, u.collection_id, u.instance_index, u.start_time
""".strip()


def build_events_query(cell: str) -> str:
    events_table = table_name(cell, "instance_events")
    return f"""
WITH {build_selected_usage_cte(cell)}
SELECT
  e.collection_id,
  e.instance_index,
  e.time,
  e.type AS event_type,
  e.missing_type,
  e.collection_type,
  e.alloc_collection_id,
  e.alloc_instance_index,
  e.scheduling_class,
  SAFE_CAST(e.priority AS FLOAT64) AS priority,
  e.machine_id,
  SAFE_CAST(e.resource_request.cpus AS FLOAT64) AS resource_request_cpu,
  SAFE_CAST(e.resource_request.memory AS FLOAT64) AS resource_request_mem
FROM {events_table} AS e
JOIN candidates AS c
USING (collection_id, instance_index)
WHERE e.time < @day_end_us
ORDER BY
  e.collection_id,
  e.instance_index,
  e.time,
  e.type,
  e.machine_id,
  e.resource_request.cpus,
  e.resource_request.memory,
  e.priority,
  e.scheduling_class,
  e.missing_type,
  e.collection_type,
  e.alloc_collection_id,
  e.alloc_instance_index
""".strip()


def build_collection_events_query(cell: str) -> str:
    collection_table = table_name(cell, "collection_events")
    return f"""
WITH {build_selected_usage_cte(cell)}
SELECT
  e.collection_id,
  e.time,
  e.type AS event_type,
  e.missing_type,
  CASE e.scheduler
    WHEN 1 THEN 'SCHEDULER_BATCH'
    WHEN 0 THEN 'SCHEDULER_DEFAULT'
    ELSE CONCAT('SCHEDULER_', CAST(e.scheduler AS STRING))
  END AS scheduler,
  CAST(e.collection_type AS STRING) AS collection_type,
  SAFE_CAST(e.scheduling_class AS FLOAT64) AS scheduling_class,
  SAFE_CAST(e.priority AS FLOAT64) AS priority,
  e.max_per_machine,
  e.max_per_switch
FROM {collection_table} AS e
JOIN (SELECT DISTINCT collection_id FROM candidates) AS c
USING (collection_id)
WHERE e.time < @day_end_us
ORDER BY
  e.collection_id,
  e.time,
  e.type,
  e.scheduler,
  e.priority,
  e.scheduling_class,
  e.missing_type,
  e.collection_type,
  e.max_per_machine,
  e.max_per_switch
""".strip()


def build_machine_events_query(cell: str) -> str:
    machine_table = table_name(cell, "machine_events")
    return f"""
SELECT
  e.time,
  e.machine_id,
  e.type AS event_type,
  SAFE_CAST(e.capacity.cpus AS FLOAT64) AS capacity_cpu,
  SAFE_CAST(e.capacity.memory AS FLOAT64) AS capacity_mem,
  e.platform_id,
  e.switch_id,
  e.missing_data_reason
FROM {machine_table} AS e
WHERE e.time < @day_end_us
ORDER BY e.time, e.machine_id, e.type
""".strip()


def _query_parameters(
    bigquery_module: object,
    day_start_us: int,
    day_end_us: int,
    max_instances: int,
    seed: int,
) -> list[object]:
    return [
        bigquery_module.ScalarQueryParameter("day_start_us", "INT64", day_start_us),
        bigquery_module.ScalarQueryParameter("day_end_us", "INT64", day_end_us),
        bigquery_module.ScalarQueryParameter("max_instances", "INT64", max_instances),
        bigquery_module.ScalarQueryParameter("seed", "INT64", seed),
    ]


def _machine_query_parameters(bigquery_module: object, day_end_us: int) -> list[object]:
    return [bigquery_module.ScalarQueryParameter("day_end_us", "INT64", day_end_us)]


def _dry_run(
    client: object,
    bigquery_module: object,
    label: str,
    query: str,
    query_parameters: list[object],
) -> int:
    job_config = bigquery_module.QueryJobConfig(
        dry_run=True,
        use_query_cache=False,
        query_parameters=query_parameters,
    )
    job = client.query(query, job_config=job_config)
    bytes_processed = int(job.total_bytes_processed)
    print(f"{label}: estimated bytes processed = {bytes_processed:,}")
    return bytes_processed


def _require_columns(frame: Any, required: tuple[str, ...], label: str) -> None:
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} query output is missing columns: {sorted(missing)}")


def _validate_query_output(label: str, frame: Any, max_instances: int) -> None:
    if label == "usage_5min":
        _require_columns(frame, USAGE_OUTPUT_COLUMNS, label)
        invalid_overlap = frame["overlap_us"].isna() | frame["overlap_us"].le(0)
        if bool(invalid_overlap.any()):
            raise ValueError("usage_5min query returned overlap_us <= 0 or NULL")
        invalid_clip = (
            frame["clipped_start_time"].isna()
            | frame["clipped_end_time"].isna()
            | frame["clipped_start_time"].ge(frame["clipped_end_time"])
        )
        if bool(invalid_clip.any()):
            raise ValueError(
                "usage_5min query returned clipped_start_time >= clipped_end_time or NULL"
            )
        selected_candidates = frame[["collection_id", "instance_index"]].drop_duplicates()
        if len(selected_candidates) > max_instances:
            raise ValueError(
                "usage_5min query returned more selected candidates than max_instances: "
                f"{len(selected_candidates)} > {max_instances}"
            )
    elif label == "instance_events":
        _require_columns(frame, INSTANCE_EVENT_OUTPUT_COLUMNS, label)
    elif label == "collection_events":
        _require_columns(frame, COLLECTION_EVENT_OUTPUT_COLUMNS, label)
    elif label == "machine_events":
        _require_columns(frame, MACHINE_EVENT_OUTPUT_COLUMNS, label)
        if frame.empty:
            raise ValueError(
                "machine_events query returned no rows before day_end_us; "
                "machine capacity state cannot be reconstructed"
            )
    else:
        raise ValueError(f"Unknown extraction query label: {label}")


def _run_to_parquet(
    client: object,
    bigquery_module: object,
    label: str,
    query: str,
    query_parameters: list[object],
    path: Path,
    max_instances: int,
) -> int:
    job_config = bigquery_module.QueryJobConfig(query_parameters=query_parameters)
    df = client.query(query, job_config=job_config).to_dataframe()
    _validate_query_output(label, df, max_instances)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"{label}: wrote {len(df):,} rows to {path}")
    return int(len(df))


def extract_google2019(
    project_id: str,
    cell: str = "a",
    day_index: int = 0,
    max_instances: int = 10_000,
    seed: int = 42,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    dry_run_only: bool = False,
    include_collection_events: bool = True,
    trace_start_offset_seconds: int = TRACE_START_US // 1_000_000,
) -> dict[str, int | str]:
    try:
        from google.cloud import bigquery
    except ImportError as exc:
        raise RuntimeError("google-cloud-bigquery is required for extraction; install requirements.txt first") from exc

    if max_instances <= 0:
        raise ValueError("max_instances must be positive")
    output_dir = Path(output_dir)
    trace_start_offset_us = trace_start_offset_seconds * 1_000_000
    day_start_us, day_end_us = day_bounds_us(
        day_index,
        trace_start_offset_seconds=trace_start_offset_seconds,
    )
    client = bigquery.Client(project=project_id)
    selected_query_parameters = _query_parameters(
        bigquery,
        day_start_us,
        day_end_us,
        max_instances,
        seed,
    )

    queries = {
        "usage_5min": build_usage_query(cell),
        "instance_events": build_events_query(cell),
        "machine_events": build_machine_events_query(cell),
    }
    if include_collection_events:
        queries["collection_events"] = build_collection_events_query(cell)
    query_parameters = {
        label: (
            _machine_query_parameters(bigquery, day_end_us)
            if label == "machine_events"
            else selected_query_parameters
        )
        for label in queries
    }

    sql_dir = output_dir / "sql"
    sql_dir.mkdir(parents=True, exist_ok=True)
    sql_files: dict[str, str] = {}
    for label, query in queries.items():
        sql_path = sql_dir / f"{label}.sql"
        sql_path.write_text(query + "\n", encoding="utf-8")
        sql_files[label] = str(sql_path)

    estimated = {
        label: _dry_run(client, bigquery, label, query, query_parameters[label])
        for label, query in queries.items()
    }
    row_counts: dict[str, int] = {}
    if not dry_run_only:
        row_counts = {
            label: _run_to_parquet(
                client,
                bigquery,
                label,
                query,
                query_parameters[label],
                output_dir / f"{label}.parquet",
                max_instances,
            )
            for label, query in queries.items()
        }
    metadata = {
        "source": "Google ClusterData 2019 BigQuery public dataset",
        "cell": cell,
        "day_index": day_index,
        "trace_start_offset_us": trace_start_offset_us,
        "day_start_us": day_start_us,
        "day_end_us": day_end_us,
        "max_instances": max_instances,
        "seed": seed,
        "resource_unit": RESOURCE_UNIT,
        "sampling_policy": {
            "key": ["collection_id", "instance_index"],
            "hash": (
                "FARM_FINGERPRINT(CONCAT(CAST(collection_id AS STRING), ':', "
                "CAST(instance_index AS STRING), ':', CAST(@seed AS STRING)))"
            ),
            "minimum_usage_rows": MIN_USAGE_ROWS,
            "ordering": ["sample_hash", "collection_id", "instance_index"],
            "limit": "@max_instances",
        },
        "usage_interval_policy": USAGE_INTERVAL_POLICY,
        "estimated_bytes_processed": estimated,
        "row_counts": row_counts,
        "dry_run_only": dry_run_only,
        "sql_files": sql_files,
        "tables": {
            "instance_usage": table_name(cell, "instance_usage"),
            "instance_events": table_name(cell, "instance_events"),
            "collection_events": table_name(cell, "collection_events"),
            "machine_events": table_name(cell, "machine_events"),
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if dry_run_only:
        return {
            "output_dir": str(output_dir),
            "dry_run_only": 1,
            "estimated_total_bytes_processed": int(sum(estimated.values())),
        }
    return {
        "output_dir": str(output_dir),
        "dry_run_only": 0,
        "estimated_total_bytes_processed": int(sum(estimated.values())),
        "usage_rows": int(row_counts.get("usage_5min", 0)),
        "event_rows": int(row_counts.get("instance_events", 0)),
        "collection_event_rows": int(row_counts.get("collection_events", 0)),
        "machine_event_rows": int(row_counts.get("machine_events", 0)),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract one bounded Google ClusterData 2019 day from BigQuery.")
    parser.add_argument("--project_id", required=True, help="GCP project billed for BigQuery queries.")
    parser.add_argument("--cell", default="a")
    parser.add_argument("--day_index", type=int, default=0)
    parser.add_argument(
        "--trace_start_offset_seconds",
        type=int,
        default=TRACE_START_US // 1_000_000,
    )
    parser.add_argument("--max_instances", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry_run_only", action="store_true", help="Print estimated bytes and exit without downloading.")
    parser.add_argument(
        "--include_collection_events",
        action="store_true",
        help="Deprecated compatibility flag; collection_events is included by default.",
    )
    parser.add_argument("--skip_collection_events", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = extract_google2019(
        project_id=args.project_id,
        cell=args.cell,
        day_index=args.day_index,
        max_instances=args.max_instances,
        seed=args.seed,
        output_dir=args.output_dir,
        dry_run_only=args.dry_run_only,
        include_collection_events=not args.skip_collection_events,
        trace_start_offset_seconds=args.trace_start_offset_seconds,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
