from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import google2019_toy.extract_bigquery as extractor


def _normalized_sql(query: str) -> str:
    return " ".join(query.split())


def _valid_usage_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "collection_id": 1,
                "instance_index": 0,
                "machine_id": 100,
                "collection_type": 0,
                "alloc_collection_id": np.nan,
                "alloc_instance_index": np.nan,
                "start_time": 500_000_000,
                "end_time": 700_000_000,
                "clipped_start_time": 600_000_000,
                "clipped_end_time": 700_000_000,
                "overlap_us": 100_000_000,
                "cpu_usage": np.nan,
                "cpu_usage_distribution": [0.05, 0.1, 0.2],
                "mem_usage": 0.2,
                "max_cpu_usage": np.nan,
                "max_mem_usage": 0.4,
                "assigned_memory": np.nan,
                "t5": 1,
                "t5_day": 0,
            }
        ]
    )


def test_day_bounds_include_trace_start_offset() -> None:
    assert extractor.TRACE_START_US == 600_000_000
    assert extractor.day_bounds_us(0) == (600_000_000, 87_000_000_000)
    assert extractor.day_bounds_us(1) == (87_000_000_000, 173_400_000_000)
    assert extractor.day_bounds_us(0, trace_start_offset_seconds=0) == (
        0,
        86_400_000_000,
    )


def test_usage_sql_uses_overlap_clipping_and_required_schema() -> None:
    query = extractor.build_usage_query("a")
    normalized = _normalized_sql(query)

    assert "start_time < @day_end_us" in query
    assert "end_time > @day_start_us" in query
    assert "end_time > start_time" in query
    assert "start_time >= @day_start_us" not in query
    assert "end_time - start_time >=" not in query
    assert "GREATEST(start_time, @day_start_us) AS clipped_start_time" in query
    assert "LEAST(end_time, @day_end_us) AS clipped_end_time" in query
    assert "AS overlap_us" in query
    assert "DIV(start_time, 300000000) AS t5" in normalized
    assert (
        "DIV( GREATEST(start_time, @day_start_us) - @day_start_us, "
        "300 * 1000000 ) AS t5_day"
    ) in normalized

    for expression in (
        "machine_id",
        "collection_type",
        "alloc_collection_id",
        "alloc_instance_index",
        "maximum_usage.cpus",
        "maximum_usage.memory",
        "FROM UNNEST(cpu_usage_distribution)",
        "AS cpu_usage_distribution",
        "SAFE_CAST(assigned_memory AS FLOAT64) AS assigned_memory",
    ):
        assert expression in query
    assert "COALESCE" not in query.upper()


def test_selected_candidate_sampling_contract_is_unchanged() -> None:
    normalized = _normalized_sql(extractor.build_usage_query("a"))

    assert (
        "FARM_FINGERPRINT( CONCAT(CAST(collection_id AS STRING), ':', "
        "CAST(instance_index AS STRING), ':', CAST(@seed AS STRING)) ) AS sample_hash"
    ) in normalized
    assert "COUNT(*) AS usage_rows" in normalized
    assert "HAVING usage_rows >= 12" in normalized
    assert "ORDER BY sample_hash, collection_id, instance_index" in normalized
    assert "LIMIT @max_instances" in normalized
    assert "alloc_collection_id IS NULL OR alloc_collection_id = 0" in normalized


def test_event_queries_keep_pre_day_history_and_raw_schemas() -> None:
    instance_query = extractor.build_events_query("a")
    collection_query = extractor.build_collection_events_query("a")

    assert "WHERE e.time < @day_end_us" in instance_query
    assert "e.time >= @day_start_us" not in instance_query
    normalized_instance = _normalized_sql(instance_query)
    assert (
        "ORDER BY e.collection_id, e.instance_index, e.time, e.type, "
        "e.machine_id, e.resource_request.cpus, e.resource_request.memory, "
        "e.priority, e.scheduling_class, e.missing_type, e.collection_type, "
        "e.alloc_collection_id, e.alloc_instance_index"
    ) in normalized_instance
    for column in (
        "e.missing_type",
        "e.collection_type",
        "e.alloc_collection_id",
        "e.alloc_instance_index",
        "e.scheduling_class",
        "e.machine_id",
        "AS resource_request_cpu",
        "AS resource_request_mem",
    ):
        assert column in instance_query

    assert "WHERE e.time < @day_end_us" in collection_query
    assert "e.time >= @day_start_us" not in collection_query
    normalized_collection = _normalized_sql(collection_query)
    assert (
        "ORDER BY e.collection_id, e.time, e.type, e.scheduler, e.priority, "
        "e.scheduling_class, e.missing_type, e.collection_type, "
        "e.max_per_machine, e.max_per_switch"
    ) in normalized_collection
    assert "WHEN 0 THEN 'SCHEDULER_DEFAULT'" in collection_query
    assert "WHEN 1 THEN 'SCHEDULER_BATCH'" in collection_query
    assert "e.max_per_machine" in collection_query
    assert "e.max_per_switch" in collection_query


def test_machine_event_query_is_unfiltered_by_selected_vms() -> None:
    query = extractor.build_machine_events_query("a")

    assert "clusterdata_2019_a.machine_events" in query
    assert "SAFE_CAST(e.capacity.cpus AS FLOAT64) AS capacity_cpu" in query
    assert "SAFE_CAST(e.capacity.memory AS FLOAT64) AS capacity_mem" in query
    assert "e.platform_id" in query
    assert "e.switch_id" in query
    assert "e.missing_data_reason" in query
    assert "WHERE e.time < @day_end_us" in query
    assert "@day_start_us" not in query
    assert "JOIN candidates" not in query
    assert "instance_usage" not in query


def test_output_validation_rejects_bad_intervals_candidate_overflow_and_empty_machines() -> None:
    valid = _valid_usage_frame()
    extractor._validate_query_output("usage_5min", valid, max_instances=1)

    bad_overlap = valid.copy()
    bad_overlap.loc[0, "overlap_us"] = 0
    with pytest.raises(ValueError, match="overlap_us <= 0"):
        extractor._validate_query_output("usage_5min", bad_overlap, max_instances=1)

    bad_clip = valid.copy()
    bad_clip.loc[0, "clipped_start_time"] = bad_clip.loc[0, "clipped_end_time"]
    with pytest.raises(ValueError, match="clipped_start_time >= clipped_end_time"):
        extractor._validate_query_output("usage_5min", bad_clip, max_instances=1)

    too_many = pd.concat(
        [valid, valid.assign(collection_id=2, instance_index=1)],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="more selected candidates"):
        extractor._validate_query_output("usage_5min", too_many, max_instances=1)

    empty_machines = pd.DataFrame(columns=extractor.MACHINE_EVENT_OUTPUT_COLUMNS)
    with pytest.raises(ValueError, match="machine capacity state cannot be reconstructed"):
        extractor._validate_query_output("machine_events", empty_machines, max_instances=1)


def test_parquet_writer_preserves_null_resource_values(tmp_path) -> None:
    frame = _valid_usage_frame()

    class FakeBigQuery:
        class QueryJobConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

    class FakeClient:
        def query(self, query, job_config):
            return SimpleNamespace(to_dataframe=lambda: frame.copy())

    path = tmp_path / "usage_5min.parquet"
    rows = extractor._run_to_parquet(
        FakeClient(),
        FakeBigQuery,
        "usage_5min",
        "SELECT 1",
        [],
        path,
        max_instances=1,
    )

    written = pd.read_parquet(path)
    assert rows == 1
    assert pd.isna(written.loc[0, "cpu_usage"])
    assert np.asarray(written.loc[0, "cpu_usage_distribution"]).tolist() == [
        0.05,
        0.1,
        0.2,
    ]
    assert pd.isna(written.loc[0, "max_cpu_usage"])
    assert pd.isna(written.loc[0, "assigned_memory"])


def test_extraction_registers_all_outputs_sql_audit_and_metadata(monkeypatch, tmp_path) -> None:
    from google.cloud import bigquery

    monkeypatch.setattr(bigquery, "Client", lambda project: object())
    estimated = {
        "usage_5min": 10,
        "instance_events": 20,
        "collection_events": 30,
        "machine_events": 40,
    }
    row_counts = {
        "usage_5min": 100,
        "instance_events": 200,
        "collection_events": 300,
        "machine_events": 400,
    }
    seen_paths = {}

    def fake_dry_run(client, bigquery_module, label, query, query_parameters):
        return estimated[label]

    def fake_run_to_parquet(
        client,
        bigquery_module,
        label,
        query,
        query_parameters,
        path,
        max_instances,
    ):
        seen_paths[label] = path
        return row_counts[label]

    monkeypatch.setattr(extractor, "_dry_run", fake_dry_run)
    monkeypatch.setattr(extractor, "_run_to_parquet", fake_run_to_parquet)

    summary = extractor.extract_google2019(
        project_id="billing-project",
        cell="a",
        day_index=0,
        max_instances=10_000,
        seed=42,
        output_dir=tmp_path,
    )

    expected_labels = {
        "usage_5min",
        "instance_events",
        "collection_events",
        "machine_events",
    }
    assert set(seen_paths) == expected_labels
    assert {path.name for path in seen_paths.values()} == {
        f"{label}.parquet" for label in expected_labels
    }
    assert summary["estimated_total_bytes_processed"] == sum(estimated.values())
    assert summary["machine_event_rows"] == 400

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["trace_start_offset_us"] == 600_000_000
    assert metadata["day_start_us"] == 600_000_000
    assert metadata["day_end_us"] == 87_000_000_000
    assert metadata["resource_unit"] == extractor.RESOURCE_UNIT
    assert metadata["usage_interval_policy"] == extractor.USAGE_INTERVAL_POLICY
    assert metadata["sampling_policy"]["minimum_usage_rows"] == 12
    assert metadata["estimated_bytes_processed"] == estimated
    assert metadata["row_counts"] == row_counts
    assert set(metadata["tables"]) == {
        "instance_usage",
        "instance_events",
        "collection_events",
        "machine_events",
    }
    assert set(metadata["sql_files"]) == expected_labels
    for label in expected_labels:
        assert (tmp_path / "sql" / f"{label}.sql").is_file()


def test_dry_run_cost_includes_machine_events(monkeypatch, tmp_path) -> None:
    from google.cloud import bigquery

    monkeypatch.setattr(bigquery, "Client", lambda project: object())
    costs = {
        "usage_5min": 1,
        "instance_events": 2,
        "collection_events": 3,
        "machine_events": 4,
    }
    labels = []

    def fake_dry_run(client, bigquery_module, label, query, query_parameters):
        labels.append(label)
        return costs[label]

    monkeypatch.setattr(extractor, "_dry_run", fake_dry_run)
    summary = extractor.extract_google2019(
        project_id="billing-project",
        output_dir=tmp_path,
        dry_run_only=True,
    )

    assert set(labels) == set(costs)
    assert summary["estimated_total_bytes_processed"] == 10
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["estimated_bytes_processed"] == costs
    assert metadata["row_counts"] == {}
