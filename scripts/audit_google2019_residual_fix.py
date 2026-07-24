#!/usr/bin/env python3
"""Audit the Google 2019 residual preprocessing fix before/after outputs.

This script is deliberately read-only.  It compares a previously generated
processed directory with a newly generated one, and emits a machine-readable
JSON report covering the diagnostics that matter for the residual-fix review:

* candidate/class counts and unresolved event-state ambiguity;
* interval-overlap reconstruction diagnostics;
* coverage_us validity and partial-window prevalence;
* source maximum_usage CPU flags and converted q_cpu outliers;
* representative-machine unit conversion and server capacities;
* baseline OD/Spot/Batch selections compared by the stable raw VM key; and
* the SHA256 digest of the unchanged mathematical model source.

By default the selection audit invokes the current server-min ``build_instance``
for both directories.  A previously captured selection JSON can be supplied for
either side to avoid rebuilding it.  Selection snapshots may contain either VM
IDs or records with ``collection_id`` and ``instance_index`` under the keys
``on_demand``, ``spot``, and ``batch_jobs``.

Run from any directory with the repository virtual environment, for example::

    .venv/bin/python scripts/audit_google2019_residual_fix.py \
      --after-dir data/processed/notion_toy_google2019_v3_deterministic_union_coverage \
      --before-selection-json /tmp/google2019_pre_fix_baseline_selection.json \
      --output review_bundles/google2019_residual_fix_audit.json
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


T5_US = 300_000_000
DEFAULT_BEFORE_DIR = Path(
    "data/processed/notion_toy_google2019_v2_resource_scaled_validation"
)
DEFAULT_BASELINE_CONFIG = Path(
    "experiments/2607-notion-server-min-vmp/configs/baseline.yaml"
)
DEFAULT_MODEL_PATH = Path(
    "experiments/2607-notion-server-min-vmp/src/notion_server_min_vmp/model.py"
)
EXPECTED_OUTPUTS = (
    "vm_requests.csv",
    "vm_usage_5min_scenarios.csv",
    "vm_usage_hourly_scenarios.csv",
    "servers.csv",
    "batch_families.csv",
    "metadata.json",
    "scaling_diagnostics.json",
    "preprocessing_diagnostics.json",
)
SELECTION_CLASSES = ("on_demand", "spot", "batch_jobs")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve(path: str | Path, root: Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def _load_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return {}
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_scalar(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            return None
        return int(number) if number.is_integer() else number
    return str(value)


def _key_component(value: Any) -> str:
    normalized = _json_scalar(value)
    return "<null>" if normalized is None else str(normalized)


def _stable_key(collection_id: Any, instance_index: Any) -> str:
    return f"{_key_component(collection_id)}:{_key_component(instance_index)}"


def _boolean_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False).astype(bool)
    normalized = series.astype("string").str.strip().str.lower()
    unknown = normalized.notna() & ~normalized.isin(
        ["true", "false", "1", "0", "yes", "no", "y", "n", ""]
    )
    if bool(unknown.any()):
        examples = sorted(normalized.loc[unknown].dropna().unique().tolist())[:5]
        raise ValueError(f"Cannot interpret boolean values {examples}")
    return normalized.isin(["true", "1", "yes", "y"])


def _numeric_summary(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {
            "count": 0,
            "min": None,
            "p50": None,
            "mean": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "p50": float(np.quantile(values, 0.50)),
        "mean": float(values.mean()),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def _read_requests(processed_dir: Path) -> pd.DataFrame:
    path = processed_dir / "vm_requests.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    requests = pd.read_csv(path, low_memory=False, dtype={"vm_id": str})
    required = {"vm_id", "collection_id", "instance_index", "class", "q_cpu"}
    missing = required - set(requests.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    if requests["vm_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate vm_id values")
    requests["vm_id"] = requests["vm_id"].astype(str)
    requests["stable_key"] = [
        _stable_key(collection_id, instance_index)
        for collection_id, instance_index in zip(
            requests["collection_id"], requests["instance_index"]
        )
    ]
    if requests["stable_key"].duplicated().any():
        raise ValueError(f"{path} contains duplicate raw VM keys")
    return requests


def _class_counts(requests: pd.DataFrame) -> dict[str, Any]:
    counts = requests["class"].astype(str).value_counts().sort_index()
    return {
        "total": int(len(requests)),
        "by_class": {str(key): int(value) for key, value in counts.items()},
    }


def _diagnostics_documents(processed_dir: Path) -> list[dict[str, Any]]:
    return [
        _load_json(processed_dir / name, required=False)
        for name in (
            "preprocessing_diagnostics.json",
            "scaling_diagnostics.json",
            "metadata.json",
        )
    ]


def _first_diagnostic(
    documents: Iterable[Mapping[str, Any]], key: str
) -> dict[str, Any]:
    for document in documents:
        value = document.get(key)
        if isinstance(value, dict) and value:
            return copy.deepcopy(value)
    return {}


def _event_ambiguity_audit(
    requests: pd.DataFrame, documents: list[dict[str, Any]]
) -> dict[str, Any]:
    diagnostic = _first_diagnostic(documents, "event_state_ambiguity_audit")
    output_flags: dict[str, Any] = {}
    for prefix in ("arrival_state", "scheduler_state"):
        flag = f"{prefix}_ambiguous"
        reason = f"{prefix}_ambiguity_reason"
        candidate_count = f"{prefix}_candidate_count"
        if flag not in requests.columns:
            output_flags[prefix] = {"column_available": False}
            continue
        mask = _boolean_series(requests[flag])
        entry: dict[str, Any] = {
            "column_available": True,
            "flagged_output_vm_count": int(mask.sum()),
            "flagged_output_vm_count_by_class": {
                str(key): int(value)
                for key, value in requests.loc[mask, "class"]
                .astype(str)
                .value_counts()
                .sort_index()
                .items()
            },
        }
        if reason in requests.columns:
            entry["reason_counts"] = {
                str(key): int(value)
                for key, value in requests.loc[mask, reason]
                .fillna("")
                .astype(str)
                .value_counts()
                .sort_index()
                .items()
            }
        if candidate_count in requests.columns:
            entry["candidate_count_summary"] = _numeric_summary(
                requests.loc[mask, candidate_count]
            )
        output_flags[prefix] = entry
    return {
        "diagnostic_available": bool(diagnostic),
        "pre_sampling_diagnostic": diagnostic,
        "emitted_vm_flags": output_flags,
    }


def _overlap_audit(documents: list[dict[str, Any]]) -> dict[str, Any]:
    return _first_diagnostic(documents, "usage_interval_overlap_audit")


def _outlier_audit(
    requests: pd.DataFrame, documents: list[dict[str, Any]]
) -> dict[str, Any]:
    diagnostic = _first_diagnostic(documents, "maximum_usage_cpu_outlier_audit")
    q_cpu = pd.to_numeric(requests["q_cpu"], errors="coerce")
    q_mask = q_cpu.gt(1.0)
    result: dict[str, Any] = {
        "policy": "audit_only_no_cap_removal_replacement_or_rescaling",
        "diagnostic_available": bool(diagnostic),
        "preprocessing_diagnostic": diagnostic,
        "converted_q_cpu_gt_one": {
            "count": int(q_mask.sum()),
            "count_by_class": {
                str(key): int(value)
                for key, value in requests.loc[q_mask, "class"]
                .astype(str)
                .value_counts()
                .sort_index()
                .items()
            },
            "summary": _numeric_summary(q_cpu.loc[q_mask]),
        },
    }
    flag = "maximum_usage_cpu_outlier"
    if flag not in requests.columns:
        result["source_maximum_usage_cpu_gt_one"] = {"column_available": False}
        return result
    source_mask = _boolean_series(requests[flag])
    source_entry: dict[str, Any] = {
        "column_available": True,
        "count": int(source_mask.sum()),
        "count_by_class": {
            str(key): int(value)
            for key, value in requests.loc[source_mask, "class"]
            .astype(str)
            .value_counts()
            .sort_index()
            .items()
        },
        "stable_keys": requests.loc[source_mask, "stable_key"].astype(str).tolist(),
    }
    if "max_cpu_usage" in requests.columns:
        source_entry["converted_max_cpu_usage_summary"] = _numeric_summary(
            requests.loc[source_mask, "max_cpu_usage"]
        )
    result["source_maximum_usage_cpu_gt_one"] = source_entry
    return result


def _histogram_quantile(histogram: Counter[int], quantile: float) -> int | None:
    total = sum(histogram.values())
    if total <= 0:
        return None
    rank = quantile * (total - 1)
    cumulative = 0
    for value, count in sorted(histogram.items()):
        cumulative += count
        if cumulative > rank:
            return int(value)
    return int(max(histogram))


def _coverage_audit(processed_dir: Path, chunksize: int) -> dict[str, Any]:
    path = processed_dir / "vm_usage_5min_scenarios.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    available = set(pd.read_csv(path, nrows=0).columns)
    required = {"scenario_id", "vm_id", "t5_day", "coverage_us"}
    missing = required - available
    if missing:
        return {
            "columns_available": False,
            "missing_columns": sorted(missing),
        }

    usecols = sorted(required | ({"coverage_ratio"} & available))
    histogram: Counter[int] = Counter()
    scenario_zero_rows = 0
    full_rows = 0
    partial_rows = 0
    invalid_nonfinite = 0
    invalid_noninteger = 0
    invalid_nonpositive = 0
    invalid_over_t5 = 0
    ratio_mismatch_rows = 0
    adjacent_duplicate_key_rows = 0
    key_order_violations = 0
    previous_key: tuple[str, int] | None = None
    t5_min: int | None = None
    t5_max: int | None = None
    overlap_conflict_rows = 0
    duplicated_timeline_us_sum = 0

    optional_audit = {
        "overlap_conflict_flag",
        "duplicated_timeline_us",
    } & available
    usecols = sorted(set(usecols) | optional_audit)

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, dtype={"vm_id": str}):
        scenario = pd.to_numeric(chunk["scenario_id"], errors="coerce")
        zero = chunk.loc[scenario.eq(0)].copy()
        if zero.empty:
            continue
        scenario_zero_rows += int(len(zero))

        coverage = pd.to_numeric(zero["coverage_us"], errors="coerce").to_numpy(float)
        finite = np.isfinite(coverage)
        integer = finite & np.equal(coverage, np.floor(coverage))
        positive = integer & (coverage > 0)
        within = positive & (coverage <= T5_US)
        invalid_nonfinite += int((~finite).sum())
        invalid_noninteger += int((finite & ~integer).sum())
        invalid_nonpositive += int((integer & (coverage <= 0)).sum())
        invalid_over_t5 += int((positive & (coverage > T5_US)).sum())
        valid_values = coverage[within].astype(np.int64)
        histogram.update(int(value) for value in valid_values)
        full_rows += int((valid_values == T5_US).sum())
        partial_rows += int((valid_values < T5_US).sum())

        if "coverage_ratio" in zero.columns:
            ratio = pd.to_numeric(zero["coverage_ratio"], errors="coerce").to_numpy(float)
            expected = coverage / T5_US
            mismatch = ~np.isfinite(ratio) | ~np.isclose(
                ratio, expected, atol=1e-12, rtol=0.0
            )
            ratio_mismatch_rows += int(mismatch.sum())

        t5_values = pd.to_numeric(zero["t5_day"], errors="coerce")
        finite_t5 = t5_values[np.isfinite(t5_values.to_numpy(float))].astype(int)
        if not finite_t5.empty:
            local_min = int(finite_t5.min())
            local_max = int(finite_t5.max())
            t5_min = local_min if t5_min is None else min(t5_min, local_min)
            t5_max = local_max if t5_max is None else max(t5_max, local_max)

        # Canonical output is sorted by (scenario_id, vm_id, t5_day), so a
        # repeated canonical key must be adjacent.  We also count ordering
        # violations explicitly rather than silently assuming the invariant.
        for vm_id, t5_value in zip(zero["vm_id"].astype(str), t5_values):
            if not math.isfinite(float(t5_value)):
                continue
            key = (vm_id, int(t5_value))
            if previous_key is not None:
                if key == previous_key:
                    adjacent_duplicate_key_rows += 1
                elif key < previous_key:
                    key_order_violations += 1
            previous_key = key

        if "overlap_conflict_flag" in zero.columns:
            overlap_conflict_rows += int(_boolean_series(zero["overlap_conflict_flag"]).sum())
        if "duplicated_timeline_us" in zero.columns:
            duplicated = pd.to_numeric(
                zero["duplicated_timeline_us"], errors="coerce"
            ).to_numpy(float)
            duplicated_timeline_us_sum += int(np.nansum(duplicated))

    valid_rows = int(sum(histogram.values()))
    coverage_sum = int(sum(value * count for value, count in histogram.items()))
    return {
        "columns_available": True,
        "source_of_truth": "coverage_us",
        "scenario_zero_rows": scenario_zero_rows,
        "valid_coverage_rows": valid_rows,
        "invalid_coverage_rows": scenario_zero_rows - valid_rows,
        "invalid_breakdown": {
            "nonfinite": invalid_nonfinite,
            "noninteger": invalid_noninteger,
            "nonpositive": invalid_nonpositive,
            "greater_than_300_seconds": invalid_over_t5,
        },
        "coverage_us": {
            "min": int(min(histogram)) if histogram else None,
            "p50": _histogram_quantile(histogram, 0.50),
            "mean": float(coverage_sum / valid_rows) if valid_rows else None,
            "p95": _histogram_quantile(histogram, 0.95),
            "max": int(max(histogram)) if histogram else None,
            "sum": coverage_sum,
        },
        "full_window_rows": full_rows,
        "partial_window_rows": partial_rows,
        "partial_window_fraction": float(partial_rows / valid_rows) if valid_rows else None,
        "observed_30min_slot_equivalents": float(coverage_sum / (6 * T5_US)),
        "coverage_ratio_mismatch_rows": ratio_mismatch_rows
        if "coverage_ratio" in available
        else None,
        "t5_day_min": t5_min,
        "t5_day_max": t5_max,
        "adjacent_duplicate_vm_t5_rows": adjacent_duplicate_key_rows,
        "canonical_key_order_violations": key_order_violations,
        "overlap_conflict_bucket_rows": overlap_conflict_rows
        if "overlap_conflict_flag" in available
        else None,
        "duplicated_timeline_us_sum_from_bucket_audit": duplicated_timeline_us_sum
        if "duplicated_timeline_us" in available
        else None,
    }


def _capacity_audit(
    processed_dir: Path, documents: list[dict[str, Any]]
) -> dict[str, Any]:
    metadata = _load_json(processed_dir / "metadata.json", required=False)
    scaling = _load_json(processed_dir / "scaling_diagnostics.json", required=False)
    representative = metadata.get("representative_machine_capacity_raw")
    if not isinstance(representative, dict):
        representative = scaling.get("representative_machine_capacity_raw", {})
    resource_scale = metadata.get("resource_scale", {})
    automatic = metadata.get(
        "automatic_utilization_calibration_applied",
        scaling.get("automatic_utilization_calibration_applied"),
    )
    servers_path = processed_dir / "servers.csv"
    server_summary: dict[str, Any] = {"file_available": servers_path.is_file()}
    if servers_path.is_file():
        servers = pd.read_csv(servers_path)
        for resource in ("C_cpu", "C_mem"):
            if resource in servers.columns:
                values = pd.to_numeric(servers[resource], errors="coerce")
                finite = values[np.isfinite(values.to_numpy(float))]
                server_summary[resource] = {
                    "unique": sorted(float(value) for value in finite.unique()),
                    "all_exactly_one": bool(len(finite) == len(values) and finite.eq(1.0).all()),
                }
            else:
                server_summary[resource] = {"column_available": False}
        server_summary["row_count"] = int(len(servers))
    return {
        "representative_machine_capacity_raw": representative,
        "representative_machine_selection_method": metadata.get(
            "representative_machine_selection_method",
            scaling.get("representative_machine_selection_method"),
        ),
        "unit_conversion": metadata.get(
            "unit_conversion", scaling.get("unit_conversion_factors", {})
        ),
        "resource_scale": resource_scale,
        "automatic_utilization_calibration_applied": automatic,
        "servers": server_summary,
    }


def _manifest(processed_dir: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for name in EXPECTED_OUTPUTS:
        path = processed_dir / name
        files[name] = {
            "exists": path.is_file(),
            "size_bytes": int(path.stat().st_size) if path.is_file() else None,
        }
    return {
        "processed_dir": str(processed_dir),
        "expected_outputs": files,
        "missing": [name for name, value in files.items() if not value["exists"]],
    }


def _selection_entries_from_ids(
    vm_ids: Iterable[Any], requests: pd.DataFrame
) -> list[dict[str, Any]]:
    indexed = requests.set_index("vm_id", drop=False)
    result: list[dict[str, Any]] = []
    for raw_vm_id in vm_ids:
        vm_id = str(raw_vm_id)
        if vm_id not in indexed.index:
            raise ValueError(f"Selection references unknown vm_id {vm_id}")
        row = indexed.loc[vm_id]
        result.append(
            {
                "vm_id": vm_id,
                "collection_id": _json_scalar(row["collection_id"]),
                "instance_index": _json_scalar(row["instance_index"]),
                "class": str(row["class"]),
                "stable_key": str(row["stable_key"]),
            }
        )
    return result


def _selection_entries_from_records(
    records: Iterable[Any], requests: pd.DataFrame
) -> list[dict[str, Any]]:
    records = list(records)
    if not records:
        return []
    if all(not isinstance(record, Mapping) for record in records):
        return _selection_entries_from_ids(records, requests)

    by_vm = requests.set_index("vm_id", drop=False)
    by_key = requests.set_index("stable_key", drop=False)
    result: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("Selection JSON mixes record and VM-ID entries")
        if "collection_id" in record and "instance_index" in record:
            key = _stable_key(record["collection_id"], record["instance_index"])
            if key not in by_key.index:
                raise ValueError(f"Selection references unknown stable key {key}")
            row = by_key.loc[key]
        elif "vm_id" in record:
            vm_id = str(record["vm_id"])
            if vm_id not in by_vm.index:
                raise ValueError(f"Selection references unknown vm_id {vm_id}")
            row = by_vm.loc[vm_id]
        else:
            raise ValueError(
                "Selection record requires collection_id/instance_index or vm_id"
            )
        result.append(
            {
                "vm_id": str(row["vm_id"]),
                "collection_id": _json_scalar(row["collection_id"]),
                "instance_index": _json_scalar(row["instance_index"]),
                "class": str(row["class"]),
                "stable_key": str(row["stable_key"]),
            }
        )
    return result


def _selection_from_json(path: Path, requests: pd.DataFrame) -> dict[str, Any]:
    snapshot = _load_json(path)
    selected: dict[str, list[dict[str, Any]]] = {}
    for class_name in SELECTION_CLASSES:
        records = snapshot.get(class_name, [])
        if not isinstance(records, list):
            raise ValueError(f"{path}: {class_name} must be a list")
        selected[class_name] = _selection_entries_from_records(records, requests)
    return {
        "source": "selection_json",
        "source_path": str(path),
        "selected": selected,
        "sets": snapshot.get("sets", {}),
    }


def _selection_from_build_instance(
    processed_dir: Path,
    requests: pd.DataFrame,
    baseline_config: Path,
    repo_root: Path,
) -> dict[str, Any]:
    experiment_src = (
        repo_root
        / "experiments/2607-notion-server-min-vmp/src"
    )
    if str(experiment_src) not in sys.path:
        sys.path.insert(0, str(experiment_src))
    from notion_server_min_vmp.data import build_instance, load_config  # noqa: PLC0415

    config = load_config(baseline_config)
    config = copy.deepcopy(config)
    config["data"]["google_dir"] = str(processed_dir)
    instance = build_instance(config, config_path=baseline_config)
    sampling = instance.metadata.get("sampling", {})
    selected_ids = sampling.get("selected_ids", {})
    class_ids = {
        "on_demand": list(instance.I),
        "spot": list(instance.J),
        "batch_jobs": list(selected_ids.get("batch_jobs", [])),
    }
    selected = {
        class_name: _selection_entries_from_ids(vm_ids, requests)
        for class_name, vm_ids in class_ids.items()
    }
    return {
        "source": "server_min_build_instance",
        "baseline_config": str(baseline_config),
        "selected": selected,
        "sets": {
            "I": len(instance.I),
            "J": len(instance.J),
            "K": len(instance.K),
            "S": len(instance.S),
            "T": len(instance.T),
            "Xi": len(instance.Xi),
        },
        "sampling_metadata": sampling,
        "coverage_loader_audit": instance.metadata.get("workload_scenarios", {}).get(
            "scenario_zero_usage_loader", {}
        ),
        "batch_coverage_totals": {
            key: instance.metadata.get("batch", {}).get(key)
            for key in (
                "total_workload_slot_units",
                "total_cpu_resource_slot_volume",
                "total_memory_resource_slot_volume",
                "workload_definition",
            )
        },
    }


def _selection_audit(
    processed_dir: Path,
    requests: pd.DataFrame,
    baseline_config: Path,
    selection_json: Path | None,
    no_build_instance: bool,
    repo_root: Path,
) -> dict[str, Any]:
    if selection_json is not None:
        return _selection_from_json(selection_json, requests)
    if no_build_instance:
        return {
            "source": "not_computed",
            "reason": "--no-build-instance was supplied without a selection JSON",
            "selected": {},
        }
    return _selection_from_build_instance(
        processed_dir, requests, baseline_config, repo_root
    )


def _selection_comparison(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    before_selected = before.get("selected", {})
    after_selected = after.get("selected", {})
    for class_name in SELECTION_CLASSES:
        before_rows = list(before_selected.get(class_name, []))
        after_rows = list(after_selected.get(class_name, []))
        before_by_key = {str(row["stable_key"]): row for row in before_rows}
        after_by_key = {str(row["stable_key"]): row for row in after_rows}
        before_order = list(before_by_key)
        after_order = list(after_by_key)
        removed_keys = [key for key in before_order if key not in after_by_key]
        added_keys = [key for key in after_order if key not in before_by_key]
        shared_before = [key for key in before_order if key in after_by_key]
        shared_after = [key for key in after_order if key in before_by_key]
        result[class_name] = {
            "before_count": len(before_rows),
            "after_count": len(after_rows),
            "stable_key_set_unchanged": not removed_keys and not added_keys,
            "shared_stable_key_order_unchanged": shared_before == shared_after,
            "removed": [before_by_key[key] for key in removed_keys],
            "added": [after_by_key[key] for key in added_keys],
        }
    return result


def _selected_outlier_audit(
    requests: pd.DataFrame, selection: Mapping[str, Any]
) -> dict[str, Any]:
    indexed = requests.set_index("stable_key", drop=False)
    flag_available = "maximum_usage_cpu_outlier" in requests.columns
    result: dict[str, Any] = {}
    for class_name in SELECTION_CLASSES:
        rows = selection.get("selected", {}).get(class_name, [])
        keys = [str(row["stable_key"]) for row in rows]
        if not keys:
            result[class_name] = {
                "selected_count": 0,
                "converted_q_cpu_gt_one_count": 0,
                "source_maximum_usage_cpu_gt_one_count": 0 if flag_available else None,
                "outliers": [],
            }
            continue
        missing = [key for key in keys if key not in indexed.index]
        if missing:
            raise ValueError(f"Selection contains unknown stable keys {missing[:5]}")
        selected = indexed.loc[keys].copy()
        q_mask = pd.to_numeric(selected["q_cpu"], errors="coerce").gt(1.0)
        source_mask = (
            _boolean_series(selected["maximum_usage_cpu_outlier"])
            if flag_available
            else pd.Series(False, index=selected.index)
        )
        union_mask = q_mask | source_mask
        entries = []
        for row_index, row in selected.loc[union_mask].iterrows():
            entries.append(
                {
                    "stable_key": str(row_index),
                    "vm_id": str(row["vm_id"]),
                    "converted_q_cpu": float(row["q_cpu"]),
                    "source_maximum_usage_cpu_gt_one": bool(source_mask.loc[row_index])
                    if flag_available
                    else None,
                }
            )
        result[class_name] = {
            "selected_count": len(keys),
            "converted_q_cpu_gt_one_count": int(q_mask.sum()),
            "source_maximum_usage_cpu_gt_one_count": int(source_mask.sum())
            if flag_available
            else None,
            "outliers": entries,
        }
    return result


def _directory_audit(
    processed_dir: Path,
    baseline_config: Path,
    selection_json: Path | None,
    no_build_instance: bool,
    chunksize: int,
    repo_root: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    requests = _read_requests(processed_dir)
    documents = _diagnostics_documents(processed_dir)
    selection = _selection_audit(
        processed_dir,
        requests,
        baseline_config,
        selection_json,
        no_build_instance,
        repo_root,
    )
    outliers = _outlier_audit(requests, documents)
    outliers["baseline_selected"] = _selected_outlier_audit(requests, selection)
    return (
        {
            "manifest": _manifest(processed_dir),
            "class_counts": _class_counts(requests),
            "event_state_ambiguity": _event_ambiguity_audit(requests, documents),
            "usage_interval_overlap": _overlap_audit(documents),
            "maximum_usage_cpu_outliers": outliers,
            "coverage": _coverage_audit(processed_dir, chunksize),
            "capacity_and_scaling": _capacity_audit(processed_dir, documents),
            "baseline_selection": selection,
        },
        requests,
    )


def _class_count_delta(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, int]:
    before_counts = before["class_counts"]["by_class"]
    after_counts = after["class_counts"]["by_class"]
    return {
        class_name: int(after_counts.get(class_name, 0) - before_counts.get(class_name, 0))
        for class_name in sorted(set(before_counts) | set(after_counts))
    }


def _representative_pair_equal(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> bool:
    first = before["capacity_and_scaling"].get("representative_machine_capacity_raw", {})
    second = after["capacity_and_scaling"].get("representative_machine_capacity_raw", {})
    try:
        return bool(
            math.isclose(float(first["cpu"]), float(second["cpu"]), abs_tol=0.0, rel_tol=0.0)
            and math.isclose(
                float(first["mem"]), float(second["mem"]), abs_tol=0.0, rel_tol=0.0
            )
        )
    except (KeyError, TypeError, ValueError):
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-dir", type=Path, default=DEFAULT_BEFORE_DIR)
    parser.add_argument("--after-dir", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path, default=DEFAULT_BASELINE_CONFIG)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--before-selection-json", type=Path)
    parser.add_argument("--after-selection-json", type=Path)
    parser.add_argument(
        "--no-build-instance",
        action="store_true",
        help="Do not call build_instance when a selection snapshot is absent.",
    )
    parser.add_argument("--chunksize", type=int, default=400_000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.chunksize <= 0:
        raise ValueError("--chunksize must be positive")
    repo_root = _repo_root()
    before_dir = _resolve(args.before_dir, repo_root)
    after_dir = _resolve(args.after_dir, repo_root)
    baseline_config = _resolve(args.baseline_config, repo_root)
    model_path = _resolve(args.model_path, repo_root)
    before_selection = (
        _resolve(args.before_selection_json, repo_root)
        if args.before_selection_json is not None
        else None
    )
    after_selection = (
        _resolve(args.after_selection_json, repo_root)
        if args.after_selection_json is not None
        else None
    )
    if not before_dir.is_dir():
        raise FileNotFoundError(before_dir)
    if not after_dir.is_dir():
        raise FileNotFoundError(after_dir)
    if not baseline_config.is_file():
        raise FileNotFoundError(baseline_config)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    before, _ = _directory_audit(
        before_dir,
        baseline_config,
        before_selection,
        args.no_build_instance,
        args.chunksize,
        repo_root,
    )
    after, _ = _directory_audit(
        after_dir,
        baseline_config,
        after_selection,
        args.no_build_instance,
        args.chunksize,
        repo_root,
    )
    report = {
        "schema_version": 1,
        "audit_semantics": {
            "read_only": True,
            "selection_identity": "collection_id:instance_index",
            "coverage_source_of_truth": "coverage_us",
            "maximum_usage_cpu_outlier_policy": (
                "audit only; no cap, removal, replacement, or rescaling"
            ),
        },
        "model_source": {
            "path": str(model_path),
            "sha256": _sha256(model_path),
        },
        "before": before,
        "after": after,
        "comparison": {
            "class_count_delta_after_minus_before": _class_count_delta(before, after),
            "representative_machine_capacity_unchanged": _representative_pair_equal(
                before, after
            ),
            "baseline_selection_by_stable_key": _selection_comparison(
                before["baseline_selection"], after["baseline_selection"]
            ),
        },
    }
    serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output is not None:
        output = _resolve(args.output, repo_root)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
    sys.stdout.write(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
