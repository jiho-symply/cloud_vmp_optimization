#!/usr/bin/env python3
"""Build an unscaled CPU-heavy OD20 fixture with frozen SP10/BJ10 IDs.

On-demand candidates are ranked by coverage-weighted scenario-0 mean CPU
usage after applying the same active-slot coverage eligibility rule as the
existing unscaled micro-data builder.  The top 20 are selected.

Spot and batch-candidate IDs, including their within-class order, are copied
from the existing ``micro_top10_unscaled_v1`` reference fixture.  All request
fields and all stored scenario-0 usage fields are then copied from the
canonical v5 source.  No q, usage, memory, lifecycle, or provenance value is
scaled or otherwise transformed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent
BASE_BUILDER_PATH = SCRIPT_DIR / "build_micro_top10_unscaled_dataset.py"
DEFAULT_SOURCE_DIR = Path(
    "data/processed/notion_toy_google2019_v5_mode_capacity_bounded"
)
DEFAULT_REFERENCE_DIR = Path(
    "data/processed/notion_toy_google2019_micro_top10_unscaled_v1"
)
DEFAULT_OUTPUT_DIR = Path(
    "data/processed/"
    "notion_toy_google2019_micro_cpu_top20_od_sp10_bj10_unscaled_v1"
)
DATASET_NAME = DEFAULT_OUTPUT_DIR.name
OD_COUNT = 20
REFERENCE_COUNT = 10
ALL_CLASSES = ("on_demand", "spot", "batch_candidate")
FROZEN_CLASSES = ("spot", "batch_candidate")
CLASS_ORDER = {name: position for position, name in enumerate(ALL_CLASSES)}


def _load_base_builder() -> ModuleType:
    if not BASE_BUILDER_PATH.is_file():
        raise FileNotFoundError(
            f"Missing base data builder: {BASE_BUILDER_PATH}"
        )
    spec = importlib.util.spec_from_file_location(
        "_micro_top10_unscaled_base_builder",
        BASE_BUILDER_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load base data builder: {BASE_BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_base_builder()


def _path_label(path: Path) -> str:
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def _assert_disjoint_roots(paths: dict[str, Path]) -> None:
    items = list(paths.items())
    for position, (left_name, left) in enumerate(items):
        for right_name, right in items[position + 1 :]:
            if (
                left == right
                or left in right.parents
                or right in left.parents
            ):
                raise ValueError(
                    f"{left_name} and {right_name} must be disjoint paths "
                    "(neither equal nor an ancestor of the other): "
                    f"{left} vs {right}"
                )


def _required_paths(
    root: Path,
    names: tuple[str, ...],
    *,
    label: str,
) -> dict[str, Path]:
    paths = {name: root / name for name in names}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {label} files: {missing}")
    return paths


def _optional_path(paths: dict[str, Path], root: Path, name: str) -> None:
    path = root / name
    if path.is_file():
        paths[name] = path


def _hashes(paths: dict[str, Path]) -> dict[str, str]:
    return {name: BASE._sha256(path) for name, path in paths.items()}


def _read_requests(path: Path, *, label: str) -> pd.DataFrame:
    requests = pd.read_csv(path, dtype={"vm_id": str})
    required = {
        "vm_id",
        "class",
        "arrival_t5",
        "departure_t5",
        "q_cpu",
        "q_mem",
        "resource_request_cpu",
        "resource_request_mem",
    }
    missing = required - set(requests.columns)
    if missing:
        raise ValueError(f"{label} is missing columns {sorted(missing)}")
    if requests["vm_id"].duplicated().any():
        raise ValueError(f"{label} contains duplicate vm_id")
    requests["vm_id"] = requests["vm_id"].astype(str)
    requests["class"] = requests["class"].astype(str)
    unknown = set(requests["class"]) - set(ALL_CLASSES)
    if unknown:
        raise ValueError(f"{label} contains unknown classes {sorted(unknown)}")
    for column in ("q_cpu", "q_mem"):
        requests[column] = BASE._finite_numeric(
            requests[column], f"{label} {column}"
        )
        if bool(requests[column].le(0).any() or requests[column].gt(1).any()):
            raise ValueError(f"{label} {column} must be in (0, 1]")
    return requests


def _read_reference_ids(
    reference_requests: pd.DataFrame,
    manifest_path: Path,
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    manifest = pd.read_csv(manifest_path, dtype={"vm_id": str})
    required = {
        "vm_id",
        "class",
        "selection_rank_within_class",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(
            "Reference selection_manifest.csv is missing columns "
            f"{sorted(missing)}"
        )
    manifest["vm_id"] = manifest["vm_id"].astype(str)
    manifest["class"] = manifest["class"].astype(str)
    if manifest["vm_id"].duplicated().any():
        raise ValueError("Reference selection manifest has duplicate vm_id")

    ids_by_class: dict[str, list[str]] = {}
    for vm_class in FROZEN_CLASSES:
        request_ids = reference_requests.loc[
            reference_requests["class"].eq(vm_class), "vm_id"
        ].astype(str).tolist()
        class_manifest = manifest.loc[
            manifest["class"].eq(vm_class)
        ].copy()
        ranks = pd.to_numeric(
            class_manifest["selection_rank_within_class"],
            errors="raise",
        ).astype(int)
        class_manifest["selection_rank_within_class"] = ranks
        class_manifest = class_manifest.sort_values(
            ["selection_rank_within_class", "vm_id"],
            kind="mergesort",
        ).reset_index(drop=True)
        manifest_ids = class_manifest["vm_id"].astype(str).tolist()
        expected_ranks = list(range(1, REFERENCE_COUNT + 1))
        if class_manifest["selection_rank_within_class"].tolist() != (
            expected_ranks
        ):
            raise ValueError(
                f"Reference {vm_class} ranks must be {expected_ranks}"
            )
        if len(request_ids) != REFERENCE_COUNT:
            raise ValueError(
                f"Reference must contain exactly {REFERENCE_COUNT} "
                f"{vm_class} requests; found {len(request_ids)}"
            )
        if request_ids != manifest_ids:
            raise ValueError(
                f"Reference {vm_class} request order differs from its "
                "selection-manifest order"
            )
        ids_by_class[vm_class] = manifest_ids
    return ids_by_class, manifest


def _assert_reference_is_canonical_subset(
    canonical_requests: pd.DataFrame,
    canonical_usage: pd.DataFrame,
    reference_requests: pd.DataFrame,
    reference_usage: pd.DataFrame,
    reference_ids: dict[str, list[str]],
) -> None:
    frozen_ids = set().union(
        *(set(reference_ids[vm_class]) for vm_class in FROZEN_CLASSES)
    )
    canonical_request_subset = canonical_requests.loc[
        canonical_requests["vm_id"].isin(frozen_ids)
    ].sort_values("vm_id").reset_index(drop=True)
    reference_request_subset = reference_requests.loc[
        reference_requests["vm_id"].isin(frozen_ids)
    ].sort_values("vm_id").reset_index(drop=True)
    missing_columns = set(canonical_request_subset.columns) - set(
        reference_request_subset.columns
    )
    if missing_columns:
        raise ValueError(
            "Reference requests cannot be compared with canonical requests; "
            f"missing columns {sorted(missing_columns)}"
        )
    if len(canonical_request_subset) != len(frozen_ids):
        missing_ids = frozen_ids - set(canonical_request_subset["vm_id"])
        raise ValueError(
            f"Reference SP/BJ IDs missing from canonical requests: "
            f"{sorted(missing_ids)}"
        )
    try:
        pd.testing.assert_frame_equal(
            reference_request_subset[canonical_request_subset.columns],
            canonical_request_subset,
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as error:
        raise ValueError(
            "Reference SP/BJ request values differ from the canonical source"
        ) from error

    canonical_usage_subset = canonical_usage.loc[
        canonical_usage["vm_id"].isin(frozen_ids)
    ].sort_values(["vm_id", "t5_day"]).reset_index(drop=True)
    reference_usage_subset = reference_usage.loc[
        reference_usage["vm_id"].isin(frozen_ids)
    ].sort_values(["vm_id", "t5_day"]).reset_index(drop=True)
    missing_usage_columns = set(canonical_usage_subset.columns) - set(
        reference_usage_subset.columns
    )
    if missing_usage_columns:
        raise ValueError(
            "Reference usage cannot be compared with canonical usage; "
            f"missing columns {sorted(missing_usage_columns)}"
        )
    if set(canonical_usage_subset["vm_id"]) != frozen_ids:
        missing_ids = frozen_ids - set(canonical_usage_subset["vm_id"])
        raise ValueError(
            f"Reference SP/BJ IDs missing from canonical usage: "
            f"{sorted(missing_ids)}"
        )
    try:
        pd.testing.assert_frame_equal(
            reference_usage_subset[canonical_usage_subset.columns],
            canonical_usage_subset,
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as error:
        raise ValueError(
            "Reference SP/BJ scenario-0 usage differs from the canonical source"
        ) from error


def _select_rows(
    requests: pd.DataFrame,
    summaries: pd.DataFrame,
    canonical_usage: pd.DataFrame,
    reference_ids: dict[str, list[str]],
    reference_manifest: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, list[str]],
]:
    candidates = requests.merge(
        summaries,
        on="vm_id",
        how="inner",
        validate="one_to_one",
    )
    candidates["selection_score_avg_cpu_plus_mem"] = (
        candidates["coverage_weighted_avg_cpu_usage"]
        + candidates["coverage_weighted_avg_mem_usage"]
    )

    eligible_od_ids = BASE._eligible_service_ids(
        requests,
        canonical_usage,
        "on_demand",
    )
    od_candidates = candidates.loc[
        candidates["class"].eq("on_demand")
        & candidates["vm_id"].isin(eligible_od_ids)
    ].copy()
    od_candidates = od_candidates.sort_values(
        ["coverage_weighted_avg_cpu_usage", "vm_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    if len(od_candidates) < OD_COUNT:
        raise ValueError(
            f"Need {OD_COUNT} eligible on_demand rows, "
            f"found {len(od_candidates)}"
        )
    od_candidates["eligible_rank_within_class"] = np.arange(
        1, len(od_candidates) + 1
    )
    od_candidates["eligible_candidate_count"] = len(od_candidates)
    od = od_candidates.head(OD_COUNT).copy()
    od["selection_rank_within_class"] = np.arange(1, OD_COUNT + 1)
    od["selection_method"] = (
        "top coverage-weighted scenario-0 mean CPU among eligible on-demand"
    )
    od["selection_score_name"] = "coverage_weighted_avg_cpu_usage"
    od["selection_score_value"] = od[
        "coverage_weighted_avg_cpu_usage"
    ]
    od["eligibility_rule"] = (
        "at least one observed 5-minute row in every active 30-minute slot"
    )
    od["reference_selection_rank_within_class"] = pd.NA
    od["reference_eligible_rank_within_class"] = pd.NA
    od["reference_selection_score_avg_cpu_plus_mem"] = np.nan

    canonical_ids = set(candidates["vm_id"].astype(str))
    usage_ids = set(summaries["vm_id"].astype(str))
    eligible_spot_ids = BASE._eligible_service_ids(
        requests,
        canonical_usage,
        "spot",
    )
    selected_parts = [od]
    selected_ids: dict[str, list[str]] = {
        "on_demand": od["vm_id"].astype(str).tolist()
    }

    for vm_class in FROZEN_CLASSES:
        ids = reference_ids[vm_class]
        if not set(ids).issubset(canonical_ids):
            missing = set(ids) - canonical_ids
            raise ValueError(
                f"Reference {vm_class} IDs missing from canonical candidates: "
                f"{sorted(missing)}"
            )
        if vm_class == "spot" and not set(ids).issubset(eligible_spot_ids):
            ineligible = set(ids) - eligible_spot_ids
            raise ValueError(
                "Frozen reference spot IDs no longer satisfy active-slot "
                f"coverage eligibility: {sorted(ineligible)}"
            )
        if vm_class == "batch_candidate" and not set(ids).issubset(usage_ids):
            missing = set(ids) - usage_ids
            raise ValueError(
                "Frozen reference batch IDs lack scenario-0 usage: "
                f"{sorted(missing)}"
            )

        indexed = candidates.set_index("vm_id", drop=False)
        frame = indexed.loc[ids].reset_index(drop=True).copy()
        if not frame["class"].eq(vm_class).all():
            raise ValueError(
                f"Reference {vm_class} IDs have mismatched canonical classes"
            )
        class_reference = reference_manifest.loc[
            reference_manifest["class"].eq(vm_class)
        ].copy()
        class_reference[
            "selection_rank_within_class"
        ] = pd.to_numeric(
            class_reference["selection_rank_within_class"],
            errors="raise",
        ).astype(int)
        class_reference = class_reference.sort_values(
            ["selection_rank_within_class", "vm_id"],
            kind="mergesort",
        ).reset_index(drop=True)
        if class_reference["vm_id"].astype(str).tolist() != ids:
            raise ValueError(
                f"Reference {vm_class} manifest order changed during selection"
            )

        frame["selection_rank_within_class"] = np.arange(
            1, REFERENCE_COUNT + 1
        )
        frame["selection_method"] = (
            "fixed exact ID and within-class order from reference fixture"
        )
        frame["selection_score_name"] = "not re-ranked"
        frame["selection_score_value"] = np.nan
        frame["eligibility_rule"] = (
            "reference fixture membership, revalidated against canonical data"
        )
        frame["eligible_candidate_count"] = pd.NA
        frame["eligible_rank_within_class"] = pd.NA
        frame["reference_selection_rank_within_class"] = (
            class_reference["selection_rank_within_class"].to_numpy()
        )
        if "eligible_rank_within_class" in class_reference:
            frame["reference_eligible_rank_within_class"] = pd.to_numeric(
                class_reference["eligible_rank_within_class"],
                errors="raise",
            ).astype(int).to_numpy()
        else:
            frame["reference_eligible_rank_within_class"] = pd.NA
        if "selection_score_avg_cpu_plus_mem" in class_reference:
            frame["reference_selection_score_avg_cpu_plus_mem"] = (
                pd.to_numeric(
                    class_reference["selection_score_avg_cpu_plus_mem"],
                    errors="raise",
                )
                .astype(float)
                .to_numpy()
            )
        else:
            frame["reference_selection_score_avg_cpu_plus_mem"] = np.nan
        selected_parts.append(frame)
        selected_ids[vm_class] = ids

    selected = pd.concat(selected_parts, ignore_index=True, sort=False)
    return selected, od_candidates, selected_ids


def _assert_exact_canonical_output(
    canonical_requests: pd.DataFrame,
    canonical_usage: pd.DataFrame,
    output_requests: pd.DataFrame,
    output_usage: pd.DataFrame,
    selected_ids: dict[str, list[str]],
) -> None:
    expected_ids = set().union(
        *(set(ids) for ids in selected_ids.values())
    )
    canonical_request_subset = canonical_requests.loc[
        canonical_requests["vm_id"].isin(expected_ids)
    ].sort_values("vm_id").reset_index(drop=True)
    output_request_subset = output_requests.sort_values(
        "vm_id"
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        output_request_subset[canonical_request_subset.columns],
        canonical_request_subset,
        check_dtype=False,
        check_exact=True,
    )

    canonical_usage_subset = canonical_usage.loc[
        canonical_usage["vm_id"].isin(expected_ids)
    ].sort_values(["vm_id", "t5_day"]).reset_index(drop=True)
    output_usage_subset = output_usage.sort_values(
        ["vm_id", "t5_day"]
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        output_usage_subset[canonical_usage_subset.columns],
        canonical_usage_subset,
        check_dtype=False,
        check_exact=True,
    )


def _validate_output(
    canonical_requests: pd.DataFrame,
    canonical_usage: pd.DataFrame,
    output_requests: pd.DataFrame,
    output_usage: pd.DataFrame,
    manifest: pd.DataFrame,
    od_candidates: pd.DataFrame,
    selected_ids: dict[str, list[str]],
    reference_ids: dict[str, list[str]],
) -> dict[str, Any]:
    expected_counts = {
        "on_demand": OD_COUNT,
        "spot": REFERENCE_COUNT,
        "batch_candidate": REFERENCE_COUNT,
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

    expected_ids = set().union(
        *(set(ids) for ids in selected_ids.values())
    )
    if len(expected_ids) != sum(expected_counts.values()):
        raise ValueError("Selected IDs overlap across workload classes")
    if set(output_requests["vm_id"].astype(str)) != expected_ids:
        raise ValueError("Derived request IDs differ from selected IDs")
    if set(output_usage["vm_id"].astype(str)) != expected_ids:
        raise ValueError("Derived usage IDs differ from selected IDs")

    for vm_class in ALL_CLASSES:
        actual_order = output_requests.loc[
            output_requests["class"].eq(vm_class), "vm_id"
        ].astype(str).tolist()
        if actual_order != selected_ids[vm_class]:
            raise ValueError(
                f"Derived {vm_class} order differs from selected order"
            )
    for vm_class in FROZEN_CLASSES:
        if selected_ids[vm_class] != reference_ids[vm_class]:
            raise ValueError(
                f"Derived {vm_class} IDs/order differ from reference"
            )

    expected_od_ids = (
        od_candidates.head(OD_COUNT)["vm_id"].astype(str).tolist()
    )
    if selected_ids["on_demand"] != expected_od_ids:
        raise ValueError(
            "Derived on-demand IDs are not the eligible CPU-only top 20"
        )
    od_ranks = manifest.loc[
        manifest["class"].eq("on_demand"),
        "eligible_rank_within_class",
    ].astype(int).tolist()
    if od_ranks != list(range(1, OD_COUNT + 1)):
        raise ValueError(
            f"On-demand eligible ranks are not 1..{OD_COUNT}: {od_ranks}"
        )

    _assert_exact_canonical_output(
        canonical_requests,
        canonical_usage,
        output_requests,
        output_usage,
        selected_ids,
    )

    indexed_q = output_requests.set_index("vm_id")
    actual_over_q: dict[str, int] = {}
    for resource in ("cpu", "mem"):
        q = BASE._finite_numeric(
            indexed_q[f"q_{resource}"],
            f"derived q_{resource}",
        )
        if bool(q.le(0).any() or q.gt(1).any()):
            raise ValueError(f"Derived q_{resource} must be in (0, 1]")
        actual = BASE._finite_numeric(
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
        "total_selected_vm_ids": len(expected_ids),
        "scenario_ids_in_usage_file": [0],
        "scenario_zero_usage_rows": int(len(output_usage)),
        "selection_manifest_rows": int(len(manifest)),
        "selected_ids": selected_ids,
        "actual_usage_above_q_row_count": actual_over_q,
        "service_packing": BASE._service_packing_diagnostics(
            output_requests,
            output_usage,
        ),
        "configured_q_packing": BASE._configured_q_packing_diagnostics(
            output_requests
        ),
        "batch_source_workload": BASE._batch_diagnostics(
            output_requests,
            output_usage,
        ),
        "invariants": {
            "canonical_requests_copied_without_value_changes": True,
            "canonical_scenario_zero_usage_copied_without_value_changes": True,
            "on_demand_is_eligible_cpu_only_top20": True,
            "spot_ids_and_order_match_reference_exactly": True,
            "batch_ids_and_order_match_reference_exactly": True,
            "no_resource_scaling": True,
            "no_usage_rescaling": True,
            "no_memory_pressure_transform": True,
            "no_lifecycle_or_provenance_transform": True,
            "selection_does_not_use_a_specific_time_slot": True,
            "exact_class_counts": True,
            "scenario_zero_only": True,
            "unique_usage_keys": True,
        },
    }


def _numeric_distributions(
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
) -> dict[str, Any]:
    distributions: dict[str, Any] = {}
    for vm_class in ALL_CLASSES:
        selected_class = selected.loc[selected["class"].eq(vm_class)]
        values: dict[str, Any] = {
            "selected_count": int(len(selected_class)),
            "q_cpu": BASE._numeric_summary(
                selected_class["q_cpu"].rename("q_cpu")
            ),
            "q_mem": BASE._numeric_summary(
                selected_class["q_mem"].rename("q_mem")
            ),
            "coverage_weighted_avg_cpu_usage": BASE._numeric_summary(
                selected_class["coverage_weighted_avg_cpu_usage"].rename(
                    "coverage_weighted_avg_cpu_usage"
                )
            ),
            "coverage_weighted_avg_mem_usage": BASE._numeric_summary(
                selected_class["coverage_weighted_avg_mem_usage"].rename(
                    "coverage_weighted_avg_mem_usage"
                )
            ),
        }
        if vm_class == "on_demand":
            values["eligible_candidate_count"] = int(len(candidates))
            values["eligible_candidate_cpu_distribution"] = (
                BASE._numeric_summary(
                    candidates[
                        "coverage_weighted_avg_cpu_usage"
                    ].rename("coverage_weighted_avg_cpu_usage")
                )
            )
        distributions[vm_class] = values
    return distributions


def _manifest(selected: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "vm_id",
        "class",
        "selection_rank_within_class",
        "selection_method",
        "selection_score_name",
        "selection_score_value",
        "eligible_rank_within_class",
        "eligible_candidate_count",
        "eligibility_rule",
        "reference_selection_rank_within_class",
        "reference_eligible_rank_within_class",
        "reference_selection_score_avg_cpu_plus_mem",
        "q_cpu",
        "q_mem",
        "coverage_weighted_avg_cpu_usage",
        "coverage_weighted_avg_mem_usage",
        "observed_coverage_us",
        "observed_5min_rows",
        "arrival_t5",
        "departure_t5",
        "lifetime_t5",
    ]
    manifest = selected[
        [column for column in columns if column in selected.columns]
    ].copy()
    manifest["resource_transform"] = "none; exact canonical values copied"
    manifest["_class_order"] = manifest["class"].map(CLASS_ORDER)
    return (
        manifest.sort_values(
            ["_class_order", "selection_rank_within_class"],
            kind="mergesort",
        )
        .drop(columns="_class_order")
        .reset_index(drop=True)
    )


def _ordered_output_requests(
    canonical_requests: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    selected_id_set = set(selected["vm_id"].astype(str))
    output = canonical_requests.loc[
        canonical_requests["vm_id"].isin(selected_id_set)
    ].copy()
    rank = selected.set_index("vm_id")["selection_rank_within_class"]
    output["_class_order"] = output["class"].map(CLASS_ORDER)
    output["_selection_order"] = output["vm_id"].map(rank)
    return (
        output.sort_values(
            ["_class_order", "_selection_order"],
            kind="mergesort",
        )
        .drop(columns=["_class_order", "_selection_order"])
        .reset_index(drop=True)
    )


def _replace_output_with_rollback(
    temp: Path,
    output: Path,
    *,
    overwrite: bool,
) -> None:
    if not output.exists():
        temp.rename(output)
        return
    if not overwrite:
        raise FileExistsError(
            f"Output already exists: {output}; pass --overwrite intentionally"
        )

    backup = output.with_name(f".{output.name}.old-{os.getpid()}")
    if backup.exists():
        shutil.rmtree(backup)
    output.rename(backup)
    try:
        temp.rename(output)
    except BaseException:
        if not output.exists() and backup.exists():
            backup.rename(output)
        raise
    else:
        shutil.rmtree(backup)


def build_micro_cpu_top20_od_sp10_bj10_unscaled_dataset(
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    reference_dir: str | Path = DEFAULT_REFERENCE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    num_scenarios: int = 10,
    chunksize: int = 400_000,
    overwrite: bool = False,
) -> dict[str, Any]:
    source = Path(source_dir).expanduser().resolve()
    reference = Path(reference_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    _assert_disjoint_roots(
        {
            "source_dir": source,
            "reference_dir": reference,
            "output_dir": output,
        }
    )
    if num_scenarios <= 0:
        raise ValueError("num_scenarios must be positive")
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output}; pass --overwrite intentionally"
        )

    source_paths = _required_paths(
        source,
        ("vm_requests.csv", "vm_usage_5min_scenarios.csv", "servers.csv"),
        label="canonical source",
    )
    _optional_path(source_paths, source, "metadata.json")
    reference_paths = _required_paths(
        reference,
        (
            "vm_requests.csv",
            "vm_usage_5min_scenarios.csv",
            "selection_manifest.csv",
        ),
        label="reference fixture",
    )
    for name in ("selection_policy.json", "validation_report.json"):
        _optional_path(reference_paths, reference, name)
    source_hashes_before = _hashes(source_paths)
    reference_hashes_before = _hashes(reference_paths)
    builder_paths = {
        "builder_script": Path(__file__).resolve(),
        "base_builder_script": BASE_BUILDER_PATH,
    }
    builder_hashes_before = _hashes(builder_paths)

    canonical_requests = _read_requests(
        source_paths["vm_requests.csv"],
        label="canonical vm_requests.csv",
    )
    canonical_usage = BASE._read_scenario_zero(
        source_paths["vm_usage_5min_scenarios.csv"],
        chunksize=chunksize,
    )
    reference_requests = _read_requests(
        reference_paths["vm_requests.csv"],
        label="reference vm_requests.csv",
    )
    reference_usage = BASE._read_scenario_zero(
        reference_paths["vm_usage_5min_scenarios.csv"],
        chunksize=chunksize,
    )
    reference_ids, reference_manifest = _read_reference_ids(
        reference_requests,
        reference_paths["selection_manifest.csv"],
    )
    _assert_reference_is_canonical_subset(
        canonical_requests,
        canonical_usage,
        reference_requests,
        reference_usage,
        reference_ids,
    )

    summaries = BASE._coverage_weighted_summaries(canonical_usage)
    selected, od_candidates, selected_ids = _select_rows(
        canonical_requests,
        summaries,
        canonical_usage,
        reference_ids,
        reference_manifest,
    )
    output_requests = _ordered_output_requests(canonical_requests, selected)
    selected_id_set = set(selected["vm_id"].astype(str))
    output_usage = canonical_usage.loc[
        canonical_usage["vm_id"].isin(selected_id_set)
    ].copy()
    output_usage = output_usage.sort_values(
        ["scenario_id", "vm_id", "t5_day"],
        kind="mergesort",
    ).reset_index(drop=True)
    manifest = _manifest(selected)
    report = _validate_output(
        canonical_requests,
        canonical_usage,
        output_requests,
        output_usage,
        manifest,
        od_candidates,
        selected_ids,
        reference_ids,
    )
    report["selected_distribution"] = _numeric_distributions(
        od_candidates,
        selected,
    )
    report["canonical_source_integrity"] = {
        "unchanged": True,
        "sha256_before": source_hashes_before,
    }
    report["reference_fixture_integrity"] = {
        "unchanged": True,
        "sha256_before": reference_hashes_before,
    }
    report["builder_integrity"] = {
        "unchanged": True,
        "files": {
            name: _path_label(path) for name, path in builder_paths.items()
        },
        "sha256_before": builder_hashes_before,
    }

    temp = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    try:
        output_requests.to_csv(temp / "vm_requests.csv", index=False)
        output_usage.to_csv(
            temp / "vm_usage_5min_scenarios.csv",
            index=False,
        )
        manifest.to_csv(temp / "selection_manifest.csv", index=False)
        shutil.copy2(source_paths["servers.csv"], temp / "servers.csv")
        pd.DataFrame(
            {
                "scenario_id": list(range(num_scenarios)),
                "probability": [1.0 / num_scenarios] * num_scenarios,
            }
        ).to_csv(temp / "scenario_probabilities.csv", index=False)

        written_requests = pd.read_csv(
            temp / "vm_requests.csv",
            dtype={"vm_id": str},
        )
        written_usage = pd.read_csv(
            temp / "vm_usage_5min_scenarios.csv",
            dtype={"vm_id": str},
        )
        _assert_exact_canonical_output(
            canonical_requests,
            canonical_usage,
            written_requests,
            written_usage,
            selected_ids,
        )

        source_hashes_after = _hashes(source_paths)
        reference_hashes_after = _hashes(reference_paths)
        builder_hashes_after = _hashes(builder_paths)
        if source_hashes_after != source_hashes_before:
            raise RuntimeError(
                "Canonical source changed while building the derived data set"
            )
        if reference_hashes_after != reference_hashes_before:
            raise RuntimeError(
                "Reference fixture changed while building the derived data set"
            )
        if builder_hashes_after != builder_hashes_before:
            raise RuntimeError(
                "A data-builder script changed while building the derived "
                "data set"
            )
        report["canonical_source_integrity"]["sha256_after"] = (
            source_hashes_after
        )
        report["reference_fixture_integrity"]["sha256_after"] = (
            reference_hashes_after
        )
        report["builder_integrity"]["sha256_after"] = builder_hashes_after

        policy = {
            "artifact_kind": (
                "trace-derived unscaled CPU-heavy OD20 fixture with "
                "reference-frozen SP10/BJ10"
            ),
            "canonical_data_modified": False,
            "reference_fixture_modified": False,
            "source_dir": _path_label(source),
            "reference_dir": _path_label(reference),
            "selection": {
                "classes": list(ALL_CLASSES),
                "on_demand": {
                    "count": OD_COUNT,
                    "score": "coverage-weighted mean(cpu_usage)",
                    "order": (
                        "score descending, vm_id ascending deterministic "
                        "tie-break"
                    ),
                    "eligibility": (
                        "at least one observed 5-minute row in every active "
                        "30-minute slot"
                    ),
                    "uses_specific_time_slot": False,
                },
                "spot": {
                    "count": REFERENCE_COUNT,
                    "method": (
                        "exact IDs and within-class order copied from "
                        "reference fixture"
                    ),
                    "preservation_scope": (
                        "stored request rows and canonical scenario-0 usage "
                        "rows; downstream synthetic scenarios are regenerated "
                        "by the experiment loader"
                    ),
                    "ids": reference_ids["spot"],
                },
                "batch_candidate": {
                    "count": REFERENCE_COUNT,
                    "method": (
                        "exact IDs and within-class order copied from "
                        "reference fixture"
                    ),
                    "ids": reference_ids["batch_candidate"],
                },
            },
            "transformation": {
                "q_cpu": "copied unchanged from canonical source",
                "q_mem": "copied unchanged from canonical source",
                "resource_requests": "copied unchanged from canonical source",
                "actual_cpu_usage": "copied unchanged from canonical source",
                "actual_mem_usage": "copied unchanged from canonical source",
                "lifecycle_and_provenance": (
                    "copied unchanged from canonical source"
                ),
                "usage_rescaling": None,
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
                "cross_fixture_reproducibility_note": (
                    "The frozen spot source rows are identical, but the "
                    "current loader uses one RNG stream for the complete "
                    "OD+spot set; changing the OD set can therefore change "
                    "synthetic spot draws in scenarios 1..N-1."
                ),
            },
            "canonical_source_integrity": report[
                "canonical_source_integrity"
            ],
            "reference_fixture_integrity": report[
                "reference_fixture_integrity"
            ],
            "builder_integrity": report["builder_integrity"],
            "validation_summary": report,
        }
        BASE._json_dump(temp / "validation_report.json", report)
        BASE._json_dump(temp / "selection_policy.json", policy)

        canonical_metadata: dict[str, Any] = {}
        source_metadata = source_paths.get("metadata.json")
        if source_metadata is not None:
            with source_metadata.open("r", encoding="utf-8") as stream:
                loaded = json.load(stream)
            if isinstance(loaded, dict):
                canonical_metadata = loaded
        metadata: dict[str, Any] = {
            "dataset_name": DATASET_NAME,
            "artifact_kind": policy["artifact_kind"],
            "source": canonical_metadata.get("source"),
            "source_note": canonical_metadata.get("source_note"),
            "canonical_source_dir": _path_label(source),
            "reference_fixture_dir": _path_label(reference),
            "seed": canonical_metadata.get("seed"),
            "num_vms": OD_COUNT + 2 * REFERENCE_COUNT,
            "class_counts": {
                "on_demand": OD_COUNT,
                "spot": REFERENCE_COUNT,
                "batch_candidate": REFERENCE_COUNT,
            },
            "num_servers": int(
                len(pd.read_csv(source_paths["servers.csv"]))
            ),
            "num_scenarios": num_scenarios,
            "stored_num_scenarios": 1,
            "stored_scenario_ids": [0],
            "horizon_t5": canonical_metadata.get("horizon_t5", 288),
            "horizon_hours": canonical_metadata.get("horizon_hours", 24),
            "default_model_usage_file": "vm_usage_5min_scenarios.csv",
            "micro_cpu_top20_od_sp10_bj10_unscaled_derivation": policy,
            "canonical_source_metadata": canonical_metadata,
        }
        BASE._json_dump(temp / "metadata.json", metadata)
        (temp / "README.md").write_text(
            "# Google 2019 CPU-heavy OD20 + frozen SP10/BJ10 fixture\n\n"
            "This directory is derived from the canonical v5 pool without "
            "modifying it. The 20 eligible on-demand VMs with the largest "
            "coverage-weighted scenario-0 mean CPU usage are selected. The "
            "10 spot and 10 batch-candidate IDs, including their within-class "
            "order, are copied exactly from the existing unscaled top-10 "
            "reference fixture.\n\n"
            "All output request rows and scenario-0 usage rows are copied "
            "from the canonical source. Configured q, resource requests, "
            "CPU and memory usage, lifecycle, and provenance values are not "
            "changed. There is no target-q scaling, usage rescaling, or "
            "memory-pressure transform.\n\n"
            "The frozen spot guarantee applies to the stored request and "
            "canonical scenario-0 usage rows. The current optimization loader "
            "regenerates later OD/SP scenarios from one RNG stream, so changing "
            "the OD set can change synthetic spot draws even though the stored "
            "spot source data is identical.\n\n"
            "The workload-class labels are Google trace proxy classes, not "
            "observed public-cloud purchase types. The optimization loader "
            "later generates the remaining workload scenarios and converts "
            "the selected batch candidates into flexible workload families. "
            "See `selection_policy.json`, `selection_manifest.csv`, and "
            "`validation_report.json` for exact provenance and validation.\n",
            encoding="utf-8",
        )

        _replace_output_with_rollback(temp, output, overwrite=overwrite)
    except BaseException:
        if temp.exists():
            shutil.rmtree(temp)
        raise
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=DEFAULT_REFERENCE_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-scenarios", type=int, default=10)
    parser.add_argument("--chunksize", type=int, default=400_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = build_micro_cpu_top20_od_sp10_bj10_unscaled_dataset(
        source_dir=args.source_dir,
        reference_dir=args.reference_dir,
        output_dir=args.output_dir,
        num_scenarios=args.num_scenarios,
        chunksize=args.chunksize,
        overwrite=args.overwrite,
    )
    packing = report["service_packing"]
    print(
        "Built unscaled CPU-heavy OD20 + frozen SP10/BJ10 data: "
        f"selected VMs={report['total_selected_vm_ids']}, "
        f"max active service VMs={packing['maximum_active_vms']}, "
        "max scenario-0 exact-minimum servers="
        f"{packing['maximum_exact_minimum_servers']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
