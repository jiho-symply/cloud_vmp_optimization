from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from google2019_toy.build_toy_instance import (
    audit_assigned_machine_cpu_capacity,
    build_hourly_usage_scenarios,
    build_toy_instance,
    build_vm_requests,
    derive_representative_machine_capacity,
    drop_vms_exceeding_representative_capacity,
)


TRACE_START_US = 600 * 1_000_000
DAY_US = 24 * 60 * 60 * 1_000_000
STEP_US = 300 * 1_000_000

REPOSITORY_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "experiments" / "2607-notion-server-min-vmp").is_dir()
)
SERVER_MIN_ROOT = REPOSITORY_ROOT / "experiments" / "2607-notion-server-min-vmp"
SERVER_MIN_SRC = SERVER_MIN_ROOT / "src"
if str(SERVER_MIN_SRC) not in sys.path:
    sys.path.insert(0, str(SERVER_MIN_SRC))

from notion_server_min_vmp.data import (  # noqa: E402
    CPU,
    MEM,
    _build_batch_families,
    _build_service_scenarios,
    build_instance,
    load_config,
)
from notion_server_min_vmp.model import build_model  # noqa: E402


def _usage_interval(
    *,
    collection_id: int = 1,
    instance_index: int = 0,
    start_offset_us: int,
    end_offset_us: int,
    cpu_usage: float | None,
    mem_usage: float | None,
    max_cpu_usage: float | None,
    max_mem_usage: float | None,
    machine_id: str = "machine-a",
) -> dict[str, object]:
    start_time = TRACE_START_US + start_offset_us
    end_time = TRACE_START_US + end_offset_us
    clipped_start = max(start_time, TRACE_START_US)
    clipped_end = min(end_time, TRACE_START_US + DAY_US)
    return {
        "collection_id": collection_id,
        "instance_index": instance_index,
        "machine_id": machine_id,
        "collection_type": 0,
        "alloc_collection_id": None,
        "alloc_instance_index": None,
        "start_time": start_time,
        "end_time": end_time,
        "clipped_start_time": clipped_start,
        "clipped_end_time": clipped_end,
        "overlap_us": clipped_end - clipped_start,
        "cpu_usage": cpu_usage,
        "mem_usage": mem_usage,
        "max_cpu_usage": max_cpu_usage,
        "max_mem_usage": max_mem_usage,
        "assigned_memory": None,
        "t5": clipped_start // STEP_US,
        "t5_day": (clipped_start - TRACE_START_US) // STEP_US,
    }


def _request_for_single_vm(
    usage_rows: list[dict[str, object]],
    *,
    events: pd.DataFrame | None = None,
    collection_events: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return build_vm_requests(
        pd.DataFrame(usage_rows),
        events=events,
        collection_events=collection_events,
        max_instances=1,
        seed=42,
        min_usage_rows=1,
        day_start_us=TRACE_START_US,
    )


def _instance_event(
    event_type: str,
    *,
    cpu: float,
    mem: float,
    time: int | None = None,
    machine_id: str = "machine-a",
    collection_id: int = 1,
    instance_index: int = 0,
) -> dict[str, object]:
    return {
        "collection_id": collection_id,
        "instance_index": instance_index,
        "time": TRACE_START_US - 1 if time is None else time,
        "event_type": event_type,
        "missing_type": 0,
        "machine_id": machine_id,
        "priority": 120,
        "scheduling_class": 1,
        "resource_request_cpu": cpu,
        "resource_request_mem": mem,
    }


def test_sub_five_minute_usage_interval_is_preserved() -> None:
    vm_requests, observed = _request_for_single_vm(
        [
            _usage_interval(
                start_offset_us=30_000_000,
                end_offset_us=90_000_000,
                cpu_usage=0.25,
                mem_usage=0.40,
                max_cpu_usage=0.30,
                max_mem_usage=0.45,
            )
        ]
    )

    assert len(vm_requests) == 1
    assert observed["t5_day"].tolist() == [0]
    assert observed.loc[0, "cpu_usage"] == pytest.approx(0.25)
    assert observed.loc[0, "mem_usage"] == pytest.approx(0.40)


def test_day_boundary_is_clipped_and_average_usage_is_duration_weighted() -> None:
    vm_requests, observed = _request_for_single_vm(
        [
            _usage_interval(
                start_offset_us=-60_000_000,
                end_offset_us=120_000_000,
                cpu_usage=0.20,
                mem_usage=0.30,
                max_cpu_usage=0.25,
                max_mem_usage=0.35,
            ),
            _usage_interval(
                start_offset_us=120_000_000,
                end_offset_us=300_000_000,
                cpu_usage=0.80,
                mem_usage=0.90,
                max_cpu_usage=0.85,
                max_mem_usage=0.95,
            ),
        ]
    )

    assert observed["t5_day"].tolist() == [0]
    assert observed.loc[0, "cpu_usage"] == pytest.approx((0.20 * 120 + 0.80 * 180) / 300)
    assert observed.loc[0, "mem_usage"] == pytest.approx((0.30 * 120 + 0.90 * 180) / 300)
    assert vm_requests.loc[0, "max_cpu_usage"] == pytest.approx(0.85)
    assert vm_requests.loc[0, "max_mem_usage"] == pytest.approx(0.95)


def test_usage_interval_is_split_across_each_overlapped_five_minute_bucket() -> None:
    _, observed = _request_for_single_vm(
        [
            _usage_interval(
                start_offset_us=270_000_000,
                end_offset_us=330_000_000,
                cpu_usage=0.20,
                mem_usage=0.40,
                max_cpu_usage=0.25,
                max_mem_usage=0.45,
            ),
            _usage_interval(
                start_offset_us=330_000_000,
                end_offset_us=600_000_000,
                cpu_usage=0.60,
                mem_usage=0.20,
                max_cpu_usage=0.65,
                max_mem_usage=0.25,
            ),
        ]
    )

    assert observed["t5_day"].tolist() == [0, 1]
    assert observed.loc[0, "cpu_usage"] == pytest.approx(0.20)
    assert observed.loc[1, "cpu_usage"] == pytest.approx((0.20 * 30 + 0.60 * 270) / 300)
    assert observed.loc[1, "mem_usage"] == pytest.approx((0.40 * 30 + 0.20 * 270) / 300)


def test_null_usage_is_excluded_per_resource_and_is_never_zero_filled() -> None:
    vm_requests, observed = _request_for_single_vm(
        [
            _usage_interval(
                start_offset_us=0,
                end_offset_us=150_000_000,
                cpu_usage=None,
                mem_usage=0.40,
                max_cpu_usage=None,
                max_mem_usage=0.50,
            ),
            _usage_interval(
                start_offset_us=150_000_000,
                end_offset_us=300_000_000,
                cpu_usage=0.80,
                mem_usage=None,
                max_cpu_usage=0.90,
                max_mem_usage=None,
            ),
        ]
    )

    assert observed.loc[0, "cpu_usage"] == pytest.approx(0.80)
    assert observed.loc[0, "mem_usage"] == pytest.approx(0.40)
    assert vm_requests.loc[0, "max_cpu_usage"] == pytest.approx(0.90)
    assert vm_requests.loc[0, "max_mem_usage"] == pytest.approx(0.50)
    assert not (observed[["cpu_usage", "mem_usage"]] == 0.0).any().any()


def test_null_only_usage_cannot_be_emitted_as_zero_demand() -> None:
    with pytest.raises(ValueError, match=r"(?i)CPU|memory|usage|bucket|candidate"):
        _request_for_single_vm(
            [
                _usage_interval(
                    start_offset_us=0,
                    end_offset_us=300_000_000,
                    cpu_usage=None,
                    mem_usage=None,
                    max_cpu_usage=None,
                    max_mem_usage=None,
                )
            ]
        )


def test_maximum_usage_survives_when_null_averages_exclude_the_actual_bucket() -> None:
    vm_requests, observed = _request_for_single_vm(
        [
            _usage_interval(
                start_offset_us=0,
                end_offset_us=300_000_000,
                cpu_usage=0.10,
                mem_usage=0.10,
                max_cpu_usage=0.20,
                max_mem_usage=0.20,
            ),
            # This bucket cannot be emitted as actual workload, but its valid
            # source maxima still constrain the VM's configured resources.
            _usage_interval(
                start_offset_us=300_000_000,
                end_offset_us=600_000_000,
                cpu_usage=None,
                mem_usage=None,
                max_cpu_usage=0.90,
                max_mem_usage=0.80,
            ),
        ]
    )

    assert observed["t5_day"].tolist() == [0]
    assert vm_requests.loc[0, "max_cpu_usage"] == pytest.approx(0.90)
    assert vm_requests.loc[0, "max_mem_usage"] == pytest.approx(0.80)
    assert vm_requests.loc[0, "q_cpu"] == pytest.approx(0.90)
    assert vm_requests.loc[0, "q_mem"] == pytest.approx(0.80)


def test_instance_request_and_priority_use_one_coherent_arrival_state() -> None:
    usage = [
        _usage_interval(
            start_offset_us=600_000_000,
            end_offset_us=900_000_000,
            cpu_usage=0.20,
            mem_usage=0.30,
            max_cpu_usage=0.50,
            max_mem_usage=0.40,
        )
    ]
    events = pd.DataFrame(
        [
            {
                "collection_id": 1,
                "instance_index": 0,
                "time": TRACE_START_US - 100_000_000,
                "event_type": 0,
                "priority": 121,
                "resource_request_cpu": 0.30,
                "resource_request_mem": 0.70,
            },
            {
                "collection_id": 1,
                "instance_index": 0,
                "time": TRACE_START_US + 500_000_000,
                "event_type": 3,
                "priority": 130,
                "resource_request_cpu": 0.80,
                "resource_request_mem": 0.20,
            },
            {
                "collection_id": 1,
                "instance_index": 0,
                "time": TRACE_START_US + 700_000_000,
                "event_type": 8,
                "priority": 5,
                "resource_request_cpu": 0.40,
                "resource_request_mem": 0.90,
            },
        ]
    )

    vm_requests, _ = _request_for_single_vm(usage, events=events)
    request = vm_requests.iloc[0]

    assert request["resource_request_cpu"] == pytest.approx(0.80)
    assert request["resource_request_mem"] == pytest.approx(0.20)
    assert request["priority"] == pytest.approx(130)
    assert request["q_cpu"] == pytest.approx(0.80)
    assert request["q_mem"] == pytest.approx(0.40)
    assert request["class"] == "on_demand"


def test_event_state_uses_exact_first_observation_not_five_minute_bucket_start() -> None:
    usage = [
        _usage_interval(
            start_offset_us=150_000_000,
            end_offset_us=300_000_000,
            cpu_usage=0.20,
            mem_usage=0.20,
            max_cpu_usage=0.25,
            max_mem_usage=0.25,
        )
    ]
    events = pd.DataFrame(
        [
            {
                "collection_id": 1,
                "instance_index": 0,
                "time": TRACE_START_US + 50_000_000,
                "event_type": 3,
                "priority": 120,
                "resource_request_cpu": 0.20,
                "resource_request_mem": 0.80,
            },
            {
                "collection_id": 1,
                "instance_index": 0,
                "time": TRACE_START_US + 140_000_000,
                "event_type": 3,
                "priority": 130,
                "resource_request_cpu": 0.70,
                "resource_request_mem": 0.30,
            },
        ]
    )

    vm_requests, _ = _request_for_single_vm(usage, events=events)
    request = vm_requests.iloc[0]

    assert request["resource_request_cpu"] == pytest.approx(0.70)
    assert request["resource_request_mem"] == pytest.approx(0.30)
    assert request["priority"] == pytest.approx(130)


def test_post_arrival_fallback_uses_one_earliest_state_row_without_field_composition() -> None:
    usage = [
        _usage_interval(
            start_offset_us=150_000_000,
            end_offset_us=300_000_000,
            cpu_usage=0.30,
            mem_usage=0.30,
            max_cpu_usage=0.40,
            max_mem_usage=0.40,
        )
    ]
    events = pd.DataFrame(
        [
            {
                "collection_id": 1,
                "instance_index": 0,
                "time": TRACE_START_US + 160_000_000,
                "event_type": 3,
                "priority": 130,
                "resource_request_cpu": 0.80,
                "resource_request_mem": None,
            },
            {
                "collection_id": 1,
                "instance_index": 0,
                "time": TRACE_START_US + 170_000_000,
                "event_type": 8,
                "priority": None,
                "resource_request_cpu": None,
                "resource_request_mem": 0.90,
            },
        ]
    )

    vm_requests, _ = _request_for_single_vm(usage, events=events)
    request = vm_requests.iloc[0]

    assert request["resource_request_cpu"] == pytest.approx(0.80)
    assert pd.isna(request["resource_request_mem"]) or request["resource_request_mem"] == pytest.approx(0.0)
    assert request["q_cpu"] == pytest.approx(0.80)
    assert request["q_mem"] == pytest.approx(0.40)


def test_post_arrival_fallback_skips_all_null_state_before_coherent_snapshot() -> None:
    usage = [
        _usage_interval(
            start_offset_us=150_000_000,
            end_offset_us=300_000_000,
            cpu_usage=0.20,
            mem_usage=0.20,
            max_cpu_usage=0.25,
            max_mem_usage=0.25,
        )
    ]
    events = pd.DataFrame(
        [
            {
                "collection_id": 1,
                "instance_index": 0,
                "time": TRACE_START_US + 160_000_000,
                "event_type": 3,
                "priority": None,
                "scheduling_class": None,
                "resource_request_cpu": None,
                "resource_request_mem": None,
            },
            {
                "collection_id": 1,
                "instance_index": 0,
                "time": TRACE_START_US + 170_000_000,
                "event_type": 3,
                "priority": 130,
                "scheduling_class": 1,
                "resource_request_cpu": 0.80,
                "resource_request_mem": 0.70,
            },
        ]
    )

    vm_requests, _ = _request_for_single_vm(usage, events=events)
    request = vm_requests.iloc[0]

    assert request["priority"] == pytest.approx(130)
    assert request["resource_request_cpu"] == pytest.approx(0.80)
    assert request["resource_request_mem"] == pytest.approx(0.70)
    assert request["q_cpu"] == pytest.approx(0.80)
    assert request["q_mem"] == pytest.approx(0.70)
    assert request["class"] == "on_demand"


@pytest.mark.parametrize("arrival_scheduler", [1, "SCHEDULER_BATCH"])
def test_collection_scheduler_uses_state_at_vm_arrival(arrival_scheduler: object) -> None:
    usage = [
        _usage_interval(
            start_offset_us=600_000_000,
            end_offset_us=900_000_000,
            cpu_usage=0.20,
            mem_usage=0.30,
            max_cpu_usage=0.25,
            max_mem_usage=0.35,
        )
    ]
    events = pd.DataFrame(
        [
            {
                "collection_id": 1,
                "instance_index": 0,
                "time": TRACE_START_US + 500_000_000,
                "event_type": 3,
                "priority": 130,
                "resource_request_cpu": 0.20,
                "resource_request_mem": 0.30,
            }
        ]
    )
    collection_events = pd.DataFrame(
        [
            {
                "collection_id": 1,
                "time": TRACE_START_US + 700_000_000,
                "event_type": 3,
                "scheduler": 0,
                "collection_type": 0,
            },
            {
                "collection_id": 1,
                "time": TRACE_START_US + 500_000_000,
                "event_type": 3,
                "scheduler": arrival_scheduler,
                "collection_type": 0,
            },
        ]
    )

    vm_requests, _ = _request_for_single_vm(
        usage,
        events=events,
        collection_events=collection_events,
    )

    assert vm_requests.loc[0, "scheduler"] == "SCHEDULER_BATCH"
    assert vm_requests.loc[0, "class"] == "batch_candidate"


def test_q_is_maximum_of_arrival_request_and_observed_maximum_usage() -> None:
    usage = [
        _usage_interval(
            start_offset_us=0,
            end_offset_us=300_000_000,
            cpu_usage=0.20,
            mem_usage=0.30,
            max_cpu_usage=0.60,
            max_mem_usage=0.70,
        )
    ]
    events = pd.DataFrame(
        [
            {
                "collection_id": 1,
                "instance_index": 0,
                "time": TRACE_START_US - 1,
                "event_type": 3,
                "priority": 120,
                "resource_request_cpu": 0.40,
                "resource_request_mem": 0.80,
            }
        ]
    )

    vm_requests, _ = _request_for_single_vm(usage, events=events)

    assert vm_requests.loc[0, "q_cpu"] == pytest.approx(0.60)
    assert vm_requests.loc[0, "q_mem"] == pytest.approx(0.80)


def test_q_does_not_use_the_legacy_one_point_two_times_p95_rule() -> None:
    vm_requests, _ = _request_for_single_vm(
        [
            _usage_interval(
                start_offset_us=0,
                end_offset_us=300_000_000,
                cpu_usage=0.20,
                mem_usage=0.30,
                max_cpu_usage=0.23,
                max_mem_usage=0.31,
            )
        ]
    )

    assert vm_requests.loc[0, "q_cpu"] == pytest.approx(0.23)
    assert vm_requests.loc[0, "q_mem"] == pytest.approx(0.31)
    assert vm_requests.loc[0, "q_cpu"] != pytest.approx(1.2 * 0.20)
    assert vm_requests.loc[0, "q_mem"] != pytest.approx(1.2 * 0.30)


@pytest.mark.parametrize("shuffle_seed", [0, 3, 17])
def test_same_time_submit_enable_schedule_uses_semantic_precedence(
    shuffle_seed: int,
) -> None:
    usage = [
        _usage_interval(
            start_offset_us=0,
            end_offset_us=STEP_US,
            cpu_usage=0.1,
            mem_usage=0.1,
            max_cpu_usage=0.2,
            max_mem_usage=0.2,
        )
    ]
    events = pd.DataFrame(
        [
            _instance_event("SUBMIT", cpu=0.30, mem=0.31),
            _instance_event("ENABLE", cpu=0.40, mem=0.41),
            _instance_event("SCHEDULE", cpu=0.50, mem=0.51),
        ]
    ).sample(frac=1.0, random_state=shuffle_seed)

    requests, _ = _request_for_single_vm(usage, events=events)

    assert requests.loc[0, "resource_request_cpu"] == pytest.approx(0.50)
    assert requests.loc[0, "resource_request_mem"] == pytest.approx(0.51)
    assert not bool(requests.loc[0, "arrival_state_ambiguous"])


def test_same_time_evict_then_submit_retry_uses_new_submit_state() -> None:
    usage = [
        _usage_interval(
            start_offset_us=0,
            end_offset_us=STEP_US,
            cpu_usage=0.1,
            mem_usage=0.1,
            max_cpu_usage=0.2,
            max_mem_usage=0.2,
        )
    ]
    events = pd.DataFrame(
        [
            _instance_event("SUBMIT", cpu=0.70, mem=0.71),
            _instance_event("EVICT", cpu=0.20, mem=0.21),
        ]
    )

    requests, _ = _request_for_single_vm(usage, events=events)

    assert requests.loc[0, "resource_request_cpu"] == pytest.approx(0.70)
    assert requests.loc[0, "resource_request_mem"] == pytest.approx(0.71)


def test_same_time_update_running_follows_schedule_and_machine_match_is_preferred() -> None:
    usage = [
        _usage_interval(
            start_offset_us=0,
            end_offset_us=STEP_US,
            cpu_usage=0.1,
            mem_usage=0.1,
            max_cpu_usage=0.2,
            max_mem_usage=0.2,
            machine_id="machine-a",
        )
    ]
    events = pd.DataFrame(
        [
            _instance_event("SCHEDULE", cpu=0.80, mem=0.81, machine_id="machine-b"),
            _instance_event("SCHEDULE", cpu=0.60, mem=0.61, machine_id="machine-a"),
            _instance_event("UPDATE_RUNNING", cpu=0.90, mem=0.91, machine_id="machine-a"),
        ]
    )

    requests, _ = _request_for_single_vm(usage, events=events)

    assert requests.loc[0, "resource_request_cpu"] == pytest.approx(0.90)
    assert requests.loc[0, "resource_request_mem"] == pytest.approx(0.91)
    assert requests.loc[0, "arrival_machine_id"] == "machine-a"


@pytest.mark.parametrize("shuffle_seed", [2, 13])
def test_observed_machine_match_precedes_nonmatching_later_running_transition(
    shuffle_seed: int,
) -> None:
    usage = [
        _usage_interval(
            start_offset_us=0,
            end_offset_us=STEP_US,
            cpu_usage=0.1,
            mem_usage=0.1,
            max_cpu_usage=0.2,
            max_mem_usage=0.2,
            machine_id="machine-a",
        )
    ]
    events = pd.DataFrame(
        [
            _instance_event("SCHEDULE", cpu=0.60, mem=0.61, machine_id="machine-a"),
            _instance_event(
                "UPDATE_RUNNING", cpu=0.90, mem=0.91, machine_id="machine-b"
            ),
        ]
    ).sample(frac=1.0, random_state=shuffle_seed)

    requests, _ = _request_for_single_vm(usage, events=events)

    assert requests.loc[0, "resource_request_cpu"] == pytest.approx(0.60)
    assert requests.loc[0, "resource_request_mem"] == pytest.approx(0.61)
    assert requests.loc[0, "arrival_machine_id"] == "machine-a"


@pytest.mark.parametrize("shuffle_seed", [1, 9])
def test_sparse_null_and_single_nonnull_same_time_values_are_not_ambiguous(
    shuffle_seed: int,
) -> None:
    usage = [
        _usage_interval(
            start_offset_us=0,
            end_offset_us=STEP_US,
            cpu_usage=0.1,
            mem_usage=0.1,
            max_cpu_usage=0.2,
            max_mem_usage=0.2,
        )
    ]
    first = _instance_event("SCHEDULE", cpu=0.30, mem=0.40)
    first["resource_request_mem"] = None
    second = _instance_event("SCHEDULE", cpu=0.30, mem=0.40)
    second["resource_request_cpu"] = None
    events = pd.DataFrame([first, second]).sample(frac=1.0, random_state=shuffle_seed)
    collection_events = pd.DataFrame(
        [
            {
                "collection_id": 1,
                "time": TRACE_START_US - 1,
                "event_type": "SCHEDULE",
                "missing_type": 0,
                "scheduler": None,
            },
            {
                "collection_id": 1,
                "time": TRACE_START_US - 1,
                "event_type": "SCHEDULE",
                "missing_type": 0,
                "scheduler": 1,
            },
        ]
    ).sample(frac=1.0, random_state=shuffle_seed)

    requests, _ = _request_for_single_vm(
        usage, events=events, collection_events=collection_events
    )

    assert not bool(requests.loc[0, "arrival_state_ambiguous"])
    assert not bool(requests.loc[0, "scheduler_state_ambiguous"])
    assert requests.loc[0, "resource_request_cpu"] == pytest.approx(0.30)
    assert requests.loc[0, "resource_request_mem"] == pytest.approx(0.40)
    assert requests.loc[0, "scheduler"] == "SCHEDULER_BATCH"


def test_conflicting_same_time_instance_state_is_flagged_and_strictly_excluded() -> None:
    usage_rows = [
        _usage_interval(
            collection_id=collection_id,
            start_offset_us=0,
            end_offset_us=STEP_US,
            cpu_usage=0.1,
            mem_usage=0.1,
            max_cpu_usage=0.2,
            max_mem_usage=0.2,
        )
        for collection_id in [1, 2]
    ]
    events = pd.DataFrame(
        [
            _instance_event("SCHEDULE", cpu=0.30, mem=0.40, collection_id=1),
            _instance_event("SCHEDULE", cpu=0.70, mem=0.20, collection_id=1),
            _instance_event("SCHEDULE", cpu=0.30, mem=0.40, collection_id=2),
        ]
    )

    legacy, _ = build_vm_requests(
        pd.DataFrame(usage_rows),
        events=events,
        min_usage_rows=1,
        max_instances=2,
        day_start_us=TRACE_START_US,
        strict_event_state=False,
    )
    ambiguous = legacy.loc[legacy["collection_id"].eq(1)].iloc[0]
    assert bool(ambiguous["arrival_state_ambiguous"])
    assert ambiguous["arrival_state_candidate_count"] == 2
    assert ambiguous["arrival_state_ambiguity_reason"] == "same_time_conflicting_instance_state"

    strict, _ = build_vm_requests(
        pd.DataFrame(usage_rows),
        events=events,
        min_usage_rows=1,
        max_instances=2,
        day_start_us=TRACE_START_US,
    )
    assert strict["collection_id"].tolist() == [2]
    audit = strict.attrs["event_state_ambiguity"]
    assert audit["excluded_ambiguous_candidate_count"] == 1


def test_collection_scheduler_conflict_is_flagged_and_strictly_excluded() -> None:
    usage_rows = [
        _usage_interval(
            collection_id=collection_id,
            start_offset_us=0,
            end_offset_us=STEP_US,
            cpu_usage=0.1,
            mem_usage=0.1,
            max_cpu_usage=0.2,
            max_mem_usage=0.2,
        )
        for collection_id in [1, 2]
    ]
    events = pd.DataFrame(
        [
            _instance_event("SCHEDULE", cpu=0.3, mem=0.3, collection_id=1),
            _instance_event("SCHEDULE", cpu=0.3, mem=0.3, collection_id=2),
        ]
    )
    collection_events = pd.DataFrame(
        [
            {
                "collection_id": 1,
                "time": TRACE_START_US - 1,
                "event_type": "SCHEDULE",
                "missing_type": 0,
                "scheduler": 0,
            },
            {
                "collection_id": 1,
                "time": TRACE_START_US - 1,
                "event_type": "SCHEDULE",
                "missing_type": 0,
                "scheduler": 1,
            },
            {
                "collection_id": 2,
                "time": TRACE_START_US - 1,
                "event_type": "SCHEDULE",
                "missing_type": 0,
                "scheduler": 0,
            },
        ]
    )

    legacy, _ = build_vm_requests(
        pd.DataFrame(usage_rows),
        events=events,
        collection_events=collection_events,
        min_usage_rows=1,
        max_instances=2,
        day_start_us=TRACE_START_US,
        strict_event_state=False,
    )
    assert bool(legacy.loc[legacy["collection_id"].eq(1), "scheduler_state_ambiguous"].iloc[0])

    strict, _ = build_vm_requests(
        pd.DataFrame(usage_rows),
        events=events,
        collection_events=collection_events,
        min_usage_rows=1,
        max_instances=2,
        day_start_us=TRACE_START_US,
    )
    assert strict["collection_id"].tolist() == [2]


def test_ambiguous_filter_precedes_unchanged_hash_head_and_backfills() -> None:
    keys = [(10, 0), (20, 0), (30, 0)]
    ranked = sorted(
        keys,
        key=lambda key: (
            int.from_bytes(
                hashlib.blake2b(f"42:{key[0]}:{key[1]}".encode(), digest_size=8).digest(),
                "big",
            ),
            key,
        ),
    )
    ambiguous_key = ranked[0]
    usage_rows = [
        _usage_interval(
            collection_id=collection_id,
            instance_index=instance_index,
            start_offset_us=0,
            end_offset_us=STEP_US,
            cpu_usage=0.1,
            mem_usage=0.1,
            max_cpu_usage=0.2,
            max_mem_usage=0.2,
        )
        for collection_id, instance_index in keys
    ]
    events = [
        _instance_event(
            "SCHEDULE",
            cpu=0.3,
            mem=0.3,
            collection_id=collection_id,
            instance_index=instance_index,
        )
        for collection_id, instance_index in keys
    ]
    events.append(
        _instance_event(
            "SCHEDULE",
            cpu=0.8,
            mem=0.2,
            collection_id=ambiguous_key[0],
            instance_index=ambiguous_key[1],
        )
    )

    requests, _ = build_vm_requests(
        pd.DataFrame(usage_rows),
        events=pd.DataFrame(events),
        min_usage_rows=1,
        max_instances=2,
        seed=42,
        day_start_us=TRACE_START_US,
    )

    selected = list(zip(requests["collection_id"], requests["instance_index"]))
    assert selected == ranked[1:]


def test_event_and_usage_shuffle_leave_outputs_value_equivalent() -> None:
    usage = pd.DataFrame(
        [
            _usage_interval(
                start_offset_us=0,
                end_offset_us=200_000_000,
                cpu_usage=0.2,
                mem_usage=0.4,
                max_cpu_usage=0.5,
                max_mem_usage=0.6,
            ),
            _usage_interval(
                start_offset_us=100_000_000,
                end_offset_us=STEP_US,
                cpu_usage=0.8,
                mem_usage=0.2,
                max_cpu_usage=0.9,
                max_mem_usage=0.5,
            ),
        ]
    )
    events = pd.DataFrame(
        [
            _instance_event("SUBMIT", cpu=0.3, mem=0.4),
            _instance_event("SCHEDULE", cpu=0.7, mem=0.6),
        ]
    )
    baseline_requests, baseline_usage = _request_for_single_vm(
        usage.to_dict("records"), events=events
    )
    shuffled_requests, shuffled_usage = _request_for_single_vm(
        usage.sample(frac=1.0, random_state=7).to_dict("records"),
        events=events.sample(frac=1.0, random_state=11),
    )

    pd.testing.assert_frame_equal(baseline_requests, shuffled_requests)
    pd.testing.assert_frame_equal(baseline_usage, shuffled_usage)


def test_exact_duplicate_and_partial_overlap_use_elementary_union_weighting() -> None:
    duplicate = _usage_interval(
        start_offset_us=0,
        end_offset_us=200_000_000,
        cpu_usage=0.2,
        mem_usage=0.4,
        max_cpu_usage=0.5,
        max_mem_usage=0.6,
    )
    duplicate["cpu_usage_distribution"] = np.asarray([0.02] * 10 + [0.5])
    equal_distinct_array_duplicate = dict(duplicate)
    equal_distinct_array_duplicate["cpu_usage_distribution"] = np.asarray(
        [0.02] * 10 + [0.5]
    )
    overlap = _usage_interval(
        start_offset_us=100_000_000,
        end_offset_us=STEP_US,
        cpu_usage=0.8,
        mem_usage=0.2,
        max_cpu_usage=0.9,
        max_mem_usage=0.5,
    )
    overlap["cpu_usage_distribution"] = np.asarray([0.03] * 10 + [0.9])
    requests, observed = _request_for_single_vm(
        [duplicate, equal_distinct_array_duplicate, overlap]
    )

    # [0,100): .2; [100,200): mean(.2,.8)=.5; [200,300): .8
    assert observed.loc[0, "cpu_usage"] == pytest.approx(0.5)
    assert observed.loc[0, "mem_usage"] == pytest.approx(0.3)
    assert observed.loc[0, "coverage_us"] == STEP_US
    assert observed.loc[0, "duplicated_timeline_us"] == 100_000_000
    assert observed.loc[0, "overlap_source_row_count"] == 2
    assert bool(observed.loc[0, "overlap_conflict_flag"])
    audit = requests.attrs["usage_overlap_diagnostics"]
    assert audit["exact_duplicate_row_count"] == 1
    assert audit["overlap_affected_bucket_count"] == 1
    assert audit["max_concurrent_source_rows"] == 2


def test_overlapping_null_cpu_does_not_pollute_cpu_or_memory_denominator() -> None:
    _, observed = _request_for_single_vm(
        [
            _usage_interval(
                start_offset_us=0,
                end_offset_us=STEP_US,
                cpu_usage=None,
                mem_usage=0.2,
                max_cpu_usage=None,
                max_mem_usage=0.3,
            ),
            _usage_interval(
                start_offset_us=0,
                end_offset_us=STEP_US,
                cpu_usage=0.8,
                mem_usage=0.6,
                max_cpu_usage=0.9,
                max_mem_usage=0.7,
            ),
        ]
    )

    assert observed.loc[0, "cpu_usage"] == pytest.approx(0.8)
    assert observed.loc[0, "mem_usage"] == pytest.approx(0.4)
    assert 0 < observed.loc[0, "coverage_us"] <= STEP_US


def test_hourly_cpu_and_memory_are_coverage_duration_weighted_means() -> None:
    usage = pd.DataFrame(
        [
            {
                "scenario_id": 0,
                "vm_id": "vm0",
                "t5_day": 0,
                "t_hour": 0,
                "cpu_usage": 0.2,
                "mem_usage": 0.4,
                "coverage_us": STEP_US,
            },
            {
                "scenario_id": 0,
                "vm_id": "vm0",
                "t5_day": 1,
                "t_hour": 0,
                "cpu_usage": 0.8,
                "mem_usage": 0.9,
                "coverage_us": 1_000_000,
            },
        ]
    )

    hourly = build_hourly_usage_scenarios(usage)

    assert hourly.loc[0, "cpu_usage"] == pytest.approx(
        (0.2 * STEP_US + 0.8 * 1_000_000) / (STEP_US + 1_000_000)
    )
    assert hourly.loc[0, "mem_usage"] == pytest.approx(
        (0.4 * STEP_US + 0.9 * 1_000_000) / (STEP_US + 1_000_000)
    )


def test_cpu_p100_outlier_is_retained_and_audited_without_cap_in_legacy_helper_mode() -> None:
    requests, _ = _request_for_single_vm(
        [
            _usage_interval(
                start_offset_us=0,
                end_offset_us=STEP_US,
                cpu_usage=0.2,
                mem_usage=0.2,
                max_cpu_usage=2.5,
                max_mem_usage=0.3,
            )
        ]
    )

    assert bool(requests.loc[0, "cpu_p100_gt_one"])
    assert requests.loc[0, "max_cpu_usage"] == pytest.approx(2.5)
    assert requests.loc[0, "q_cpu"] == pytest.approx(2.5)
    assert requests.attrs["vm_request_derivation"]["cpu_p100_gt_one_count"] == 1


def test_assigned_machine_cpu_capacity_audit_reports_task_alloc_fractions() -> None:
    vector = lambda maximum: [0.01 * index for index in range(10)] + [maximum]
    usage = pd.DataFrame(
        [
            {
                "collection_id": 1,
                "instance_index": 0,
                "machine_id": "task-machine",
                "collection_type": 0,
                "start_time": TRACE_START_US,
                "end_time": TRACE_START_US + STEP_US,
                "max_cpu_usage": 0.40,
                "cpu_usage_distribution": vector(0.60),
            },
            {
                "collection_id": 1,
                "instance_index": 0,
                "machine_id": "task-machine",
                "collection_type": 0,
                "start_time": TRACE_START_US,
                "end_time": TRACE_START_US + STEP_US,
                "max_cpu_usage": 0.40,
                "cpu_usage_distribution": np.asarray(vector(0.60)),
            },
            {
                "collection_id": 2,
                "instance_index": 0,
                "machine_id": "alloc-machine",
                "collection_type": "ALLOC_SET",
                "start_time": TRACE_START_US,
                "end_time": TRACE_START_US + STEP_US,
                "max_cpu_usage": 0.30,
                "cpu_usage_distribution": json.dumps(vector(0.20)),
            },
            {
                "collection_id": 3,
                "instance_index": 0,
                "machine_id": "task-machine",
                "collection_type": "JOB",
                "start_time": TRACE_START_US,
                "end_time": TRACE_START_US + STEP_US,
                "max_cpu_usage": 0.20,
                "cpu_usage_distribution": np.asarray(vector(0.30)),
            },
            {
                "collection_id": 4,
                "instance_index": 0,
                "machine_id": "unmatched-machine",
                "collection_type": 0,
                "start_time": TRACE_START_US,
                "end_time": TRACE_START_US + STEP_US,
                "max_cpu_usage": 9.0,
                "cpu_usage_distribution": vector(9.0),
            },
        ]
    )
    machine_events = pd.DataFrame(
        [
            {
                "time": 0,
                "machine_id": "task-machine",
                "event_type": "ADD",
                "capacity_cpu": 0.50,
            },
            {
                "time": 0,
                "machine_id": "alloc-machine",
                "event_type": 1,
                "capacity_cpu": 0.25,
            },
        ]
    )

    audit = audit_assigned_machine_cpu_capacity(
        usage,
        machine_events,
        day_start_us=TRACE_START_US,
        day_end_us=TRACE_START_US + DAY_US,
        selected_keys=usage.loc[usage["collection_id"].isin([1, 2])],
    )

    raw = audit["scopes"]["raw_extracted_top_level_instances"]
    assert raw["instance_count"] == 4
    assert audit["exact_duplicate_source_row_count_removed"] == 1
    assert raw["source_usage_row_count"] == 4
    assert raw["machine_history_unmatched_instance_count"] == 1
    assert raw["maximum_usage"]["over_capacity_instance_count"] == 1
    assert raw["cpu_usage_distribution"]["over_capacity_instance_count"] == 1
    assert raw["either_metric"]["over_capacity_instance_count"] == 2
    assert raw["either_metric"]["fraction_of_all_instances"] == pytest.approx(0.5)
    assert raw["either_metric"]["fraction_of_assessable_instances"] == pytest.approx(2 / 3)
    assert raw["by_instance_kind"]["task"]["instance_count"] == 3
    assert raw["by_instance_kind"]["task"]["either_metric"][
        "fraction_of_all_instances"
    ] == pytest.approx(1 / 3)
    assert raw["by_instance_kind"]["alloc"]["instance_count"] == 1
    assert raw["by_instance_kind"]["alloc"]["either_metric"][
        "fraction_of_all_instances"
    ] == pytest.approx(1.0)
    selected = audit["scopes"]["selected_model_input_instances"]
    assert selected["instance_count"] == 2
    assert selected["either_metric"]["fraction_of_all_instances"] == pytest.approx(1.0)
    assert audit["cpu_usage_distribution_unexpected_length_row_count"] == 0


def test_capacity_change_reports_any_and_all_overlapping_capacity_exceedance() -> None:
    usage = pd.DataFrame(
        [
            {
                "collection_id": 1,
                "instance_index": 0,
                "machine_id": "updated",
                "collection_type": 0,
                "start_time": TRACE_START_US,
                "end_time": TRACE_START_US + STEP_US,
                "max_cpu_usage": 0.40,
                "cpu_usage_distribution": [0.01] * 10 + [0.40],
            }
        ]
    )
    machine_events = pd.DataFrame(
        [
            {
                "time": 0,
                "machine_id": "updated",
                "event_type": "ADD",
                "capacity_cpu": 0.25,
            },
            {
                "time": TRACE_START_US + STEP_US // 2,
                "machine_id": "updated",
                "event_type": "UPDATE",
                "capacity_cpu": 0.50,
            },
        ]
    )

    audit = audit_assigned_machine_cpu_capacity(
        usage,
        machine_events,
        day_start_us=TRACE_START_US,
        day_end_us=TRACE_START_US + DAY_US,
    )

    maximum = audit["scopes"]["raw_extracted_top_level_instances"]["maximum_usage"]
    assert maximum["over_capacity_instance_count"] == 1
    assert maximum["over_all_overlapping_capacities_instance_count"] == 0
    assert maximum["capacity_change_ambiguous_source_row_count"] == 1


def _expected_sampled_keys(keys: list[tuple[int, int]], seed: int, limit: int) -> list[tuple[int, int]]:
    def hash_value(key: tuple[int, int]) -> int:
        payload = f"{seed}:{key[0]}:{key[1]}".encode("utf-8")
        return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big", signed=False)

    return sorted(keys, key=lambda key: (hash_value(key), key[0], key[1]))[:limit]


def test_deterministic_hash_sampling_is_unchanged_and_input_order_independent() -> None:
    keys = [(10, 0), (20, 0), (30, 0), (40, 0)]
    usage = pd.DataFrame(
        [
            _usage_interval(
                collection_id=collection_id,
                instance_index=instance_index,
                start_offset_us=0,
                end_offset_us=300_000_000,
                cpu_usage=0.10,
                mem_usage=0.10,
                max_cpu_usage=0.11,
                max_mem_usage=0.12,
            )
            for collection_id, instance_index in keys
        ]
    )

    first, _ = build_vm_requests(
        usage,
        max_instances=2,
        seed=42,
        min_usage_rows=1,
        day_start_us=TRACE_START_US,
    )
    shuffled, _ = build_vm_requests(
        usage.sample(frac=1.0, random_state=7).reset_index(drop=True),
        max_instances=2,
        seed=42,
        min_usage_rows=1,
        day_start_us=TRACE_START_US,
    )

    actual = list(first[["collection_id", "instance_index"]].itertuples(index=False, name=None))
    shuffled_actual = list(shuffled[["collection_id", "instance_index"]].itertuples(index=False, name=None))
    assert actual == _expected_sampled_keys(keys, seed=42, limit=2)
    assert shuffled_actual == actual


def _machine_events_with_dominant_joint_shape() -> pd.DataFrame:
    rows = [
        # Two machines carry this joint pair for the complete requested day.
        {"time": 0, "machine_id": "dominant-1", "event_type": 1, "capacity_cpu": 0.50, "capacity_mem": 0.25},
        {"time": 0, "machine_id": "dominant-2", "event_type": 1, "capacity_cpu": 0.50, "capacity_mem": 0.25},
        # A competing pair is active for only one hour of the requested day.
        {"time": 0, "machine_id": "short", "event_type": 1, "capacity_cpu": 0.25, "capacity_mem": 0.50},
        {"time": TRACE_START_US + 3_600_000_000, "machine_id": "short", "event_type": 2, "capacity_cpu": None, "capacity_mem": None},
        # Exercise numeric UPDATE plus string REMOVE without changing the winner.
        {"time": 0, "machine_id": "updated", "event_type": 1, "capacity_cpu": 0.10, "capacity_mem": 0.10},
        {"time": TRACE_START_US + 600_000_000, "machine_id": "updated", "event_type": 3, "capacity_cpu": 0.20, "capacity_mem": 0.20},
        {"time": TRACE_START_US + 1_200_000_000, "machine_id": "updated", "event_type": 2, "capacity_cpu": None, "capacity_mem": None},
    ]
    frame = pd.DataFrame(rows)
    frame["platform_id"] = "platform"
    frame["switch_id"] = "switch"
    frame["missing_data_reason"] = None
    return frame


def test_remove_clears_capacity_before_a_sparse_add_starts_new_lifecycle() -> None:
    events = pd.DataFrame(
        [
            {
                "time": 0,
                "machine_id": "stable",
                "event_type": 1,
                "capacity_cpu": 0.50,
                "capacity_mem": 0.25,
            },
            {
                "time": 0,
                "machine_id": "reused",
                "event_type": 1,
                "capacity_cpu": 0.80,
                "capacity_mem": 0.40,
            },
            {
                "time": TRACE_START_US + 100_000_000,
                "machine_id": "reused",
                "event_type": 2,
                "capacity_cpu": None,
                "capacity_mem": None,
            },
            {
                "time": TRACE_START_US + 200_000_000,
                "machine_id": "reused",
                "event_type": 1,
                "capacity_cpu": 0.20,
                "capacity_mem": None,
            },
        ]
    )

    representative, diagnostics = derive_representative_machine_capacity(
        events,
        day_start_us=TRACE_START_US,
        day_end_us=TRACE_START_US + DAY_US,
    )

    assert representative["capacity_cpu"] == pytest.approx(0.50)
    assert representative["capacity_mem"] == pytest.approx(0.25)
    pairs = {
        (row["capacity_cpu"], row["capacity_mem"])
        for row in diagnostics["joint_shape_weights"]
    }
    assert pairs == {(0.50, 0.25), (0.80, 0.40)}
    assert (0.20, 0.40) not in pairs
    assert diagnostics["active_duration_without_positive_joint_capacity_us"] == (
        DAY_US - 200_000_000
    )


def _write_raw_fixture(
    raw_dir: Path,
    *,
    machine_events: pd.DataFrame | None,
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    usage = pd.DataFrame(
        [
            _usage_interval(
                start_offset_us=t5 * STEP_US,
                end_offset_us=(t5 + 1) * STEP_US,
                cpu_usage=0.10,
                mem_usage=0.05,
                max_cpu_usage=0.20,
                max_mem_usage=0.10,
                machine_id="dominant-1",
            )
            for t5 in range(12)
        ]
    )
    usage["cpu_usage_distribution"] = [
        np.asarray([0.02] * 10 + [0.20], dtype=float)
        for _ in range(len(usage))
    ]
    usage.to_parquet(raw_dir / "usage_5min.parquet", index=False)
    pd.DataFrame(
        [
            {
                "collection_id": 1,
                "instance_index": 0,
                "time": TRACE_START_US - 1,
                "event_type": 3,
                "missing_type": None,
                "collection_type": 0,
                "alloc_collection_id": None,
                "alloc_instance_index": None,
                "scheduling_class": 1,
                "priority": 120,
                "machine_id": "dominant-1",
                "resource_request_cpu": 0.15,
                "resource_request_mem": 0.075,
            },
            {
                "collection_id": 1,
                "instance_index": 0,
                "time": TRACE_START_US + 12 * STEP_US,
                "event_type": 6,
                "missing_type": None,
                "collection_type": 0,
                "alloc_collection_id": None,
                "alloc_instance_index": None,
                "scheduling_class": None,
                "priority": None,
                "machine_id": "dominant-1",
                "resource_request_cpu": None,
                "resource_request_mem": None,
            },
        ]
    ).to_parquet(raw_dir / "instance_events.parquet", index=False)
    pd.DataFrame(
        [
            {
                "collection_id": 1,
                "time": TRACE_START_US - 1,
                "event_type": 3,
                "missing_type": None,
                "scheduler": "SCHEDULER_DEFAULT",
                "collection_type": 0,
                "scheduling_class": 1,
                "priority": 120,
                "max_per_machine": None,
                "max_per_switch": None,
            }
        ]
    ).to_parquet(raw_dir / "collection_events.parquet", index=False)
    if machine_events is not None:
        machine_events.to_parquet(raw_dir / "machine_events.parquet", index=False)
    (raw_dir / "metadata.json").write_text(
        json.dumps(
            {
                "trace_start_offset_us": TRACE_START_US,
                "day_start_us": TRACE_START_US,
                "day_end_us": TRACE_START_US + DAY_US,
            }
        ),
        encoding="utf-8",
    )


def test_representative_joint_machine_shape_and_unit_conversion_are_exact(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    _write_raw_fixture(raw_dir, machine_events=_machine_events_with_dominant_joint_shape())

    build_toy_instance(
        raw_dir=raw_dir,
        output_dir=output_dir,
        max_instances=1,
        seed=42,
        num_servers=2,
        num_scenarios=1,
    )

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((output_dir / "scaling_diagnostics.json").read_text(encoding="utf-8"))
    preprocessing = json.loads(
        (output_dir / "preprocessing_diagnostics.json").read_text(encoding="utf-8")
    )
    requests = pd.read_csv(output_dir / "vm_requests.csv")
    usage = pd.read_csv(output_dir / "vm_usage_5min_scenarios.csv")
    servers = pd.read_csv(output_dir / "servers.csv")

    assert metadata["representative_machine_capacity_raw"] == {"cpu": 0.50, "mem": 0.25}
    assert metadata["unit_conversion"] == {"cpu_divisor": 0.50, "mem_divisor": 0.25}
    assert metadata["representative_machine_selection_method"]
    assert metadata["preprocessing_diagnostics_file"] == "preprocessing_diagnostics.json"
    assert metadata["resource_scale"] == {
        "automatic_utilization_calibration_applied": False,
        "cpu_divisor": "representative_machine_capacity_raw.cpu",
        "mem_divisor": "representative_machine_capacity_raw.mem",
        "converted_server_capacity": {"cpu": 1.0, "mem": 1.0},
    }
    assert diagnostics["num_machine_capacity_pairs"] == 4
    assert diagnostics["selected_capacity_pair_weight"] == pytest.approx(2 * DAY_US)
    assert "usage_interval_overlap_audit" in preprocessing
    assert "event_state_ambiguity_audit" in preprocessing
    assert "cpu_p100_audit" in preprocessing
    assert "episode_quality_filter" in preprocessing
    assert "assigned_machine_cpu_capacity_audit" in preprocessing
    assert "distribution[10]" in preprocessing["policies"]["cpu_maximum"]
    capacity_audit = preprocessing["assigned_machine_cpu_capacity_audit"]
    raw_capacity_scope = capacity_audit["scopes"]["raw_extracted_top_level_instances"]
    assert raw_capacity_scope["instance_count"] == 1
    assert raw_capacity_scope["maximum_usage"]["over_capacity_instance_count"] == 0
    assert capacity_audit["cpu_usage_distribution_column_present"] is True
    assert "assigned_machine_cpu_capacity_audit" in metadata
    assert requests.loc[0, "resource_request_cpu"] == pytest.approx(0.15 / 0.50)
    assert requests.loc[0, "resource_request_mem"] == pytest.approx(0.075 / 0.25)
    assert requests.loc[0, "max_cpu_usage"] == pytest.approx(0.20 / 0.50)
    assert requests.loc[0, "max_mem_usage"] == pytest.approx(0.10 / 0.25)
    assert requests.loc[0, "q_cpu"] == pytest.approx(0.20 / 0.50)
    assert requests.loc[0, "q_mem"] == pytest.approx(0.10 / 0.25)
    assert usage.loc[0, "cpu_usage"] == pytest.approx(0.10 / 0.50)
    assert usage.loc[0, "mem_usage"] == pytest.approx(0.05 / 0.25)
    assert servers["C_cpu"].tolist() == pytest.approx([1.0, 1.0])
    assert servers["C_mem"].tolist() == pytest.approx([1.0, 1.0])


def test_representative_shape_tie_break_is_lexicographic(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    tied = pd.DataFrame(
        [
            {"time": 0, "machine_id": "z", "event_type": "ADD", "capacity_cpu": 0.60, "capacity_mem": 0.30},
            {"time": 0, "machine_id": "a", "event_type": "ADD", "capacity_cpu": 0.30, "capacity_mem": 0.60},
            # A short-lived third machine exercises string UPDATE/REMOVE.
            {"time": TRACE_START_US, "machine_id": "short", "event_type": "ADD", "capacity_cpu": 0.10, "capacity_mem": 0.10},
            {"time": TRACE_START_US + 60_000_000, "machine_id": "short", "event_type": "UPDATE", "capacity_cpu": 0.20, "capacity_mem": 0.20},
            {"time": TRACE_START_US + 120_000_000, "machine_id": "short", "event_type": "REMOVE", "capacity_cpu": None, "capacity_mem": None},
        ]
    )
    tied["platform_id"] = "platform"
    tied["switch_id"] = "switch"
    tied["missing_data_reason"] = None
    _write_raw_fixture(raw_dir, machine_events=tied.sample(frac=1.0, random_state=7))
    usage = pd.read_parquet(raw_dir / "usage_5min.parquet")
    usage["machine_id"] = "a"
    usage.to_parquet(raw_dir / "usage_5min.parquet", index=False)
    events = pd.read_parquet(raw_dir / "instance_events.parquet")
    events["machine_id"] = "a"
    events.to_parquet(raw_dir / "instance_events.parquet", index=False)

    build_toy_instance(
        raw_dir=raw_dir,
        output_dir=output_dir,
        max_instances=1,
        seed=42,
        num_servers=1,
        num_scenarios=1,
    )

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["representative_machine_capacity_raw"] == {"cpu": 0.30, "mem": 0.60}


def test_vms_above_representative_capacity_are_dropped_without_clipping() -> None:
    converted_requests = pd.DataFrame(
        [
            {"vm_id": "fit", "class": "on_demand", "q_cpu": 0.8, "q_mem": 0.9},
            {"vm_id": "exact", "class": "spot", "q_cpu": 1.0, "q_mem": 1.0},
            {"vm_id": "cpu-big", "class": "spot", "q_cpu": 1.1, "q_mem": 0.2},
            {"vm_id": "mem-big", "class": "batch_candidate", "q_cpu": 0.2, "q_mem": 1.1},
            {"vm_id": "both-big", "class": "batch_candidate", "q_cpu": 1.2, "q_mem": 1.3},
        ]
    )
    raw_requests = converted_requests.copy()
    converted_usage = pd.DataFrame(
        {"vm_id": converted_requests["vm_id"], "cpu_usage": 0.1, "mem_usage": 0.1}
    )
    raw_usage = converted_usage.copy()

    raw_kept, raw_usage_kept, converted_kept, usage_kept, diagnostics = (
        drop_vms_exceeding_representative_capacity(
            raw_requests,
            raw_usage,
            converted_requests,
            converted_usage,
        )
    )

    assert set(converted_kept["vm_id"]) == {"fit", "exact"}
    assert set(raw_kept["vm_id"]) == {"fit", "exact"}
    assert set(raw_usage_kept["vm_id"]) == {"fit", "exact"}
    assert set(usage_kept["vm_id"]) == {"fit", "exact"}
    assert converted_kept.set_index("vm_id").loc["exact", "q_cpu"] == pytest.approx(1.0)
    assert diagnostics["dropped_vm_count"] == 3
    assert diagnostics["dropped_cpu_exceeds_count"] == 2
    assert diagnostics["dropped_mem_exceeds_count"] == 2
    assert diagnostics["dropped_both_exceed_count"] == 1
    assert diagnostics["configured_resource_clipping_applied"] is False
    assert diagnostics["sampling_replacement_applied"] is False


def test_resource_conversion_does_not_depend_on_server_count(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_raw_fixture(raw_dir, machine_events=_machine_events_with_dominant_joint_shape())
    # This demand is deliberately above one converted server's old 75% target,
    # so the retired target-utilization calibration would produce different
    # values for one versus six servers.
    raw_usage = pd.read_parquet(raw_dir / "usage_5min.parquet")
    raw_usage["cpu_usage"] = 0.40
    raw_usage["mem_usage"] = 0.20
    raw_usage["max_cpu_usage"] = 0.45
    raw_usage["cpu_usage_distribution"] = [
        np.asarray([0.08] * 10 + [0.45], dtype=float)
        for _ in range(len(raw_usage))
    ]
    raw_usage["max_mem_usage"] = 0.22
    raw_usage.to_parquet(raw_dir / "usage_5min.parquet", index=False)
    outputs = []
    for num_servers in (1, 6):
        output_dir = tmp_path / f"processed-{num_servers}"
        build_toy_instance(
            raw_dir=raw_dir,
            output_dir=output_dir,
            max_instances=1,
            seed=42,
            num_servers=num_servers,
            num_scenarios=1,
        )
        outputs.append(
            (
                pd.read_csv(output_dir / "vm_requests.csv")[["q_cpu", "q_mem"]],
                pd.read_csv(output_dir / "vm_usage_5min_scenarios.csv")[["cpu_usage", "mem_usage"]],
            )
        )

    pd.testing.assert_frame_equal(outputs[0][0], outputs[1][0])
    pd.testing.assert_frame_equal(outputs[0][1], outputs[1][1])


@pytest.mark.parametrize("empty_machine_table", [False, True])
def test_missing_or_empty_machine_events_fails_clearly(tmp_path: Path, empty_machine_table: bool) -> None:
    raw_dir = tmp_path / "raw"
    machine_events = None
    if empty_machine_table:
        machine_events = pd.DataFrame(
            columns=[
                "time",
                "machine_id",
                "event_type",
                "capacity_cpu",
                "capacity_mem",
                "platform_id",
                "switch_id",
                "missing_data_reason",
            ]
        )
    _write_raw_fixture(raw_dir, machine_events=machine_events)

    with pytest.raises((FileNotFoundError, ValueError), match=r"(?i)machine[_ ]events|machine event"):
        build_toy_instance(
            raw_dir=raw_dir,
            output_dir=tmp_path / "processed",
            max_instances=1,
            seed=42,
            num_servers=1,
            num_scenarios=1,
        )


def test_server_min_loader_uses_csv_q_without_recomputing_it() -> None:
    requests = pd.DataFrame(
        [
            {
                "vm_id": "od0",
                "class": "on_demand",
                "arrival_t5": 0,
                "departure_t5": 6,
                "q_cpu": 0.91,
                "q_mem": 0.82,
                "resource_request_cpu": 0.20,
                "resource_request_mem": 0.30,
            },
            {
                "vm_id": "batch0",
                "class": "batch_candidate",
                "arrival_t5": 0,
                "departure_t5": 6,
                "q_cpu": 0.73,
                "q_mem": 0.64,
                "resource_request_cpu": 0.10,
                "resource_request_mem": 0.10,
            },
        ]
    )
    usage = pd.DataFrame(
        [
            {"vm_id": "od0", "t5_day": 0, "cpu_usage": 0.40, "mem_usage": 0.50},
            {"vm_id": "batch0", "t5_day": 0, "cpu_usage": 0.20, "mem_usage": 0.30},
        ]
    )

    q_od, _, _, _, _ = _build_service_scenarios(
        requests,
        usage,
        ["od0"],
        [],
        {"od0": [0]},
        [0],
        t5_per_slot=6,
        seed=42,
        cpu_sigma=0.22,
        mem_sigma=0.08,
    )
    _, q_batch, _, _, _ = _build_batch_families(
        requests,
        usage,
        ["batch0"],
        t5_per_slot=6,
        max_families=10,
        pair_round_digits=12,
    )

    assert q_od[("od0", CPU)] == pytest.approx(0.91)
    assert q_od[("od0", MEM)] == pytest.approx(0.82)
    assert q_batch[("batch000", CPU)] == pytest.approx(0.73)
    assert q_batch[("batch000", MEM)] == pytest.approx(0.64)


def _write_nyiso_price_fixture(path: Path) -> None:
    local = pd.date_range("2019-06-01", periods=48, freq="30min", tz="America/New_York")
    pd.DataFrame(
        {
            "timestamp_utc": local.tz_convert("UTC"),
            "site_id": "NYC",
            "rt_lbmp_usd_per_mwh": np.full(48, 50.0),
        }
    ).to_csv(path, index=False)


def _one_vm_server_min_config(tmp_path: Path, processed_dir: Path) -> dict[str, object]:
    config = copy.deepcopy(load_config(SERVER_MIN_ROOT / "configs" / "baseline.yaml"))
    prices = tmp_path / "prices.csv"
    _write_nyiso_price_fixture(prices)
    config["workspace_root"] = str(tmp_path)
    config["experiment"]["num_scenarios"] = 1
    config["experiment"]["num_servers"] = 1
    config["experiment"]["class_counts"] = {
        "on_demand": 1,
        "spot": 0,
        "batch_jobs": 0,
    }
    config["data"]["google_dir"] = str(processed_dir)
    config["data"]["nyiso_prices"] = str(prices)
    config["data"]["additional_source_files"] = []
    config["electricity_price"]["known_interpolated_raw_slots"] = []
    return config


def test_processed_fixture_builds_server_min_model_without_rescaling_csv_q(tmp_path: Path) -> None:
    pytest.importorskip("gurobipy")
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    _write_raw_fixture(raw_dir, machine_events=_machine_events_with_dominant_joint_shape())
    build_toy_instance(
        raw_dir=raw_dir,
        output_dir=processed_dir,
        max_instances=1,
        seed=42,
        num_servers=1,
        num_scenarios=1,
    )
    request = pd.read_csv(processed_dir / "vm_requests.csv").iloc[0]
    instance = build_instance(_one_vm_server_min_config(tmp_path, processed_dir))

    vm_id = instance.I[0]
    assert instance.q_od[(vm_id, CPU)] == pytest.approx(request["q_cpu"])
    assert instance.q_od[(vm_id, MEM)] == pytest.approx(request["q_mem"])

    artifacts = build_model(instance, name="google2019_resource_semantics_smoke")
    artifacts.model.Params.OutputFlag = 0
    assert artifacts.model.NumVars > 0
    assert artifacts.model.NumConstrs > 0
    artifacts.model.dispose()
