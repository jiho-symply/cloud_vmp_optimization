from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from google2019_toy.build_toy_instance import (
    HORIZON_CENSORED,
    build_spot_preemption_scenarios,
    build_vm_requests,
)


DAY_START_US = 600 * 1_000_000
DAY_US = 24 * 60 * 60 * 1_000_000
DAY_END_US = DAY_START_US + DAY_US
STEP_US = 300 * 1_000_000


def _distribution(p100: float = 0.40) -> list[float]:
    return [0.01 * index for index in range(10)] + [p100]


def _usage_row(
    collection_id: int,
    start_offset_us: int,
    end_offset_us: int,
    *,
    machine_id: str = "machine-ok",
    distribution: object = None,
    raw_max_cpu: float = 0.45,
) -> dict[str, object]:
    start = DAY_START_US + start_offset_us
    end = DAY_START_US + end_offset_us
    return {
        "collection_id": collection_id,
        "instance_index": 0,
        "machine_id": machine_id,
        "collection_type": 0,
        "alloc_collection_id": None,
        "alloc_instance_index": None,
        "start_time": start,
        "end_time": end,
        "clipped_start_time": max(start, DAY_START_US),
        "clipped_end_time": min(end, DAY_END_US),
        "overlap_us": min(end, DAY_END_US) - max(start, DAY_START_US),
        "cpu_usage": 0.20,
        "mem_usage": 0.30,
        "max_cpu_usage": raw_max_cpu,
        "max_mem_usage": 0.35,
        "assigned_memory": 0.40,
        "cpu_usage_distribution": (
            _distribution() if distribution is None else distribution
        ),
    }


def _event(
    collection_id: int,
    offset_us: int,
    event_type: object,
    *,
    priority: float | None = 1,
    request_cpu: float | None = 0.10,
    request_mem: float | None = 0.10,
) -> dict[str, object]:
    return {
        "collection_id": collection_id,
        "instance_index": 0,
        "time": DAY_START_US + offset_us,
        "event_type": event_type,
        "missing_type": 0,
        "machine_id": "machine-ok",
        "priority": priority,
        "scheduling_class": 1,
        "resource_request_cpu": request_cpu,
        "resource_request_mem": request_mem,
    }


def _machine_event(
    machine_id: str,
    time_us: int,
    event_type: object,
    capacity_cpu: float | None,
) -> dict[str, object]:
    return {
        "time": time_us,
        "machine_id": machine_id,
        "event_type": event_type,
        "capacity_cpu": capacity_cpu,
        "capacity_mem": 1.0 if capacity_cpu is not None else None,
    }


def _build(
    usage_rows: list[dict[str, object]],
    event_rows: list[dict[str, object]],
    machine_rows: list[dict[str, object]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if machine_rows is None:
        machine_rows = [_machine_event("machine-ok", 0, "ADD", 1.0)]
    return build_vm_requests(
        pd.DataFrame(usage_rows),
        events=pd.DataFrame(event_rows),
        machine_events=pd.DataFrame(machine_rows),
        max_instances=100,
        seed=42,
        min_usage_rows=1,
        day_start_us=DAY_START_US,
        day_end_us=DAY_END_US,
        episode_unit=True,
    )


@pytest.mark.parametrize(
    ("event_type", "reason"),
    [(4, "EVICT"), (5, "FAIL"), (6, "FINISH"), (7, "KILL"), (8, "LOST")],
)
def test_terminal_event_codes_close_a_lifecycle_episode(
    event_type: int,
    reason: str,
) -> None:
    requests, _ = _build(
        [_usage_row(1, 0, STEP_US)],
        [_event(1, 0, "SCHEDULE"), _event(1, STEP_US, event_type)],
    )

    episode = requests.iloc[0]
    assert episode["termination_reason"] == reason
    assert episode["episode_start_time_us"] == DAY_START_US
    assert episode["episode_end_time_us"] == DAY_START_US + STEP_US
    assert not bool(episode["horizon_censored"])
    assert bool(episode["termination_time_uncertain"]) is (reason == "LOST")


def test_usage_inferred_episode_without_terminal_spans_the_full_horizon() -> None:
    requests, _ = _build([_usage_row(7, 2 * STEP_US, 3 * STEP_US)], [])

    episode = requests.iloc[0]
    assert episode["termination_reason"] == HORIZON_CENSORED
    assert bool(episode["horizon_censored"])
    assert bool(episode["episode_start_inferred_from_usage"])
    assert episode["episode_start_time_us"] == DAY_START_US
    assert episode["episode_end_time_us"] == DAY_END_US
    assert episode["arrival_t5"] == 0
    assert episode["departure_t5"] == 288
    assert episode["lifetime_t5"] == 288


def test_empty_distribution_drops_only_the_affected_sibling_episode() -> None:
    events = [
        _event(1, 0, "SCHEDULE"),
        _event(1, STEP_US, "EVICT"),
        _event(1, STEP_US, "SCHEDULE", request_cpu=0.20),
        _event(1, 2 * STEP_US, "FINISH"),
    ]
    requests, observed = _build(
        [
            _usage_row(1, 0, STEP_US, distribution=_distribution(0.30)),
            _usage_row(1, STEP_US, 2 * STEP_US, distribution=[]),
        ],
        events,
    )

    assert requests[["collection_id", "instance_index", "episode_index"]].values.tolist() == [
        [1, 0, 0]
    ]
    assert observed["episode_index"].unique().tolist() == [0]
    quality = requests.attrs["episode_quality_filter"]
    assert quality["episode_count_with_usage_before_quality_filter"] == 2
    assert quality["episode_count_retained_after_quality_filter"] == 1
    assert quality["drop_reason_counts_nonexclusive"][
        "empty_cpu_usage_distribution"
    ] == 1


def test_cpu_maximum_uses_last_distribution_element_and_ignores_raw_maximum() -> None:
    # An intentionally non-monotone fixture distinguishes p100 (last) from
    # max(array), while raw maximum_usage is deliberately much larger.
    row = _usage_row(
        2,
        0,
        STEP_US,
        distribution=[0.90] + [0.10] * 9 + [0.40],
        raw_max_cpu=9.0,
    )
    requests, _ = _build(
        [row, dict(row)],
        [_event(2, 0, "SCHEDULE"), _event(2, STEP_US, "FINISH")],
    )

    assert requests.loc[0, "max_cpu_usage"] == pytest.approx(0.40)
    assert requests.loc[0, "q_cpu"] == pytest.approx(0.40)
    assert not bool(requests.loc[0, "cpu_p100_gt_one"])
    assert requests.attrs["episode_quality_filter"][
        "exact_duplicate_source_row_count_removed"
    ] == 1


def test_missing_and_incomplete_machine_capacity_drop_whole_episodes() -> None:
    usage = [
        _usage_row(1, 0, STEP_US, machine_id="machine-ok"),
        _usage_row(2, 0, STEP_US, machine_id="machine-missing"),
        _usage_row(3, 0, STEP_US, machine_id="machine-gap"),
    ]
    events = []
    for collection_id in (1, 2, 3):
        events.extend(
            [
                _event(collection_id, 0, "SCHEDULE"),
                _event(collection_id, STEP_US, "FINISH"),
            ]
        )
    machines = [
        _machine_event("machine-ok", 0, "ADD", 1.0),
        _machine_event("machine-gap", 0, "ADD", 1.0),
        _machine_event("machine-gap", DAY_START_US + STEP_US // 2, "REMOVE", None),
    ]

    requests, _ = _build(usage, events, machines)

    assert requests["collection_id"].tolist() == [1]
    quality = requests.attrs["episode_quality_filter"]
    assert quality["episode_count_dropped"] == 2
    assert quality["drop_reason_counts_nonexclusive"]["machine_history_missing"] == 1
    # The no-history episode is also incomplete, so this count is intentionally
    # nonexclusive and includes both rejected episodes.
    assert quality["drop_reason_counts_nonexclusive"]["machine_capacity_incomplete"] == 2


def test_evict_preemption_is_scoped_to_its_episode_not_its_parent_instance() -> None:
    events = [
        _event(5, 0, "SCHEDULE"),
        _event(5, STEP_US, "EVICT"),
        _event(5, STEP_US, "SCHEDULE"),
        _event(5, 2 * STEP_US, "FINISH"),
    ]
    requests, observed = _build(
        [
            _usage_row(5, 0, STEP_US, distribution=_distribution(0.30)),
            _usage_row(5, STEP_US, 2 * STEP_US, distribution=_distribution(0.35)),
        ],
        events,
    )

    assert len(requests) == 2
    paths = build_spot_preemption_scenarios(
        requests,
        observed,
        events=pd.DataFrame(events),
        num_scenarios=1,
        seed=42,
    )
    later_vm_id = requests.loc[requests["episode_index"].eq(1), "vm_id"].iloc[0]
    later = paths.loc[
        paths["scenario_id"].eq(0) & paths["vm_id"].eq(later_vm_id)
    ]

    assert later["t5_day"].tolist() == [1]
    assert later["active"].tolist() == [1]
    assert later["preempted"].tolist() == [0]
