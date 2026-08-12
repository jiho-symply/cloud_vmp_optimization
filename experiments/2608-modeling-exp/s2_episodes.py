"""instance_events와 usage의 parent instance 키 → 실행 회차 episode parquet.
SCHEDULE/UPDATE_RUNNING과 terminal event로 episode 경계를 재구성하는 단계다.
검증 수치: horizon overlap 25351, parent instance 5000, usage-inferred 0.
추출 SQL이 이미 time < day_end_us로 필터링하며 time NULL은 없다."""

from __future__ import annotations

import numpy as np
import pandas as pd

from paths import S2_EPISODES_PATH, WORK_DIR
from raw_tables import load_instance_events, load_usage


DAY_START_US = 600_000_000
DAY_END_US = 87_000_000_000
HORIZON_CENSORED = "HORIZON_CENSORED"

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
EPISODE_COLUMNS = [
    "collection_id",
    "instance_index",
    "episode_index",
    "lifecycle_start_time_us",
    "episode_start_time_us",
    "episode_end_time_us",
    "termination_time_us",
    "termination_reason",
    "horizon_censored",
    "termination_time_uncertain",
    "episode_start_inferred_from_update",
    "episode_start_inferred_from_usage",
]


def _instance_event_name(value: object) -> str:
    """instance event 값을 문서화된 사건 이름으로 반환한다."""
    return INSTANCE_EVENT_TYPE_BY_CODE[int(value)]


def _prepare_events(instance_events: pd.DataFrame) -> pd.DataFrame:
    """episode 재구성에 필요한 사건 이름을 붙인 DataFrame을 반환한다."""
    events = instance_events[
        ["collection_id", "instance_index", "time", "event_type"]
    ].copy()
    names = events["event_type"].map(_instance_event_name)
    events["event_name"] = names
    return events


def _append_episode(
    rows: list[dict[str, object]],
    key: tuple[object, object],
    episode_index: int,
    lifecycle_start: int,
    lifecycle_end: int,
    termination_reason: str,
    start_inferred_from_update: bool,
) -> None:
    """horizon과 겹치는 episode 행을 rows에 추가한다."""
    episode_start = max(int(DAY_START_US), int(lifecycle_start))
    episode_end = min(int(DAY_END_US), int(lifecycle_end))
    if episode_end <= episode_start:
        return
    horizon_censored = termination_reason == HORIZON_CENSORED
    rows.append(
        {
            "collection_id": key[0],
            "instance_index": key[1],
            "episode_index": int(episode_index),
            "lifecycle_start_time_us": int(lifecycle_start),
            "episode_start_time_us": episode_start,
            "episode_end_time_us": episode_end,
            "termination_time_us": np.nan if horizon_censored else int(lifecycle_end),
            "termination_reason": termination_reason,
            "horizon_censored": bool(horizon_censored),
            "termination_time_uncertain": termination_reason == "LOST",
            "episode_start_inferred_from_update": bool(start_inferred_from_update),
            "episode_start_inferred_from_usage": False,
        }
    )


def _terminal_reason(names: set[str]) -> str | None:
    """같은 시각 terminal 사건의 lifecycle precedence 최우선 이름을 반환한다."""
    terminal_names = sorted(
        names & TERMINAL_INSTANCE_EVENTS,
        key=lambda name: INSTANCE_EVENT_LIFECYCLE_ORDER[name],
    )
    return terminal_names[-1] if terminal_names else None


def _episode_start_name(names: set[str]) -> str | None:
    """episode를 시작할 사건 이름을 SCHEDULE 우선으로 반환한다."""
    if "SCHEDULE" in names:
        return "SCHEDULE"
    if "UPDATE_RUNNING" in names:
        return "UPDATE_RUNNING"
    return None


def _episodes_for_key(
    key: tuple[object, object],
    history: pd.DataFrame,
) -> list[dict[str, object]]:
    """한 parent instance의 episode 행 목록을 반환한다."""
    rows: list[dict[str, object]] = []
    running = False
    running_start = 0
    start_inferred_from_update = False
    episode_index = -1
    for event_time, cohort in history.groupby("time", sort=True, dropna=False):
        event_time = int(event_time)
        names = set(cohort["event_name"].astype(str))
        # 동시 terminal 사건은 lifecycle precedence 최우선 사건으로 닫는다.
        reason = _terminal_reason(names)
        if reason is not None:
            if running and event_time > running_start:
                _append_episode(
                    rows, key, episode_index, running_start, event_time,
                    reason, start_inferred_from_update,
                )
            running = False
        # SCHEDULE이 없으면 UPDATE_RUNNING을 방어적 시작점으로 삼는다.
        start_name = _episode_start_name(names)
        if start_name is not None and not running:
            episode_index += 1
            running = True
            running_start = event_time
            start_inferred_from_update = start_name == "UPDATE_RUNNING"
    if running and running_start < DAY_END_US:
        _append_episode(
            rows, key, episode_index, running_start, DAY_END_US,
            HORIZON_CENSORED, start_inferred_from_update,
        )
    return rows


def _build_lifecycle_episodes(
    instance_events: pd.DataFrame,
    usage: pd.DataFrame,
) -> pd.DataFrame:
    """usage parent instance별로 horizon과 겹치는 episode를 반환한다."""
    events = _prepare_events(instance_events)
    usage_keys = usage[["collection_id", "instance_index"]].drop_duplicates()
    histories = {
        key: group
        for key, group in events.groupby(
            ["collection_id", "instance_index"], sort=False, dropna=False
        )
    }
    rows: list[dict[str, object]] = []
    for key_row in usage_keys.itertuples(index=False):
        key = (key_row.collection_id, key_row.instance_index)
        history = histories.get(key, pd.DataFrame(columns=events.columns))
        rows.extend(_episodes_for_key(key, history))
    episodes = pd.DataFrame(rows, columns=EPISODE_COLUMNS)
    return episodes.sort_values(
        ["collection_id", "instance_index", "episode_index"], kind="stable"
    ).reset_index(drop=True)


def main() -> None:
    """s2 episode 산출물을 디스크에 저장한다."""
    instance_events = load_instance_events()
    usage = load_usage()
    episodes = _build_lifecycle_episodes(instance_events, usage)
    usage_keys = usage[["collection_id", "instance_index"]].drop_duplicates()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    episodes.to_parquet(S2_EPISODES_PATH, index=False)
    print(f"horizon_overlap_episode_count={len(episodes)}")
    print(f"parent_instance_count={len(usage_keys)}")
    print(
        "usage_inferred_episode_count="
        f"{int(episodes['episode_start_inferred_from_usage'].sum())}"
    )


if __name__ == "__main__":
    main()
