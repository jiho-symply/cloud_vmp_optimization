"""s3 usage와 raw event → 품질·bucket·상태 필터를 거친 episode usage를 저장한다.
결과는 결정론적 blake2b 순서의 상위 5,000 episode다.
검증: quality 20065/24883, event ambiguity 57/5471, selected 5000."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .paths import WORK_DIR
from .raw_tables import load_collection_events, load_instance_events


S3_PATH = WORK_DIR / "s3_episode_usage.parquet"
OUTPUT_PATH = WORK_DIR / "s4_selected_episodes.parquet"
EPISODE_KEYS = ["collection_id", "instance_index", "episode_index"]
INSTANCE_STATE = [
    "priority", "scheduling_class", "resource_request_cpu", "resource_request_mem",
]
COLLECTION_STATE = ["scheduler", "collection_type", "priority", "scheduling_class"]
EVENT_NAMES = {
    0: "SUBMIT", 1: "QUEUE", 2: "ENABLE", 3: "SCHEDULE", 4: "EVICT",
    5: "FAIL", 6: "FINISH", 7: "KILL", 8: "LOST", 9: "UPDATE_PENDING",
    10: "UPDATE_RUNNING",
}
EVENT_PRECEDENCE = {
    "EVICT": 0, "FAIL": 1, "FINISH": 2, "KILL": 3, "LOST": 4,
    "SUBMIT": 5, "QUEUE": 6, "ENABLE": 7, "UPDATE_PENDING": 8,
    "SCHEDULE": 9, "UPDATE_RUNNING": 10,
}
STATE_EVENTS = {"SCHEDULE", "UPDATE_RUNNING"}


def _stable_key_hash(collection_id: object, instance_index: object,
                     episode_index: object) -> int:
    """episode key의 8바이트 blake2b 정수 hash를 반환한다."""
    payload = f"42:{collection_id}:{instance_index}:{episode_index}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _value_key(value: object) -> str:
    """event tie-break용 결정론적 값 key를 반환한다."""
    if pd.isna(value):
        return "0:<NULL>"
    if isinstance(value, (bool, np.bool_)):
        return f"1:{int(value)}"
    if isinstance(value, (int, np.integer, float, np.floating)):
        return f"2:{float(value):.17g}"
    return f"3:{str(value)}"


def _normalize_events(events: pd.DataFrame, collection: bool) -> pd.DataFrame:
    """raw instance 또는 collection event를 sparse-state 입력으로 반환한다."""
    if collection:
        columns = [
            "collection_id", "time", "event_type", "missing_type", "scheduler",
            "collection_type", "scheduling_class", "priority",
        ]
        out = events[columns].copy()
        group_keys = ["collection_id"]
        sort_values = ["scheduler", "collection_type", "scheduling_class", "priority"]
    else:
        columns = [
            "collection_id", "instance_index", "time", "event_type", "missing_type",
            "machine_id", "priority", "scheduling_class", "resource_request_cpu",
            "resource_request_mem",
        ]
        out = events[columns].copy()
        group_keys = ["collection_id", "instance_index"]
        sort_values = [
            "normal_missing_row_rank", "machine_id", "priority", "scheduling_class",
            "resource_request_cpu", "resource_request_mem",
        ]
    out = out.drop_duplicates().copy()
    out["event_name"] = out["event_type"].map(EVENT_NAMES).fillna("UNKNOWN")
    out["event_precedence"] = out["event_name"].map(EVENT_PRECEDENCE).fillna(-1)
    out["normal_missing_row_rank"] = (
        out["missing_type"].isna() | out["missing_type"].eq(0)
    ).astype(int)
    out["_event_key"] = out["event_name"].map(_value_key)
    for column in sort_values:
        out[f"_{column}_key"] = out[column].map(_value_key)
    if "_machine_id_key" not in out:
        out["machine_id"] = np.nan
        out["_machine_id_key"] = "0:<NULL>"
    sort_columns = [*group_keys, "time", "event_precedence"]
    sort_columns.extend(f"_{column}_key" for column in sort_values)
    return out.sort_values(
        sort_columns, kind="stable", na_position="first"
    ).reset_index(drop=True)
def _resolve_cohort(records: list[tuple], machine_keys: set[str],
                    state_columns: list[str], scheduler_conflict: bool,
                    current: dict[str, object],
                    unresolved: set[str]) -> tuple[dict[str, object], set[str], bool]:
    """한 timestamp cohort를 반영한 sparse state와 모호성 상태를 반환한다."""
    matches = [int(r[0] in STATE_EVENTS and r[4] in machine_keys) for r in records]
    candidates = [record for record, match in zip(records, matches) if match]
    candidates = candidates or records
    max_precedence = max(record[1] for record in candidates)
    candidates = [record for record in candidates if record[1] == max_precedence]
    max_missing_rank = max(record[2] for record in candidates)
    candidates = [record for record in candidates if record[2] == max_missing_rank]
    cohorts: dict[object, list[tuple]] = {}
    for record in candidates:
        key = record[0] if scheduler_conflict else (record[0], record[4])
        cohorts.setdefault(key, []).append(record)
    conflicts: set[str] = set()
    for rows in cohorts.values():
        for index, column in enumerate(state_columns):
            values = {row[6][index] for row in rows if row[6][index] != "0:<NULL>"}
            conflicts.update({column} if len(values) > 1 else set())
    ordered = sorted(
        zip(records, matches),
        key=lambda item: (item[1], item[0][1], item[0][2], _value_key(item[0][0]),
                          item[0][4], *item[0][6]),
    )
    overwritten: set[str] = set()
    for record, _ in ordered:
        for index, column in enumerate(state_columns):
            value = record[5][index]
            if not pd.isna(value):
                current[column] = value
                overwritten.add(column)
    unresolved.difference_update(overwritten)
    unresolved.update(conflicts)
    has_state = any(pd.notna(value) for value in current.values())
    return current, unresolved, has_state

def _resolve_sparse_state_at_time(history: pd.DataFrame, arrival_time: int,
                                  state_columns: list[str],
                                  arrival_machines: tuple[object, ...] = (),
                                  scheduler_conflict: bool = False) -> bool:
    """arrival 시점 sparse state가 모호한지 반환한다."""
    if history.empty:
        return False
    machine_keys = {_value_key(value) for value in arrival_machines}
    columns = ["time", "event_name", "event_precedence", "normal_missing_row_rank",
               "machine_id", "_machine_id_key"] + state_columns
    rows = []
    for row in history[columns].itertuples(index=False, name=None):
        values = tuple(row[6:])
        value_keys = tuple(_value_key(value) for value in values)
        rows.append((row[0], row[1], row[2], row[3], row[4], row[5], values, value_keys))
    rows.sort(key=lambda row: -np.inf if pd.isna(row[0]) else row[0])
    current = {column: np.nan for column in state_columns}
    unresolved: set[str] = set()
    latest: bool | None = None
    position = 0
    while position < len(rows):
        start = position
        event_time = rows[position][0]
        position += 1
        while position < len(rows) and rows[position][0] == event_time:
            position += 1
        if not pd.isna(event_time) and int(event_time) > arrival_time and latest is not None:
            return latest
        cohort = [row[1:] for row in rows[start:position]]
        current, unresolved, has_state = _resolve_cohort(
            cohort, machine_keys, state_columns, scheduler_conflict, current, unresolved
        )
        if has_state:
            snapshot = bool(unresolved)
            if pd.isna(event_time) or int(event_time) <= arrival_time:
                latest = snapshot
            else:
                return snapshot
    return bool(latest) if latest is not None else False

def _event_state_mask(candidates: pd.DataFrame, instance_events: pd.DataFrame,
                      collection_events: pd.DataFrame) -> pd.Series:
    """candidate별 arrival·scheduler state 모호성 mask를 반환한다."""
    instance = _normalize_events(instance_events, collection=False)
    collection = _normalize_events(collection_events, collection=True)
    instance_groups = {
        key: instance.loc[index]
        for key, index in instance.groupby(
            ["collection_id", "instance_index"], sort=False
        ).groups.items()
    }
    collection_groups = {
        key: collection.loc[index]
        for key, index in collection.groupby("collection_id", sort=False).groups.items()
    }
    ambiguous: list[bool] = []
    for row in candidates.itertuples(index=False):
        key = (row.collection_id, row.instance_index)
        history = instance_groups.get(key, instance.iloc[0:0])
        history = history.loc[
            history["time"].lt(int(row.episode_end_time_us)) | history["time"].isna()
        ]
        machine_ids = row.arrival_machine_ids
        if not isinstance(machine_ids, tuple):
            machine_ids = tuple(np.asarray(machine_ids).tolist())
        arrival_ambiguous = _resolve_sparse_state_at_time(
            history, int(row.episode_start_time_us), INSTANCE_STATE, machine_ids
        )
        scheduler_ambiguous = _resolve_sparse_state_at_time(
            collection_groups.get(row.collection_id, collection.iloc[0:0]),
            int(row.episode_start_time_us), COLLECTION_STATE, scheduler_conflict=True,
        )
        ambiguous.append(arrival_ambiguous or scheduler_ambiguous)
    return pd.Series(ambiguous, index=candidates.index, dtype=bool)


def _select_episodes(usage: pd.DataFrame, instance_events: pd.DataFrame,
                     collection_events: pd.DataFrame) -> tuple[pd.DataFrame, int, int, int]:
    """네 단계 순서로 selected episode usage와 필터 전후 수를 반환한다."""
    grouped = usage.groupby(EPISODE_KEYS, dropna=False)
    # 품질 결함은 bucket·상태보다 먼저 제거해
    # 불완전한 trace가 후보가 되지 않게 한다.
    quality = grouped[["cpu_complete", "capacity_complete"]].all()
    quality_keys = quality.index[quality.all(axis=1)]
    quality_usage = usage.merge(quality_keys.to_frame(index=False), on=EPISODE_KEYS)
    # 온전한 품질 episode만 재구성된 5분 bucket 수로
    # 최소 관측 길이를 판정한다.
    bucket_counts = quality_usage.groupby(EPISODE_KEYS, dropna=False).size()
    candidates = bucket_counts.loc[bucket_counts.ge(12)].reset_index(name="bucket_count")
    arrival = quality_usage.groupby(EPISODE_KEYS, as_index=False, dropna=False).agg(
        episode_start_time_us=("episode_start_time_us", "first"),
        episode_end_time_us=("episode_end_time_us", "first"),
        arrival_machine_ids=("arrival_machine_ids", "first"),
    )
    candidates = candidates.merge(arrival, on=EPISODE_KEYS, validate="one_to_one")
    # 상태 충돌은 임의의 priority/request를 고르는 대신 episode 전체를 버린다.
    ambiguous = _event_state_mask(candidates, instance_events, collection_events)
    event_candidate_count = len(candidates)
    candidates = candidates.loc[~ambiguous].copy()
    # 마지막에 hash를 적용해야 필터 순서가 선택 집합의
    # 의미를 바꾸지 않는다.
    candidates["sample_hash"] = [
        _stable_key_hash(row.collection_id, row.instance_index, row.episode_index)
        for row in candidates.itertuples(index=False)
    ]
    selected = candidates.sort_values(
        ["sample_hash", *EPISODE_KEYS], kind="stable"
    ).head(5000)
    result = usage.merge(selected[EPISODE_KEYS], on=EPISODE_KEYS, how="inner")
    result = result.sort_values(EPISODE_KEYS + ["t5_day"], kind="stable")
    return (
        result.reset_index(drop=True), len(quality_keys), event_candidate_count,
        int(ambiguous.sum()),
    )


def main() -> None:
    """s4 selected episode usage parquet을 저장한다."""
    usage = pd.read_parquet(S3_PATH)
    instance_events = load_instance_events()
    collection_events = load_collection_events()
    result, quality_count, event_candidate_count, ambiguity_count = _select_episodes(
        usage, instance_events, collection_events
    )
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_PATH, index=False)
    print(f"quality_episode_count={quality_count}")
    print(f"event_state_candidate_count={event_candidate_count}")
    print(f"event_state_ambiguous_count={ambiguity_count}")
    print(f"selected_episode_count={len(result[EPISODE_KEYS].drop_duplicates())}")


if __name__ == "__main__":
    main()
