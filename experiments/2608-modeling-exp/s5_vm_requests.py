"""selected episode usage와 raw event → raw trace 단위 VM 요청 parquet을 만든다.
episode별 요청·관측 최댓값·proxy class를 한 행으로 요약한다.
검증: 5,000행, max_mem_usage_fallback_from_average 7건, 나머지 fallback 0건.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from paths import WORK_DIR
from raw_tables import load_collection_events, load_instance_events
from s4_select_episodes import _normalize_events, _stable_key_hash, _value_key


S4_PATH = WORK_DIR / "s4_selected_episodes.parquet"
OUTPUT_PATH = WORK_DIR / "s5_vm_requests_raw.parquet"
EPISODE_KEYS = ["collection_id", "instance_index", "episode_index"]
DAY_START_US = 600_000_000
T5_US = 300_000_000
EPSILON = 1e-6
STATE_EVENTS = {"SCHEDULE", "UPDATE_RUNNING"}
INSTANCE_STATE = [
    "priority", "scheduling_class", "resource_request_cpu", "resource_request_mem",
]
COLLECTION_STATE = ["scheduler", "collection_type", "priority", "scheduling_class"]
VM_COLUMNS = [
    "vm_id", "stable_episode_key", "collection_id", "instance_index", "episode_index",
    "episode_start_time_us", "episode_end_time_us", "termination_time_us",
    "termination_reason", "horizon_censored", "termination_time_uncertain",
    "episode_start_inferred_from_update", "episode_start_inferred_from_usage", "class",
    "class_rule", "scheduler", "arrival_t5", "departure_t5", "lifetime_t5",
    "arrival_time_us", "q_cpu", "q_mem", "resource_request_cpu", "resource_request_mem",
    "p95_cpu_usage", "p95_mem_usage", "avg_cpu_usage", "avg_mem_usage", "max_cpu_usage",
    "max_mem_usage", "priority", "scheduling_class", "arrival_machine_id",
    "arrival_state_ambiguous", "arrival_state_ambiguity_reason",
    "arrival_state_candidate_count", "scheduler_state_ambiguous",
    "scheduler_state_ambiguity_reason", "scheduler_state_candidate_count",
    "arrival_state_used_post_arrival", "scheduler_state_used_post_arrival",
    "resource_request_cpu_fallback_zero", "resource_request_mem_fallback_zero",
    "max_cpu_usage_fallback_from_average", "max_mem_usage_fallback_from_average",
    "cpu_p100_gt_one", "spot_revenue", "migration_energy_coeff",
]
CLASS_RULES = {
    "spot": (
        "spot: proxy label for priority < 120 or missing priority; Google 2019 tiers are "
        "0-99 free/best-effort, 100-115 BE/BEB, 116-119 mid; not an observed cloud purchase class"
    ),
    "on_demand": (
        "on_demand: proxy label for priority >= 120 production-tier Google 2019 workload; "
        "not an observed cloud purchase class"
    ),
    "batch_candidate": (
        "batch_candidate: proxy label because collection scheduler == SCHEDULER_BATCH; "
        "not an observed cloud purchase class"
    ),
}


def _resolve_state(history: pd.DataFrame, arrival_time_us: int,
                   state_columns: list[str], arrival_machines: tuple[object, ...] = ()
                   ) -> tuple[dict[str, object], bool, bool, str, int, object]:
    """arrival 시점 sparse state와 상태 선택 메타데이터를 반환한다."""
    empty = {column: np.nan for column in state_columns}
    if history.empty: return empty, False, False, "", 0, np.nan
    machine_keys = {_value_key(value) for value in arrival_machines}
    columns = ["time", "event_name", "event_precedence", "normal_missing_row_rank",
               "machine_id", *state_columns]
    rows = [(raw[0], raw[1], int(raw[2]), int(raw[3]), raw[4],
             tuple(raw[5:]), tuple(_value_key(value) for value in raw[5:]))
            for raw in history[columns].itertuples(index=False, name=None)]
    rows.sort(key=lambda row: -np.inf if pd.isna(row[0]) else row[0])
    current, latest, current_machine, position = dict(empty), None, np.nan, 0
    while position < len(rows):
        start, event_time, position = position, rows[position][0], position + 1
        while position < len(rows) and rows[position][0] == event_time: position += 1
        if not pd.isna(event_time) and int(event_time) > arrival_time_us and latest:
            return latest[0], False, latest[1], latest[2], latest[3], latest[4]
        cohort = rows[start:position]
        matches = [int(row[1] in STATE_EVENTS and _value_key(row[4]) in machine_keys)
                   for row in cohort]
        ordered = sorted(zip(cohort, matches), key=lambda item: (
            item[1], item[0][2], item[0][3], _value_key(item[0][1]),
            _value_key(item[0][4]), *item[0][6],
        ))
        for row, _ in ordered:
            current.update({
                column: value for column, value in zip(state_columns, row[5])
                if not pd.isna(value)
            })
            current_machine = row[4] if not pd.isna(row[4]) else current_machine
        if not any(pd.notna(value) for value in current.values()): continue
        snapshot = (dict(current), False, "", 0, current_machine)
        if pd.isna(event_time) or int(event_time) <= arrival_time_us: latest = snapshot
        else: return snapshot[0], True, snapshot[1], snapshot[2], snapshot[3], snapshot[4]
    latest = latest or (empty, False, "", 0, np.nan)
    return latest[0], False, latest[1], latest[2], latest[3], latest[4]


def _event_state_row(candidate: object, instance_groups: dict, collection_groups: dict,
                     empty_instance: pd.DataFrame, empty_collection: pd.DataFrame) -> dict:
    """candidate episode 한 건의 instance·collection state dict를 반환한다."""
    key = (candidate.collection_id, candidate.instance_index)
    history = instance_groups.get(key, empty_instance)
    history = history.loc[
        history["time"].lt(int(candidate.episode_end_time_us)) | history["time"].isna()
    ]
    machines = candidate.arrival_machine_ids
    if not isinstance(machines, tuple):
        machines = tuple(np.asarray(machines).tolist())
    state, post, ambiguous, reason, count, machine = _resolve_state(
        history, int(candidate.episode_start_time_us), INSTANCE_STATE, machines
    )
    cstate, cpost, cambiguous, creason, ccount, _ = _resolve_state(
        collection_groups.get(candidate.collection_id, empty_collection),
        int(candidate.episode_start_time_us), COLLECTION_STATE,
    )
    return {
        **{column: getattr(candidate, column) for column in EPISODE_KEYS}, **state,
        "arrival_machine_id": machine, "arrival_state_used_post_arrival": post,
        "arrival_state_ambiguous": ambiguous, "arrival_state_ambiguity_reason": reason,
        "arrival_state_candidate_count": count, "scheduler": cstate["scheduler"],
        "collection_type": cstate["collection_type"], "collection_priority": cstate["priority"],
        "scheduler_state_used_post_arrival": cpost,
        "scheduler_state_ambiguous": cambiguous,
        "scheduler_state_ambiguity_reason": creason,
        "scheduler_state_candidate_count": ccount,
    }


def _event_state_table(candidates: pd.DataFrame, instance_events: pd.DataFrame,
                       collection_events: pd.DataFrame) -> pd.DataFrame:
    """candidate episode별 arrival instance·collection state를 반환한다."""
    instance = _normalize_events(instance_events, collection=False)
    collection = _normalize_events(collection_events, collection=True).copy()
    collection["scheduler"] = collection["scheduler"].fillna("").astype(str).str.strip().str.upper()
    collection["scheduler"] = collection["scheduler"].replace({
        "0": "SCHEDULER_DEFAULT", "1": "SCHEDULER_BATCH",
    })
    instance_groups = {
        key: group for key, group in instance.groupby(
            ["collection_id", "instance_index"], sort=False
        )
    }
    collection_groups = {
        key: group for key, group in collection.groupby("collection_id", sort=False)
    }
    empty_instance, empty_collection = instance.iloc[:0], collection.iloc[:0]
    # s4가 모호한 episode를 제외했으므로 선택된 state만 저장한다.
    rows = [
        _event_state_row(candidate, instance_groups, collection_groups,
                         empty_instance, empty_collection)
        for candidate in candidates.itertuples(index=False)
    ]
    result = pd.DataFrame(rows)
    result["arrival_machine_id"] = result["arrival_machine_id"].astype(float)
    return result


def _aggregate_requests(usage: pd.DataFrame) -> pd.DataFrame:
    """selected episode usage를 raw trace 단위 집계 행으로 반환한다."""
    grouped = usage.groupby(EPISODE_KEYS, dropna=False)
    requests = grouped.agg(
        arrival_t5=("t5_day", "min"),
        departure_t5=("t5_day", lambda values: int(values.max()) + 1),
        first_observed_us=("first_observed_us", "min"),
        last_observed_us=("last_observed_us", "max"),
        p95_cpu_usage=("cpu_usage", lambda values: float(values.quantile(0.95))),
        p95_mem_usage=("mem_usage", lambda values: float(values.quantile(0.95))),
        avg_cpu_usage=("cpu_usage", "mean"), avg_mem_usage=("mem_usage", "mean"),
        max_average_cpu_usage=("cpu_usage", "max"),
        max_average_mem_usage=("mem_usage", "max"),
        source_max_cpu_usage=("vm_source_max_cpu_usage", "max"),
        source_max_mem_usage=("vm_source_max_mem_usage", "max"),
        episode_start_time_us=("episode_start_time_us", "first"),
        episode_end_time_us=("episode_end_time_us", "first"),
        termination_time_us=("termination_time_us", "first"),
        termination_reason=("termination_reason", "first"),
        horizon_censored=("horizon_censored", "first"),
        termination_time_uncertain=("termination_time_uncertain", "first"),
        episode_start_inferred_from_update=("episode_start_inferred_from_update", "first"),
        episode_start_inferred_from_usage=("episode_start_inferred_from_usage", "first"),
    ).reset_index()
    requests["arrival_time_us"] = requests["episode_start_time_us"].astype("int64")
    requests["arrival_t5"] = np.floor(
        (requests["episode_start_time_us"] - DAY_START_US) / T5_US
    ).astype(int).clip(0, 287)
    requests["departure_t5"] = np.ceil(
        (requests["episode_end_time_us"] - DAY_START_US) / T5_US
    ).astype(int).clip(1, 288)
    requests["arrival_machine_ids"] = usage.groupby(
        EPISODE_KEYS, sort=False
    )["arrival_machine_ids"].first().to_numpy()
    return requests


def _apply_resource_requests(requests: pd.DataFrame) -> pd.DataFrame:
    """arrival request와 episode maximum으로 q_cpu·q_mem을 계산한다."""
    for resource in ["cpu", "mem"]:
        request_column = f"resource_request_{resource}"
        source_column = f"source_max_{resource}_usage"
        average_column = f"max_average_{resource}_usage"
        fallback_column = f"max_{resource}_usage_fallback_from_average"
        requests[request_column] = pd.to_numeric(requests[request_column], errors="coerce")
        # 누락된 도착 요청은 0으로 두어 관측 maximum만으로도
        # q를 정할 수 있게 한다.
        requests[f"{request_column}_fallback_zero"] = requests[request_column].isna()
        requests[request_column] = requests[request_column].fillna(0.0)
        requests[source_column] = pd.to_numeric(requests[source_column], errors="coerce")
        requests[fallback_column] = requests[source_column].isna()
        # source maximum 누락은 원본 규칙대로 episode의 평균 사용량
        # maximum으로 보완한다.
        requests[f"max_{resource}_usage"] = requests[source_column].where(
            ~requests[fallback_column], requests[average_column]
        )
        # 요청보다 많이 사용한 episode도 수용하도록 q는 두 값 중
        # 큰 값으로 정한다.
        values = np.maximum(
            requests[request_column].to_numpy(dtype=float),
            requests[f"max_{resource}_usage"].to_numpy(dtype=float),
        )
        requests[f"q_{resource}"] = np.maximum(values, EPSILON)
    return requests


def _add_stable_ids(requests: pd.DataFrame) -> pd.DataFrame:
    """requests에 stable episode key와 VM ID를 붙인 DataFrame을 반환한다."""
    requests["sample_hash"] = [
        _stable_key_hash(row.collection_id, row.instance_index, row.episode_index)
        for row in requests.itertuples(index=False)
    ]
    requests = requests.sort_values(["sample_hash", *EPISODE_KEYS]).reset_index(drop=True)
    requests["stable_episode_key"] = [
        f"collection={_value_key(row.collection_id)}|"
        f"instance={_value_key(row.instance_index)}|episode={int(row.episode_index)}"
        for row in requests.itertuples(index=False)
    ]
    requests["vm_id"] = [
        "ep_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
        for key in requests["stable_episode_key"]
    ]
    return requests


def _finish_requests(requests: pd.DataFrame, instance_events: pd.DataFrame,
                     collection_events: pd.DataFrame) -> pd.DataFrame:
    """state·class·stable VM ID를 붙인 raw VM 요청 행을 반환한다."""
    state = _event_state_table(requests, instance_events, collection_events)
    requests = requests.merge(state, on=EPISODE_KEYS, how="left", validate="one_to_one")
    requests = _apply_resource_requests(requests)
    requests["cpu_p100_gt_one"] = requests["source_max_cpu_usage"].gt(1.0)
    requests["lifetime_t5"] = requests["departure_t5"] - requests["arrival_t5"]
    requests["scheduler"] = requests["scheduler"].fillna("")
    priority = pd.to_numeric(requests["priority"], errors="coerce")
    # class는 구매 유형이 아니라 priority와 collection scheduler로
    # 만든 proxy label이다.
    requests["class"] = "spot"
    high_priority = priority >= 120
    requests.loc[high_priority, "class"] = "on_demand"
    batch = requests["scheduler"].eq("SCHEDULER_BATCH")
    requests.loc[batch, "class"] = "batch_candidate"
    requests["class_rule"] = requests["class"].map(CLASS_RULES)
    requests["spot_revenue"] = np.where(
        requests["class"].eq("spot"), 0.05 * requests["q_cpu"], 0.0
    )
    requests["migration_energy_coeff"] = np.where(
        requests["class"].eq("on_demand"),
        0.01 * requests["q_cpu"] + 0.005 * requests["q_mem"], 0.0,
    )
    requests = _add_stable_ids(requests)
    return requests[VM_COLUMNS].copy()


def main() -> None:
    """s5 raw VM request parquet을 저장한다."""
    usage = pd.read_parquet(S4_PATH)
    instance_events = load_instance_events()
    collection_events = load_collection_events()
    requests = _aggregate_requests(usage)
    result = _finish_requests(requests, instance_events, collection_events)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_PATH, index=False)
    print(f"vm_count={len(result)}")
    for column in [
        "resource_request_cpu_fallback_zero", "resource_request_mem_fallback_zero",
        "max_cpu_usage_fallback_from_average", "max_mem_usage_fallback_from_average",
    ]:
        print(f"{column}_count={int(result[column].sum())}")


if __name__ == "__main__":
    main()
