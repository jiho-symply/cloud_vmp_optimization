"""machine_events 입력 → 대표 용량 JSON과 머신 구간 parquet를 생성한다.
horizon 내 활성 용량의 가중 최빈값을 고정한다.
검증: cpu=.591796875 mem=.33349609375 n=13 w=283330432576447 active=820416271910336."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from paths import S1_MACHINE_INTERVALS_PATH, S1_REPRESENTATIVE_PATH, WORK_DIR
from raw_tables import load_machine_events


DAY_START_US = 600_000_000
DAY_END_US = 87_000_000_000


def _prepared_events(machine_events: pd.DataFrame, day_end_us: int) -> pd.DataFrame:
    """day_end_us 이전의 머신 이벤트를 정렬한 DataFrame을 반환한다."""
    events = pd.DataFrame({
        "time": machine_events["time"],
        "machine_id": machine_events["machine_id"],
        "event_type": machine_events["event_type"],
        "capacity_cpu": machine_events["capacity_cpu"],
        "capacity_mem": machine_events["capacity_mem"],
        "row_order": np.arange(len(machine_events), dtype=np.int64),
    })
    events = events.loc[events["time"].lt(int(day_end_us))].copy()
    return events.sort_values(["machine_id", "time", "row_order"], kind="stable")


def _apply_machine_event(
    row: object, active: bool, cpu: float | None, mem: float | None
) -> tuple[bool, float | None, float | None]:
    """머신 이벤트를 적용한 active 상태와 CPU·메모리 용량을 반환한다."""
    code = int(row.event_type)
    row_cpu = None if pd.isna(row.capacity_cpu) else float(row.capacity_cpu)
    row_mem = None if pd.isna(row.capacity_mem) else float(row.capacity_mem)
    if code in {1, 3}:
        if row_cpu is not None:
            cpu = row_cpu
        if row_mem is not None:
            mem = row_mem
    if code == 1:
        active = True
    elif code == 2:
        active = False
        # REMOVE는 머신 lifecycle을 닫는다.
        # 이후 sparse ADD가 제거된 머신의 용량을 물려받지 않게 한다.
        cpu = None
        mem = None
    return active, cpu, mem


def _machine_cpu_capacity_intervals(
    machine_events: pd.DataFrame, day_start_us: int, day_end_us: int
) -> pd.DataFrame:
    """horizon 내 머신별 활성 용량 구간을 반환한다."""
    events = _prepared_events(machine_events, day_end_us)
    interval_rows: list[dict[str, object]] = []
    for machine_id, history in events.groupby("machine_id", sort=False, dropna=False):
        active = False
        cpu: float | None = None
        mem: float | None = None
        rows = list(history.itertuples(index=False))
        position = 0
        # horizon 시작 시점의 상태를 복원한다.
        # day_start_us 이전 이벤트를 모두 소화한 뒤 cursor를 둔다.
        while position < len(rows) and int(rows[position].time) < int(day_start_us):
            active, cpu, mem = _apply_machine_event(rows[position], active, cpu, mem)
            position += 1
        cursor = int(day_start_us)
        while position < len(rows):
            row = rows[position]
            event_time = max(int(day_start_us), min(int(day_end_us), int(row.time)))
            if event_time > cursor and active:
                interval_rows.append(
                    {"machine_id": machine_id, "start_time": cursor, "end_time": event_time,
                     "capacity_cpu": cpu, "capacity_mem": mem}
                )
            active, cpu, mem = _apply_machine_event(row, active, cpu, mem)
            cursor = event_time
            position += 1
        if cursor < int(day_end_us) and active:
            interval_rows.append(
                {"machine_id": machine_id, "start_time": cursor, "end_time": int(day_end_us),
                 "capacity_cpu": cpu, "capacity_mem": mem}
            )
    return pd.DataFrame(
        interval_rows,
        columns=["machine_id", "start_time", "end_time", "capacity_cpu", "capacity_mem"],
    )


def _joint_shape_weights(intervals: pd.DataFrame) -> pd.DataFrame:
    """(cpu, mem) 용량 쌍별 active duration 합계를 반환한다."""
    positive = intervals.loc[
        intervals["capacity_cpu"].gt(0) & intervals["capacity_mem"].gt(0)
    ].copy()
    positive["active_duration_us"] = positive["end_time"] - positive["start_time"]
    return positive.groupby(
        ["capacity_cpu", "capacity_mem"], as_index=False, sort=True
    )["active_duration_us"].sum()


def _representative_capacity(shape_weights: pd.DataFrame) -> dict[str, float | int | str]:
    """active duration 기준 대표 용량과 선택 가중치를 반환한다."""
    # 동점 처리 결과가 바뀌므로 cpu 오름차순을 유지한다.
    # mem 오름차순도 함께 유지한다.
    ranked = shape_weights.sort_values(
        ["active_duration_us", "capacity_cpu", "capacity_mem"],
        ascending=[False, True, True],
    )
    selected = ranked.iloc[0]
    representative = {
        "capacity_cpu": float(selected["capacity_cpu"]),
        "capacity_mem": float(selected["capacity_mem"]),
        "selected_weight_us": int(selected["active_duration_us"]),
        "method": "active_duration_weighted_mode_of_joint_cpu_mem_machine_shape",
    }
    return representative


def main() -> None:
    """s1 산출물을 디스크에 저장한다."""
    machine_events = load_machine_events()
    interval_frame = _machine_cpu_capacity_intervals(machine_events, DAY_START_US, DAY_END_US)
    shape_weights = _joint_shape_weights(interval_frame)
    representative = _representative_capacity(shape_weights)
    total_active_us = int((interval_frame["end_time"] - interval_frame["start_time"]).sum())
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    S1_REPRESENTATIVE_PATH.write_text(json.dumps(representative, indent=2) + "\n", encoding="utf-8")
    interval_frame.to_parquet(S1_MACHINE_INTERVALS_PATH, index=False)
    print(f"capacity_cpu={representative['capacity_cpu']}")
    print(f"capacity_mem={representative['capacity_mem']}")
    print(f"joint_shape_count={len(shape_weights)}")
    print(f"selected_shape_weight_us={representative['selected_weight_us']}")
    print(f"total_active_machine_duration_us={total_active_us}")


if __name__ == "__main__":
    main()
