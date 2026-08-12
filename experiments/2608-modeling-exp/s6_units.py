"""raw VM 요청과 representative machine 용량 → 모델 단위 산출물을 만든다.
모든 자원을 대표 용량으로 나누고 q가 1을 넘는 VM은 통째로 제거한다.
검증: cpu divisor .591796875, mem divisor .33349609375, 5,000개 유지.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from paths import S1_REPRESENTATIVE_PATH, WORK_DIR
from s5_vm_requests import VM_COLUMNS


EXPERIMENT_DIR = WORK_DIR.parent
S4_PATH = WORK_DIR / "s4_selected_episodes.parquet"
S5_PATH = WORK_DIR / "s5_vm_requests_raw.parquet"
OUT_DIR = EXPERIMENT_DIR / "out"
VM_OUTPUT_PATH = OUT_DIR / "vm_requests.csv"
OBSERVED_OUTPUT_PATH = WORK_DIR / "s6_observed_usage.parquet"
EPISODE_KEYS = ["collection_id", "instance_index", "episode_index"]
OBSERVED_COLUMNS = [
    "vm_id", "collection_id", "instance_index", "episode_index", "t5", "t5_day",
    "t_hour", "cpu_usage", "mem_usage", "max_cpu_usage", "max_mem_usage",
    "assigned_memory", "coverage_us", "coverage_ratio", "overlap_source_row_count",
    "overlap_conflict_flag", "duplicated_timeline_us",
]
EPSILON = 1e-6


def _convert_units(vm: pd.DataFrame, observed: pd.DataFrame,
                   capacity: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """raw trace 자원을 representative-machine fraction으로 반환한다."""
    cpu_divisor = float(capacity["capacity_cpu"])
    mem_divisor = float(capacity["capacity_mem"])
    converted_vm = vm.copy()
    converted_observed = observed.copy()
    for column in ["q_cpu", "resource_request_cpu", "p95_cpu_usage", "avg_cpu_usage",
                   "max_cpu_usage"]:
        converted_vm[column] = pd.to_numeric(converted_vm[column], errors="coerce") / cpu_divisor
    for column in ["q_mem", "resource_request_mem", "p95_mem_usage", "avg_mem_usage",
                   "max_mem_usage"]:
        converted_vm[column] = pd.to_numeric(converted_vm[column], errors="coerce") / mem_divisor
    for column in ["cpu_usage", "max_cpu_usage"]:
        converted_observed[column] = converted_observed[column] / cpu_divisor
    for column in ["mem_usage", "max_mem_usage", "assigned_memory"]:
        converted_observed[column] = converted_observed[column] / mem_divisor
    converted_vm["q_cpu"] = converted_vm["q_cpu"].clip(lower=EPSILON)
    converted_vm["q_mem"] = converted_vm["q_mem"].clip(lower=EPSILON)
    converted_vm["spot_revenue"] = np.where(
        converted_vm["class"].eq("spot"), 0.05 * converted_vm["q_cpu"], 0.0
    )
    converted_vm["migration_energy_coeff"] = np.where(
        converted_vm["class"].eq("on_demand"),
        0.01 * converted_vm["q_cpu"] + 0.005 * converted_vm["q_mem"], 0.0,
    )
    return converted_vm, converted_observed


def _filter_capacity(
    vm: pd.DataFrame, observed: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """representative machine에 들어가는 VM과 관측 행만 반환한다."""
    exceeds = vm["q_cpu"].gt(1.0 + 1e-12) | vm["q_mem"].gt(1.0 + 1e-12)
    retained_ids = set(vm.loc[~exceeds, "vm_id"].astype(str))
    kept_vm = vm.loc[~exceeds].copy()
    kept_observed = observed.loc[observed["vm_id"].astype(str).isin(retained_ids)].copy()
    return kept_vm, kept_observed, int(exceeds.sum())


def _observed_usage(usage: pd.DataFrame, vm: pd.DataFrame) -> pd.DataFrame:
    """selected episode usage에 stable VM ID를 붙인 관측 trace를 반환한다."""
    key_to_vm = vm[EPISODE_KEYS + ["vm_id"]]
    observed = usage.merge(key_to_vm, on=EPISODE_KEYS, how="inner")
    observed = observed[OBSERVED_COLUMNS]
    return observed.sort_values(["vm_id", "t5_day"]).reset_index(drop=True)


def main() -> None:
    """s6 CSV와 observed usage parquet을 저장한다."""
    raw_vm = pd.read_parquet(S5_PATH)
    selected_usage = pd.read_parquet(S4_PATH)
    capacity = json.loads(S1_REPRESENTATIVE_PATH.read_text(encoding="utf-8"))
    observed_raw = _observed_usage(selected_usage, raw_vm)
    converted_vm, converted_observed = _convert_units(raw_vm, observed_raw, capacity)
    converted_vm, converted_observed, dropped = _filter_capacity(
        converted_vm, converted_observed
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    converted_vm[VM_COLUMNS].to_csv(VM_OUTPUT_PATH, index=False)
    converted_observed.to_parquet(OBSERVED_OUTPUT_PATH, index=False)
    print(f"cpu_divisor={capacity['capacity_cpu']}")
    print(f"mem_divisor={capacity['capacity_mem']}")
    print(f"source_vm_count={len(raw_vm)}")
    print(f"dropped_vm_count={dropped}")
    print(f"retained_vm_count={len(converted_vm)}")


if __name__ == "__main__":
    main()
