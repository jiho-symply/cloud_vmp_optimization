"""s6 관측 usage와 VM 요청 → 5분·시간별 workload scenario CSV를 만든다.
scenario 0은 관측값이고 scenario 1–4는 평균보존 lognormal 변형이다.
검증: canonical 5분·시간별 CSV와 직접 비교한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .paths import WORK_DIR


EXPERIMENT_DIR = WORK_DIR.parent
OBSERVED_PATH = WORK_DIR / "s6_observed_usage.parquet"
VM_REQUEST_PATH = EXPERIMENT_DIR / "out/vm_requests.csv"
OUT_DIR = EXPERIMENT_DIR / "out"
FIVE_MIN_OUTPUT_PATH = OUT_DIR / "vm_usage_5min_scenarios.csv"
HOURLY_OUTPUT_PATH = OUT_DIR / "vm_usage_hourly_scenarios.csv"
SCENARIO_COUNT = 5
SEED = 42
USAGE_COLUMNS = [
    "scenario_id", "vm_id", "t5_day", "t_hour", "cpu_usage", "mem_usage",
]
AUDIT_COLUMNS = [
    "coverage_us", "coverage_ratio", "max_cpu_usage", "max_mem_usage",
    "assigned_memory", "overlap_source_row_count", "overlap_conflict_flag",
    "duplicated_timeline_us",
]
HOURLY_COLUMNS = ["scenario_id", "vm_id", "t_hour", "cpu_usage", "mem_usage"]


def _merge_usage_inputs(
    observed: pd.DataFrame, vm_requests: pd.DataFrame
) -> pd.DataFrame:
    """관측 usage에 VM 요청 열을 붙인 DataFrame을 반환한다."""
    request_columns = ["vm_id", "class", "q_cpu", "q_mem"]
    base = observed.merge(vm_requests[request_columns], on="vm_id", how="left")
    # coverage_us는 정수 duration이므로 CSV에서 canonical의 표기를 유지한다.
    base["coverage_us"] = base["coverage_us"].astype("int64")
    return base


def _scenario_frame(
    base: pd.DataFrame, scenario_id: int, rng: np.random.Generator
) -> pd.DataFrame:
    """한 scenario의 5분 usage와 관측 audit 열을 반환한다."""
    scenario = base.copy()
    scenario["scenario_id"] = scenario_id
    if scenario_id > 0:
        cpu_sigma, mem_sigma = 0.22, 0.08
        cpu_noise = rng.lognormal(
            mean=-0.5 * cpu_sigma**2, sigma=cpu_sigma, size=len(scenario)
        )
        mem_noise = rng.lognormal(
            mean=-0.5 * mem_sigma**2, sigma=mem_sigma, size=len(scenario)
        )
        scenario["cpu_usage"] = scenario["cpu_usage"] * cpu_noise
        scenario["mem_usage"] = scenario["mem_usage"] * mem_noise
        scenario["cpu_usage"] = np.clip(
            scenario["cpu_usage"].to_numpy(dtype=float),
            0.0,
            scenario["q_cpu"].to_numpy(dtype=float),
        )
        scenario["mem_usage"] = np.clip(
            scenario["mem_usage"].to_numpy(dtype=float),
            0.0,
            scenario["q_mem"].to_numpy(dtype=float),
        )
    scenario["cpu_usage"] = scenario["cpu_usage"].clip(lower=0.0)
    scenario["mem_usage"] = scenario["mem_usage"].clip(lower=0.0)
    audit_columns = [column for column in AUDIT_COLUMNS if column in scenario.columns]
    return scenario[USAGE_COLUMNS + audit_columns]


def _hourly_frame(usage_5min: pd.DataFrame) -> pd.DataFrame:
    """5분 scenario를 VM·시간별 duration 가중평균 DataFrame으로 반환한다."""
    work = usage_5min.copy()
    work["coverage_us"] = pd.to_numeric(work["coverage_us"], errors="coerce")
    work["cpu_coverage_weighted"] = work["cpu_usage"] * work["coverage_us"]
    work["mem_coverage_weighted"] = work["mem_usage"] * work["coverage_us"]
    hourly = (
        work.groupby(["scenario_id", "vm_id", "t_hour"], as_index=False)
        .agg(
            cpu_coverage_weighted=("cpu_coverage_weighted", "sum"),
            mem_coverage_weighted=("mem_coverage_weighted", "sum"),
            coverage_us=("coverage_us", "sum"),
        )
        .sort_values(["scenario_id", "vm_id", "t_hour"])
        .reset_index(drop=True)
    )
    hourly["cpu_usage"] = hourly["cpu_coverage_weighted"] / hourly["coverage_us"]
    hourly["mem_usage"] = hourly["mem_coverage_weighted"] / hourly["coverage_us"]
    return hourly[HOURLY_COLUMNS]


def main() -> None:
    """5분·시간별 scenario CSV를 순차적으로 저장한다."""
    observed = pd.read_parquet(OBSERVED_PATH)
    vm_requests = pd.read_csv(VM_REQUEST_PATH, float_precision="round_trip")
    base = _merge_usage_inputs(observed, vm_requests)
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for scenario_id in range(SCENARIO_COUNT):
        five_min = _scenario_frame(base, scenario_id, rng)
        hourly = _hourly_frame(five_min)
        mode = "w" if scenario_id == 0 else "a"
        five_min.to_csv(
            FIVE_MIN_OUTPUT_PATH, index=False, mode=mode, header=scenario_id == 0
        )
        hourly.to_csv(
            HOURLY_OUTPUT_PATH, index=False, mode=mode, header=scenario_id == 0
        )
        print(
            f"scenario_id={scenario_id}, five_min_rows={len(five_min)}, "
            f"hourly_rows={len(hourly)}"
        )


if __name__ == "__main__":
    main()
