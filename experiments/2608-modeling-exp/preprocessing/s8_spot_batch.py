"""s6 VM·관측 usage → spot 선점과 batch family/workload CSV를 만든다.
실제 EVICT 이후를 비활성으로 두고 batch VM은 최대 10개 family로 묶는다.
검증: canonical의 세 CSV와 행 순서·문자열까지 일치해야 한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .paths import WORK_DIR
from .raw_tables import load_instance_events


EXPERIMENT_DIR = WORK_DIR.parent
OBSERVED_PATH = WORK_DIR / "s6_observed_usage.parquet"
VM_REQUEST_PATH = EXPERIMENT_DIR / "out/vm_requests.csv"
OUT_DIR = EXPERIMENT_DIR / "out"
T5_US = 300_000_000
HORIZON_T5 = 288
SCENARIO_COUNT = 5
SEED = 42
EPSILON = 1e-6
SPOT_COLUMNS = ["scenario_id", "vm_id", "t5_day", "t_hour", "active", "preempted"]
BATCH_FAMILY_COLUMNS = [
    "family_id", "q_cpu_B", "q_mem_B", "W_k", "rho_cpu_B", "rho_mem_B",
    "base_cpu", "base_mem", "startup_cpu", "startup_mem",
]
BATCH_WORKLOAD_COLUMNS = [
    "scenario_id", "family_id", "t_hour", "workload_volume", "cpu_workload",
    "mem_workload",
]


def _is_evict_event(value: object) -> bool:
    """숫자 4 또는 이름에 EVICT가 든 사건인지 반환한다."""
    if pd.isna(value):
        return False
    if isinstance(value, (int, np.integer, float, np.floating)):
        return int(value) == 4
    text = str(value).strip().upper()
    return "EVICT" in text or text == "4"


def _eviction_t5_by_vm(vm_requests: pd.DataFrame, events: pd.DataFrame | None,
                       min_t5: int, day_start_us: int | None = None) -> dict[str, int]:
    """VM별 관측 구간 안 최초 EVICT 5분 슬롯을 반환한다."""
    has_termination = {"termination_reason", "termination_time_us"}.issubset(
        vm_requests.columns
    )
    has_episode_ids = vm_requests["vm_id"].astype(str).str.startswith("ep_").all()
    if has_termination and has_episode_ids:
        evicted = vm_requests.loc[
            vm_requests["termination_reason"].astype(str).eq("EVICT")
            & pd.to_numeric(vm_requests["termination_time_us"], errors="coerce").notna()
        ].copy()
        if evicted.empty: return {}
        origin = int(day_start_us) if day_start_us is not None else int(min_t5) * T5_US
        evicted["t5_day"] = np.floor(
            (pd.to_numeric(evicted["termination_time_us"], errors="coerce") - origin)
            / T5_US
        ).astype(int)
        evicted = evicted.loc[evicted["t5_day"].between(0, HORIZON_T5 - 1)]
        return evicted.set_index("vm_id")["t5_day"].astype(int).to_dict()
    if events is None or events.empty: return {}
    event_columns = ["collection_id", "instance_index", "time", "event_type"]
    normalized = events[event_columns].copy().drop_duplicates()
    normalized["time"] = pd.to_numeric(normalized["time"], errors="coerce")
    joined = normalized.merge(
        vm_requests[["vm_id", "collection_id", "instance_index"]],
        on=["collection_id", "instance_index"],
        how="inner",
    )
    joined = joined.loc[joined["event_type"].map(_is_evict_event)].dropna(subset=["time"])
    if joined.empty: return {}
    origin = int(day_start_us) if day_start_us is not None else int(min_t5) * T5_US
    joined["t5_day"] = np.floor((joined["time"] - origin) / T5_US).astype(int)
    joined = joined.loc[joined["t5_day"].between(0, HORIZON_T5 - 1)]
    return joined.groupby("vm_id")["t5_day"].min().astype(int).to_dict()


def _spot_preemption_t5(vm: object, t5_values: object, actual_evict: dict[str, int],
                        scenario_id: int, rng: np.random.Generator) -> object:
    """spot VM의 scenario별 선점 슬롯을 반환한다."""
    if vm.vm_id in actual_evict:
        return actual_evict[vm.vm_id]
    # 실제 EVICT가 없는 VM은 관측 선점 시점을 알 수 없어
    # priority 기반 합성 hazard를 쓴다.
    priority = 0.0 if pd.isna(vm.priority) else float(vm.priority)
    hazard = min(0.08, 0.003 + max(0.0, 100.0 - priority) / 100.0 * 0.025)
    preempt_t5 = None
    for t5 in t5_values:
        if scenario_id > 0 and rng.random() < hazard:
            preempt_t5 = t5
            break
    return preempt_t5


def build_spot_preemption_scenarios(
    vm_requests: pd.DataFrame, observed_usage: pd.DataFrame,
    events: pd.DataFrame | None = None, num_scenarios: int = SCENARIO_COUNT,
    seed: int = SEED,
) -> pd.DataFrame:
    """spot VM별 scenario·5분 슬롯의 active 상태를 반환한다."""
    spot_vms = vm_requests.loc[vm_requests["class"].eq("spot")].copy()
    if spot_vms.empty:
        return pd.DataFrame(columns=SPOT_COLUMNS)
    rng = np.random.default_rng(seed + 17)
    min_t5 = int(observed_usage["t5"].min()) if "t5" in observed_usage else 0
    day_start_us = observed_usage.attrs.get("day_start_us")
    actual_evict = _eviction_t5_by_vm(vm_requests, events, min_t5, day_start_us)
    rows: list[dict[str, int | str]] = []
    for vm in spot_vms.itertuples(index=False):
        observed_times = observed_usage.loc[
            observed_usage["vm_id"].eq(vm.vm_id), "t5_day"
        ]
        t5_values = (
            sorted(int(t) for t in observed_times.unique())
            if not observed_times.empty else range(int(vm.arrival_t5), int(vm.departure_t5))
        )
        for scenario_id in range(num_scenarios):
            preempt_t5 = _spot_preemption_t5(
                vm, t5_values, actual_evict, scenario_id, rng
            )
            for t5 in t5_values:
                active = int(preempt_t5 is None or t5 < int(preempt_t5))
                rows.append({
                    "scenario_id": scenario_id, "vm_id": vm.vm_id,
                    "t5_day": int(t5), "t_hour": int(t5) // 12,
                    "active": active, "preempted": 1 - active,
                })
    return pd.DataFrame(rows, columns=SPOT_COLUMNS).sort_values(
        ["scenario_id", "vm_id", "t5_day"]
    ).reset_index(drop=True)


def _assign_batch_families(batch: pd.DataFrame, max_families: int) -> pd.DataFrame:
    """batch VM에 q 크기 또는 균등 bucket 기반 family ID를 붙인다."""
    # 개별 batch VM 대신 유사한 q의 family volume으로 묶어
    # workload 규모를 축약한다.
    rounded = batch.assign(cpu_bin=batch["q_cpu"].round(2), mem_bin=batch["q_mem"].round(2))
    group_count = rounded[["cpu_bin", "mem_bin"]].drop_duplicates().shape[0]
    if group_count <= max_families:
        family_keys = rounded[["cpu_bin", "mem_bin"]].drop_duplicates().sort_values(
            ["cpu_bin", "mem_bin"]
        )
        family_keys["family_id"] = [f"batch{k:02d}" for k in range(len(family_keys))]
        return rounded.merge(family_keys, on=["cpu_bin", "mem_bin"], how="left")
    ordered = batch.sort_values(["q_cpu", "q_mem", "vm_id"]).reset_index(drop=True)
    ordered["family_bucket"] = np.floor(
        np.arange(len(ordered)) * max_families / len(ordered)
    ).astype(int)
    ordered["family_id"] = ordered["family_bucket"].map(lambda i: f"batch{i:02d}")
    return ordered


def _build_batch_workload(batch: pd.DataFrame, families: pd.DataFrame,
                          observed_usage: pd.DataFrame, num_scenarios: int,
                          seed: int) -> pd.DataFrame:
    """family·시간별 관측 workload와 합성 scenario를 반환한다."""
    observed_batch = observed_usage.merge(batch[["vm_id", "family_id"]], on="vm_id", how="inner")
    hourly = observed_batch.groupby(["family_id", "t_hour"], as_index=False).agg(
        workload_volume=("vm_id", "nunique"),
        cpu_workload=("cpu_usage", "sum"),
        mem_workload=("mem_usage", "sum"),
    )
    rng = np.random.default_rng(seed + 29)
    family_q_cpu = families.set_index("family_id")["q_cpu_B"].to_dict()
    family_q_mem = families.set_index("family_id")["q_mem_B"].to_dict()
    frames = []
    for scenario_id in range(num_scenarios):
        scenario = hourly.copy()
        scenario["scenario_id"] = scenario_id
        if scenario_id > 0 and not scenario.empty:
            scale = rng.lognormal(mean=-0.5 * 0.12**2, sigma=0.12, size=len(scenario))
            scenario["workload_volume"] = scenario["workload_volume"] * scale
            scenario["cpu_workload"] = scenario["cpu_workload"] * scale
            scenario["mem_workload"] = scenario["mem_workload"] * rng.lognormal(
                mean=-0.5 * 0.06**2, sigma=0.06, size=len(scenario)
            )
            scenario["workload_volume"] = scenario["workload_volume"].clip(lower=0.0)
            cpu_upper = scenario["family_id"].map(family_q_cpu).astype(float)
            mem_upper = scenario["family_id"].map(family_q_mem).astype(float)
            scenario["cpu_workload"] = np.clip(
                scenario["cpu_workload"].to_numpy(dtype=float),
                0.0, (cpu_upper * scenario["workload_volume"]).to_numpy(dtype=float),
            )
            scenario["mem_workload"] = np.clip(
                scenario["mem_workload"].to_numpy(dtype=float),
                0.0, (mem_upper * scenario["workload_volume"]).to_numpy(dtype=float),
            )
        frames.append(scenario[BATCH_WORKLOAD_COLUMNS])
    workload = pd.concat(frames, ignore_index=True)
    return workload.sort_values(["scenario_id", "family_id", "t_hour"]).reset_index(drop=True)


def build_batch_outputs(
    vm_requests: pd.DataFrame,
    observed_usage: pd.DataFrame,
    max_families: int = 10,
    num_scenarios: int = SCENARIO_COUNT,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """batch VM을 family 요약과 시간별 workload로 반환한다."""
    batch = vm_requests.loc[vm_requests["class"].eq("batch_candidate")].copy()
    if batch.empty:
        return (
            pd.DataFrame(columns=BATCH_FAMILY_COLUMNS),
            pd.DataFrame(columns=BATCH_WORKLOAD_COLUMNS),
        )
    assigned = _assign_batch_families(batch, max_families)
    family_stats = []
    for family_id, group in assigned.groupby("family_id", sort=True):
        q_cpu = float(group["q_cpu"].max())
        q_mem = float(group["q_mem"].max())
        avg_cpu = float(group["avg_cpu_usage"].mean())
        avg_mem = float(group["avg_mem_usage"].mean())
        family_stats.append({
            "family_id": family_id, "q_cpu_B": q_cpu, "q_mem_B": q_mem,
            "W_k": int(len(group)),
            "rho_cpu_B": min(1.0, avg_cpu / max(q_cpu, EPSILON)),
            "rho_mem_B": min(1.0, avg_mem / max(q_mem, EPSILON)),
            "base_cpu": avg_cpu, "base_mem": avg_mem,
            "startup_cpu": 0.10 * q_cpu, "startup_mem": 0.05 * q_mem,
        })
    families = pd.DataFrame(family_stats, columns=BATCH_FAMILY_COLUMNS)
    workload = _build_batch_workload(
        assigned, families, observed_usage, num_scenarios, seed
    )
    return families, workload


def main() -> None:
    """spot·batch 산출물 세 개를 저장한다."""
    vm_requests = pd.read_csv(VM_REQUEST_PATH, float_precision="round_trip")
    observed_usage = pd.read_parquet(OBSERVED_PATH)
    events = load_instance_events()
    spot = build_spot_preemption_scenarios(vm_requests, observed_usage, events)
    families, workload = build_batch_outputs(vm_requests, observed_usage)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    spot.to_csv(OUT_DIR / "spot_preemption_scenarios.csv", index=False)
    families.to_csv(OUT_DIR / "batch_families.csv", index=False)
    workload.to_csv(OUT_DIR / "batch_workload.csv", index=False)
    print(
        f"spot_rows={len(spot)}, batch_families={len(families)}, "
        f"batch_workload={len(workload)}"
    )


if __name__ == "__main__":
    main()
