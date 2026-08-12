"""trace와 무관한 서버·전기요금·확률·모델 파라미터 입력을 만든다.
고정 설정은 num_servers=6, num_scenarios=5, seed=42이고 24시간을 사용한다.
검증: canonical의 네 static 산출물과 행 순서·문자열까지 일치해야 한다.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .paths import WORK_DIR


EXPERIMENT_DIR = WORK_DIR.parent
OUT_DIR = EXPERIMENT_DIR / "out"
NUM_SERVERS = 6
NUM_SCENARIOS = 5
SEED = 42
HORIZON_HOURS = 24
ENERGY_COLUMNS = [
    "scenario_id", "t_hour", "day_ahead_price", "real_time_price", "sell_price",
    "renewable_generation", "ess_capacity", "ess_charge_max", "ess_discharge_max",
    "ess_charge_efficiency", "ess_discharge_efficiency",
]


def build_servers(num_servers: int = NUM_SERVERS) -> pd.DataFrame:
    """대표 서버별 용량·전력·최소 on/off 시간을 반환한다."""
    return pd.DataFrame([
        {
            "server_id": f"s{i:03d}", "C_cpu": 1.0, "C_mem": 1.0,
            "E_idle": 0.35, "E_cpu": 0.65, "min_on_time": 1, "min_off_time": 1,
        }
        for i in range(num_servers)
    ])


def _energy_rows(num_scenarios: int, seed: int) -> list[dict[str, float | int]]:
    """시나리오별 24시간 전기요금과 renewable 행을 반환한다."""
    rng = np.random.default_rng(seed + 43)
    hours = np.arange(HORIZON_HOURS)
    base_price = 45.0 + 8.0 * np.sin((hours - 7) / 24.0 * 2 * np.pi)
    base_price += 10.0 * np.exp(-0.5 * ((hours - 18) / 3.0) ** 2)
    base_price = np.maximum(base_price, 20.0)
    solar_shape = np.exp(-0.5 * ((hours - 13) / 3.0) ** 2)
    solar_shape[(hours < 6) | (hours > 19)] = 0.0
    rows = []
    for scenario_id in range(num_scenarios):
        da_noise = rng.normal(0.0, 1.5) if scenario_id > 0 else 0.0
        rt_noise = (
            rng.normal(0.0, 4.0, size=HORIZON_HOURS)
            if scenario_id > 0 else np.zeros(HORIZON_HOURS)
        )
        renewable_noise = (
            rng.lognormal(mean=-0.5 * 0.18**2, sigma=0.18, size=HORIZON_HOURS)
            if scenario_id > 0 else np.ones(HORIZON_HOURS)
        )
        day_ahead = np.maximum(base_price + da_noise, 0.0)
        real_time = np.maximum(day_ahead + rt_noise, -20.0)
        renewable = np.maximum(0.0, 0.75 * solar_shape * renewable_noise)
        sell_price = np.minimum(day_ahead * 0.85, real_time * 0.95)
        for hour in hours:
            rows.append({
                "scenario_id": scenario_id, "t_hour": int(hour),
                "day_ahead_price": float(day_ahead[hour]),
                "real_time_price": float(real_time[hour]),
                "sell_price": float(sell_price[hour]),
                "renewable_generation": float(renewable[hour]),
                "ess_capacity": 1.0, "ess_charge_max": 0.25,
                "ess_discharge_max": 0.25, "ess_charge_efficiency": 0.92,
                "ess_discharge_efficiency": 0.92,
            })
    return rows


def build_energy_scenarios(
    num_scenarios: int = NUM_SCENARIOS, seed: int = SEED
) -> pd.DataFrame:
    """시나리오별 24시간 energy 입력을 반환한다."""
    return pd.DataFrame(_energy_rows(num_scenarios, seed), columns=ENERGY_COLUMNS)


def build_scenario_probabilities(num_scenarios: int = NUM_SCENARIOS) -> pd.DataFrame:
    """각 scenario에 동일한 확률을 부여한 DataFrame을 반환한다."""
    probability = 1.0 / num_scenarios
    return pd.DataFrame([
        {"scenario_id": scenario_id, "probability": probability}
        for scenario_id in range(num_scenarios)
    ], columns=["scenario_id", "probability"])


def build_model_params() -> dict[str, float]:
    """확률 제약·ESS 초기 상태·효율 모델 파라미터를 반환한다."""
    return {
        "alpha": 0.95, "epsilon": 0.05, "soc_init": 0.50,
        "ess_charge_efficiency": 0.92, "ess_discharge_efficiency": 0.92,
    }


def main() -> None:
    """static 산출물 네 개를 저장한다."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_servers().to_csv(OUT_DIR / "servers.csv", index=False)
    build_energy_scenarios().to_csv(OUT_DIR / "energy_scenarios.csv", index=False)
    build_scenario_probabilities().to_csv(
        OUT_DIR / "scenario_probabilities.csv", index=False
    )
    model_params = json.dumps(build_model_params(), indent=2)
    (OUT_DIR / "model_params.json").write_text(model_params, encoding="utf-8")
    print("servers=6, energy_scenarios=120, scenario_probabilities=5")


if __name__ == "__main__":
    main()
