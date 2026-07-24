from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from renewable_trace.sites import Site, site_from_mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "nrel_renewables_2019_v0.yaml"


@dataclass(frozen=True)
class ApiConfig:
    solar_endpoint: str
    wind_endpoint: str
    min_seconds_between_calls: float
    max_retries: int
    timeout_seconds: float


@dataclass(frozen=True)
class SolarConfig:
    attributes: list[str]
    ac_rated_power_w: float
    dc_ac_ratio: float
    inverter_nominal_efficiency: float
    system_losses: float
    gamma_pdc: float
    surface_azimuth: float
    default_albedo: float

    @property
    def module_pdc0_w(self) -> float:
        return self.ac_rated_power_w * self.dc_ac_ratio

    @property
    def inverter_pdc0_w(self) -> float:
        return self.ac_rated_power_w / self.inverter_nominal_efficiency


@dataclass(frozen=True)
class WindConfig:
    attributes: list[str]
    rated_power_mw: float
    hub_height_m: float
    cut_in_speed_mps: float
    rated_speed_mps: float
    cut_out_speed_mps: float
    wind_loss_factor: float


@dataclass(frozen=True)
class ScalingConfig:
    assumed_mean_demand_mw_per_site: float
    target_average_renewable_to_demand_ratio: float
    solar_energy_mix: float
    wind_energy_mix: float


@dataclass(frozen=True)
class OutputPaths:
    raw_solar_dir: Path
    raw_wind_dir: Path
    processed_dir: Path

    @property
    def cf_5min_parquet(self) -> Path:
        return self.processed_dir / "renewable_cf_2019_5min.parquet"

    @property
    def cf_5min_csv(self) -> Path:
        return self.processed_dir / "renewable_cf_2019_5min.csv.gz"

    @property
    def cf_30min_parquet(self) -> Path:
        return self.processed_dir / "renewable_cf_2019_30min.parquet"

    @property
    def cf_30min_csv(self) -> Path:
        return self.processed_dir / "renewable_cf_2019_30min.csv.gz"

    @property
    def power_30min_parquet(self) -> Path:
        return self.processed_dir / "renewable_power_2019_30min_assumed_100mw.parquet"

    @property
    def power_30min_csv(self) -> Path:
        return self.processed_dir / "renewable_power_2019_30min_assumed_100mw.csv.gz"

    @property
    def summary_by_site_csv(self) -> Path:
        return self.processed_dir / "renewable_cf_2019_summary_by_site.csv"

    @property
    def capacity_assumptions_csv(self) -> Path:
        return self.processed_dir / "site_capacity_assumptions_2019_assumed_100mw.csv"


@dataclass(frozen=True)
class PipelineConfig:
    year: int
    raw_interval_minutes: int
    placement_interval_minutes: int
    timezone: str
    leap_day: bool
    expected_raw_rows_per_site: int
    expected_cf_5min_total_rows: int
    expected_cf_30min_total_rows: int
    api: ApiConfig
    solar: SolarConfig
    wind: WindConfig
    scaling: ScalingConfig
    paths: OutputPaths
    sites: list[Site]

    @property
    def expected_start_utc(self):
        import pandas as pd

        return pd.Timestamp(f"{self.year}-01-01 00:00:00", tz="UTC")

    @property
    def expected_end_utc(self):
        import pandas as pd

        return pd.Timestamp(f"{self.year}-12-31 23:55:00", tz="UTC")


def resolve_repo_path(path_value: str | Path, repo_root: Path = REPO_ROOT) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else repo_root / path


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> PipelineConfig:
    config_path = resolve_repo_path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return parse_config(raw)


def parse_config(raw: dict[str, Any]) -> PipelineConfig:
    expected = raw["expected_rows"]
    api = raw["api"]
    solar = raw["solar"]
    wind = raw["wind"]
    scaling = raw["scaling"]
    paths = raw["paths"]

    return PipelineConfig(
        year=int(raw["year"]),
        raw_interval_minutes=int(raw["raw_interval_minutes"]),
        placement_interval_minutes=int(raw["placement_interval_minutes"]),
        timezone=str(raw["timezone"]),
        leap_day=bool(raw["leap_day"]),
        expected_raw_rows_per_site=int(expected["raw_5min_per_site"]),
        expected_cf_5min_total_rows=int(expected["cf_5min_total"]),
        expected_cf_30min_total_rows=int(expected["cf_30min_total"]),
        api=ApiConfig(
            solar_endpoint=str(api["solar_endpoint"]),
            wind_endpoint=str(api["wind_endpoint"]),
            min_seconds_between_calls=float(api["min_seconds_between_calls"]),
            max_retries=int(api["max_retries"]),
            timeout_seconds=float(api["timeout_seconds"]),
        ),
        solar=SolarConfig(
            attributes=[str(value) for value in solar["attributes"]],
            ac_rated_power_w=float(solar["ac_rated_power_w"]),
            dc_ac_ratio=float(solar["dc_ac_ratio"]),
            inverter_nominal_efficiency=float(solar["inverter_nominal_efficiency"]),
            system_losses=float(solar["system_losses"]),
            gamma_pdc=float(solar["gamma_pdc"]),
            surface_azimuth=float(solar["surface_azimuth"]),
            default_albedo=float(solar["default_albedo"]),
        ),
        wind=WindConfig(
            attributes=[str(value) for value in wind["attributes"]],
            rated_power_mw=float(wind["rated_power_mw"]),
            hub_height_m=float(wind["hub_height_m"]),
            cut_in_speed_mps=float(wind["cut_in_speed_mps"]),
            rated_speed_mps=float(wind["rated_speed_mps"]),
            cut_out_speed_mps=float(wind["cut_out_speed_mps"]),
            wind_loss_factor=float(wind["wind_loss_factor"]),
        ),
        scaling=ScalingConfig(
            assumed_mean_demand_mw_per_site=float(scaling["assumed_mean_demand_mw_per_site"]),
            target_average_renewable_to_demand_ratio=float(
                scaling["target_average_renewable_to_demand_ratio"]
            ),
            solar_energy_mix=float(scaling["solar_energy_mix"]),
            wind_energy_mix=float(scaling["wind_energy_mix"]),
        ),
        paths=OutputPaths(
            raw_solar_dir=resolve_repo_path(paths["raw_solar_dir"]),
            raw_wind_dir=resolve_repo_path(paths["raw_wind_dir"]),
            processed_dir=resolve_repo_path(paths["processed_dir"]),
        ),
        sites=[site_from_mapping(value) for value in raw["sites"]],
    )
