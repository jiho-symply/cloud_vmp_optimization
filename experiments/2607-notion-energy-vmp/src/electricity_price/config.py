from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from electricity_price.zones import NyisoZone, zone_from_mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "nyiso_prices_2019_v0.yaml"


@dataclass(frozen=True)
class NyisoApiConfig:
    da_url_template: str
    rt_url_template: str
    max_retries: int
    timeout_seconds: float
    min_seconds_between_calls: float


@dataclass(frozen=True)
class NyisoPaths:
    raw_da_dir: Path
    raw_rt_dir: Path
    processed_dir: Path

    @property
    def price_5min_parquet(self) -> Path:
        return self.processed_dir / "electricity_price_nyiso_2019_5min.parquet"

    @property
    def price_5min_csv(self) -> Path:
        return self.processed_dir / "electricity_price_nyiso_2019_5min.csv.gz"

    @property
    def price_30min_parquet(self) -> Path:
        return self.processed_dir / "electricity_price_nyiso_2019_30min.parquet"

    @property
    def price_30min_csv(self) -> Path:
        return self.processed_dir / "electricity_price_nyiso_2019_30min.csv.gz"

    @property
    def summary_by_zone_csv(self) -> Path:
        return self.processed_dir / "electricity_price_nyiso_2019_summary_by_zone.csv"


@dataclass(frozen=True)
class NyisoPriceConfig:
    market: str
    year: int
    timezone_raw: str
    timezone_output: str
    raw_interval_rt_minutes: int
    da_native_interval_minutes: int
    final_interval_minutes: int
    placement_interval_minutes: int
    expected_price_5min_total_rows: int
    expected_price_30min_total_rows: int
    expected_price_5min_rows_per_site: int
    expected_price_30min_rows_per_site: int
    api: NyisoApiConfig
    paths: NyisoPaths
    zones: list[NyisoZone]


def resolve_repo_path(path_value: str | Path, repo_root: Path = REPO_ROOT) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else repo_root / path


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> NyisoPriceConfig:
    config_path = resolve_repo_path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return parse_config(raw)


def parse_config(raw: dict[str, Any]) -> NyisoPriceConfig:
    expected = raw["expected_rows"]
    api = raw["api"]
    paths = raw["paths"]
    return NyisoPriceConfig(
        market=str(raw["market"]),
        year=int(raw["year"]),
        timezone_raw=str(raw["timezone_raw"]),
        timezone_output=str(raw["timezone_output"]),
        raw_interval_rt_minutes=int(raw["raw_interval_rt_minutes"]),
        da_native_interval_minutes=int(raw["da_native_interval_minutes"]),
        final_interval_minutes=int(raw["final_interval_minutes"]),
        placement_interval_minutes=int(raw["placement_interval_minutes"]),
        expected_price_5min_total_rows=int(expected["price_5min_total"]),
        expected_price_30min_total_rows=int(expected["price_30min_total"]),
        expected_price_5min_rows_per_site=int(expected["price_5min_per_site"]),
        expected_price_30min_rows_per_site=int(expected["price_30min_per_site"]),
        api=NyisoApiConfig(
            da_url_template=str(api["da_url_template"]),
            rt_url_template=str(api["rt_url_template"]),
            max_retries=int(api["max_retries"]),
            timeout_seconds=float(api["timeout_seconds"]),
            min_seconds_between_calls=float(api["min_seconds_between_calls"]),
        ),
        paths=NyisoPaths(
            raw_da_dir=resolve_repo_path(paths["raw_da_dir"]),
            raw_rt_dir=resolve_repo_path(paths["raw_rt_dir"]),
            processed_dir=resolve_repo_path(paths["processed_dir"]),
        ),
        zones=[zone_from_mapping(value) for value in raw["zones"]],
    )
