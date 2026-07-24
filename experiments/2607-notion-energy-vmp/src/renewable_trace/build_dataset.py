from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from renewable_trace.config import PipelineConfig
from renewable_trace.nrel_csv import read_nrel_timeseries_csv
from renewable_trace.solar_cf import resource_to_solar_cf
from renewable_trace.wind_cf import resource_to_wind_cf


@dataclass(frozen=True)
class BuildOutputs:
    cf_5min_parquet: Path
    cf_5min_csv: Path
    cf_30min_parquet: Path
    cf_30min_csv: Path
    power_30min_parquet: Path
    power_30min_csv: Path
    summary_by_site_csv: Path
    capacity_assumptions_csv: Path


def build_processed_datasets(config: PipelineConfig) -> BuildOutputs:
    config.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    cf_5min = build_capacity_factor_5min(config)
    write_dataframe_pair(cf_5min, config.paths.cf_5min_parquet, config.paths.cf_5min_csv)

    summary = summarize_by_site(cf_5min)
    summary.to_csv(config.paths.summary_by_site_csv, index=False)

    cf_30min = aggregate_to_30min(cf_5min, config)
    write_dataframe_pair(cf_30min, config.paths.cf_30min_parquet, config.paths.cf_30min_csv)

    capacity_assumptions = build_capacity_assumptions(cf_5min, config)
    capacity_assumptions.to_csv(config.paths.capacity_assumptions_csv, index=False)

    power_30min = build_scaled_power_30min(cf_30min, capacity_assumptions)
    write_dataframe_pair(power_30min, config.paths.power_30min_parquet, config.paths.power_30min_csv)

    return BuildOutputs(
        cf_5min_parquet=config.paths.cf_5min_parquet,
        cf_5min_csv=config.paths.cf_5min_csv,
        cf_30min_parquet=config.paths.cf_30min_parquet,
        cf_30min_csv=config.paths.cf_30min_csv,
        power_30min_parquet=config.paths.power_30min_parquet,
        power_30min_csv=config.paths.power_30min_csv,
        summary_by_site_csv=config.paths.summary_by_site_csv,
        capacity_assumptions_csv=config.paths.capacity_assumptions_csv,
    )


def build_capacity_factor_5min(config: PipelineConfig) -> pd.DataFrame:
    site_frames = []
    for site in config.sites:
        solar_raw = read_nrel_timeseries_csv(
            config.paths.raw_solar_dir / f"{site.site_id}.csv",
            expected_fields=config.solar.attributes,
            expected_rows=config.expected_raw_rows_per_site,
            expected_start_utc=config.expected_start_utc,
            expected_end_utc=config.expected_end_utc,
        )
        wind_raw = read_nrel_timeseries_csv(
            config.paths.raw_wind_dir / f"{site.site_id}.csv",
            expected_fields=config.wind.attributes,
            expected_rows=config.expected_raw_rows_per_site,
            expected_start_utc=config.expected_start_utc,
            expected_end_utc=config.expected_end_utc,
        )
        solar = resource_to_solar_cf(solar_raw.frame, site, config.solar)
        wind = resource_to_wind_cf(wind_raw.frame, site)
        site_cf = solar.merge(wind, on=["timestamp_utc", "site_id", "state", "lat", "lon"], how="inner")
        if len(site_cf) != config.expected_raw_rows_per_site:
            raise ValueError(
                f"{site.site_id} merged solar/wind rows={len(site_cf)}; "
                f"expected {config.expected_raw_rows_per_site}"
            )
        site_frames.append(site_cf)

    cf_5min = pd.concat(site_frames, ignore_index=True)
    cf_5min = cf_5min[
        ["timestamp_utc", "site_id", "state", "lat", "lon", "solar_cf", "wind_cf"]
    ].sort_values(["site_id", "timestamp_utc"])
    validate_capacity_factor_dataset(
        cf_5min,
        expected_rows=config.expected_cf_5min_total_rows,
        label="5-minute capacity-factor dataset",
    )
    return cf_5min.reset_index(drop=True)


def aggregate_to_30min(cf_5min: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    frame = cf_5min.copy()
    frame["timestamp_utc"] = frame["timestamp_utc"].dt.floor(f"{config.placement_interval_minutes}min")
    group_columns = ["timestamp_utc", "site_id", "state", "lat", "lon"]
    grouped = frame.groupby(group_columns, as_index=False, sort=True).agg(
        solar_cf=("solar_cf", "mean"),
        wind_cf=("wind_cf", "mean"),
        source_rows=("solar_cf", "size"),
    )
    invalid_groups = grouped[grouped["source_rows"] != 6]
    if not invalid_groups.empty:
        raise ValueError(
            "30-minute aggregation found groups without exactly six 5-minute rows: "
            f"{invalid_groups.head().to_dict(orient='records')}"
        )
    cf_30min = grouped.drop(columns=["source_rows"])
    validate_capacity_factor_dataset(
        cf_30min,
        expected_rows=config.expected_cf_30min_total_rows,
        label="30-minute capacity-factor dataset",
    )
    return cf_30min


def summarize_by_site(cf_5min: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (site_id, state, lat, lon), group in cf_5min.groupby(["site_id", "state", "lat", "lon"], sort=True):
        rows.append(
            {
                "site_id": site_id,
                "state": state,
                "lat": lat,
                "lon": lon,
                "solar_cf_mean": group["solar_cf"].mean(),
                "solar_cf_p05": group["solar_cf"].quantile(0.05),
                "solar_cf_p50": group["solar_cf"].quantile(0.50),
                "solar_cf_p95": group["solar_cf"].quantile(0.95),
                "wind_cf_mean": group["wind_cf"].mean(),
                "wind_cf_p05": group["wind_cf"].quantile(0.05),
                "wind_cf_p50": group["wind_cf"].quantile(0.50),
                "wind_cf_p95": group["wind_cf"].quantile(0.95),
                "missing_solar_cf_count": int(group["solar_cf"].isna().sum()),
                "missing_wind_cf_count": int(group["wind_cf"].isna().sum()),
            }
        )
    return pd.DataFrame(rows)


def build_capacity_assumptions(cf_5min: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    demand = config.scaling.assumed_mean_demand_mw_per_site
    target_avg_renewable = config.scaling.target_average_renewable_to_demand_ratio * demand
    target_avg_solar = config.scaling.solar_energy_mix * target_avg_renewable
    target_avg_wind = config.scaling.wind_energy_mix * target_avg_renewable

    rows = []
    for (site_id, state, lat, lon), group in cf_5min.groupby(["site_id", "state", "lat", "lon"], sort=True):
        solar_mean = group["solar_cf"].mean()
        wind_mean = group["wind_cf"].mean()
        if pd.isna(solar_mean) or solar_mean <= 0:
            raise ValueError(f"{site_id} has invalid solar_cf mean: {solar_mean}")
        if pd.isna(wind_mean) or wind_mean <= 0:
            raise ValueError(f"{site_id} has invalid wind_cf mean: {wind_mean}")
        rows.append(
            {
                "site_id": site_id,
                "state": state,
                "lat": lat,
                "lon": lon,
                "assumed_mean_demand_mw": demand,
                "target_avg_renewable_mw": target_avg_renewable,
                "target_avg_solar_mw": target_avg_solar,
                "target_avg_wind_mw": target_avg_wind,
                "solar_cf_mean": solar_mean,
                "wind_cf_mean": wind_mean,
                "solar_capacity_mw": target_avg_solar / solar_mean,
                "wind_capacity_mw": target_avg_wind / wind_mean,
            }
        )
    return pd.DataFrame(rows)


def build_scaled_power_30min(cf_30min: pd.DataFrame, capacity_assumptions: pd.DataFrame) -> pd.DataFrame:
    capacity_columns = [
        "site_id",
        "assumed_mean_demand_mw",
        "solar_capacity_mw",
        "wind_capacity_mw",
    ]
    frame = cf_30min.merge(capacity_assumptions[capacity_columns], on="site_id", how="left")
    if frame[["solar_capacity_mw", "wind_capacity_mw"]].isna().any().any():
        raise ValueError("Missing capacity assumptions for at least one site")
    frame["solar_power_mw"] = frame["solar_capacity_mw"] * frame["solar_cf"]
    frame["wind_power_mw"] = frame["wind_capacity_mw"] * frame["wind_cf"]
    frame["renewable_power_mw"] = frame["solar_power_mw"] + frame["wind_power_mw"]
    frame["renewable_energy_mwh"] = frame["renewable_power_mw"] * 0.5
    if (frame["renewable_power_mw"] < 0).any() or (frame["renewable_energy_mwh"] < 0).any():
        raise ValueError("Scaled renewable power output contains negative values")
    return frame[
        [
            "timestamp_utc",
            "site_id",
            "state",
            "lat",
            "lon",
            "assumed_mean_demand_mw",
            "solar_capacity_mw",
            "wind_capacity_mw",
            "solar_power_mw",
            "wind_power_mw",
            "renewable_power_mw",
            "renewable_energy_mwh",
        ]
    ].sort_values(["site_id", "timestamp_utc"]).reset_index(drop=True)


def validate_capacity_factor_dataset(frame: pd.DataFrame, expected_rows: int, label: str) -> None:
    if len(frame) != expected_rows:
        raise ValueError(f"{label} has {len(frame)} rows; expected {expected_rows}")
    if frame.duplicated(["site_id", "timestamp_utc"]).any():
        raise ValueError(f"{label} has duplicated site_id/timestamp_utc pairs")
    for column in ["solar_cf", "wind_cf"]:
        if frame[column].dropna().lt(0.0).any() or frame[column].dropna().gt(1.0).any():
            raise ValueError(f"{label} column {column} contains values outside [0, 1]")


def write_dataframe_pair(frame: pd.DataFrame, parquet_path: Path, csv_path: Path) -> None:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(parquet_path, index=False)
    frame.to_csv(csv_path, index=False, compression="gzip")
