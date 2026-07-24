from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from electricity_price.config import NyisoPriceConfig
from electricity_price.nyiso_csv import (
    PRICE_COLUMNS,
    align_day_ahead_to_realtime_grid,
    read_lbmp_zips,
    regularize_price_grid,
    report_extreme_prices,
    validate_price_frame,
)


@dataclass(frozen=True)
class NyisoBuildOutputs:
    price_5min_parquet: Path
    price_5min_csv: Path
    price_30min_parquet: Path
    price_30min_csv: Path
    summary_by_zone_csv: Path


def build_processed_datasets(config: NyisoPriceConfig) -> NyisoBuildOutputs:
    config.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    price_5min = build_5min_prices(config)
    write_dataframe_pair(price_5min, config.paths.price_5min_parquet, config.paths.price_5min_csv)

    price_30min = aggregate_to_30min(price_5min, config)
    write_dataframe_pair(price_30min, config.paths.price_30min_parquet, config.paths.price_30min_csv)

    summary = summarize_by_zone(price_5min)
    summary.to_csv(config.paths.summary_by_zone_csv, index=False)

    return NyisoBuildOutputs(
        price_5min_parquet=config.paths.price_5min_parquet,
        price_5min_csv=config.paths.price_5min_csv,
        price_30min_parquet=config.paths.price_30min_parquet,
        price_30min_csv=config.paths.price_30min_csv,
        summary_by_zone_csv=config.paths.summary_by_zone_csv,
    )


def build_5min_prices(config: NyisoPriceConfig) -> pd.DataFrame:
    da_zip_paths = sorted(config.paths.raw_da_dir.glob("*.zip"))
    rt_zip_paths = sorted(config.paths.raw_rt_dir.glob("*.zip"))
    if len(da_zip_paths) != 12:
        raise FileNotFoundError(f"Expected 12 NYISO DA ZIPs under {config.paths.raw_da_dir}, found {len(da_zip_paths)}")
    if len(rt_zip_paths) != 12:
        raise FileNotFoundError(f"Expected 12 NYISO RT ZIPs under {config.paths.raw_rt_dir}, found {len(rt_zip_paths)}")

    da = read_lbmp_zips(da_zip_paths, dataset="da", zones=config.zones, year=config.year)
    rt = read_lbmp_zips(rt_zip_paths, dataset="rt", zones=config.zones, year=config.year)
    rt_grid = pd.date_range(
        pd.Timestamp(f"{config.year}-01-01 00:00:00", tz=config.timezone_raw).tz_convert("UTC"),
        periods=config.expected_price_5min_rows_per_site,
        freq=f"{config.final_interval_minutes}min",
    )
    da_grid = pd.date_range(
        pd.Timestamp(f"{config.year}-01-01 00:00:00", tz=config.timezone_raw).tz_convert("UTC"),
        periods=config.expected_price_5min_rows_per_site // 12,
        freq=f"{config.da_native_interval_minutes}min",
    )
    da = regularize_price_grid(da, dataset="da", zones=config.zones, grid=da_grid)
    rt = regularize_price_grid(rt, dataset="rt", zones=config.zones, grid=rt_grid)
    price_5min = align_day_ahead_to_realtime_grid(da, rt)
    validate_price_frame(
        price_5min,
        expected_rows=config.expected_price_5min_total_rows,
        expected_rows_per_site=config.expected_price_5min_rows_per_site,
        label="NYISO 5-minute price dataset",
    )
    return price_5min


def aggregate_to_30min(price_5min: pd.DataFrame, config: NyisoPriceConfig) -> pd.DataFrame:
    frame = price_5min.copy()
    frame["timestamp_utc"] = frame["timestamp_utc"].dt.floor(f"{config.placement_interval_minutes}min")
    group_columns = ["timestamp_utc", "market", "site_id", "nyiso_zone", "ptid"]
    aggregations = {column: (column, "mean") for column in PRICE_COLUMNS}
    aggregations["source_rows"] = ("rt_lbmp_usd_per_mwh", "size")
    grouped = frame.groupby(group_columns, as_index=False, sort=True).agg(**aggregations)
    invalid = grouped[grouped["source_rows"] != 6]
    if not invalid.empty:
        raise ValueError(
            "NYISO 30-minute aggregation found groups without exactly six 5-minute rows: "
            f"{invalid.head().to_dict(orient='records')}"
        )
    price_30min = grouped.drop(columns=["source_rows"])
    validate_price_frame(
        price_30min,
        expected_rows=config.expected_price_30min_total_rows,
        expected_rows_per_site=config.expected_price_30min_rows_per_site,
        label="NYISO 30-minute price dataset",
    )
    return price_30min


def summarize_by_zone(price_5min: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (site_id, nyiso_zone, ptid), group in price_5min.groupby(["site_id", "nyiso_zone", "ptid"], sort=True):
        rows.append(
            {
                "site_id": site_id,
                "nyiso_zone": nyiso_zone,
                "ptid": int(ptid),
                "da_lbmp_mean": group["da_lbmp_usd_per_mwh"].mean(),
                "da_lbmp_p01": group["da_lbmp_usd_per_mwh"].quantile(0.01),
                "da_lbmp_p05": group["da_lbmp_usd_per_mwh"].quantile(0.05),
                "da_lbmp_p50": group["da_lbmp_usd_per_mwh"].quantile(0.50),
                "da_lbmp_p95": group["da_lbmp_usd_per_mwh"].quantile(0.95),
                "da_lbmp_p99": group["da_lbmp_usd_per_mwh"].quantile(0.99),
                "da_lbmp_min": group["da_lbmp_usd_per_mwh"].min(),
                "da_lbmp_max": group["da_lbmp_usd_per_mwh"].max(),
                "rt_lbmp_mean": group["rt_lbmp_usd_per_mwh"].mean(),
                "rt_lbmp_p01": group["rt_lbmp_usd_per_mwh"].quantile(0.01),
                "rt_lbmp_p05": group["rt_lbmp_usd_per_mwh"].quantile(0.05),
                "rt_lbmp_p50": group["rt_lbmp_usd_per_mwh"].quantile(0.50),
                "rt_lbmp_p95": group["rt_lbmp_usd_per_mwh"].quantile(0.95),
                "rt_lbmp_p99": group["rt_lbmp_usd_per_mwh"].quantile(0.99),
                "rt_lbmp_min": group["rt_lbmp_usd_per_mwh"].min(),
                "rt_lbmp_max": group["rt_lbmp_usd_per_mwh"].max(),
                "negative_da_lbmp_count": int(group["da_lbmp_usd_per_mwh"].lt(0).sum()),
                "negative_rt_lbmp_count": int(group["rt_lbmp_usd_per_mwh"].lt(0).sum()),
                "missing_da_lbmp_count": int(group["da_lbmp_usd_per_mwh"].isna().sum()),
                "missing_rt_lbmp_count": int(group["rt_lbmp_usd_per_mwh"].isna().sum()),
            }
        )
    return pd.DataFrame(rows)


def write_dataframe_pair(frame: pd.DataFrame, parquet_path: Path, csv_path: Path) -> None:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(parquet_path, index=False)
    frame.to_csv(csv_path, index=False, compression="gzip")
