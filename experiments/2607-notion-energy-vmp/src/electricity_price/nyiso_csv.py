from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytz

from electricity_price.zones import NyisoZone, zone_lookup


PRICE_COLUMNS = [
    "da_lbmp_usd_per_mwh",
    "rt_lbmp_usd_per_mwh",
    "da_loss_raw_usd_per_mwh",
    "rt_loss_raw_usd_per_mwh",
    "da_congestion_raw_usd_per_mwh",
    "rt_congestion_raw_usd_per_mwh",
]

OUTPUT_5MIN_COLUMNS = [
    "timestamp_utc",
    "market",
    "site_id",
    "nyiso_zone",
    "ptid",
    "da_lbmp_usd_per_mwh",
    "rt_lbmp_usd_per_mwh",
    "da_loss_raw_usd_per_mwh",
    "rt_loss_raw_usd_per_mwh",
    "da_congestion_raw_usd_per_mwh",
    "rt_congestion_raw_usd_per_mwh",
]


class NyisoPriceError(ValueError):
    pass


def parse_lbmp_csv(
    file_obj,
    dataset: str,
    zones: list[NyisoZone],
    timezone_raw: str = "America/New_York",
    rt_interval_minutes: int = 5,
) -> pd.DataFrame:
    if dataset not in {"da", "rt"}:
        raise ValueError("dataset must be 'da' or 'rt'")

    raw = pd.read_csv(file_obj)
    required = [
        "Time Stamp",
        "Name",
        "PTID",
        "LBMP ($/MWHr)",
        "Marginal Cost Losses ($/MWHr)",
        "Marginal Cost Congestion ($/MWHr)",
    ]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise NyisoPriceError(f"Missing NYISO LBMP columns: {missing}")

    lookup = zone_lookup(zones)
    frame = raw[raw["Name"].isin(lookup)].copy()
    if frame.empty:
        return empty_price_frame(dataset)

    raw_local_naive = pd.to_datetime(frame["Time Stamp"], errors="raise")
    raw_timestamp_utc = local_timestamp_blocks_to_utc(raw_local_naive, timezone_raw)
    if dataset == "rt":
        frame["timestamp_utc"] = (
            raw_timestamp_utc - pd.Timedelta(minutes=rt_interval_minutes)
        ).dt.round(f"{rt_interval_minutes}min")
        local_year = frame["timestamp_utc"].dt.tz_convert(timezone_raw).dt.year
    else:
        frame["timestamp_utc"] = raw_timestamp_utc
        local_year = raw_local_naive.dt.year
    frame["_local_year"] = local_year
    frame["market"] = "NYISO"
    frame["site_id"] = frame["Name"].map(lambda name: lookup[name].site_id)
    frame["nyiso_zone"] = frame["Name"]
    frame["ptid"] = frame["PTID"].astype(int)

    if dataset == "da":
        frame["da_lbmp_usd_per_mwh"] = pd.to_numeric(frame["LBMP ($/MWHr)"], errors="raise")
        frame["da_loss_raw_usd_per_mwh"] = pd.to_numeric(
            frame["Marginal Cost Losses ($/MWHr)"], errors="raise"
        )
        frame["da_congestion_raw_usd_per_mwh"] = pd.to_numeric(
            frame["Marginal Cost Congestion ($/MWHr)"], errors="raise"
        )
        columns = [
            "timestamp_utc",
            "_local_year",
            "market",
            "site_id",
            "nyiso_zone",
            "ptid",
            "da_lbmp_usd_per_mwh",
            "da_loss_raw_usd_per_mwh",
            "da_congestion_raw_usd_per_mwh",
        ]
    else:
        frame["rt_lbmp_usd_per_mwh"] = pd.to_numeric(frame["LBMP ($/MWHr)"], errors="raise")
        frame["rt_loss_raw_usd_per_mwh"] = pd.to_numeric(
            frame["Marginal Cost Losses ($/MWHr)"], errors="raise"
        )
        frame["rt_congestion_raw_usd_per_mwh"] = pd.to_numeric(
            frame["Marginal Cost Congestion ($/MWHr)"], errors="raise"
        )
        columns = [
            "timestamp_utc",
            "_local_year",
            "market",
            "site_id",
            "nyiso_zone",
            "ptid",
            "rt_lbmp_usd_per_mwh",
            "rt_loss_raw_usd_per_mwh",
            "rt_congestion_raw_usd_per_mwh",
        ]

    return (
        collapse_duplicate_prices(frame[columns], dataset=dataset)
        .sort_values(["site_id", "timestamp_utc"])
        .reset_index(drop=True)
    )


def local_timestamp_blocks_to_utc(local_naive: pd.Series, timezone_raw: str) -> pd.Series:
    block_id = local_naive.ne(local_naive.shift()).cumsum()
    blocks = pd.DataFrame({"block_id": block_id, "local_naive": local_naive}).drop_duplicates("block_id")
    try:
        localized = blocks["local_naive"].dt.tz_localize(
            timezone_raw,
            ambiguous="infer",
            nonexistent="shift_forward",
        )
    except pytz.AmbiguousTimeError:
        localized = blocks["local_naive"].dt.tz_localize(
            timezone_raw,
            ambiguous=False,
            nonexistent="shift_forward",
        )
    blocks["timestamp_utc"] = localized.dt.tz_convert("UTC")
    mapped = block_id.map(blocks.set_index("block_id")["timestamp_utc"])
    return pd.Series(pd.DatetimeIndex(mapped), index=local_naive.index)


def empty_price_frame(dataset: str) -> pd.DataFrame:
    base = ["timestamp_utc", "_local_year", "market", "site_id", "nyiso_zone", "ptid"]
    if dataset == "da":
        return pd.DataFrame(
            columns=[
                *base,
                "da_lbmp_usd_per_mwh",
                "da_loss_raw_usd_per_mwh",
                "da_congestion_raw_usd_per_mwh",
            ]
        )
    return pd.DataFrame(
        columns=[
            *base,
            "rt_lbmp_usd_per_mwh",
            "rt_loss_raw_usd_per_mwh",
            "rt_congestion_raw_usd_per_mwh",
        ]
    )


def collapse_duplicate_prices(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    if dataset == "da":
        value_columns = [
            "da_lbmp_usd_per_mwh",
            "da_loss_raw_usd_per_mwh",
            "da_congestion_raw_usd_per_mwh",
        ]
    elif dataset == "rt":
        value_columns = [
            "rt_lbmp_usd_per_mwh",
            "rt_loss_raw_usd_per_mwh",
            "rt_congestion_raw_usd_per_mwh",
        ]
    else:
        raise ValueError("dataset must be 'da' or 'rt'")
    group_columns = ["timestamp_utc", "_local_year", "market", "site_id", "nyiso_zone", "ptid"]
    return (
        frame.groupby(group_columns, as_index=False, sort=True)
        .agg(**{column: (column, "mean") for column in value_columns})
        .sort_values(["site_id", "timestamp_utc"])
        .reset_index(drop=True)
    )


def read_lbmp_zip(zip_path: Path, dataset: str, zones: list[NyisoZone], year: int) -> pd.DataFrame:
    frames = []
    with ZipFile(zip_path) as archive:
        for member in sorted(name for name in archive.namelist() if name.endswith(".csv")):
            with archive.open(member) as file:
                frame = parse_lbmp_csv(file, dataset=dataset, zones=zones)
                if not frame.empty:
                    frame = frame[frame["_local_year"] == year]
                    frames.append(frame)
    if not frames:
        return empty_price_frame(dataset)
    return pd.concat(frames, ignore_index=True).sort_values(["site_id", "timestamp_utc"])


def read_lbmp_zips(zip_paths: Iterable[Path], dataset: str, zones: list[NyisoZone], year: int) -> pd.DataFrame:
    frames = [read_lbmp_zip(path, dataset=dataset, zones=zones, year=year) for path in sorted(zip_paths)]
    if not frames:
        return empty_price_frame(dataset)
    combined = pd.concat(frames, ignore_index=True)
    return (
        collapse_duplicate_prices(combined, dataset=dataset)
        .sort_values(["site_id", "timestamp_utc"])
        .reset_index(drop=True)
    )


def align_day_ahead_to_realtime_grid(da: pd.DataFrame, rt: pd.DataFrame) -> pd.DataFrame:
    outputs = []
    da_columns = [
        "timestamp_utc",
        "site_id",
        "da_lbmp_usd_per_mwh",
        "da_loss_raw_usd_per_mwh",
        "da_congestion_raw_usd_per_mwh",
    ]
    for site_id, rt_site in rt.groupby("site_id", sort=True):
        da_site = da.loc[da["site_id"] == site_id, da_columns].sort_values("timestamp_utc")
        if da_site.empty:
            raise NyisoPriceError(f"No day-ahead rows found for site_id={site_id}")
        merged = pd.merge_asof(
            rt_site.sort_values("timestamp_utc"),
            da_site,
            on="timestamp_utc",
            by="site_id",
            direction="backward",
            tolerance=pd.Timedelta(minutes=59),
        )
        outputs.append(merged)
    aligned = pd.concat(outputs, ignore_index=True)
    aligned = aligned[OUTPUT_5MIN_COLUMNS].sort_values(["site_id", "timestamp_utc"]).reset_index(drop=True)
    return aligned


def validate_price_frame(
    frame: pd.DataFrame,
    expected_rows: int | None,
    expected_rows_per_site: int | None,
    label: str = "NYISO price frame",
) -> None:
    if expected_rows is not None and len(frame) != expected_rows:
        raise NyisoPriceError(f"{label} has {len(frame)} rows; expected {expected_rows}")
    if frame.duplicated(["timestamp_utc", "site_id"]).any():
        raise NyisoPriceError(f"{label} has duplicated timestamp_utc/site_id pairs")
    for column in ["da_lbmp_usd_per_mwh", "rt_lbmp_usd_per_mwh"]:
        if column in frame.columns and frame[column].isna().any():
            raise NyisoPriceError(f"{label} has missing values in {column}")
    if expected_rows_per_site is not None:
        counts = frame.groupby("site_id").size()
        bad_counts = counts[counts != expected_rows_per_site]
        if not bad_counts.empty:
            raise NyisoPriceError(
                f"{label} has unexpected rows per site: {bad_counts.to_dict()}, "
                f"expected {expected_rows_per_site}"
            )


def regularize_price_grid(
    frame: pd.DataFrame,
    dataset: str,
    zones: list[NyisoZone],
    grid: pd.DatetimeIndex,
) -> pd.DataFrame:
    if dataset == "da":
        value_columns = [
            "da_lbmp_usd_per_mwh",
            "da_loss_raw_usd_per_mwh",
            "da_congestion_raw_usd_per_mwh",
        ]
    elif dataset == "rt":
        value_columns = [
            "rt_lbmp_usd_per_mwh",
            "rt_loss_raw_usd_per_mwh",
            "rt_congestion_raw_usd_per_mwh",
        ]
    else:
        raise ValueError("dataset must be 'da' or 'rt'")

    outputs = []
    for zone in zones:
        site = frame[frame["site_id"] == zone.site_id].copy()
        if site.empty:
            raise NyisoPriceError(f"No NYISO {dataset} rows found for site_id={zone.site_id}")
        site = collapse_duplicate_prices(site, dataset=dataset).set_index("timestamp_utc").sort_index()
        site = site.reindex(grid)
        site.index.name = "timestamp_utc"
        site["market"] = "NYISO"
        site["site_id"] = zone.site_id
        site["nyiso_zone"] = zone.nyiso_zone
        site["ptid"] = site["ptid"].ffill().bfill().astype(int)
        site["_local_year"] = site["_local_year"].ffill().bfill().astype(int)
        for column in value_columns:
            site[column] = site[column].interpolate(method="time").ffill().bfill()
        outputs.append(site.reset_index())
    return pd.concat(outputs, ignore_index=True).sort_values(["site_id", "timestamp_utc"]).reset_index(drop=True)


def report_extreme_prices(frame: pd.DataFrame, threshold: float = 1000.0) -> pd.DataFrame:
    columns = [column for column in ["da_lbmp_usd_per_mwh", "rt_lbmp_usd_per_mwh"] if column in frame.columns]
    if not columns:
        return frame.iloc[0:0].copy()
    mask = pd.Series(False, index=frame.index)
    for column in columns:
        mask = mask | frame[column].abs().gt(threshold)
    return frame.loc[mask].copy()
