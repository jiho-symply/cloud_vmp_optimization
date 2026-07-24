from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


TIMESTAMP_COLUMNS = ["year", "month", "day", "hour", "minute"]

FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "ghi": ("ghi", "global_horizontal_irradiance"),
    "dni": ("dni", "direct_normal_irradiance"),
    "dhi": ("dhi", "diffuse_horizontal_irradiance"),
    "air_temperature": ("air_temperature", "temperature", "temp_air", "air_temp"),
    "wind_speed": ("wind_speed", "windspeed", "wind_speed_10m"),
    "solar_zenith_angle": ("solar_zenith_angle", "solar_zenith"),
    "surface_albedo": ("surface_albedo", "albedo"),
    "surface_pressure": ("surface_pressure", "pressure"),
    "windspeed_80m": (
        "windspeed_80m",
        "wind_speed_80m",
        "windspeed_at_80m",
        "wind_speed_at_80m",
    ),
    "winddirection_80m": (
        "winddirection_80m",
        "wind_direction_80m",
        "winddirection_at_80m",
        "wind_direction_at_80m",
    ),
}


class NrelCsvError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedNrelCsv:
    metadata: dict[str, str]
    frame: pd.DataFrame
    column_map: dict[str, str]


def normalize_column_name(name: str) -> str:
    text = name.strip().lower()
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\[[^]]*\]", "", text)
    text = text.replace("%", "percent")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def _compact(value: str) -> str:
    return normalize_column_name(value).replace("_", "")


def read_metadata_rows(path: Path) -> dict[str, str]:
    header_row_index = find_timeseries_header_row(path)
    metadata_rows: list[list[str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        for index, row in enumerate(reader):
            if index >= header_row_index:
                break
            metadata_rows.append(row)

    metadata = parse_metadata_rows(metadata_rows, path)
    if not metadata:
        raise NrelCsvError(f"{path} has no metadata rows before the time-series header")
    return metadata


def parse_metadata_rows(rows: list[list[str]], path: Path) -> dict[str, str]:
    if len(rows) == 2:
        keys = [key.strip() for key in rows[0]]
        values = [value.strip() for value in rows[1]]
        return dict(zip(keys, values))
    if len(rows) == 1:
        row = [value.strip() for value in rows[0]]
        if len(row) >= 2 and len(row) % 2 == 0:
            return dict(zip(row[0::2], row[1::2]))
        return {f"metadata_column_{index}": value for index, value in enumerate(row)}
    if not rows:
        raise NrelCsvError(f"{path} has no metadata rows before the time-series header")
    metadata: dict[str, str] = {}
    for row_index, row in enumerate(rows, start=1):
        metadata[f"metadata_row_{row_index}"] = ",".join(row)
    return metadata


def looks_like_standard_metadata_line(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 10_000:
        return False
    first_line = path.open("r", encoding="utf-8-sig", errors="replace").readline().strip()
    if "," not in first_line:
        return False
    normalized = normalize_column_name(first_line)
    markers = ("source", "location_id", "siteid", "latitude", "longitude", "time_zone", "timezone", "elevation")
    return any(marker in normalized for marker in markers)


def find_timeseries_header_row(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        for index, row in enumerate(reader):
            normalized = [normalize_column_name(column) for column in row[:5]]
            if normalized == TIMESTAMP_COLUMNS:
                return index
    raise NrelCsvError(f"{path} does not contain a Year,Month,Day,Hour,Minute time-series header")


def read_nrel_timeseries_csv(
    path: str | Path,
    expected_fields: list[str] | tuple[str, ...],
    expected_rows: int | None = None,
    expected_start_utc: pd.Timestamp | None = None,
    expected_end_utc: pd.Timestamp | None = None,
) -> ParsedNrelCsv:
    csv_path = Path(path)
    metadata = read_metadata_rows(csv_path)
    header_row_index = find_timeseries_header_row(csv_path)
    frame = pd.read_csv(csv_path, skiprows=header_row_index)
    if frame.empty:
        raise NrelCsvError(f"{csv_path} contains no time-series rows")

    original_columns = list(frame.columns)
    normalized_columns = [normalize_column_name(column) for column in original_columns]
    frame.columns = normalized_columns

    missing_timestamp = [column for column in TIMESTAMP_COLUMNS if column not in frame.columns]
    if missing_timestamp:
        raise NrelCsvError(f"Missing timestamp columns in {csv_path}: {missing_timestamp}")

    timestamp = pd.to_datetime(
        {
            "year": frame["year"],
            "month": frame["month"],
            "day": frame["day"],
            "hour": frame["hour"],
            "minute": frame["minute"],
        },
        utc=True,
        errors="raise",
    )
    frame.insert(0, "timestamp_utc", timestamp)

    if frame["timestamp_utc"].duplicated().any():
        duplicates = frame.loc[frame["timestamp_utc"].duplicated(), "timestamp_utc"].head(3).tolist()
        raise NrelCsvError(f"{csv_path} contains duplicate timestamps: {duplicates}")
    if not frame["timestamp_utc"].is_monotonic_increasing:
        raise NrelCsvError(f"{csv_path} timestamps are not strictly increasing")
    if expected_rows is not None and len(frame) != expected_rows:
        raise NrelCsvError(f"{csv_path} has {len(frame)} rows; expected {expected_rows}")
    if expected_start_utc is not None and frame["timestamp_utc"].iloc[0] != expected_start_utc:
        raise NrelCsvError(
            f"{csv_path} first timestamp is {frame['timestamp_utc'].iloc[0]}; "
            f"expected {expected_start_utc}"
        )
    if expected_end_utc is not None and frame["timestamp_utc"].iloc[-1] != expected_end_utc:
        raise NrelCsvError(
            f"{csv_path} last timestamp is {frame['timestamp_utc'].iloc[-1]}; "
            f"expected {expected_end_utc}"
        )

    column_map = map_required_columns(frame.columns, expected_fields)
    rename_map = {source: target for target, source in column_map.items() if source != target}
    frame = frame.rename(columns=rename_map)
    return ParsedNrelCsv(metadata=metadata, frame=frame, column_map=column_map)


def map_required_columns(columns: list[str] | pd.Index, expected_fields: list[str] | tuple[str, ...]) -> dict[str, str]:
    available = {normalize_column_name(column): normalize_column_name(column) for column in columns}
    available_compact = {_compact(column): normalize_column_name(column) for column in columns}
    result: dict[str, str] = {}
    missing: list[str] = []

    for expected in expected_fields:
        normalized_expected = normalize_column_name(expected)
        candidates = FIELD_ALIASES.get(normalized_expected, (normalized_expected,))
        matched = None
        for candidate in candidates:
            candidate_normalized = normalize_column_name(candidate)
            if candidate_normalized in available:
                matched = available[candidate_normalized]
                break
            compact_candidate = _compact(candidate)
            if compact_candidate in available_compact:
                matched = available_compact[compact_candidate]
                break

        if matched is None:
            matched = _token_fuzzy_match(normalized_expected, columns)

        if matched is None:
            missing.append(normalized_expected)
        else:
            result[normalized_expected] = matched

    if missing:
        raise NrelCsvError(f"Missing required columns: {', '.join(missing)}")
    return result


def _token_fuzzy_match(expected: str, columns: list[str] | pd.Index) -> str | None:
    expected_compact = _compact(expected)
    expected_tokens = [token for token in normalize_column_name(expected).split("_") if token]
    for column in columns:
        normalized = normalize_column_name(column)
        compact = _compact(normalized)
        if compact == expected_compact:
            return normalized
        if expected_tokens and all(token in normalized.split("_") or token in compact for token in expected_tokens):
            return normalized
    return None


def count_timeseries_rows(path: Path) -> int:
    header_row_index = find_timeseries_header_row(path)
    return len(pd.read_csv(path, skiprows=header_row_index, usecols=[0]))


def repair_missing_timeseries_newlines(content: bytes) -> tuple[bytes, int]:
    text = content.decode("utf-8-sig", errors="replace")
    repaired, count = re.subn(
        r"(?<=[0-9])(?=(?:2018|2019|2020),\d{1,2},\d{1,2},\d{1,2},\d{1,2},)",
        "\n",
        text,
    )
    return repaired.encode("utf-8"), count
