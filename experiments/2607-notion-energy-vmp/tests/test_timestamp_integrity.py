from pathlib import Path

import pandas as pd
import pytest

from renewable_trace.nrel_csv import (
    NrelCsvError,
    count_timeseries_rows,
    read_nrel_timeseries_csv,
    repair_missing_timeseries_newlines,
)


def write_fake_nrel_csv(path: Path, rows: list[dict], extra_columns: list[str] | None = None) -> None:
    extra_columns = extra_columns or ["GHI", "DNI"]
    header = ["Year", "Month", "Day", "Hour", "Minute", *extra_columns]
    lines = [
        "Source,Location ID,Latitude,Longitude,Time Zone",
        "Fake,123,36.7783,-119.4179,0",
        ",".join(header),
    ]
    for row in rows:
        values = [str(row[column]) for column in header]
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_parser_creates_timezone_aware_utc_timestamps(tmp_path):
    path = tmp_path / "fake.csv"
    write_fake_nrel_csv(
        path,
        [
            {"Year": 2019, "Month": 1, "Day": 1, "Hour": 0, "Minute": 0, "GHI": 0, "DNI": 0},
            {"Year": 2019, "Month": 1, "Day": 1, "Hour": 0, "Minute": 5, "GHI": 1, "DNI": 2},
        ],
    )

    parsed = read_nrel_timeseries_csv(path, expected_fields=["ghi", "dni"], expected_rows=2)

    assert str(parsed.frame["timestamp_utc"].dtype) == "datetime64[ns, UTC]"
    assert parsed.frame["timestamp_utc"].iloc[0] == pd.Timestamp("2019-01-01T00:00:00Z")
    assert parsed.frame["timestamp_utc"].iloc[1] == pd.Timestamp("2019-01-01T00:05:00Z")


def test_parser_detects_duplicate_timestamps(tmp_path):
    path = tmp_path / "duplicate.csv"
    duplicate_row = {"Year": 2019, "Month": 1, "Day": 1, "Hour": 0, "Minute": 0, "GHI": 0, "DNI": 0}
    write_fake_nrel_csv(path, [duplicate_row, duplicate_row])

    with pytest.raises(NrelCsvError, match="duplicate timestamps"):
        read_nrel_timeseries_csv(path, expected_fields=["ghi", "dni"], expected_rows=2)


def test_parser_raises_clear_error_for_missing_required_columns(tmp_path):
    path = tmp_path / "missing.csv"
    write_fake_nrel_csv(
        path,
        [{"Year": 2019, "Month": 1, "Day": 1, "Hour": 0, "Minute": 0, "GHI": 0}],
        extra_columns=["GHI"],
    )

    with pytest.raises(NrelCsvError, match="Missing required columns.*dni"):
        read_nrel_timeseries_csv(path, expected_fields=["ghi", "dni"], expected_rows=1)


def test_parser_supports_wind_csv_with_single_key_value_metadata_row(tmp_path):
    path = tmp_path / "wind_single_metadata.csv"
    path.write_text(
        "SiteID,443463,Site Timezone,-8,Data Timezone,0,Longitude,-119.42865,Latitude,36.774742\n"
        "Year,Month,Day,Hour,Minute,wind speed at 80m (m/s),wind direction at 80m (deg)\n"
        "2019,1,1,0,0,0.69,314.88\n"
        "2019,1,1,0,5,0.87,321.17\n",
        encoding="utf-8",
    )

    parsed = read_nrel_timeseries_csv(
        path,
        expected_fields=["windspeed_80m", "winddirection_80m"],
        expected_rows=2,
    )

    assert count_timeseries_rows(path) == 2
    assert parsed.metadata["SiteID"] == "443463"
    assert parsed.frame["timestamp_utc"].tolist() == [
        pd.Timestamp("2019-01-01T00:00:00Z"),
        pd.Timestamp("2019-01-01T00:05:00Z"),
    ]
    assert parsed.frame["windspeed_80m"].tolist() == [0.69, 0.87]


def test_repair_missing_timeseries_newlines_splits_concatenated_year_rows():
    text = (
        "Year,Month,Day,Hour,Minute,wind speed at 80m (m/s),wind direction at 80m (deg)\n"
        "2019,1,1,1,5,2.64,329.942019,1,1,1,10,2.96,329.79\n"
    )

    repaired, repair_count = repair_missing_timeseries_newlines(text.encode("utf-8"))

    assert repair_count == 1
    assert b"329.94\n2019,1,1,1,10" in repaired
