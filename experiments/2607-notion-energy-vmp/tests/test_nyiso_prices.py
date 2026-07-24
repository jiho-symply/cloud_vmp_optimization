from io import StringIO

import pandas as pd
import pytest

from electricity_price.nyiso_csv import (
    NyisoPriceError,
    align_day_ahead_to_realtime_grid,
    collapse_duplicate_prices,
    local_timestamp_blocks_to_utc,
    parse_lbmp_csv,
    regularize_price_grid,
    validate_price_frame,
)
from electricity_price.zones import NYISO_ZONES


def test_parse_da_lbmp_uses_new_york_interval_start_and_preserves_negative_prices():
    csv_text = """Time Stamp,Name,PTID,LBMP ($/MWHr),Marginal Cost Losses ($/MWHr),Marginal Cost Congestion ($/MWHr)
01/01/2019 00:00,WEST,61752,-3.50,-0.10,-1.00
01/01/2019 00:00,PJM,61847,99.00,1.00,2.00
"""

    frame = parse_lbmp_csv(StringIO(csv_text), dataset="da", zones=NYISO_ZONES)

    assert len(frame) == 1
    assert frame.loc[0, "timestamp_utc"] == pd.Timestamp("2019-01-01T05:00:00Z")
    assert frame.loc[0, "site_id"] == "WEST"
    assert frame.loc[0, "da_lbmp_usd_per_mwh"] == -3.50
    assert frame.loc[0, "da_loss_raw_usd_per_mwh"] == -0.10
    assert frame.loc[0, "da_congestion_raw_usd_per_mwh"] == -1.00


def test_parse_rt_lbmp_converts_interval_end_to_interval_start_before_utc():
    csv_text = """Time Stamp,Name,PTID,LBMP ($/MWHr),Marginal Cost Losses ($/MWHr),Marginal Cost Congestion ($/MWHr)
01/01/2019 00:05:00,CAPITL,61757,7.01,0.43,0.00
"""

    frame = parse_lbmp_csv(StringIO(csv_text), dataset="rt", zones=NYISO_ZONES)

    assert len(frame) == 1
    assert frame.loc[0, "timestamp_utc"] == pd.Timestamp("2019-01-01T05:00:00Z")
    assert frame.loc[0, "site_id"] == "CAPITL"
    assert frame.loc[0, "rt_lbmp_usd_per_mwh"] == 7.01


def test_parse_rt_lbmp_handles_spring_forward_by_localizing_interval_end_first():
    csv_text = """Time Stamp,Name,PTID,LBMP ($/MWHr),Marginal Cost Losses ($/MWHr),Marginal Cost Congestion ($/MWHr)
03/10/2019 03:00:00,WEST,61752,17.10,-0.21,-1.17
"""

    frame = parse_lbmp_csv(StringIO(csv_text), dataset="rt", zones=NYISO_ZONES)

    assert frame.loc[0, "timestamp_utc"] == pd.Timestamp("2019-03-10T06:55:00Z")


def test_parse_rt_lbmp_rounds_irregular_dispatch_interval_to_5min_grid():
    csv_text = """Time Stamp,Name,PTID,LBMP ($/MWHr),Marginal Cost Losses ($/MWHr),Marginal Cost Congestion ($/MWHr)
01/01/2019 00:19:33,WEST,61752,6.51,0.09,-8.21
"""

    frame = parse_lbmp_csv(StringIO(csv_text), dataset="rt", zones=NYISO_ZONES)

    assert frame.loc[0, "timestamp_utc"] == pd.Timestamp("2019-01-01T05:15:00Z")


def test_parse_rt_lbmp_uses_interval_start_local_year_for_midnight_year_boundary():
    csv_text = """Time Stamp,Name,PTID,LBMP ($/MWHr),Marginal Cost Losses ($/MWHr),Marginal Cost Congestion ($/MWHr)
01/01/2020 00:00:00,WEST,61752,8.43,-0.42,-1.38
"""

    frame = parse_lbmp_csv(StringIO(csv_text), dataset="rt", zones=NYISO_ZONES)

    assert frame.loc[0, "timestamp_utc"] == pd.Timestamp("2020-01-01T04:55:00Z")
    assert frame.loc[0, "_local_year"] == 2019


def test_align_day_ahead_to_realtime_grid_forward_fills_hourly_da_prices():
    da = pd.DataFrame(
        {
            "timestamp_utc": [
                pd.Timestamp("2019-01-01T05:00:00Z"),
                pd.Timestamp("2019-01-01T06:00:00Z"),
            ],
            "market": "NYISO",
            "site_id": "WEST",
            "nyiso_zone": "WEST",
            "ptid": 61752,
            "da_lbmp_usd_per_mwh": [10.0, 20.0],
            "da_loss_raw_usd_per_mwh": [1.0, 2.0],
            "da_congestion_raw_usd_per_mwh": [0.5, 0.6],
        }
    )
    rt = pd.DataFrame(
        {
            "timestamp_utc": [
                pd.Timestamp("2019-01-01T05:00:00Z"),
                pd.Timestamp("2019-01-01T05:55:00Z"),
                pd.Timestamp("2019-01-01T06:00:00Z"),
            ],
            "market": "NYISO",
            "site_id": "WEST",
            "nyiso_zone": "WEST",
            "ptid": 61752,
            "rt_lbmp_usd_per_mwh": [11.0, 12.0, 21.0],
            "rt_loss_raw_usd_per_mwh": [1.1, 1.2, 2.1],
            "rt_congestion_raw_usd_per_mwh": [0.1, 0.2, 0.3],
        }
    )

    aligned = align_day_ahead_to_realtime_grid(da, rt)

    assert aligned["da_lbmp_usd_per_mwh"].tolist() == [10.0, 10.0, 20.0]
    assert aligned["rt_lbmp_usd_per_mwh"].tolist() == [11.0, 12.0, 21.0]


def test_collapse_duplicate_prices_averages_repeated_site_timestamp_rows():
    frame = pd.DataFrame(
        {
            "timestamp_utc": [
                pd.Timestamp("2019-01-01T05:15:00Z"),
                pd.Timestamp("2019-01-01T05:15:00Z"),
            ],
            "_local_year": [2019, 2019],
            "market": ["NYISO", "NYISO"],
            "site_id": ["WEST", "WEST"],
            "nyiso_zone": ["WEST", "WEST"],
            "ptid": [61752, 61752],
            "rt_lbmp_usd_per_mwh": [6.0, 10.0],
            "rt_loss_raw_usd_per_mwh": [1.0, 3.0],
            "rt_congestion_raw_usd_per_mwh": [0.0, 2.0],
        }
    )

    collapsed = collapse_duplicate_prices(frame, dataset="rt")

    assert len(collapsed) == 1
    assert collapsed.loc[0, "rt_lbmp_usd_per_mwh"] == 8.0
    assert collapsed.loc[0, "rt_loss_raw_usd_per_mwh"] == 2.0
    assert collapsed.loc[0, "rt_congestion_raw_usd_per_mwh"] == 1.0


def test_regularize_price_grid_fills_missing_slots():
    frame = pd.DataFrame(
        {
            "timestamp_utc": [
                pd.Timestamp("2019-01-01T05:00:00Z"),
                pd.Timestamp("2019-01-01T05:10:00Z"),
            ],
            "_local_year": [2019, 2019],
            "market": ["NYISO", "NYISO"],
            "site_id": ["WEST", "WEST"],
            "nyiso_zone": ["WEST", "WEST"],
            "ptid": [61752, 61752],
            "rt_lbmp_usd_per_mwh": [10.0, 20.0],
            "rt_loss_raw_usd_per_mwh": [1.0, 3.0],
            "rt_congestion_raw_usd_per_mwh": [0.0, 2.0],
        }
    )
    grid = pd.date_range("2019-01-01T05:00:00Z", periods=3, freq="5min")

    regularized = regularize_price_grid(frame, dataset="rt", zones=[NYISO_ZONES[0]], grid=grid)

    assert len(regularized) == 3
    assert regularized["rt_lbmp_usd_per_mwh"].tolist() == [10.0, 15.0, 20.0]
    assert regularized["rt_loss_raw_usd_per_mwh"].tolist() == [1.0, 2.0, 3.0]


def test_validate_price_frame_detects_duplicate_site_timestamp_pairs():
    frame = pd.DataFrame(
        {
            "timestamp_utc": [
                pd.Timestamp("2019-01-01T05:00:00Z"),
                pd.Timestamp("2019-01-01T05:00:00Z"),
            ],
            "site_id": ["WEST", "WEST"],
            "da_lbmp_usd_per_mwh": [1.0, 1.0],
            "rt_lbmp_usd_per_mwh": [1.0, 1.0],
        }
    )

    with pytest.raises(NyisoPriceError, match="duplicated timestamp_utc/site_id"):
        validate_price_frame(frame, expected_rows=None, expected_rows_per_site=None)


def test_local_timestamp_blocks_falls_back_for_single_ambiguous_fall_hour():
    local_naive = pd.Series(pd.to_datetime(["2019-11-03 01:00:00"]))

    timestamp_utc = local_timestamp_blocks_to_utc(local_naive, "America/New_York")

    assert timestamp_utc.iloc[0] == pd.Timestamp("2019-11-03T06:00:00Z")
