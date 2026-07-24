from pathlib import Path

import requests

from renewable_trace.config import ApiConfig, OutputPaths, PipelineConfig, ScalingConfig, SolarConfig, WindConfig
from renewable_trace.fetch_nrel import FetchTask, NrelDownloader, raw_csv_is_valid
from renewable_trace.nrel_csv import NrelCsvError
from renewable_trace.sites import Site


class StaticSession:
    def __init__(self, response):
        self.response = response

    def get(self, *args, **kwargs):
        return self.response


def make_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        year=2019,
        raw_interval_minutes=5,
        placement_interval_minutes=30,
        timezone="UTC",
        leap_day=False,
        expected_raw_rows_per_site=105120,
        expected_cf_5min_total_rows=1051200,
        expected_cf_30min_total_rows=175200,
        api=ApiConfig(
            solar_endpoint="https://example.test/solar.csv",
            wind_endpoint="https://example.test/wind.csv",
            min_seconds_between_calls=0,
            max_retries=0,
            timeout_seconds=1,
        ),
        solar=SolarConfig(
            attributes=["ghi"],
            ac_rated_power_w=1_000_000,
            dc_ac_ratio=1.2,
            inverter_nominal_efficiency=0.96,
            system_losses=0.14,
            gamma_pdc=-0.004,
            surface_azimuth=180.0,
            default_albedo=0.2,
        ),
        wind=WindConfig(
            attributes=["windspeed_80m"],
            rated_power_mw=1.5,
            hub_height_m=80,
            cut_in_speed_mps=3.5,
            rated_speed_mps=12.0,
            cut_out_speed_mps=25.0,
            wind_loss_factor=0.90,
        ),
        scaling=ScalingConfig(
            assumed_mean_demand_mw_per_site=100,
            target_average_renewable_to_demand_ratio=2,
            solar_energy_mix=0.2,
            wind_energy_mix=0.8,
        ),
        paths=OutputPaths(
            raw_solar_dir=tmp_path / "solar",
            raw_wind_dir=tmp_path / "wind",
            processed_dir=tmp_path / "processed",
        ),
        sites=[Site(site_id="CA", state="California", lat=36.7783, lon=-119.4179)],
    )


def test_http_error_includes_status_and_response_body_without_api_key(tmp_path):
    response = requests.Response()
    response.status_code = 400
    response._content = b'{"errors":["bad wind attribute"],"api_key":"SECRET"}'
    response.url = "https://example.test/wind.csv?api_key=SECRET"

    task = FetchTask(
        dataset="wind",
        site=Site(site_id="CA", state="California", lat=36.7783, lon=-119.4179),
        endpoint="https://example.test/wind.csv",
        csv_path=tmp_path / "CA.csv",
        metadata_path=tmp_path / "CA.metadata.json",
        params={"api_key": "SECRET"},
    )
    downloader = NrelDownloader(make_config(tmp_path), session=StaticSession(response))

    try:
        downloader.fetch_one(task, force=True)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected RuntimeError")

    assert "HTTP 400" in message
    assert "bad wind attribute" in message
    assert "SECRET" not in message


def test_validation_error_includes_response_preview_without_api_key(tmp_path):
    response = requests.Response()
    response.status_code = 200
    response._content = b'{"errors":["invalid attributes"],"api_key":"SECRET"}'
    response.headers["Content-Type"] = "application/json"

    task = FetchTask(
        dataset="solar",
        site=Site(site_id="CA", state="California", lat=36.7783, lon=-119.4179),
        endpoint="https://example.test/solar.csv",
        csv_path=tmp_path / "CA.csv",
        metadata_path=tmp_path / "CA.metadata.json",
        params={"api_key": "SECRET"},
    )
    downloader = NrelDownloader(make_config(tmp_path), session=StaticSession(response))

    try:
        downloader.fetch_one(task, force=True)
    except NrelCsvError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected NrelCsvError")

    assert "Downloaded solar CSV for CA failed validation" in message
    assert "invalid attributes" in message
    assert "application/json" in message
    assert "SECRET" not in message


def test_raw_csv_is_valid_returns_true_for_standard_metadata_and_expected_rows(tmp_path):
    path = tmp_path / "valid.csv"
    dummy_keys = [f"Dummy Key {index}" for index in range(1200)]
    dummy_values = ["value" for _ in dummy_keys]
    path.write_text(
        ",".join(["Source", "Location ID", "Latitude", "Longitude", "Time Zone", *dummy_keys])
        + "\n"
        + ",".join(["NSRDB", "1", "36.78", "-119.41", "0", *dummy_values])
        + "\n"
        + "Year,Month,Day,Hour,Minute,GHI\n"
        + "2019,1,1,0,0,0\n"
        + "2019,1,1,0,5,1\n",
        encoding="utf-8",
    )

    assert path.stat().st_size > 10_000
    assert raw_csv_is_valid(path, expected_rows=2)
