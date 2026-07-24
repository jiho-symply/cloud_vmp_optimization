from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import urlencode

import requests

from renewable_trace.config import PipelineConfig
from renewable_trace.nrel_csv import (
    NrelCsvError,
    count_timeseries_rows,
    looks_like_standard_metadata_line,
    read_metadata_rows,
    repair_missing_timeseries_newlines,
)
from renewable_trace.sites import Site


LOGGER = logging.getLogger(__name__)
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
DatasetKind = Literal["solar", "wind"]


@dataclass(frozen=True)
class Credentials:
    api_key: str
    email: str
    full_name: str | None = None
    affiliation: str | None = None


@dataclass(frozen=True)
class FetchTask:
    dataset: DatasetKind
    site: Site
    endpoint: str
    csv_path: Path
    metadata_path: Path
    params: dict[str, str]

    @property
    def sanitized_params(self) -> dict[str, str]:
        sanitized = dict(self.params)
        if "api_key" in sanitized:
            sanitized["api_key"] = "<redacted>"
        return sanitized

    @property
    def sanitized_url(self) -> str:
        return f"{self.endpoint}?{urlencode(self.sanitized_params)}"


@dataclass(frozen=True)
class DownloadedPayload:
    content: bytes
    content_type: str


def read_credentials(require: bool = True) -> Credentials:
    api_key = os.environ.get("NREL_API_KEY") or os.environ.get("NLR_API_KEY")
    email = os.environ.get("NREL_API_EMAIL")
    if require and not api_key:
        raise RuntimeError("Set NREL_API_KEY or NLR_API_KEY before fetching NREL/NLR data")
    if require and not email:
        raise RuntimeError("Set NREL_API_EMAIL before fetching NREL/NLR data")
    return Credentials(
        api_key=api_key or "<missing>",
        email=email or "<missing>",
        full_name=os.environ.get("NREL_API_FULL_NAME"),
        affiliation=os.environ.get("NREL_API_AFFILIATION"),
    )


def build_fetch_tasks(config: PipelineConfig, credentials: Credentials) -> list[FetchTask]:
    tasks: list[FetchTask] = []
    for site in config.sites:
        tasks.append(
            FetchTask(
                dataset="solar",
                site=site,
                endpoint=config.api.solar_endpoint,
                csv_path=config.paths.raw_solar_dir / f"{site.site_id}.csv",
                metadata_path=config.paths.raw_solar_dir / f"{site.site_id}.metadata.json",
                params=build_common_params(config, site, credentials)
                | {"attributes": ",".join(config.solar.attributes)},
            )
        )
        tasks.append(
            FetchTask(
                dataset="wind",
                site=site,
                endpoint=config.api.wind_endpoint,
                csv_path=config.paths.raw_wind_dir / f"{site.site_id}.csv",
                metadata_path=config.paths.raw_wind_dir / f"{site.site_id}.metadata.json",
                params=build_common_params(config, site, credentials)
                | {"attributes": ",".join(config.wind.attributes)},
            )
        )
    return tasks


def build_common_params(config: PipelineConfig, site: Site, credentials: Credentials) -> dict[str, str]:
    params = {
        "wkt": site.point_wkt,
        "names": str(config.year),
        "interval": str(config.raw_interval_minutes),
        "utc": "true",
        "leap_day": "false" if not config.leap_day else "true",
        "email": credentials.email,
        "api_key": credentials.api_key,
    }
    if credentials.full_name:
        params["full_name"] = credentials.full_name
    if credentials.affiliation:
        params["affiliation"] = credentials.affiliation
    return params


class NrelDownloader:
    def __init__(self, config: PipelineConfig, session: requests.Session | None = None):
        self.config = config
        self.session = session or requests.Session()
        self._last_call_at: float | None = None

    def fetch_all(self, force: bool = False) -> list[Path]:
        credentials = read_credentials(require=True)
        tasks = build_fetch_tasks(self.config, credentials)
        paths: list[Path] = []
        for task in tasks:
            paths.append(self.fetch_one(task, force=force))
        return paths

    def fetch_one(self, task: FetchTask, force: bool = False) -> Path:
        task.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not force and raw_csv_is_valid(task.csv_path, self.config.expected_raw_rows_per_site):
            LOGGER.info("Skipping valid raw %s file for %s", task.dataset, task.site.site_id)
            write_metadata_json(task)
            return task.csv_path

        tmp_path = task.csv_path.with_suffix(task.csv_path.suffix + ".tmp")
        payload = self._request_with_retries(task)
        content, repair_count = repair_missing_timeseries_newlines(payload.content)
        if repair_count:
            LOGGER.warning(
                "Repaired %s missing time-series newline(s) in %s CSV for %s",
                repair_count,
                task.dataset,
                task.site.site_id,
            )
        tmp_path.write_bytes(content)
        validation_error = raw_csv_validation_error(tmp_path, self.config.expected_raw_rows_per_site)
        if validation_error is not None:
            preview = file_preview(tmp_path, task.params.get("api_key", ""))
            tmp_path.unlink(missing_ok=True)
            raise NrelCsvError(
                f"Downloaded {task.dataset} CSV for {task.site.site_id} failed validation: "
                f"{validation_error}; content-type={payload.content_type or '<missing>'}; "
                f"preview={preview}"
            )
        tmp_path.replace(task.csv_path)
        write_metadata_json(task)
        LOGGER.info("Downloaded %s file for %s to %s", task.dataset, task.site.site_id, task.csv_path)
        return task.csv_path

    def _request_with_retries(self, task: FetchTask) -> DownloadedPayload:
        backoff = 2.0
        last_error: Exception | None = None
        for attempt in range(self.config.api.max_retries + 1):
            self._wait_for_rate_limit()
            try:
                response = self.session.get(
                    task.endpoint,
                    params=task.params,
                    timeout=self.config.api.timeout_seconds,
                )
                if response.status_code in RETRY_STATUS_CODES:
                    last_error = RuntimeError(
                        api_error_message(response, task)
                    )
                    raise last_error
                if response.status_code >= 400:
                    raise RuntimeError(api_error_message(response, task))
                return DownloadedPayload(
                    content=response.content,
                    content_type=response.headers.get("Content-Type", ""),
                )
            except (requests.Timeout, requests.ConnectionError, RuntimeError) as exc:
                last_error = exc
                if attempt >= self.config.api.max_retries:
                    break
                sleep_seconds = backoff * (2**attempt)
                LOGGER.warning(
                    "Retrying %s %s after %s; attempt %s/%s",
                    task.dataset,
                    task.site.site_id,
                    type(exc).__name__,
                    attempt + 1,
                    self.config.api.max_retries,
                )
                time.sleep(sleep_seconds)
        raise RuntimeError(f"Failed fetching {task.dataset} for {task.site.site_id}: {last_error}")

    def _wait_for_rate_limit(self) -> None:
        now = time.monotonic()
        if self._last_call_at is not None:
            elapsed = now - self._last_call_at
            remaining = self.config.api.min_seconds_between_calls - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_call_at = time.monotonic()


def raw_csv_is_valid(path: Path, expected_rows: int) -> bool:
    return raw_csv_validation_error(path, expected_rows) is None


def raw_csv_validation_error(path: Path, expected_rows: int) -> str | None:
    if not path.exists():
        return "file does not exist"
    size = path.stat().st_size
    if size <= 10_000:
        return f"file size is {size} bytes, expected > 10000 bytes"
    if not looks_like_standard_metadata_line(path):
        return "first line is not an NREL/NLR Standard Time Series metadata line"
    try:
        row_count = count_timeseries_rows(path)
    except Exception as exc:
        return f"could not parse time-series rows: {type(exc).__name__}: {exc}"
    if row_count != expected_rows:
        return f"time-series row count is {row_count}, expected {expected_rows}"
    return None


def api_error_message(response: requests.Response, task: FetchTask) -> str:
    body = sanitize_secret(response.text, task.params.get("api_key", ""))
    body = " ".join(body.split())
    if len(body) > 1000:
        body = body[:1000] + "..."
    return (
        f"HTTP {response.status_code} while fetching {task.dataset} for {task.site.site_id}. "
        f"Response body: {body}"
    )


def sanitize_secret(text: str, secret: str | None) -> str:
    sanitized = text
    if secret:
        sanitized = sanitized.replace(secret, "<redacted>")
    sanitized = re.sub(r"(api_key=)[^&\\s]+", r"\1<redacted>", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r'("api_key"\\s*:\\s*")[^"]+(")', r"\1<redacted>\2", sanitized, flags=re.IGNORECASE)
    return sanitized


def file_preview(path: Path, secret: str | None, max_bytes: int = 1000) -> str:
    raw = path.read_bytes()[:max_bytes]
    text = raw.decode("utf-8", errors="replace")
    text = sanitize_secret(text, secret)
    return " ".join(text.split())


def write_metadata_json(task: FetchTask) -> None:
    metadata = read_metadata_rows(task.csv_path)
    payload = {
        "dataset": task.dataset,
        "site_id": task.site.site_id,
        "state": task.site.state,
        "lat": task.site.lat,
        "lon": task.site.lon,
        "wkt": task.site.point_wkt,
        "source_csv": str(task.csv_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "request": {
            "endpoint": task.endpoint,
            "params": task.sanitized_params,
        },
        "nrel_metadata": metadata,
    }
    task.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    task.metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
