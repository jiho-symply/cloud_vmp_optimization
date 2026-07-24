from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile, ZipFile, is_zipfile

import requests

from electricity_price.config import NyisoPriceConfig


LOGGER = logging.getLogger(__name__)
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class NyisoFetchTask:
    dataset: str
    month: int
    url: str
    output_path: Path


def build_fetch_tasks(config: NyisoPriceConfig) -> list[NyisoFetchTask]:
    tasks = []
    for month in range(1, 13):
        yyyymm = f"{config.year}{month:02d}"
        tasks.append(
            NyisoFetchTask(
                dataset="da",
                month=month,
                url=config.api.da_url_template.format(yyyymm=yyyymm),
                output_path=config.paths.raw_da_dir / f"{yyyymm}01damlbmp_zone_csv.zip",
            )
        )
        tasks.append(
            NyisoFetchTask(
                dataset="rt",
                month=month,
                url=config.api.rt_url_template.format(yyyymm=yyyymm),
                output_path=config.paths.raw_rt_dir / f"{yyyymm}01realtime_zone_csv.zip",
            )
        )
    return tasks


class NyisoDownloader:
    def __init__(self, config: NyisoPriceConfig, session: requests.Session | None = None):
        self.config = config
        self.session = session or requests.Session()
        self._last_call_at: float | None = None

    def fetch_all(self, force: bool = False) -> list[Path]:
        outputs = []
        for task in build_fetch_tasks(self.config):
            outputs.append(self.fetch_one(task, force=force))
        return outputs

    def fetch_one(self, task: NyisoFetchTask, force: bool = False) -> Path:
        task.output_path.parent.mkdir(parents=True, exist_ok=True)
        if not force and zip_is_valid(task.output_path):
            LOGGER.info("Skipping valid NYISO %s ZIP for %04d-%02d", task.dataset, self.config.year, task.month)
            return task.output_path

        response = self._get_with_retries(task)
        tmp_path = task.output_path.with_suffix(task.output_path.suffix + ".tmp")
        tmp_path.write_bytes(response.content)
        if not zip_is_valid(tmp_path):
            preview = response.text[:500].replace("\n", " ")
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Downloaded NYISO {task.dataset} ZIP for month {task.month:02d} is invalid; "
                f"content-type={response.headers.get('Content-Type', '<missing>')}; preview={preview}"
            )
        tmp_path.replace(task.output_path)
        LOGGER.info("Downloaded NYISO %s ZIP for %04d-%02d to %s", task.dataset, self.config.year, task.month, task.output_path)
        return task.output_path

    def _get_with_retries(self, task: NyisoFetchTask) -> requests.Response:
        last_error: Exception | None = None
        for attempt in range(self.config.api.max_retries + 1):
            self._wait_for_rate_limit()
            try:
                response = self.session.get(task.url, timeout=self.config.api.timeout_seconds)
                if response.status_code in RETRY_STATUS_CODES:
                    raise RuntimeError(f"HTTP {response.status_code} from {task.url}")
                if response.status_code >= 400:
                    raise RuntimeError(f"HTTP {response.status_code} from {task.url}: {response.text[:500]}")
                return response
            except (requests.Timeout, requests.ConnectionError, RuntimeError) as exc:
                last_error = exc
                if attempt >= self.config.api.max_retries:
                    break
                time.sleep(2.0 * (2**attempt))
        raise RuntimeError(f"Failed to download {task.url}: {last_error}")

    def _wait_for_rate_limit(self) -> None:
        now = time.monotonic()
        if self._last_call_at is not None:
            remaining = self.config.api.min_seconds_between_calls - (now - self._last_call_at)
            if remaining > 0:
                time.sleep(remaining)
        self._last_call_at = time.monotonic()


def zip_is_valid(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0 or not is_zipfile(path):
        return False
    try:
        with ZipFile(path) as archive:
            return any(name.endswith(".csv") for name in archive.namelist())
    except BadZipFile:
        return False
