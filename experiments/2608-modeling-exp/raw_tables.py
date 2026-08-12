"""원시 parquet 입력 → pandas DataFrame을 반환하는 로더를 제공한다.
입력 경로와 테이블 읽기를 한 곳에 모아
후속 단계의 입출력을 고정한다."""

from __future__ import annotations

import pandas as pd

from paths import (
    RAW_COLLECTION_EVENTS_PATH,
    RAW_INSTANCE_EVENTS_PATH,
    RAW_MACHINE_EVENTS_PATH,
    RAW_USAGE_PATH,
)


def load_usage() -> pd.DataFrame:
    """usage 원시 테이블을 DataFrame으로 반환한다."""
    return pd.read_parquet(RAW_USAGE_PATH)


def load_instance_events() -> pd.DataFrame:
    """instance events 원시 테이블을 DataFrame으로 반환한다."""
    return pd.read_parquet(RAW_INSTANCE_EVENTS_PATH)


def load_collection_events() -> pd.DataFrame:
    """collection events 원시 테이블을 DataFrame으로 반환한다."""
    return pd.read_parquet(RAW_COLLECTION_EVENTS_PATH)


def load_machine_events() -> pd.DataFrame:
    """machine events 원시 테이블을 DataFrame으로 반환한다."""
    return pd.read_parquet(RAW_MACHINE_EVENTS_PATH)
