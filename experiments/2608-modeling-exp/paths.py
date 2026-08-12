"""저장소 위치 → 원시 입력과 단계 산출물 경로를 정의한다.
파일 경로를 한 곳에서 공유해 단계 간 입출력 계약을 고정한다."""

from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = EXPERIMENT_DIR.parents[1]
RAW_DIR = REPOSITORY_DIR / "data/raw/google2019_cell_a_day0_cpu_distribution"
WORK_DIR = EXPERIMENT_DIR / "work"

RAW_METADATA_PATH = RAW_DIR / "metadata.json"
RAW_USAGE_PATH = RAW_DIR / "usage_5min.parquet"
RAW_INSTANCE_EVENTS_PATH = RAW_DIR / "instance_events.parquet"
RAW_COLLECTION_EVENTS_PATH = RAW_DIR / "collection_events.parquet"
RAW_MACHINE_EVENTS_PATH = RAW_DIR / "machine_events.parquet"

S1_REPRESENTATIVE_PATH = WORK_DIR / "s1_representative_capacity.json"
S1_MACHINE_INTERVALS_PATH = WORK_DIR / "s1_machine_capacity_intervals.parquet"

S2_EPISODES_PATH = WORK_DIR / "s2_episodes.parquet"
