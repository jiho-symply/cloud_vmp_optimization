from pathlib import Path
import json
import pandas as pd

RAW_DIR = (
    Path(__file__).resolve().parents[2]
    / "data/raw/google2019_cell_a_day0_cpu_distribution"
)


def load_metadata():
    return json.loads(
        (RAW_DIR / "metadata.json").read_text(encoding="utf-8")
    )


def load_usage_df():
    usage_df = pd.read_parquet(RAW_DIR / "usage_5min.parquet")

    required_columns = {
        "collection_id",
        "instance_index",
        "machine_id",
        "start_time",
        "end_time",
        "clipped_start_time",
        "clipped_end_time",
        "overlap_us",
        "cpu_usage",
        "cpu_usage_distribution",
        "mem_usage",
        "assigned_memory",
        "t5",
        "t5_day",
    }

    missing_columns = required_columns - set(usage_df.columns)
    if missing_columns:
        raise ValueError(f"usage_df에 필요한 컬럼이 없습니다: {missing_columns}")

    return usage_df


def load_instance_events_df():
    instance_events_df = pd.read_parquet(
        RAW_DIR / "instance_events.parquet"
    )

    required_columns = {
        "collection_id",
        "instance_index",
        "time",
        "event_type",
        "scheduling_class",
        "priority",
        "machine_id",
        "resource_request_cpu",
        "resource_request_mem",
    }

    missing_columns = required_columns - set(instance_events_df.columns)
    if missing_columns:
        raise ValueError(
            f"instance_events_df에 필요한 컬럼이 없습니다: {missing_columns}"
        )

    return instance_events_df


def load_collection_events_df():
    collection_events_df = pd.read_parquet(
        RAW_DIR / "collection_events.parquet"
    )

    required_columns = {
        "collection_id",
        "time",
        "event_type",
        "scheduler",
        "scheduling_class",
        "priority",
        "max_per_machine",
        "max_per_switch",
    }

    missing_columns = required_columns - set(collection_events_df.columns)
    if missing_columns:
        raise ValueError(
            f"collection_events_df에 필요한 컬럼이 없습니다: {missing_columns}"
        )

    return collection_events_df


def load_machine_events_df():
    machine_events_df = pd.read_parquet(
        RAW_DIR / "machine_events.parquet"
    )

    required_columns = {
        "time",
        "machine_id",
        "event_type",
        "capacity_cpu",
        "capacity_mem",
        "platform_id",
        "switch_id",
    }

    missing_columns = required_columns - set(machine_events_df.columns)
    if missing_columns:
        raise ValueError(
            f"machine_events_df에 필요한 컬럼이 없습니다: {missing_columns}"
        )

    return machine_events_df

if __name__ == "__main__":
    metadata = load_metadata()
    usage_df = load_usage_df()
    instance_events_df = load_instance_events_df()
    collection_events_df = load_collection_events_df()
    machine_events_df = load_machine_events_df()

    print(metadata)
    print("usage_df:", usage_df.shape)
    print("instance_events_df:", instance_events_df.shape)
    print("collection_events_df:", collection_events_df.shape)
    print("machine_events_df:", machine_events_df.shape)
