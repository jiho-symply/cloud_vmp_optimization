import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
VMTABLE_CSV = REPO_ROOT / "data" / "raw" / "trace_data" / "vmtable" / "vmtable.csv"
CPU_READINGS_DIR = REPO_ROOT / "data" / "raw" / "trace_data" / "vm_cpu_readings"

VMTABLE_COLUMNS = [
    "vm_id",
    "subscription_id",
    "deployment_id",
    "timestamp_vm_created",
    "timestamp_vm_deleted",
    "max_cpu",
    "avg_cpu",
    "p95_max_cpu",
    "vm_category",
    "vm_virtual_core_count_bucket",
    "vm_memory_gb_bucket",
]

PROFILE_CONFIG = {
    "2601-initial-toy-model": {
        "ndays": 3,
        "sample_n": 2000,
        "random_seed": 42,
        "remove_unknown": False,
        "avg_cpu_threshold": None,
        "max_vcpu": 8,
        "output_csv": REPO_ROOT / "data" / "processed" / "2601-initial-toy-model" / "sample_vm_data.csv",
        "time_col": "hour",
        "include_max_avg": False,
        "weight_interactive_by_start": False,
    },
    "2602-baseline-toy-model": {
        "ndays": 1,
        "sample_n": 9,
        "random_seed": 42,
        "remove_unknown": True,
        "avg_cpu_threshold": 10.0,
        "max_vcpu": None,
        "output_csv": REPO_ROOT / "data" / "processed" / "2602-baseline-toy-model" / "sample_vm_data.csv",
        "time_col": "timestamp",
        "include_max_avg": True,
        "weight_interactive_by_start": False,
    },
    "2602-refined-toy-model": {
        "ndays": 1,
        "sample_n": 9,
        "random_seed": 42,
        "remove_unknown": True,
        "avg_cpu_threshold": 10.0,
        "max_vcpu": None,
        "output_csv": REPO_ROOT / "data" / "processed" / "2602-refined-toy-model" / "sample_vm_data.csv",
        "time_col": "timestamp",
        "include_max_avg": True,
        "weight_interactive_by_start": False,
    },
}

def parse_args():
    parser = argparse.ArgumentParser(description="이전 notebook 로직을 따라 sample_vm_data.csv를 다시 생성합니다.")
    parser.add_argument(
        "--profile",
        choices=PROFILE_CONFIG.keys(),
        default="2601-initial-toy-model",
        help="재현할 전처리 프로필 이름",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="vm_cpu_readings gzip 파일을 읽을 때 사용할 chunk 크기",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    profile_name = args.profile
    config = PROFILE_CONFIG[profile_name]
    output_csv = config["output_csv"]
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    vmtable = pd.read_csv(VMTABLE_CSV, header=None, names=VMTABLE_COLUMNS)
    nseconds = config["ndays"] * 86400

    vmtable = vmtable.loc[vmtable["timestamp_vm_created"] < nseconds].copy()
    vmtable["vCPU"] = vmtable["vm_virtual_core_count_bucket"].map(
        {"2": 2, "4": 4, "8": 8, "12": 12, "24": 24, ">24": 24}
    )

    if config["remove_unknown"]:
        vmtable = vmtable.loc[vmtable["vm_category"].str.lower().ne("unknown")].copy()
    if config["avg_cpu_threshold"] is not None:
        vmtable = vmtable.loc[vmtable["avg_cpu"] >= config["avg_cpu_threshold"]].copy()
    if config["max_vcpu"] is not None:
        vmtable = vmtable.loc[vmtable["vCPU"] <= config["max_vcpu"]].copy()

    # 오래 켜져 있는 interactive VM보다 중간 유입 interactive VM을 더 잘 보이게 한다.
    weights = np.ones(len(vmtable), dtype=float)
    if config["weight_interactive_by_start"]:
        is_started_on_0 = vmtable["timestamp_vm_created"] == 0
        is_interactive = vmtable["vm_category"].str.lower().eq("interactive")
        weights[is_started_on_0 & is_interactive] = 0.1
        weights[(~is_started_on_0) & is_interactive] = 10.0

    sampled_vmtable = vmtable.sample(
        n=config["sample_n"],
        replace=False,
        weights=weights,
        random_state=config["random_seed"],
        ignore_index=True,
    )

    sample_vm_ids = set(sampled_vmtable["vm_id"])
    cpu_files = sorted(
        CPU_READINGS_DIR.glob("vm_cpu_readings-file-*-of-195.csv.gz"),
        key=lambda path: int(path.name.split("-")[2]),
    )

    collected_chunks = []
    files_read = 0
    for cpu_file in cpu_files:
        files_read += 1
        file_timestamp_max = -1

        for chunk in pd.read_csv(
            cpu_file,
            header=None,
            names=["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"],
            compression="gzip",
            chunksize=args.chunk_size,
        ):
            if not chunk.empty:
                file_timestamp_max = max(file_timestamp_max, int(chunk["timestamp"].max()))
                filtered = chunk.loc[chunk["vm_id"].isin(sample_vm_ids)]
                if not filtered.empty:
                    collected_chunks.append(filtered)

        # 목표 기간 이후 구간만 남은 파일은 더 읽지 않는다.
        if file_timestamp_max >= nseconds:
            break

    sample_vm_df = pd.concat(collected_chunks, ignore_index=True)
    sample_vm_df = sample_vm_df.loc[sample_vm_df["timestamp"] < nseconds].copy()
    # 원시 5분 단위 timestamp를 실험 프로필에 맞는 시간 축으로 바꾼다.
    sample_vm_df[config["time_col"]] = sample_vm_df["timestamp"] // 3600

    agg_map = {
        "min_cpu": ("min_cpu", "min"),
        "avg_cpu": ("avg_cpu", "mean"),
        "max_cpu": ("max_cpu", "max"),
    }
    if config["include_max_avg"]:
        agg_map = {
            "min_cpu": ("min_cpu", "min"),
            "avg_cpu": ("avg_cpu", "mean"),
            "max_avg_cpu": ("avg_cpu", "max"),
            "max_cpu": ("max_cpu", "max"),
        }

    sample_vm_df = (
        sample_vm_df.groupby(["vm_id", config["time_col"]], as_index=False)
        .agg(**agg_map)
        .merge(sampled_vmtable[["vm_id", "vCPU", "vm_category"]], on="vm_id", how="left")
    )

    # CPU 비율을 실제 코어 수요 단위로 바꿔 이후 toy model에서 바로 쓸 수 있게 한다.
    sample_vm_df["min_core_usage"] = sample_vm_df["min_cpu"] * sample_vm_df["vCPU"] / 100
    sample_vm_df["avg_core_usage"] = sample_vm_df["avg_cpu"] * sample_vm_df["vCPU"] / 100
    if config["include_max_avg"]:
        sample_vm_df["max_avg_core_usage"] = sample_vm_df["max_avg_cpu"] * sample_vm_df["vCPU"] / 100
    sample_vm_df["max_core_usage"] = sample_vm_df["max_cpu"] * sample_vm_df["vCPU"] / 100

    sample_vm_df.to_csv(output_csv, index=False)

    print(f"profile: {profile_name}")
    print(f"output: {output_csv}")
    print(f"files_read: {files_read}")
    print(f"rows: {len(sample_vm_df)}")
    print(f"unique_vm: {sample_vm_df['vm_id'].nunique()}")
    print(f"category_counts: {sample_vm_df['vm_category'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
