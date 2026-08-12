"""s1부터 s9까지 전처리 단계를 순서대로 실행한다."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from preprocessing import (
    s1_machine_capacity,
    s2_episodes,
    s3_episode_usage,
    s4_select_episodes,
    s5_vm_requests,
    s6_units,
    s7_scenarios,
    s8_spot_batch,
    s9_static,
)


STEPS = [
    ("s1", s1_machine_capacity),
    ("s2", s2_episodes),
    ("s3", s3_episode_usage),
    ("s4", s4_select_episodes),
    ("s5", s5_vm_requests),
    ("s6", s6_units),
    ("s7", s7_scenarios),
    ("s8", s8_spot_batch),
    ("s9", s9_static),
]


def main() -> None:
    """선택한 전처리 단계를 순서대로 실행한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=[name for name, _ in STEPS])
    args = parser.parse_args()
    selected_steps = [
        module for name, module in STEPS
        if args.only is None or name == args.only
    ]
    for module in selected_steps:
        module.main()


if __name__ == "__main__":
    main()
