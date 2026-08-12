"""s1부터 s9까지의 단계 스크립트를 정해진 순서로 실행한다."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> None:
    """아홉 단계 스크립트를 현재 Python으로 순서대로 실행한다."""
    scripts = [
        "s1_machine_capacity.py", "s2_episodes.py", "s3_episode_usage.py",
        "s4_select_episodes.py", "s5_vm_requests.py", "s6_units.py",
        "s7_scenarios.py", "s8_spot_batch.py", "s9_static.py",
    ]
    for script in scripts:
        subprocess.run(
            [sys.executable, str(SCRIPT_DIR / script)],
            check=True,
            cwd=SCRIPT_DIR,
        )


if __name__ == "__main__":
    main()
