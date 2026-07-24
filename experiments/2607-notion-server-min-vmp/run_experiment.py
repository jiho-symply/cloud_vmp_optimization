#!/usr/bin/env python3
"""Isolated entry point for the server-minimum-time experiment."""

from __future__ import annotations

import sys
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from notion_server_min_vmp.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
