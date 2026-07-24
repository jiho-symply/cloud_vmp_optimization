#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from electricity_price.build_dataset import build_processed_datasets  # noqa: E402
from electricity_price.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from electricity_price.fetch_nyiso import NyisoDownloader, build_fetch_tasks  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch NYISO public MIS LBMP ZIPs and build 2019 electricity price traces."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--fetch", action="store_true", help="Download missing raw NYISO ZIPs.")
    parser.add_argument("--process", action="store_true", help="Build processed NYISO price datasets.")
    parser.add_argument("--fetch-only", action="store_true", help="Only download raw ZIPs.")
    parser.add_argument("--process-only", action="store_true", help="Only process existing ZIPs.")
    parser.add_argument("--force", action="store_true", help="Re-download ZIPs even if valid files exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned downloads without network calls.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def resolve_actions(args: argparse.Namespace) -> tuple[bool, bool]:
    if args.fetch_only and args.process_only:
        raise SystemExit("--fetch-only and --process-only cannot be combined")
    if args.fetch_only and (args.fetch or args.process):
        raise SystemExit("--fetch-only cannot be combined with --fetch or --process")
    if args.process_only and (args.fetch or args.process):
        raise SystemExit("--process-only cannot be combined with --fetch or --process")
    if args.fetch_only:
        return True, False
    if args.process_only:
        return False, True
    if args.fetch or args.process:
        return args.fetch, args.process
    if args.dry_run:
        return True, False
    raise SystemExit("Choose --fetch --process, --fetch-only, --process-only, or --dry-run")


def print_dry_run(config) -> None:
    print("Planned NYISO public ZIP downloads:")
    for task in build_fetch_tasks(config):
        print(f"- {task.dataset} {config.year}-{task.month:02d}: {task.url}")
        print(f"  output: {task.output_path}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    should_fetch, should_process = resolve_actions(args)
    config = load_config(args.config)

    try:
        if args.dry_run:
            print_dry_run(config)
            return
        if should_fetch:
            paths = NyisoDownloader(config).fetch_all(force=args.force)
            print(f"Raw NYISO fetch phase complete: {len(paths)} ZIP paths checked or downloaded.")
        if should_process:
            outputs = build_processed_datasets(config)
            print("Processed NYISO output files:")
            for path in outputs.__dict__.values():
                print(f"- {path}")
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logging.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
