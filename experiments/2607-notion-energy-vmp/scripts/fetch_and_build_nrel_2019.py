#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from renewable_trace.build_dataset import build_processed_datasets  # noqa: E402
from renewable_trace.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from renewable_trace.fetch_nrel import NrelDownloader, build_fetch_tasks, read_credentials  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch NLR/NREL 2019 solar/wind resource traces and build VM-placement inputs."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--fetch", action="store_true", help="Download missing raw solar and wind CSVs.")
    parser.add_argument("--process", action="store_true", help="Build processed capacity-factor datasets.")
    parser.add_argument("--fetch-only", action="store_true", help="Only download raw files.")
    parser.add_argument("--process-only", action="store_true", help="Only process existing raw files.")
    parser.add_argument("--force", action="store_true", help="Re-download raw CSVs even if valid files exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned requests without network calls.")
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
    credentials = read_credentials(require=False)
    tasks = build_fetch_tasks(config, credentials)
    print("Planned NLR/NREL direct CSV requests:")
    for task in tasks:
        print(f"- {task.dataset} {task.site.site_id}: {task.sanitized_url}")
        print(f"  output: {task.csv_path}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    load_dotenv(REPO_ROOT / ".env")

    should_fetch, should_process = resolve_actions(args)
    config = load_config(args.config)

    if args.dry_run:
        print_dry_run(config)
        return

    try:
        if should_fetch:
            downloader = NrelDownloader(config)
            fetched_paths = downloader.fetch_all(force=args.force)
            print(f"Raw fetch phase complete: {len(fetched_paths)} CSV paths checked or downloaded.")

        if should_process:
            outputs = build_processed_datasets(config)
            print("Processed output files:")
            for path in outputs.__dict__.values():
                print(f"- {path}")
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logging.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
