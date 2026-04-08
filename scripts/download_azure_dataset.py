#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import shutil
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LINKS_FILE = REPO_ROOT / "data" / "metadata" / "AzurePublicDatasetLinksV2.txt"
DEFAULT_OUTPUT_DIR = REPO_ROOT
BLOB_MARKER = "/azurepublicdatasetv2/"
CPU_READING_PATTERN = re.compile(r"vm_cpu_readings-file-(\d+)-of-\d+\.csv\.gz$")

PROFILE_METADATA = "metadata"
PROFILE_ANALYSIS = "analysis"
PROFILE_PREPARE = "prepare-dataset"
PROFILE_FULL = "full"
PROFILE_CHOICES = (
    PROFILE_METADATA,
    PROFILE_ANALYSIS,
    PROFILE_PREPARE,
    PROFILE_FULL,
)

DEFAULT_EXTRACT_RELATIVE_PATHS = {
    "trace_data/vmtable/vmtable.csv.gz",
    "trace_data/deployments/deployments.csv.gz",
    "trace_data/subscriptions/subscriptions.csv.gz",
}

METADATA_FILES = {
    "Azure 2019 Public Dataset V2 - Trace Analysis.ipynb",
    "schema.csv",
    "vm_virtual_core_bucket_definition.csv",
    "vm_memory_bucket_definition.csv",
}


@dataclass(frozen=True)
class Asset:
    url: str
    relative_path: Path

    @property
    def relative_path_str(self) -> str:
        return self.relative_path.as_posix()

    @property
    def is_cpu_reading(self) -> bool:
        return self.relative_path_str.startswith("trace_data/vm_cpu_readings/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download Azure Public Dataset V2 assets used by this workspace. "
            "The script reads AzurePublicDatasetLinksV2.txt and writes files "
            "into this repository's data/notebooks layout."
        )
    )
    parser.add_argument(
        "--profile",
        choices=PROFILE_CHOICES,
        default=PROFILE_ANALYSIS,
        help=(
            "Download scope. "
            "'metadata' gets docs/schema/small text files, "
            "'analysis' adds vmtable/deployments/subscriptions, "
            "'prepare-dataset' adds vm_cpu_readings, "
            "'full' downloads every URL in the links file."
        ),
    )
    parser.add_argument(
        "--links-file",
        type=Path,
        default=DEFAULT_LINKS_FILE,
        help="Path to AzurePublicDatasetLinksV2.txt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Workspace root directory. Defaults to the repo root.",
    )
    parser.add_argument(
        "--cpu-files",
        default="all",
        help=(
            "Subset of vm_cpu_readings shards to download, for example "
            "'1-4,7,10-12'. Use 'all' for every shard."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel download workers.",
    )
    parser.add_argument(
        "--extract-mode",
        choices=("default", "none", "all"),
        default="default",
        help=(
            "Gzip extraction policy. 'default' extracts only vmtable, "
            "deployments, and subscriptions because the notebooks expect the "
            "plain .csv paths. 'all' extracts every .gz file, which is large."
        ),
    )
    parser.add_argument(
        "--delete-archive-after-extract",
        action="store_true",
        help="Delete the downloaded .gz after a successful extraction.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files instead of skipping them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected files without downloading them.",
    )
    return parser.parse_args()


def load_assets(links_file: Path) -> list[Asset]:
    assets: list[Asset] = []
    for raw_line in links_file.read_text(encoding="utf-8").splitlines():
        url = raw_line.strip()
        if not url or url.startswith("#"):
            continue
        relative_path = extract_relative_path(url)
        assets.append(Asset(url=url, relative_path=relative_path))
    if not assets:
        raise ValueError(f"No URLs found in {links_file}")
    return assets


def extract_relative_path(url: str) -> Path:
    parsed = urllib.parse.urlparse(url)
    decoded_path = urllib.parse.unquote(parsed.path)
    if BLOB_MARKER not in decoded_path:
        raise ValueError(f"Unexpected Azure blob URL path: {url}")
    relative = decoded_path.split(BLOB_MARKER, 1)[1]
    return Path(relative)


def select_assets(
    assets: list[Asset],
    profile: str,
    cpu_file_spec: str,
) -> list[Asset]:
    cpu_indices = parse_cpu_file_spec(cpu_file_spec)
    selected: list[Asset] = []
    for asset in assets:
        rel = asset.relative_path_str
        if profile == PROFILE_FULL:
            include = True
        elif profile == PROFILE_METADATA:
            include = is_metadata_asset(rel)
        elif profile == PROFILE_ANALYSIS:
            include = is_analysis_asset(rel)
        elif profile == PROFILE_PREPARE:
            include = is_prepare_asset(rel)
        else:
            raise ValueError(f"Unsupported profile: {profile}")

        if include and asset.is_cpu_reading and cpu_indices is not None:
            shard_index = cpu_shard_index(asset.relative_path.name)
            include = shard_index in cpu_indices

        if include:
            selected.append(asset)
    return selected


def is_metadata_asset(relative_path: str) -> bool:
    return relative_path in METADATA_FILES or relative_path.startswith("azure2019_data/")


def is_analysis_asset(relative_path: str) -> bool:
    return is_metadata_asset(relative_path) or relative_path in DEFAULT_EXTRACT_RELATIVE_PATHS


def is_prepare_asset(relative_path: str) -> bool:
    return is_analysis_asset(relative_path) or relative_path.startswith("trace_data/vm_cpu_readings/")


def parse_cpu_file_spec(spec: str) -> set[int] | None:
    cleaned = spec.strip().lower()
    if cleaned == "all":
        return None

    indices: set[int] = set()
    for chunk in cleaned.split(","):
        part = chunk.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid CPU shard range: {part}")
            indices.update(range(start, end + 1))
        else:
            indices.add(int(part))
    if not indices:
        raise ValueError("No CPU shard indices were parsed from --cpu-files")
    return indices


def cpu_shard_index(filename: str) -> int:
    match = CPU_READING_PATTERN.search(filename)
    if not match:
        raise ValueError(f"Unrecognized vm_cpu_readings filename: {filename}")
    return int(match.group(1))


def should_extract(asset: Asset, extract_mode: str) -> bool:
    rel = asset.relative_path_str
    if not rel.endswith(".gz"):
        return False
    if extract_mode == "none":
        return False
    if extract_mode == "all":
        return True
    return rel in DEFAULT_EXTRACT_RELATIVE_PATHS


def extracted_path_for(asset: Asset) -> Path:
    return destination_relative_path(asset).with_suffix("")


def destination_relative_path(asset: Asset) -> Path:
    rel = asset.relative_path_str
    if rel == "Azure 2019 Public Dataset V2 - Trace Analysis.ipynb":
        return Path("notebooks") / "reference" / asset.relative_path.name
    if rel.startswith("trace_data/") or rel.startswith("azure2019_data/"):
        return Path("data") / "raw" / asset.relative_path
    return Path("data") / "metadata" / asset.relative_path.name


def download_selected_assets(
    assets: list[Asset],
    output_dir: Path,
    extract_mode: str,
    workers: int,
    overwrite: bool,
    delete_archive_after_extract: bool,
) -> tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if workers < 1:
        raise ValueError("--workers must be at least 1")

    downloaded = 0
    skipped = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(
                download_one_asset,
                asset,
                output_dir,
                extract_mode,
                overwrite,
                delete_archive_after_extract,
            ): asset
            for asset in assets
        }
        for future in concurrent.futures.as_completed(future_map):
            status = future.result()
            if status == "downloaded":
                downloaded += 1
            else:
                skipped += 1
    return downloaded, skipped


def download_one_asset(
    asset: Asset,
    output_dir: Path,
    extract_mode: str,
    overwrite: bool,
    delete_archive_after_extract: bool,
) -> str:
    archive_rel_path = destination_relative_path(asset)
    archive_path = output_dir / archive_rel_path
    extracted_path = output_dir / extracted_path_for(asset)
    needs_extract = should_extract(asset, extract_mode)

    if not overwrite and archive_path.exists():
        print(f"[skip] {archive_rel_path.as_posix()}")
        if needs_extract and not extracted_path.exists():
            extract_gzip_file(archive_path, extracted_path, overwrite=False)
        return "skipped"

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = archive_path.with_suffix(archive_path.suffix + ".part")
    if temp_path.exists():
        temp_path.unlink()

    print(f"[downloading] {archive_rel_path.as_posix()}")
    with urllib.request.urlopen(asset.url) as response, temp_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    temp_path.replace(archive_path)

    if needs_extract:
        extract_gzip_file(archive_path, extracted_path, overwrite=overwrite)
        if delete_archive_after_extract:
            archive_path.unlink()

    return "downloaded"


def extract_gzip_file(source_path: Path, target_path: Path, overwrite: bool) -> None:
    if target_path.exists() and not overwrite:
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(target_path.suffix + ".part")
    if temp_path.exists():
        temp_path.unlink()

    print(f"[extract] {source_path.as_posix()} -> {target_path.as_posix()}")
    with gzip.open(source_path, "rb") as compressed, temp_path.open("wb") as plain:
        shutil.copyfileobj(compressed, plain)
    temp_path.replace(target_path)


def print_dry_run(assets: list[Asset], output_dir: Path, extract_mode: str) -> None:
    print(f"Selected {len(assets)} files:")
    for asset in assets:
        destination = output_dir / destination_relative_path(asset)
        line = f"  {asset.relative_path_str} -> {destination.relative_to(output_dir).as_posix()}"
        if should_extract(asset, extract_mode):
            line += f" (extract -> {extracted_path_for(asset).as_posix()})"
        print(line)


def main() -> int:
    args = parse_args()

    try:
        assets = load_assets(args.links_file)
        selected = select_assets(assets, args.profile, args.cpu_files)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not selected:
        print("No files matched the selected profile/options.", file=sys.stderr)
        return 1

    if args.profile in {PROFILE_PREPARE, PROFILE_FULL} and args.cpu_files.strip().lower() == "all":
        print(
            "warning: the selected profile includes all 195 vm_cpu_readings shards. "
            "This is a large download. Use --cpu-files to limit the subset if needed.",
            file=sys.stderr,
        )

    if args.dry_run:
        print_dry_run(selected, args.output_dir.resolve(), args.extract_mode)
        return 0

    try:
        downloaded, skipped = download_selected_assets(
            assets=selected,
            output_dir=args.output_dir.resolve(),
            extract_mode=args.extract_mode,
            workers=args.workers,
            overwrite=args.overwrite,
            delete_archive_after_extract=args.delete_archive_after_extract,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        "Done. "
        f"downloaded={downloaded}, skipped={skipped}, output_dir={args.output_dir.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
