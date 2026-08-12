"""out 산출물을 canonical 디렉터리와 파일 단위로 비교한다.
모든 비교는 bounded chunk로 읽어 466MB CSV도 메모리에 올리지 않는다.
검증 대상이 아닌 canonical 전용 진단·중복 파일도 출력에 명시한다.
"""

from __future__ import annotations

from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = EXPERIMENT_DIR.parents[1]
OUT_DIR = EXPERIMENT_DIR / "out"
CANONICAL_DIR = REPOSITORY_DIR / "data/processed/notion_toy_google2019_v5_mode_capacity_bounded"
EXCLUDED_CANONICAL_ONLY = {
    "vm_usage_scenarios.csv", "metadata.json", "preprocessing_diagnostics.json",
    "scaling_diagnostics.json", "toy_instance_summary.md",
}
CHUNK_SIZE = 8 * 1024 * 1024


def _same_bytes(left: Path, right: Path) -> bool:
    """두 파일을 chunk 단위로 읽어 바이트 내용이 같은지 반환한다."""
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as left_file, right.open("rb") as right_file:
        while True:
            left_chunk = left_file.read(CHUNK_SIZE)
            right_chunk = right_file.read(CHUNK_SIZE)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def _compare_names(local_names: list[str], canonical_names: list[str]) -> list[str]:
    """local과 canonical의 공통 산출물을 비교하고 불일치 이름을 반환한다."""
    mismatches = []
    expected_names = sorted(set(canonical_names) - EXCLUDED_CANONICAL_ONLY)
    for name in sorted(set(local_names) | set(expected_names)):
        local_path = OUT_DIR / name
        canonical_path = CANONICAL_DIR / name
        if name in EXCLUDED_CANONICAL_ONLY:
            continue
        if not local_path.is_file() or not canonical_path.is_file():
            print(f"{name}: NO (missing local or canonical file)")
            mismatches.append(name)
            continue
        mode = "chunked-8MiB"
        matched = _same_bytes(local_path, canonical_path)
        status = "MATCH" if matched else "MISMATCH"
        print(f"{name}: {status} ({mode}, {local_path.stat().st_size} bytes)")
        if not matched:
            mismatches.append(name)
    return mismatches


def main() -> None:
    """out 파일과 canonical 파일의 비교 결과를 출력한다."""
    local_names = [path.name for path in OUT_DIR.iterdir() if path.is_file()]
    canonical_names = [path.name for path in CANONICAL_DIR.iterdir() if path.is_file()]
    excluded = sorted(EXCLUDED_CANONICAL_ONLY & set(canonical_names))
    print("excluded_canonical_only:")
    for name in excluded:
        print(f"  {name} (intentionally not created; comparison skipped)")
    unexpected_local = sorted(set(local_names) & EXCLUDED_CANONICAL_ONLY)
    for name in unexpected_local:
        print(f"{name}: MISMATCH (must not be created)")
    mismatches = _compare_names(local_names, canonical_names)
    mismatches.extend(name for name in unexpected_local if name not in mismatches)
    matched_count = len(set(local_names) - set(mismatches))
    mismatch_count = len(set(mismatches))
    print(f"summary: matched={matched_count}, mismatched={mismatch_count}")
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
