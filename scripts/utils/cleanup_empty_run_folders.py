from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def is_metadata_only_run_dir(run_dir: Path) -> bool:
    if not run_dir.is_dir():
        return False

    children = list(run_dir.iterdir())
    return len(children) == 1 and children[0].is_file() and children[0].name == "run_metadata_0.json"


def find_metadata_only_run_dirs(runs_root: Path) -> list[Path]:
    if not runs_root.is_dir():
        raise NotADirectoryError(f"{runs_root} is not a directory")

    return sorted(
        {
            metadata_file.parent
            for metadata_file in runs_root.rglob("run_metadata_0.json")
            if is_metadata_only_run_dir(metadata_file.parent)
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete run directories that contain only run_metadata_0.json.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="Root runs directory to scan.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print matching directories without deleting them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dirs = find_metadata_only_run_dirs(args.runs_root)

    for run_dir in run_dirs:
        if args.dry_run:
            print(run_dir)
        else:
            shutil.rmtree(run_dir)
            print(f"deleted {run_dir}")

    print(f"{'would delete' if args.dry_run else 'deleted'} {len(run_dirs)} run directories")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
