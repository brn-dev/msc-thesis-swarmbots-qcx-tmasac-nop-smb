from __future__ import annotations

import argparse
import shutil
from pathlib import Path

DELETE_ANSWERS = {"d", "del", "delete"}
RUN_MARKER_NAMES = ("run_metadata_0.json", "log.csv", "models")


def has_final_model(run_dir: Path) -> bool:
    models_dir = run_dir / "models"
    return models_dir.is_dir() and any(models_dir.glob("*_final.pt"))


def is_run_dir(path: Path) -> bool:
    return path.is_dir() and any((path / marker_name).exists() for marker_name in RUN_MARKER_NAMES)


def iter_run_dirs(target_dir: Path) -> list[Path]:
    if not target_dir.is_dir():
        raise NotADirectoryError(f"{target_dir} is not a directory")

    if is_run_dir(target_dir):
        return [target_dir]

    run_dirs = {
        child
        for child in target_dir.iterdir()
        if is_run_dir(child)
    }
    run_dirs.update(
        run_dir
        for category_dir in target_dir.iterdir()
        if category_dir.is_dir() and not is_run_dir(category_dir)
        for run_dir in category_dir.iterdir()
        if is_run_dir(run_dir)
    )
    return sorted(run_dirs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete run folders that do not contain a final model.",
    )
    parser.add_argument(
        "target_dir",
        nargs="?",
        type=Path,
        default=Path("runs"),
        help="Run directory, category directory containing runs, or parent directory containing categories.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    deleted_count = 0
    skipped_count = 0
    kept_count = 0

    for run_dir in iter_run_dirs(args.target_dir):
        if has_final_model(run_dir):
            kept_count += 1
            continue

        answer = input(f"{run_dir} has no final model. Type d/del/delete to delete, Enter to skip: ")
        if answer.strip().lower() in DELETE_ANSWERS:
            shutil.rmtree(run_dir)
            deleted_count += 1
            print(f"deleted {run_dir}")
        else:
            skipped_count += 1
            print(f"skipped {run_dir}")

    print(f"deleted {deleted_count} run directories without final models")
    print(f"skipped {skipped_count} candidate run directories")
    print(f"kept {kept_count} run directories with final models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
