from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENTS_DIR.parent


def discover_plot_scripts() -> list[Path]:
    return sorted(path for path in EXPERIMENTS_DIR.glob("thesis_*/plot_*.py") if path.is_file())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run all experiments/thesis_*/plot_*.py scripts using the current Python environment.",
    )
    parser.add_argument("--dry-run", action="store_true", help="List the scripts without running them.")
    args = parser.parse_args(argv)
    scripts = discover_plot_scripts()
    if not scripts:
        print(f"No thesis plot scripts found under {EXPERIMENTS_DIR}", file=sys.stderr)
        return 1
    if args.dry_run:
        for script in scripts:
            print(script.relative_to(REPO_ROOT).as_posix())
        return 0

    failures: list[tuple[Path, int]] = []
    for index, script in enumerate(scripts, start=1):
        print(f"\n[{index}/{len(scripts)}] {script.relative_to(REPO_ROOT).as_posix()}", flush=True)
        result = subprocess.run([sys.executable, str(script)], cwd=REPO_ROOT, check=False)
        if result.returncode != 0:
            failures.append((script, result.returncode))

    print(f"\nFinished: {len(scripts) - len(failures)}/{len(scripts)} thesis plot scripts succeeded.")
    for script, returncode in failures:
        print(f"FAILED (exit {returncode}): {script.relative_to(REPO_ROOT).as_posix()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
