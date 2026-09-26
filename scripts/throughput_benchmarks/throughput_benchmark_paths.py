from __future__ import annotations

import os
import platform
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
THROUGHPUT_BENCHMARK_RESULTS_DIR = REPO_ROOT / "throughput_benchmark_results"


def get_machine_name() -> str:
    machine_name = (
        os.environ.get("COMPUTERNAME")
        or os.environ.get("HOSTNAME")
        or platform.node()
        or "unknown-machine"
    ).strip()
    return "".join("_" if character in {"/", "\\", ":", " "} else character for character in machine_name) or "unknown-machine"


def default_throughput_benchmark_json_out(script_file: str | Path) -> Path:
    return THROUGHPUT_BENCHMARK_RESULTS_DIR / get_machine_name() / f"{Path(script_file).stem}.json"
