from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.throughput_benchmarks.throughput_benchmark_paths import THROUGHPUT_BENCHMARK_RESULTS_DIR

DEFAULT_BENCHMARK_FILE = "benchmark_mj_env_vs_mjw_env.json"
DEFAULT_MJ_WORKERS = (32, 64, 128, 256)
DEFAULT_MARKDOWN_OUTPUT = "mj_env_vs_mjw_summary.md"
DEFAULT_LATEX_OUTPUT = "mj_env_vs_mjw_summary.tex"


@dataclass(frozen=True)
class MachineTable:
    machine_name: str
    created_at: str | None
    headers: list[str]
    rows: list[list[str]]
    failure_count: int


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object.")
    return payload


def find_result_files(results_dir: Path, benchmark_file: str) -> list[Path]:
    return sorted(
        path
        for path in results_dir.glob(f"*/{benchmark_file}")
        if path.is_file()
    )


def fps_by_num_envs(results: list[dict[str, Any]], *, backend: str) -> dict[int, float]:
    values: dict[int, float] = {}
    for result in results:
        if result.get("backend") != backend:
            continue
        values[int(result["num_envs"])] = float(result["step_envs_per_second"])
    return values


def mj_fps_by_num_envs_and_workers(results: list[dict[str, Any]]) -> dict[tuple[int, int], float]:
    values: dict[tuple[int, int], float] = {}
    for result in results:
        if result.get("backend") != "mj_env":
            continue
        mj_workers = result.get("mj_workers")
        if mj_workers is None:
            continue
        values[(int(result["num_envs"]), int(mj_workers))] = float(result["step_envs_per_second"])
    return values


def format_fps(value: float) -> str:
    return f"{value:,.0f}"


def format_mj_fps(value: float | None, *, mjw_fps: float | None) -> str:
    if value is None:
        return "-"
    formatted_value = format_fps(value)
    if mjw_fps is None:
        return formatted_value

    relative_change_percent = (value / mjw_fps - 1.0) * 100.0
    return f"{formatted_value} ({relative_change_percent:+.0f}%)"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_machine_table(path: Path, *, mj_workers: tuple[int, ...]) -> MachineTable:
    payload = load_json(path)
    raw_results = payload.get("results", [])
    if not isinstance(raw_results, list):
        raise ValueError(f"{path} has a non-list 'results' field.")

    results = [
        result
        for result in raw_results
        if isinstance(result, dict)
    ]
    mjw_fps_by_envs = fps_by_num_envs(results, backend="mjw_env")
    mj_fps_by_key = mj_fps_by_num_envs_and_workers(results)
    env_counts = sorted({
        *mjw_fps_by_envs.keys(),
        *(num_envs for num_envs, _ in mj_fps_by_key),
    })

    headers = [
        "Parallel envs",
        "MJWarp FPS",
        *(f"MuJoCo FPS ({worker_count} workers)" for worker_count in mj_workers),
    ]
    rows = []
    for num_envs in env_counts:
        mjw_fps = mjw_fps_by_envs.get(num_envs)
        rows.append(
            [
                str(num_envs),
                "-" if mjw_fps is None else format_fps(mjw_fps),
                *(
                    format_mj_fps(
                        mj_fps_by_key.get((num_envs, worker_count)),
                        mjw_fps=mjw_fps,
                    )
                    for worker_count in mj_workers
                ),
            ]
        )

    created_at = payload.get("created_at")
    failures = payload.get("failures", [])
    return MachineTable(
        machine_name=path.parent.name,
        created_at=created_at if isinstance(created_at, str) and created_at else None,
        headers=headers,
        rows=rows,
        failure_count=len(failures) if isinstance(failures, list) else 0,
    )


def render_markdown_machine_table(table: MachineTable) -> str:
    lines = [f"## {table.machine_name}"]
    if table.created_at is not None:
        lines.append(f"Result timestamp: {table.created_at}")
    lines.append("")
    lines.append(markdown_table(table.headers, table.rows))

    if table.failure_count:
        lines.append("")
        lines.append(f"Failures recorded: {table.failure_count}")

    return "\n".join(lines)


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def latex_label(value: str) -> str:
    sanitized = "".join(character.lower() if character.isalnum() else "-" for character in value)
    return "-".join(part for part in sanitized.split("-") if part)


def latex_table(headers: list[str], rows: list[list[str]]) -> str:
    column_spec = "r" * len(headers)
    lines = [
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\hline",
        " & ".join(latex_escape(header) for header in headers) + r" \\",
        r"\hline",
    ]
    lines.extend(
        " & ".join(latex_escape(value) for value in row) + r" \\"
        for row in rows
    )
    lines.extend([
        r"\hline",
        r"\end{tabular}",
    ])
    return "\n".join(lines)


def render_latex_machine_table(table: MachineTable) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{latex_escape(f'MJWarp versus MuJoCo throughput on {table.machine_name}')}}}",
        rf"\label{{tab:mjwarp-mujoco-throughput-{latex_label(table.machine_name)}}}",
    ]
    if table.created_at is not None:
        lines.append(rf"% Result timestamp: {latex_escape(table.created_at)}")
    if table.failure_count:
        lines.append(rf"% Failures recorded: {table.failure_count}")
    lines.append(latex_table(table.headers, table.rows))
    lines.append(r"\end{table}")
    return "\n".join(lines)


def render_markdown_report(tables: list[MachineTable]) -> str:
    return "\n\n".join(render_markdown_machine_table(table) for table in tables)


def render_latex_report(tables: list[MachineTable]) -> str:
    return "\n\n".join(render_latex_machine_table(table) for table in tables)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print one MJWarp versus MuJoCo throughput summary table per machine result.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=THROUGHPUT_BENCHMARK_RESULTS_DIR,
        help="Directory containing per-machine throughput benchmark result folders.",
    )
    parser.add_argument(
        "--benchmark-file",
        default=DEFAULT_BENCHMARK_FILE,
        help="Benchmark JSON filename to read inside each machine folder.",
    )
    parser.add_argument(
        "--mj-workers",
        nargs="+",
        type=int,
        default=list(DEFAULT_MJ_WORKERS),
        help="MuJoCo worker-count columns to include.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=None,
        help=f"Markdown output path. Defaults to RESULTS_DIR/{DEFAULT_MARKDOWN_OUTPUT}.",
    )
    parser.add_argument(
        "--latex-output",
        type=Path,
        default=None,
        help=f"LaTeX output path. Defaults to RESULTS_DIR/{DEFAULT_LATEX_OUTPUT}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_files = find_result_files(args.results_dir, args.benchmark_file)
    if not result_files:
        raise FileNotFoundError(f"No {args.benchmark_file} files found under {args.results_dir}.")

    tables = [
        build_machine_table(path, mj_workers=tuple(args.mj_workers))
        for path in result_files
    ]
    markdown_output = args.markdown_output or args.results_dir / DEFAULT_MARKDOWN_OUTPUT
    latex_output = args.latex_output or args.results_dir / DEFAULT_LATEX_OUTPUT

    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    latex_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(render_markdown_report(tables) + "\n", encoding="utf-8")
    latex_output.write_text(render_latex_report(tables) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(tables)} machine tables to:\n"
        f"- {markdown_output}\n"
        f"- {latex_output}"
    )


if __name__ == "__main__":
    main()
