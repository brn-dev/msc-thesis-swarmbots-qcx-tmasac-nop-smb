from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.evaluate_thesis_mjw_unseen_morphologies import (
    discover_evaluation_checkpoints,
    migrate_evaluation_payload,
)
from experiments.evaluate_thesis_mjw_unseen_morphologies import main as _evaluate_main

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results" / "evaluation_50m"
EXPERIMENT_RUN_DIR = REPO_ROOT / "runs" / "thesis_mjw_po_wall_disconnected_finetune_50m"
HARD_WALL_EXPERIMENT_RUN_DIR = (
    REPO_ROOT / "runs" / "thesis_mjw_po_wall_disconnected_finetune_hard_wall_50m"
)
HARD_WALL_HEIGHT = 0.4
GLOBAL_SUMMARY_SCHEMA_VERSION = 1
GLOBAL_SUMMARY_FILENAME = "global_summary.json"
EPISODE_METRIC_KEYS = (
    "episode_return",
    "episode_length",
    "progress_reward",
    "guidance_reward",
)
CONNECTION_METRIC_OUTCOME_COUNTS = {
    "successful_connections_per_unit": "episode_count",
    "successful_connections_per_unit_successful_episodes": "success_count",
    "successful_connections_per_unit_unsuccessful_episodes": "unsuccessful_count",
}
COMMON_COMPARISON_CONFIG_KEYS = (
    "unit_counts",
    "pool_size",
    "episodes",
    "pool_seed_base",
    "rollout_seed",
    "deterministic",
    "episode_length",
    "unconnected_prob",
)


@dataclass(frozen=True)
class EvaluationCase:
    key: str
    display_name: str
    group_dir: Path
    disable_policy_connector_actions: bool
    scenario_kwargs_overrides: Mapping[str, object] | None = None


EVALUATION_CASES = (
    EvaluationCase(
        key="medium_connectors",
        display_name="Medium wall, connectors enabled",
        group_dir=EXPERIMENT_RUN_DIR / "tmasac_baseline",
        disable_policy_connector_actions=False,
    ),
    EvaluationCase(
        key="medium_no_connectors",
        display_name="Medium wall, connectors disabled",
        group_dir=EXPERIMENT_RUN_DIR / "tmasac_no_connectors",
        disable_policy_connector_actions=True,
    ),
    EvaluationCase(
        key="hard_connectors",
        display_name="Hard wall, connectors enabled",
        group_dir=HARD_WALL_EXPERIMENT_RUN_DIR / "tmasac_baseline",
        disable_policy_connector_actions=False,
        scenario_kwargs_overrides={"wall_height": HARD_WALL_HEIGHT},
    ),
    EvaluationCase(
        key="hard_no_connectors",
        display_name="Hard wall, connectors disabled",
        group_dir=HARD_WALL_EXPERIMENT_RUN_DIR / "tmasac_no_connectors",
        disable_policy_connector_actions=True,
        scenario_kwargs_overrides={"wall_height": HARD_WALL_HEIGHT},
    ),
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all 50M disconnected-start PO-wall TMASAC models on "
            "matching medium/hard walls with connectors enabled or disabled."
        )
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(case.key for case in EVALUATION_CASES),
        help="Evaluate only the selected case; repeat to select multiple cases.",
    )
    parser.add_argument("--unit-counts", nargs="+", type=int, default=(4, 5))
    parser.add_argument("--pool-size", type=int, default=50)
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--max-parallel-envs", type=int, default=512)
    parser.add_argument("--pool-seed-base", type=int, default=3_000_000)
    parser.add_argument("--rollout-seed", type=int, default=2_000_000)
    parser.add_argument("--episode-length", type=int, default=512)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--cuda_idx", "--cuda-idx", "--gpu", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Rebuild the global summary from existing case JSON files without evaluating.",
    )
    return parser.parse_args(argv)


def _selected_cases(case_keys: Sequence[str] | None) -> tuple[EvaluationCase, ...]:
    if case_keys is None:
        return EVALUATION_CASES
    selected_keys = set(case_keys)
    return tuple(case for case in EVALUATION_CASES if case.key in selected_keys)


def _shared_evaluator_args(
    *,
    args: argparse.Namespace,
    case: EvaluationCase,
    checkpoints: Sequence[Path],
) -> list[str]:
    evaluator_args = [
        "--target",
        "po_wall_tmasac",
        "--unit-counts",
        *(str(unit_count) for unit_count in args.unit_counts),
        "--pool-size",
        str(args.pool_size),
        "--episodes",
        str(args.episodes),
        "--max-parallel-envs",
        str(args.max_parallel_envs),
        "--pool-seed-base",
        str(args.pool_seed_base),
        "--rollout-seed",
        str(args.rollout_seed),
        "--episode-length",
        str(args.episode_length),
        "--output",
        str(args.output_root / f"{case.key}.json"),
    ]
    for checkpoint in checkpoints:
        evaluator_args.extend(("--po-wall-checkpoint", str(checkpoint)))
    if args.cuda_idx is not None:
        evaluator_args.extend(("--cuda-idx", str(args.cuda_idx)))
    for enabled, option in (
        (args.stochastic, "--stochastic"),
        (args.resume, "--resume"),
        (args.overwrite, "--overwrite"),
        (args.no_progress, "--no-progress"),
        (args.dry_run, "--dry-run"),
    ):
        if enabled:
            evaluator_args.append(option)
    return evaluator_args


def _summarize_values(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.pstdev(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _pooled_metric_summary(
    results: Sequence[Mapping[str, Any]],
    *,
    metric_key: str,
    outcome_count_key: str | None = None,
) -> dict[str, int | float] | None:
    weighted_summaries: list[tuple[int, Mapping[str, float]]] = []
    for result in results:
        summary = result["summary"]
        metric_summary = summary[metric_key]
        if metric_summary is None:
            continue
        if outcome_count_key == "unsuccessful_count":
            episode_count = int(summary["episode_count"]) - int(summary["success_count"])
        else:
            episode_count = int(
                summary["episode_count"]
                if outcome_count_key is None
                else summary[outcome_count_key]
            )
        observation_count = episode_count
        if metric_key in CONNECTION_METRIC_OUTCOME_COUNTS:
            observation_count *= int(result["unit_count"])
        if observation_count:
            weighted_summaries.append((observation_count, metric_summary))

    total_observation_count = sum(count for count, _summary in weighted_summaries)
    if not total_observation_count:
        return None
    pooled_mean = sum(
        count * float(metric_summary["mean"])
        for count, metric_summary in weighted_summaries
    ) / total_observation_count
    pooled_second_moment = sum(
        count
        * (
            float(metric_summary["std"]) ** 2
            + float(metric_summary["mean"]) ** 2
        )
        for count, metric_summary in weighted_summaries
    ) / total_observation_count
    pooled_variance = max(0.0, pooled_second_moment - pooled_mean**2)
    return {
        "observation_count": total_observation_count,
        "mean": float(pooled_mean),
        "std": float(pooled_variance**0.5),
        "min": min(float(summary["min"]) for _count, summary in weighted_summaries),
        "max": max(float(summary["max"]) for _count, summary in weighted_summaries),
    }


def _aggregate_connection_usage(
    results: Sequence[Mapping[str, Any]],
    *,
    outcome_key: str,
) -> dict[str, int | float | None]:
    usage_summaries = [
        result["summary"]["connection_usage_by_outcome"][outcome_key]
        for result in results
    ]
    episode_count = sum(int(summary["episode_count"]) for summary in usage_summaries)
    observed_summaries = [
        summary
        for summary in usage_summaries
        if summary["episodes_with_never_connected_unit_count"] is not None
        and summary["episodes_without_any_successful_connection_count"] is not None
    ]
    observed_episode_count = sum(
        int(summary["episode_count"]) for summary in observed_summaries
    )
    never_connected_count = sum(
        int(summary["episodes_with_never_connected_unit_count"])
        for summary in observed_summaries
    )
    no_connection_count = sum(
        int(summary["episodes_without_any_successful_connection_count"])
        for summary in observed_summaries
    )
    return {
        "episode_count": episode_count,
        "observed_episode_count": observed_episode_count,
        "observation_coverage_percent": (
            100.0 * observed_episode_count / episode_count if episode_count else None
        ),
        "episodes_with_never_connected_unit_count": (
            never_connected_count if observed_episode_count else None
        ),
        "episodes_with_never_connected_unit_rate_percent": (
            100.0 * never_connected_count / observed_episode_count
            if observed_episode_count
            else None
        ),
        "episodes_without_any_successful_connection_count": (
            no_connection_count if observed_episode_count else None
        ),
        "episodes_without_any_successful_connection_rate_percent": (
            100.0 * no_connection_count / observed_episode_count
            if observed_episode_count
            else None
        ),
    }


def _aggregate_results(results: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    episode_count = sum(int(result["summary"]["episode_count"]) for result in results)
    success_count = sum(int(result["summary"]["success_count"]) for result in results)
    unsuccessful_count = episode_count - success_count
    aggregate: dict[str, object] = {
        "run_count": len({str(result["run_id"]) for result in results}),
        "evaluation_job_count": len(results),
        "episode_count": episode_count,
        "success_count": success_count,
        "unsuccessful_count": unsuccessful_count,
        "success_rate_percent": 100.0 * success_count / episode_count,
        "evaluation_job_success_rate_percent": _summarize_values(
            [float(result["summary"]["success_rate_percent"]) for result in results]
        ),
        "connection_usage_by_outcome": {
            outcome_key: _aggregate_connection_usage(results, outcome_key=outcome_key)
            for outcome_key in ("successful_episodes", "unsuccessful_episodes")
        },
    }
    for metric_key in EPISODE_METRIC_KEYS:
        aggregate[metric_key] = _pooled_metric_summary(
            results,
            metric_key=metric_key,
        )
    for metric_key, count_key in CONNECTION_METRIC_OUTCOME_COUNTS.items():
        aggregate[metric_key] = _pooled_metric_summary(
            results,
            metric_key=metric_key,
            outcome_count_key=count_key,
        )
    return aggregate


def _aggregate_case(payload: Mapping[str, Any]) -> dict[str, object]:
    results = payload["results"]
    unit_counts = sorted({int(result["unit_count"]) for result in results})
    return {
        "config": payload["config"],
        "source_created_at": payload.get("created_at"),
        "source_completed_at": payload.get("completed_at"),
        "overall": _aggregate_results(results),
        "by_unit_count": {
            str(unit_count): _aggregate_results(
                [result for result in results if int(result["unit_count"]) == unit_count]
            )
            for unit_count in unit_counts
        },
    }


def _comparison_design(payload: Mapping[str, Any]) -> dict[str, object]:
    config = payload["config"]
    return {key: config[key] for key in COMMON_COMPARISON_CONFIG_KEYS}


def _comparison_snapshot(aggregate: Mapping[str, Any]) -> dict[str, float | None]:
    successful_usage = aggregate["connection_usage_by_outcome"]["successful_episodes"]
    return {
        "success_rate_percent": float(aggregate["success_rate_percent"]),
        "episode_return_mean": float(aggregate["episode_return"]["mean"]),
        "successful_connections_per_unit_mean": (
            float(aggregate["successful_connections_per_unit_successful_episodes"]["mean"])
            if aggregate["successful_connections_per_unit_successful_episodes"] is not None
            else None
        ),
        "successful_episodes_with_never_connected_unit_rate_percent": (
            successful_usage["episodes_with_never_connected_unit_rate_percent"]
        ),
        "successful_episodes_without_any_successful_connection_rate_percent": (
            successful_usage[
                "episodes_without_any_successful_connection_rate_percent"
            ]
        ),
    }


def _compare_aggregates(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    first_name: str,
    second_name: str,
) -> dict[str, object]:
    first_snapshot = _comparison_snapshot(first)
    second_snapshot = _comparison_snapshot(second)
    return {
        first_name: first_snapshot,
        second_name: second_snapshot,
        f"{first_name}_minus_{second_name}": {
            metric_name: (
                first_value - second_snapshot[metric_name]
                if first_value is not None and second_snapshot[metric_name] is not None
                else None
            )
            for metric_name, first_value in first_snapshot.items()
        },
    }


def _case_pair_comparisons(
    case_summaries: Mapping[str, Mapping[str, Any]],
    case_payloads: Mapping[str, Mapping[str, Any]],
    *,
    first_case_key: str,
    second_case_key: str,
    first_name: str,
    second_name: str,
) -> dict[str, object] | None:
    if first_case_key not in case_summaries or second_case_key not in case_summaries:
        return None
    if _comparison_design(case_payloads[first_case_key]) != _comparison_design(
        case_payloads[second_case_key]
    ):
        return {
            "comparable_evaluation_design": False,
            "by_unit_count": {},
            "overall": None,
        }

    first_case = case_summaries[first_case_key]
    second_case = case_summaries[second_case_key]
    unit_counts = sorted(
        set(first_case["by_unit_count"]) & set(second_case["by_unit_count"]),
        key=int,
    )
    return {
        "comparable_evaluation_design": True,
        "by_unit_count": {
            unit_count: _compare_aggregates(
                first_case["by_unit_count"][unit_count],
                second_case["by_unit_count"][unit_count],
                first_name=first_name,
                second_name=second_name,
            )
            for unit_count in unit_counts
        },
        "overall": _compare_aggregates(
            first_case["overall"],
            second_case["overall"],
            first_name=first_name,
            second_name=second_name,
        ),
    }


def build_global_summary(
    case_payloads: Mapping[str, Mapping[str, Any]],
    *,
    source_paths: Mapping[str, Path],
) -> dict[str, object]:
    case_summaries = {
        case_key: {
            "display_name": next(
                case.display_name for case in EVALUATION_CASES if case.key == case_key
            ),
            "source_file": str(source_paths[case_key].resolve()),
            **_aggregate_case(payload),
        }
        for case_key, payload in case_payloads.items()
    }
    connector_effect = {
        wall: _case_pair_comparisons(
            case_summaries,
            case_payloads,
            first_case_key=f"{wall}_connectors",
            second_case_key=f"{wall}_no_connectors",
            first_name="connector_enabled",
            second_name="connector_disabled",
        )
        for wall in ("medium", "hard")
    }
    wall_difficulty_effect = {
        connector_mode: _case_pair_comparisons(
            case_summaries,
            case_payloads,
            first_case_key=f"hard_{connector_mode}",
            second_case_key=f"medium_{connector_mode}",
            first_name="hard_wall",
            second_name="medium_wall",
        )
        for connector_mode in ("connectors", "no_connectors")
    }
    return {
        "schema_version": GLOBAL_SUMMARY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "included_cases": list(case_summaries),
        "missing_cases": [
            case.key for case in EVALUATION_CASES if case.key not in case_summaries
        ],
        "cases": case_summaries,
        "comparisons": {
            "connector_enabled_vs_disabled": connector_effect,
            "hard_vs_medium_wall": wall_difficulty_effect,
        },
        "metric_notes": {
            "pooled_statistics": (
                "Means and population standard deviations pool the source first and second "
                "moments using their exact episode or episode-unit observation counts."
            ),
            "evaluation_job_success_rate_percent": (
                "Distribution across checkpoint-by-unit-count evaluation jobs; use its standard "
                "deviation as run-level variability only within one unit-count group."
            ),
            "never_connected_unit": (
                "A unit with zero successful connector activations during the episode. All "
                "evaluated episodes start fully disconnected."
            ),
            "comparison_differences": "The first named condition minus the second.",
        },
    }


def write_global_summary(output_root: Path) -> Path:
    case_payloads: dict[str, Mapping[str, Any]] = {}
    source_paths: dict[str, Path] = {}
    for case in EVALUATION_CASES:
        source_path = output_root / f"{case.key}.json"
        if not source_path.is_file():
            continue
        payload = migrate_evaluation_payload(
            json.loads(source_path.read_text(encoding="utf-8")),
            source_description=str(source_path),
        )
        case_payloads[case.key] = payload
        source_paths[case.key] = source_path
    if not case_payloads:
        raise FileNotFoundError(f"No case evaluation JSON files found under {output_root}")

    output_path = output_root / GLOBAL_SUMMARY_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(
            build_global_summary(case_payloads, source_paths=source_paths),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if args.summary_only:
        summary_path = write_global_summary(args.output_root)
        print(f"Wrote global summary: {summary_path.resolve()}")
        return 0

    for case in _selected_cases(args.case):
        checkpoints = discover_evaluation_checkpoints((case.group_dir,))
        if not checkpoints:
            raise FileNotFoundError(
                f"No final or best checkpoints found under {case.group_dir}"
            )
        print(f"\n=== {case.display_name} ===")
        _evaluate_main(
            _shared_evaluator_args(
                args=args,
                case=case,
                checkpoints=checkpoints,
            ),
            unconnected_prob=1.0,
            disable_policy_connector_actions=case.disable_policy_connector_actions,
            scenario_kwargs_overrides=case.scenario_kwargs_overrides,
            morphology_description="unseen, fully disconnected PO-wall pools",
        )

    if not args.dry_run:
        summary_path = write_global_summary(args.output_root)
        print(f"\nWrote global summary: {summary_path.resolve()}")
    print(f"\nEvaluation results: {args.output_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
