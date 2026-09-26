from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.evaluate_thesis_mjw_unseen_morphologies import (
    BEST_CHECKPOINT_NAME,
    DEFAULT_UNIT_COUNTS,
    TARGETS,
    TargetKey,
    _build_env_and_policy,
    _checkpoint_run_id,
    _load_checkpoint,
    _selected_target_keys,
    evaluate_policy,
    make_pool_seeds,
    resolve_target_checkpoints,
)
from swarmbots.utils.recording_resolution import DEFAULT_RECORDING_HEIGHT, DEFAULT_RECORDING_WIDTH

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs" / "thesis_mjw_unseen_morphologies" / "recordings"


@dataclass(frozen=True)
class RecordingConfig:
    unit_counts: tuple[int, ...]
    pool_size: int
    episodes_per_combination: int
    pool_seed_base: int
    rollout_seed: int
    deterministic: bool
    episode_length: int
    fps: int
    fps_mode: str
    frame_stride: int
    width: int
    height: int
    camera: int | str
    unconnected_prob: float = 0.0
    disable_policy_connector_actions: bool = False
    scenario_kwargs_overrides: dict[str, object] = field(default_factory=dict)


def _safe_path_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return component or "unnamed"


def checkpoint_output_name(checkpoint_path: Path) -> str:
    path_hash = hashlib.sha256(str(checkpoint_path.resolve()).encode()).hexdigest()[:8]
    return f"{_safe_path_component(_checkpoint_run_id(checkpoint_path))}-{path_hash}"


def combination_output_dir(
    *,
    output_root: Path,
    target_key: TargetKey,
    unit_count: int,
    checkpoint_path: Path,
) -> Path:
    return (
        output_root
        / target_key
        / f"{unit_count}_agents"
        / checkpoint_output_name(checkpoint_path)
    )


def _parse_camera(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def _parse_args(
    argv: Sequence[str] | None = None,
    *,
    default_output_root: Path = DEFAULT_OUTPUT_ROOT,
    morphology_description: str = "unseen, fully pre-connected morphologies",
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Record thesis TMASAC checkpoints on {morphology_description} "
            "containing 2 through 10 agents."
        ),
    )
    parser.add_argument(
        "--target",
        choices=("all", *TARGETS),
        default="all",
        help="Model/scenario target to record (default: both).",
    )
    parser.add_argument("--unit-counts", nargs="+", type=int, default=DEFAULT_UNIT_COUNTS)
    parser.add_argument("--pool-size", type=int, default=50)
    parser.add_argument(
        "-n",
        "--episodes-per-combination",
        "--episodes",
        dest="episodes_per_combination",
        type=int,
        default=3,
        help=(
            "Episodes to record per agent-count/scenario/checkpoint combination "
            "(default: 3)."
        ),
    )
    parser.add_argument("--pool-seed-base", type=int, default=1_000_000)
    parser.add_argument("--rollout-seed", type=int, default=2_000_000)
    parser.add_argument("--episode-length", type=int, default=512)
    parser.add_argument("--stochastic", action="store_true", help="Sample policy actions instead of using modes.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--fps-mode",
        choices=("compensate_stride", "fixed"),
        default="compensate_stride",
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--width", type=int, default=DEFAULT_RECORDING_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_RECORDING_HEIGHT)
    parser.add_argument("--camera", type=_parse_camera, default=-1)
    parser.add_argument("--cuda_idx", "--cuda-idx", "--gpu", type=int, default=None)
    parser.add_argument("--output", type=Path, default=default_output_root)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable episode progress bars.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--po-wall-checkpoint", type=Path, action="append")
    parser.add_argument("--find-opening-checkpoint", type=Path, action="append")
    return parser.parse_args(argv)


def _validate_args(
    args: argparse.Namespace,
    *,
    unconnected_prob: float = 0.0,
    disable_policy_connector_actions: bool = False,
    scenario_kwargs_overrides: Mapping[str, object] | None = None,
) -> RecordingConfig:
    if not 0.0 <= unconnected_prob <= 1.0:
        raise ValueError("unconnected_prob must be in [0, 1]")
    unit_counts = tuple(dict.fromkeys(args.unit_counts))
    if not unit_counts or any(count < 2 or count > 20 for count in unit_counts):
        raise ValueError("--unit-counts must contain values in [2, 20]")
    if args.pool_size <= 0:
        raise ValueError("--pool-size must be positive")
    if args.episodes_per_combination <= 0:
        raise ValueError("--episodes-per-combination must be positive")
    if args.episode_length <= 0:
        raise ValueError("--episode-length must be positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive")
    make_pool_seeds(
        pool_seed_base=args.pool_seed_base,
        unit_count=unit_counts[0],
        pool_size=args.pool_size,
    )
    return RecordingConfig(
        unit_counts=unit_counts,
        pool_size=args.pool_size,
        episodes_per_combination=args.episodes_per_combination,
        pool_seed_base=args.pool_seed_base,
        rollout_seed=args.rollout_seed,
        deterministic=not args.stochastic,
        episode_length=args.episode_length,
        fps=args.fps,
        fps_mode=args.fps_mode,
        frame_stride=args.frame_stride,
        width=args.width,
        height=args.height,
        camera=args.camera,
        unconnected_prob=unconnected_prob,
        disable_policy_connector_actions=disable_policy_connector_actions,
        scenario_kwargs_overrides=dict(scenario_kwargs_overrides or {}),
    )


def _prepare_combination_output(output_dir: Path, *, overwrite: bool) -> None:
    existing_videos = list(output_dir.glob("*.mp4")) if output_dir.is_dir() else []
    if existing_videos and not overwrite:
        raise FileExistsError(
            f"Refusing to replace {len(existing_videos)} existing video(s) in {output_dir}. "
            "Pass --overwrite to replace them."
        )
    if overwrite:
        for video_path in existing_videos:
            video_path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def make_combination_rollout_seed(
    *,
    rollout_seed_base: int,
    combination_index: int,
) -> int:
    return rollout_seed_base + combination_index


def _write_manifest(output_root: Path, payload: dict[str, object]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "recording.json"
    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def _start_recording(
    *,
    env: Any,
    output_dir: Path,
    video_name_prefix: str,
    config: RecordingConfig,
) -> None:
    env.unwrapped.start_video_recording(
        video_folder=str(output_dir),
        video_name_prefix=video_name_prefix,
        num_episodes=config.episodes_per_combination,
        max_parallel_episodes=config.episodes_per_combination,
        fps=config.fps,
        fps_mode=config.fps_mode,
        frame_stride=config.frame_stride,
        width=config.width,
        height=config.height,
        camera=config.camera,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    unconnected_prob: float = 0.0,
    disable_policy_connector_actions: bool = False,
    scenario_kwargs_overrides: Mapping[str, object] | None = None,
    default_output_root: Path = DEFAULT_OUTPUT_ROOT,
    morphology_description: str = "unseen, fully pre-connected morphologies",
) -> int:
    args = _parse_args(
        argv,
        default_output_root=default_output_root,
        morphology_description=morphology_description,
    )
    config = _validate_args(
        args,
        unconnected_prob=unconnected_prob,
        disable_policy_connector_actions=disable_policy_connector_actions,
        scenario_kwargs_overrides=scenario_kwargs_overrides,
    )
    targets = [TARGETS[key] for key in _selected_target_keys(args.target)]
    checkpoints_by_target = {
        target.key: resolve_target_checkpoints(args, target)
        for target in targets
    }
    combination_count = (
        sum(len(checkpoints_by_target[target.key]) for target in targets)
        * len(config.unit_counts)
    )
    print(
        f"Planned {combination_count} combinations with "
        f"{config.episodes_per_combination} recorded episodes each."
    )
    for target in targets:
        print(f"{target.display_name}: {len(checkpoints_by_target[target.key])} checkpoint(s)")
        for checkpoint in checkpoints_by_target[target.key]:
            fallback_label = " [best fallback]" if checkpoint.name == BEST_CHECKPOINT_NAME else ""
            print(f"  {checkpoint}{fallback_label}")
    if args.dry_run:
        return 0

    import torch
    import warp as wp

    if not torch.cuda.is_available():
        raise RuntimeError("MJW recording requires CUDA")
    if args.cuda_idx is not None:
        if args.cuda_idx < 0 or args.cuda_idx >= torch.cuda.device_count():
            raise ValueError(f"Invalid CUDA device index: {args.cuda_idx}")
        torch.cuda.set_device(args.cuda_idx)
        wp.set_device(f"cuda:{args.cuda_idx}")
    device = torch.device("cuda")
    output_root = args.output.resolve()
    manifest: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "results": [],
    }
    manifest_config = manifest["config"]
    assert isinstance(manifest_config, dict)
    manifest_config["unit_counts"] = list(config.unit_counts)
    results = manifest["results"]
    assert isinstance(results, list)

    combination_index = 0
    for target in targets:
        checkpoints = checkpoints_by_target[target.key]
        for unit_count in config.unit_counts:
            torch.compiler.reset()
            pool_seeds = make_pool_seeds(
                pool_seed_base=config.pool_seed_base,
                unit_count=unit_count,
                pool_size=config.pool_size,
            )
            for checkpoint in checkpoints:
                rollout_seed = make_combination_rollout_seed(
                    rollout_seed_base=config.rollout_seed,
                    combination_index=combination_index,
                )
                combination_index += 1
                output_dir = combination_output_dir(
                    output_root=output_root,
                    target_key=target.key,
                    unit_count=unit_count,
                    checkpoint_path=checkpoint,
                )
                _prepare_combination_output(output_dir, overwrite=args.overwrite)
                print(
                    f"Building {target.display_name}, {unit_count} agents for {checkpoint} "
                    f"with pool seeds {pool_seeds[0]}-{pool_seeds[-1]} "
                    f"and environment seed {rollout_seed}..."
                )
                env, policy = _build_env_and_policy(
                    target=target,
                    unit_count=unit_count,
                    pool_seeds=pool_seeds,
                    num_envs=config.episodes_per_combination,
                    episode_length=config.episode_length,
                    device=device,
                    unconnected_prob=config.unconnected_prob,
                    disable_policy_connector_actions=(
                        config.disable_policy_connector_actions
                    ),
                    scenario_kwargs_overrides=config.scenario_kwargs_overrides,
                )
                try:
                    _load_checkpoint(checkpoint_path=checkpoint, env=env, policy=policy)
                    video_prefix = f"{target.key}_{unit_count}_agents_{checkpoint.stem}"
                    summary = evaluate_policy(
                        env=env,
                        policy=policy,
                        episode_count=config.episodes_per_combination,
                        deterministic=config.deterministic,
                        rollout_seed=rollout_seed,
                        progress_description=f"{target.display_name}, {unit_count} agents",
                        show_progress=not args.no_progress,
                        on_reset=lambda: _start_recording(
                            env=env,
                            output_dir=output_dir,
                            video_name_prefix=video_prefix,
                            config=config,
                        ),
                    )
                    recording_status = env.unwrapped.get_video_recording_status()
                    if recording_status["episodes_completed"] != config.episodes_per_combination:
                        raise RuntimeError(
                            "Recording finished without completing all requested episodes: "
                            f"{recording_status}"
                        )
                finally:
                    env.close()

                video_paths = sorted(output_dir.glob("*.mp4"))
                if len(video_paths) != config.episodes_per_combination:
                    raise RuntimeError(
                        f"Expected {config.episodes_per_combination} videos in {output_dir}, "
                        f"found {len(video_paths)}"
                    )
                results.append(
                    {
                        "target": target.key,
                        "checkpoint": str(checkpoint),
                        "run_id": _checkpoint_run_id(checkpoint),
                        "unit_count": unit_count,
                        "pool_size": config.pool_size,
                        "pool_seed_range_inclusive": [pool_seeds[0], pool_seeds[-1]],
                        "rollout_seed": rollout_seed,
                        "summary": summary,
                        "videos": [str(path) for path in video_paths],
                    }
                )
                _write_manifest(output_root, manifest)
                print(f"  wrote {len(video_paths)} video(s) to {output_dir}")

    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    _write_manifest(output_root, manifest)
    print(f"Wrote recording manifest to {output_root / 'recording.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
