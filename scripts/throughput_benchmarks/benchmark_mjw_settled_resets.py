from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.throughput_benchmarks.throughput_benchmark_paths import default_throughput_benchmark_json_out

DEFAULT_JSON_OUT = default_throughput_benchmark_json_out(__file__)

from swarmbots.mjw_env import MJWSwarmBotsVectorEnv
from swarmbots.mjw_env.scenarios.mjw_scenario_presets import default_wall as default_mjw_wall
from swarmbots.mjw_env.swarm.mjw_homogeneous_swarm import MJWPreConnectedUnitLocationsConfig


@dataclass(frozen=True)
class BenchmarkConfig:
    num_envs: int
    warmup_steps: int
    measured_steps: int
    action_pool_size: int
    episode_length: int
    connector_prob: float
    seed: int
    settled_reset_time: float
    settled_reset_timestep_scale: float


@dataclass(frozen=True)
class BenchmarkResult:
    mode: str
    num_envs: int
    episode_length: int
    measured_steps: int
    reset_settle_time: float
    reset_settle_timestep_scale: float
    step_seconds: float
    step_envs_per_second: float
    done_count: int


def make_mjw_unit_start_locations(pool_seeds: tuple[int, ...]) -> MJWPreConnectedUnitLocationsConfig:
    return MJWPreConnectedUnitLocationsConfig(
        num_units=5,
        num_unit_probs={
            4: 1.0,
            5: 1.0,
        },
        max_radius=1.5,
        unconnected_prob=0.02,
        z_pos=0.5,
        pool_seeds=pool_seeds,
    )


def create_env(config: BenchmarkConfig, *, use_settled_resets: bool) -> MJWSwarmBotsVectorEnv:
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_mjw_settled_resets requires CUDA.")

    first_episode_lengths = [int((i + 1) * config.episode_length / config.num_envs) for i in range(config.num_envs)]
    scenario = default_mjw_wall(
        wall_distance=1.0,
        unit_start_locations=make_mjw_unit_start_locations(pool_seeds=tuple(range(62_000, 62_005))),
        quantize_connection_twist=8,
        reset_settle_time=config.settled_reset_time if use_settled_resets else 0.0,
        reset_settle_timestep_scale=config.settled_reset_timestep_scale,
    )
    return MJWSwarmBotsVectorEnv(
        scenario=scenario,
        num_envs=config.num_envs,
        episode_length=config.episode_length,
        first_episode_lengths=first_episode_lengths,
        settle_initial_reset=False,
        device=torch.device("cuda"),
    )


def synchronize_env(env: MJWSwarmBotsVectorEnv) -> None:
    if env.device.type == "cuda":
        torch.cuda.synchronize(env.device)


def build_action_pool(env: MJWSwarmBotsVectorEnv, config: BenchmarkConfig) -> list[dict[str, torch.Tensor]]:
    actuators_shape = tuple(env.single_action_space["actuators"].shape)
    connectors_shape = tuple(env.single_action_space["connectors"].shape)
    generator = torch.Generator(device=env.device)
    generator.manual_seed(config.seed)
    return [
        {
            "actuators": (
                torch.rand((config.num_envs, *actuators_shape), device=env.device, generator=generator) * 2.0 - 1.0
            ),
            "connectors": (
                torch.rand((config.num_envs, *connectors_shape), device=env.device, generator=generator)
                < config.connector_prob
            ),
        }
        for _ in range(config.action_pool_size)
    ]


def measure_mode(
    *,
    env: MJWSwarmBotsVectorEnv,
    action_pool: list[dict[str, torch.Tensor]],
    warmup_steps: int,
    measured_steps: int,
    seed: int,
    mode: str,
) -> BenchmarkResult:
    env.reset(seed=seed)
    for step_idx in range(warmup_steps):
        env.step(action_pool[step_idx % len(action_pool)])

    synchronize_env(env)
    start = time.perf_counter()
    done_count = 0
    for step_idx in range(measured_steps):
        _, _, terminations, truncations, _ = env.step(action_pool[step_idx % len(action_pool)])
        done_count += int((terminations.sum() + truncations.sum()).item())
    synchronize_env(env)
    elapsed = time.perf_counter() - start
    return BenchmarkResult(
        mode=mode,
        num_envs=env.num_envs,
        episode_length=env.episode_length,
        measured_steps=measured_steps,
        reset_settle_time=float(env.scenario.reset_settle_time),
        reset_settle_timestep_scale=float(env.scenario.reset_settle_timestep_scale),
        step_seconds=elapsed,
        step_envs_per_second=(measured_steps * env.num_envs) / elapsed,
        done_count=done_count,
    )


def print_results(results: list[BenchmarkResult]) -> None:
    headers = [
        "mode",
        "envs",
        "ep_len",
        "steps",
        "settle_time",
        "seconds",
        "env_steps_per_second",
        "done_count",
    ]
    rows = [
        [
            result.mode,
            str(result.num_envs),
            str(result.episode_length),
            str(result.measured_steps),
            f"{result.reset_settle_time:.3f}",
            f"{result.step_seconds:.3f}",
            f"{result.step_envs_per_second:,.0f}",
            str(result.done_count),
        ]
        for result in results
    ]
    widths = [max(len(header), *(len(row[col_idx]) for row in rows)) for col_idx, header in enumerate(headers)]
    print(" | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))

    summary_headers = ["envs", "direct/settled", "settled_slowdown_pct", "episode_length"]
    summary_rows: list[list[str]] = []
    for num_envs in sorted({result.num_envs for result in results}):
        per_mode = {result.mode: result for result in results if result.num_envs == num_envs}
        if "direct_reset" not in per_mode or "settled_reset" not in per_mode:
            continue
        direct_eps = per_mode["direct_reset"].step_envs_per_second
        settled_eps = per_mode["settled_reset"].step_envs_per_second
        slowdown_pct = (1.0 - (settled_eps / direct_eps)) * 100.0
        summary_rows.append(
            [
                str(num_envs),
                f"{direct_eps / settled_eps:.2f}x",
                f"{slowdown_pct:.1f}%",
                str(per_mode["direct_reset"].episode_length),
            ]
        )

    if not summary_rows:
        return

    summary_widths = [
        max(len(header), *(len(row[col_idx]) for row in summary_rows))
        for col_idx, header in enumerate(summary_headers)
    ]
    print()
    print(" | ".join(header.ljust(summary_widths[idx]) for idx, header in enumerate(summary_headers)))
    print("-+-".join("-" * width for width in summary_widths))
    for row in summary_rows:
        print(" | ".join(value.ljust(summary_widths[idx]) for idx, value in enumerate(row)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark MJW slowdown from non-initial settled env resets by comparing "
            "reset_settle_time=0 against reset_settle_time>0 at short episode lengths."
        ),
    )
    parser.add_argument(
        "--num-envs",
        nargs="+",
        type=int,
        default=[512],
        help="Vector-env sizes to benchmark.",
    )
    parser.add_argument("--warmup-steps", type=int, default=128, help="Warmup steps per mode.")
    parser.add_argument("--steps", type=int, default=2048, help="Measured steps per mode.")
    parser.add_argument("--action-pool-size", type=int, default=16, help="Number of prebuilt random actions to cycle.")
    parser.add_argument(
        "--episode-length",
        type=int,
        default=512,
        help="Episode length used to force non-initial same-step resets during measurement.",
    )
    parser.add_argument(
        "--connector-prob",
        type=float,
        default=0.15,
        help="Connector activation probability in the action pool.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Base RNG seed.")
    parser.add_argument(
        "--settled-reset-time",
        type=float,
        default=1.0,
        help="Scenario reset_settle_time for the settled-reset mode.",
    )
    parser.add_argument(
        "--settled-reset-timestep-scale",
        type=float,
        default=3.0,
        help="Scenario reset_settle_timestep_scale for the settled-reset mode.",
    )
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT, help="Optional path for machine-readable output.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if any(num_envs <= 0 for num_envs in args.num_envs):
        raise ValueError("--num-envs values must be positive.")
    if args.warmup_steps < 0 or args.steps <= 0 or args.action_pool_size <= 0:
        raise ValueError("warmup/steps/action-pool-size must be positive, with warmup >= 0.")
    if args.episode_length <= 0:
        raise ValueError("--episode-length must be positive.")
    if not 0.0 <= args.connector_prob <= 1.0:
        raise ValueError("--connector-prob must be in [0, 1].")
    if args.settled_reset_time < 0.0:
        raise ValueError("--settled-reset-time must be >= 0.")
    if args.settled_reset_timestep_scale <= 0.0:
        raise ValueError("--settled-reset-timestep-scale must be > 0.")


def main() -> None:
    from swarmbots.learn.torch_logging import enable_torch_compile_logging

    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <5}</level> | <level>{message}</level>",
    )
    enable_torch_compile_logging()

    args = parse_args()
    validate_args(args)

    results: list[BenchmarkResult] = []
    for num_envs in args.num_envs:
        config = BenchmarkConfig(
            num_envs=num_envs,
            warmup_steps=args.warmup_steps,
            measured_steps=args.steps,
            action_pool_size=args.action_pool_size,
            episode_length=args.episode_length,
            connector_prob=args.connector_prob,
            seed=args.seed,
            settled_reset_time=args.settled_reset_time,
            settled_reset_timestep_scale=args.settled_reset_timestep_scale,
        )
        logger.info(
            f"Benchmarking settled-reset slowdown at num_envs={config.num_envs}, "
            f"episode_length={config.episode_length}, measured_steps={config.measured_steps}"
        )

        for mode_name, use_settled_resets in (
            ("direct_reset", False),
            ("settled_reset", True),
        ):
            env = create_env(config, use_settled_resets=use_settled_resets)
            try:
                action_pool = build_action_pool(env, config)
                results.append(
                    measure_mode(
                        env=env,
                        action_pool=action_pool,
                        warmup_steps=config.warmup_steps,
                        measured_steps=config.measured_steps,
                        seed=config.seed + (17_000 if use_settled_resets else 0),
                        mode=mode_name,
                    )
                )
            finally:
                env.close()

    print_results(results)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "results": [asdict(result) for result in results],
        }
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON results to {args.json_out}")


if __name__ == "__main__":
    main()


