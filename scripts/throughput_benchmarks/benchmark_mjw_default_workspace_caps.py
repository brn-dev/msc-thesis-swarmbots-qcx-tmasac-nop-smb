from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

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


VariantName = Literal["current_defaults", "previous_defaults"]


@dataclass(frozen=True)
class BenchmarkConfig:
    num_envs: int
    warmup_steps: int
    measured_steps: int
    action_pool_size: int
    episode_length: int
    connector_prob: float
    seed: int


@dataclass(frozen=True)
class VariantResult:
    variant: VariantName
    num_envs: int
    measured_steps: int
    step_seconds: float
    step_envs_per_second: float
    done_count: int
    nconmax: int
    njmax: int


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


def make_scenario() -> Any:
    return default_mjw_wall(
        wall_distance=1.0,
        unit_start_locations=make_mjw_unit_start_locations(pool_seeds=tuple(range(52_000, 52_005))),
        quantize_connection_twist=8,
    )


def _compute_previous_default_caps(scenario: Any) -> tuple[int, int]:
    model = scenario.build_model()
    n_total_connectors = scenario.swarm.num_units * scenario.swarm.config.limbs_per_unit
    previous_nconmax = max(128, n_total_connectors * 8)
    previous_njmax = max(512, int(model.nv * 8 + previous_nconmax * 6))
    return previous_nconmax, previous_njmax


def create_env(config: BenchmarkConfig, *, variant: VariantName) -> MJWSwarmBotsVectorEnv:
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_mjw_default_workspace_caps requires CUDA.")

    scenario = make_scenario()
    if variant == "current_defaults":
        return MJWSwarmBotsVectorEnv(
            scenario=scenario,
            num_envs=config.num_envs,
            episode_length=config.episode_length,
            device=torch.device("cuda"),
        )

    previous_nconmax, previous_njmax = _compute_previous_default_caps(scenario)
    return MJWSwarmBotsVectorEnv(
        scenario=scenario,
        num_envs=config.num_envs,
        episode_length=config.episode_length,
        device=torch.device("cuda"),
        nconmax=previous_nconmax,
        njmax=previous_njmax,
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


def measure_steps(
    *,
    env: MJWSwarmBotsVectorEnv,
    action_pool: list[dict[str, torch.Tensor]],
    warmup_steps: int,
    measured_steps: int,
    seed: int,
) -> tuple[float, float, int]:
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
    return elapsed, (measured_steps * env.num_envs) / elapsed, done_count


def benchmark_variant(*, config: BenchmarkConfig, variant: VariantName) -> VariantResult:
    env = create_env(config, variant=variant)
    try:
        action_pool = build_action_pool(env, config)
        step_seconds, step_envs_per_second, done_count = measure_steps(
            env=env,
            action_pool=action_pool,
            warmup_steps=config.warmup_steps,
            measured_steps=config.measured_steps,
            seed=config.seed,
        )
        caps = env.get_settings()["physics_workspace_caps"]
        return VariantResult(
            variant=variant,
            num_envs=config.num_envs,
            measured_steps=config.measured_steps,
            step_seconds=step_seconds,
            step_envs_per_second=step_envs_per_second,
            done_count=done_count,
            nconmax=int(caps["nconmax"]),
            njmax=int(caps["njmax"]),
        )
    finally:
        env.close()


def print_results(results: list[VariantResult]) -> None:
    headers = ["variant", "envs", "steps", "nconmax", "njmax", "seconds", "env_steps_per_second", "done_count"]
    rows = [
        [
            result.variant,
            str(result.num_envs),
            str(result.measured_steps),
            str(result.nconmax),
            str(result.njmax),
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

    summary_headers = ["envs", "current/previous", "current_gain_pct"]
    summary_rows: list[list[str]] = []
    for num_envs in sorted({result.num_envs for result in results}):
        per_variant = {result.variant: result for result in results if result.num_envs == num_envs}
        current = per_variant.get("current_defaults")
        previous = per_variant.get("previous_defaults")
        if current is None or previous is None:
            continue
        speed_ratio = current.step_envs_per_second / previous.step_envs_per_second
        gain_pct = ((current.step_envs_per_second / previous.step_envs_per_second) - 1.0) * 100.0
        summary_rows.append([str(num_envs), f"{speed_ratio:.2f}x", f"{gain_pct:.1f}%"])

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
        description="Benchmark MJW current workspace-cap defaults versus the previous oversized defaults.",
    )
    parser.add_argument(
        "--num-envs",
        nargs="+",
        type=int,
        default=[256, 512, 1024],
        help="Vector-env sizes to benchmark.",
    )
    parser.add_argument("--warmup-steps", type=int, default=64, help="Warmup steps per variant.")
    parser.add_argument("--steps", type=int, default=256, help="Measured steps per variant.")
    parser.add_argument("--action-pool-size", type=int, default=16, help="Number of prebuilt random actions to cycle.")
    parser.add_argument(
        "--episode-length",
        type=int,
        default=100_000,
        help="Episode length to avoid truncation noise during throughput measurement.",
    )
    parser.add_argument(
        "--connector-prob",
        type=float,
        default=0.15,
        help="Connector activation probability in the random action pool.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Base RNG seed.")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT, help="Optional path for machine-readable output.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if any(num_envs <= 0 for num_envs in args.num_envs):
        raise ValueError("--num-envs values must be positive.")
    if args.warmup_steps < 0 or args.steps <= 0 or args.action_pool_size <= 0:
        raise ValueError("warmup/steps/action-pool-size must be positive, with warmup >= 0.")
    if not 0.0 <= args.connector_prob <= 1.0:
        raise ValueError("--connector-prob must be in [0, 1].")


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
    results: list[VariantResult] = []
    for num_envs in args.num_envs:
        config = BenchmarkConfig(
            num_envs=num_envs,
            warmup_steps=args.warmup_steps,
            measured_steps=args.steps,
            action_pool_size=args.action_pool_size,
            episode_length=args.episode_length,
            connector_prob=args.connector_prob,
            seed=args.seed,
        )
        for variant in ("current_defaults", "previous_defaults"):
            logger.info(
                f"Benchmarking MJW workspace caps variant={variant}, "
                f"num_envs={config.num_envs}, warmup_steps={config.warmup_steps}, measured_steps={config.measured_steps}"
            )
            results.append(benchmark_variant(config=config, variant=variant))

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
        logger.info(f"Wrote benchmark JSON to {args.json_out}")


if __name__ == "__main__":
    main()


