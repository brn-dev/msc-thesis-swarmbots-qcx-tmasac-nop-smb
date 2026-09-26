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


DEFAULT_PAIR_SPECS = ("0.002:15", "0.003:10", "0.004:8", "0.005:6")


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
class CadenceSpec:
    timestep: float
    action_repeat: int

    @property
    def env_step_dt(self) -> float:
        return self.timestep * self.action_repeat

    @property
    def label(self) -> str:
        return f"{self.timestep:.3f}x{self.action_repeat}"


@dataclass(frozen=True)
class CadenceResult:
    num_envs: int
    timestep: float
    action_repeat: int
    env_step_dt: float
    measured_steps: int
    step_seconds: float
    step_envs_per_second: float
    terminations_count: int
    truncations_count: int
    nconmax: int
    njmax: int


def parse_cadence_spec(text: str) -> CadenceSpec:
    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError(f"Expected cadence spec '<timestep>:<action_repeat>', got {text!r}")
    timestep = float(parts[0])
    action_repeat = int(parts[1])
    if timestep <= 0.0:
        raise ValueError(f"Expected timestep > 0, got {timestep}")
    if action_repeat <= 0:
        raise ValueError(f"Expected action_repeat > 0, got {action_repeat}")
    return CadenceSpec(timestep=timestep, action_repeat=action_repeat)


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


def create_env(config: BenchmarkConfig, *, cadence: CadenceSpec) -> MJWSwarmBotsVectorEnv:
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_mjw_physics_cadence_sweep requires CUDA.")
    scenario = default_mjw_wall(
        wall_distance=1.0,
        unit_start_locations=make_mjw_unit_start_locations(pool_seeds=tuple(range(52_000, 52_005))),
        quantize_connection_twist=8,
        timestep=cadence.timestep,
        action_repeat=cadence.action_repeat,
    )
    return MJWSwarmBotsVectorEnv(
        scenario=scenario,
        num_envs=config.num_envs,
        episode_length=config.episode_length,
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


def measure_steps(
    *,
    env: MJWSwarmBotsVectorEnv,
    action_pool: list[dict[str, torch.Tensor]],
    warmup_steps: int,
    measured_steps: int,
    seed: int,
) -> tuple[float, float, int, int]:
    env.reset(seed=seed)
    for step_idx in range(warmup_steps):
        env.step(action_pool[step_idx % len(action_pool)])

    synchronize_env(env)
    start = time.perf_counter()
    terminations_count = 0
    truncations_count = 0
    for step_idx in range(measured_steps):
        _, _, terminations, truncations, _ = env.step(action_pool[step_idx % len(action_pool)])
        terminations_count += int(terminations.sum().item())
        truncations_count += int(truncations.sum().item())
    synchronize_env(env)
    elapsed = time.perf_counter() - start
    return elapsed, (measured_steps * env.num_envs) / elapsed, terminations_count, truncations_count


def benchmark_cadence(*, config: BenchmarkConfig, cadence: CadenceSpec) -> CadenceResult:
    env = create_env(config, cadence=cadence)
    try:
        action_pool = build_action_pool(env, config)
        step_seconds, step_envs_per_second, terminations_count, truncations_count = measure_steps(
            env=env,
            action_pool=action_pool,
            warmup_steps=config.warmup_steps,
            measured_steps=config.measured_steps,
            seed=config.seed,
        )
        caps = env.get_settings()["physics_workspace_caps"]
        return CadenceResult(
            num_envs=config.num_envs,
            timestep=cadence.timestep,
            action_repeat=cadence.action_repeat,
            env_step_dt=cadence.env_step_dt,
            measured_steps=config.measured_steps,
            step_seconds=step_seconds,
            step_envs_per_second=step_envs_per_second,
            terminations_count=terminations_count,
            truncations_count=truncations_count,
            nconmax=int(caps["nconmax"]),
            njmax=int(caps["njmax"]),
        )
    finally:
        env.close()


def print_results(results: list[CadenceResult], *, baseline: CadenceSpec) -> None:
    headers = [
        "envs",
        "cadence",
        "env_step_dt",
        "nconmax",
        "njmax",
        "seconds",
        "env_steps_per_second",
        "terminations",
        "truncations",
    ]
    rows = [
        [
            str(result.num_envs),
            f"{result.timestep:.3f}x{result.action_repeat}",
            f"{result.env_step_dt:.3f}",
            str(result.nconmax),
            str(result.njmax),
            f"{result.step_seconds:.3f}",
            f"{result.step_envs_per_second:,.0f}",
            str(result.terminations_count),
            str(result.truncations_count),
        ]
        for result in results
    ]
    widths = [max(len(header), *(len(row[col_idx]) for row in rows)) for col_idx, header in enumerate(headers)]
    print(" | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))

    baseline_label = baseline.label
    summary_headers = ["envs", "cadence", "vs_baseline", "gain_pct"]
    summary_rows: list[list[str]] = []
    for num_envs in sorted({result.num_envs for result in results}):
        per_label = {f"{result.timestep:.3f}x{result.action_repeat}": result for result in results if result.num_envs == num_envs}
        baseline_result = per_label.get(baseline_label)
        if baseline_result is None:
            continue
        for result in (r for r in results if r.num_envs == num_envs and f"{r.timestep:.3f}x{r.action_repeat}" != baseline_label):
            speed_ratio = result.step_envs_per_second / baseline_result.step_envs_per_second
            gain_pct = (speed_ratio - 1.0) * 100.0
            summary_rows.append([str(num_envs), f"{result.timestep:.3f}x{result.action_repeat}", f"{speed_ratio:.2f}x", f"{gain_pct:.1f}%"])

    if not summary_rows:
        return

    summary_widths = [
        max(len(header), *(len(row[col_idx]) for row in summary_rows))
        for col_idx, header in enumerate(summary_headers)
    ]
    print()
    print("Baseline:", baseline_label)
    print(" | ".join(header.ljust(summary_widths[idx]) for idx, header in enumerate(summary_headers)))
    print("-+-".join("-" * width for width in summary_widths))
    for row in summary_rows:
        print(" | ".join(value.ljust(summary_widths[idx]) for idx, value in enumerate(row)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark MJW throughput over timestep/action_repeat sweep.",
    )
    parser.add_argument(
        "--num-envs",
        nargs="+",
        type=int,
        default=[512],
        help="Vector-env sizes to benchmark.",
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=list(DEFAULT_PAIR_SPECS),
        help="Cadence pairs in '<timestep>:<action_repeat>' format.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="0.002:15",
        help="Baseline cadence used in the summary table.",
    )
    parser.add_argument("--warmup-steps", type=int, default=32, help="Warmup steps per cadence.")
    parser.add_argument("--steps", type=int, default=128, help="Measured steps per cadence.")
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


def validate_args(args: argparse.Namespace) -> tuple[list[CadenceSpec], CadenceSpec]:
    if any(num_envs <= 0 for num_envs in args.num_envs):
        raise ValueError("--num-envs values must be positive.")
    if args.warmup_steps < 0 or args.steps <= 0 or args.action_pool_size <= 0:
        raise ValueError("warmup/steps/action-pool-size must be positive, with warmup >= 0.")
    if not 0.0 <= args.connector_prob <= 1.0:
        raise ValueError("--connector-prob must be in [0, 1].")
    cadence_specs = [parse_cadence_spec(text) for text in args.pairs]
    baseline = parse_cadence_spec(args.baseline)
    baseline_label = baseline.label
    if baseline_label not in {spec.label for spec in cadence_specs}:
        raise ValueError(f"--baseline {args.baseline!r} is not present in --pairs")
    return cadence_specs, baseline


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
    cadence_specs, baseline = validate_args(args)
    results: list[CadenceResult] = []
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
        for cadence in cadence_specs:
            logger.info(
                f"Benchmarking MJW cadence={cadence.label}, env_step_dt={cadence.env_step_dt:.3f}, "
                f"num_envs={config.num_envs}, warmup_steps={config.warmup_steps}, measured_steps={config.measured_steps}"
            )
            results.append(benchmark_cadence(config=config, cadence=cadence))

    print_results(results, baseline=baseline)

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


