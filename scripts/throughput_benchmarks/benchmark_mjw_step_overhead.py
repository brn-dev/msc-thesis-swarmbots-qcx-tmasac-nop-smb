from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import mujoco_warp as mjw
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
    compile_reward_kernel: bool
    reward_kernel_compile_mode: str


@dataclass(frozen=True)
class ModeResult:
    mode: str
    num_envs: int
    measured_steps: int
    step_seconds: float
    step_envs_per_second: float
    done_count: int | None


@dataclass(slots=True)
class PhysicsBranchSnapshot:
    qpos: torch.Tensor
    qvel: torch.Tensor
    ctrl: torch.Tensor
    eq_active: torch.Tensor
    mocap_pos: torch.Tensor
    mocap_quat: torch.Tensor
    time_values: torch.Tensor
    qacc_warmstart: torch.Tensor
    act: torch.Tensor
    units_active_mask: torch.Tensor
    partner_unit: torch.Tensor
    partner_connector: torch.Tensor
    connection_twist_idx: torch.Tensor
    disconnect_potentials: torch.Tensor


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


def create_env(config: BenchmarkConfig) -> MJWSwarmBotsVectorEnv:
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_mjw_step_overhead requires CUDA.")
    scenario = default_mjw_wall(
        wall_distance=1.0,
        unit_start_locations=make_mjw_unit_start_locations(pool_seeds=tuple(range(52_000, 52_005))),
        quantize_connection_twist=8,
        compile_reward_kernel=config.compile_reward_kernel,
        reward_kernel_compile_mode=config.reward_kernel_compile_mode,
    )
    return MJWSwarmBotsVectorEnv(
        scenario=scenario,
        num_envs=config.num_envs,
        episode_length=config.episode_length,
        device=torch.device("cuda"),
        compile_tensor_operations=config.compile_reward_kernel,
        tensor_operations_compile_mode=config.reward_kernel_compile_mode,
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


def apply_actuators_only(env: MJWSwarmBotsVectorEnv, actuators: torch.Tensor) -> None:
    masked_actuators = actuators.masked_fill(~env.units_active_mask.unsqueeze(-1), 0.0)
    if env._ctrl.numel() > 0:
        env._ctrl[:, env._ctrl_flat_indices] = masked_actuators.reshape(env.num_envs, -1) * float(env.scenario.actuator_strength)


def measure_full_step(
    *,
    env: MJWSwarmBotsVectorEnv,
    mode: str,
    action_pool: list[dict[str, torch.Tensor]],
    warmup_steps: int,
    measured_steps: int,
    seed: int,
) -> ModeResult:
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
    return ModeResult(
        mode=mode,
        num_envs=env.num_envs,
        measured_steps=measured_steps,
        step_seconds=elapsed,
        step_envs_per_second=(measured_steps * env.num_envs) / elapsed,
        done_count=done_count,
    )


def capture_physics_branch_snapshot(env: MJWSwarmBotsVectorEnv) -> PhysicsBranchSnapshot:
    return PhysicsBranchSnapshot(
        qpos=env._qpos.clone(),
        qvel=env._qvel.clone(),
        ctrl=env._ctrl.clone(),
        eq_active=env._eq_active.clone(),
        mocap_pos=env._mocap_pos.clone(),
        mocap_quat=env._mocap_quat.clone(),
        time_values=env._time.clone(),
        qacc_warmstart=env._qacc_warmstart.clone(),
        act=env._act.clone(),
        units_active_mask=env.units_active_mask.clone(),
        partner_unit=env.partner_unit.clone(),
        partner_connector=env.partner_connector.clone(),
        connection_twist_idx=env.connection_twist_idx.clone(),
        disconnect_potentials=env.disconnect_potentials.clone(),
    )


def restore_physics_branch_snapshot(env: MJWSwarmBotsVectorEnv, snapshot: PhysicsBranchSnapshot) -> None:
    env._qpos.copy_(snapshot.qpos)
    env._qvel.copy_(snapshot.qvel)
    env._ctrl.copy_(snapshot.ctrl)
    env._eq_active.copy_(snapshot.eq_active)
    env._mocap_pos.copy_(snapshot.mocap_pos)
    env._mocap_quat.copy_(snapshot.mocap_quat)
    env._time.copy_(snapshot.time_values)
    env._qacc_warmstart.copy_(snapshot.qacc_warmstart)
    env._act.copy_(snapshot.act)
    env.units_active_mask.copy_(snapshot.units_active_mask)
    env.partner_unit.copy_(snapshot.partner_unit)
    env.partner_connector.copy_(snapshot.partner_connector)
    env.connection_twist_idx.copy_(snapshot.connection_twist_idx)
    env.disconnect_potentials.copy_(snapshot.disconnect_potentials)
    mjw.forward(env._model, env._data)


def time_physics_branch(
    *,
    env: MJWSwarmBotsVectorEnv,
    snapshot: PhysicsBranchSnapshot,
    action: dict[str, torch.Tensor],
    with_connectors: bool,
) -> float:
    restore_physics_branch_snapshot(env, snapshot)
    synchronize_env(env)
    start = time.perf_counter()
    if with_connectors:
        env._apply_actions(
            actuators=action["actuators"],
            connectors=action["connectors"],
        )
    else:
        apply_actuators_only(env, action["actuators"])
    env._run_physics()
    synchronize_env(env)
    return time.perf_counter() - start


def measure_connector_branch_steps(
    *,
    source_env: MJWSwarmBotsVectorEnv,
    branch_env: MJWSwarmBotsVectorEnv,
    action_pool: list[dict[str, torch.Tensor]],
    warmup_steps: int,
    measured_steps: int,
    seed: int,
) -> list[ModeResult]:
    source_env.reset(seed=seed)
    branch_env.reset(seed=seed + 1)

    for step_idx in range(warmup_steps):
        action = action_pool[step_idx % len(action_pool)]
        snapshot = capture_physics_branch_snapshot(source_env)
        time_physics_branch(env=branch_env, snapshot=snapshot, action=action, with_connectors=False)
        time_physics_branch(env=branch_env, snapshot=snapshot, action=action, with_connectors=True)
        source_env.step(action)

    actuators_only_elapsed = 0.0
    connectors_elapsed = 0.0
    for step_idx in range(measured_steps):
        action = action_pool[(warmup_steps + step_idx) % len(action_pool)]
        snapshot = capture_physics_branch_snapshot(source_env)
        actuators_only_elapsed += time_physics_branch(
            env=branch_env,
            snapshot=snapshot,
            action=action,
            with_connectors=False,
        )
        connectors_elapsed += time_physics_branch(
            env=branch_env,
            snapshot=snapshot,
            action=action,
            with_connectors=True,
        )
        source_env.step(action)

    total_env_steps = measured_steps * source_env.num_envs
    return [
        ModeResult(
            mode="physics_actuators_only_snapshot",
            num_envs=source_env.num_envs,
            measured_steps=measured_steps,
            step_seconds=actuators_only_elapsed,
            step_envs_per_second=total_env_steps / actuators_only_elapsed,
            done_count=None,
        ),
        ModeResult(
            mode="physics_with_connectors_snapshot",
            num_envs=source_env.num_envs,
            measured_steps=measured_steps,
            step_seconds=connectors_elapsed,
            step_envs_per_second=total_env_steps / connectors_elapsed,
            done_count=None,
        ),
    ]


def print_results(results: list[ModeResult]) -> None:
    headers = ["mode", "envs", "steps", "seconds", "env_steps_per_second", "done_count"]
    rows = [
        [
            result.mode,
            str(result.num_envs),
            str(result.measured_steps),
            f"{result.step_seconds:.3f}",
            f"{result.step_envs_per_second:,.0f}",
            "-" if result.done_count is None else str(result.done_count),
        ]
        for result in results
    ]
    widths = [max(len(header), *(len(row[col_idx]) for row in rows)) for col_idx, header in enumerate(headers)]
    print(" | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))

    summary_headers = [
        "envs",
        "compiled/eager",
        "compiled_gain_pct",
        "conn/act_only",
        "conn_gain_pct",
        "conn/eager",
        "conn/compiled",
    ]
    summary_rows: list[list[str]] = []
    num_envs_values = sorted({result.num_envs for result in results})
    for num_envs in num_envs_values:
        per_mode = {
            result.mode: result
            for result in results
            if result.num_envs == num_envs
        }
        eager_result = per_mode.get("full_step_eager")
        compiled_result = per_mode.get("full_step_compiled")
        actuators_only_result = per_mode.get("physics_actuators_only_snapshot")
        connectors_result = per_mode.get("physics_with_connectors_snapshot")
        if actuators_only_result is None or connectors_result is None:
            continue

        compiled_vs_eager = "-"
        compiled_gain_pct = "-"
        connectors_vs_actuators_only = (
            connectors_result.step_envs_per_second / actuators_only_result.step_envs_per_second
        )
        connector_gain_pct = f"{(connectors_vs_actuators_only - 1.0) * 100.0:.1f}%"
        connectors_vs_eager = "-"
        connectors_vs_compiled = "-"
        if eager_result is not None:
            connectors_vs_eager = f"{connectors_result.step_envs_per_second / eager_result.step_envs_per_second:.2f}x"
        if compiled_result is not None:
            connectors_vs_compiled = f"{connectors_result.step_envs_per_second / compiled_result.step_envs_per_second:.2f}x"
        if eager_result is not None and compiled_result is not None:
            compiled_ratio = compiled_result.step_envs_per_second / eager_result.step_envs_per_second
            compiled_vs_eager = f"{compiled_ratio:.2f}x"
            compiled_gain_pct = f"{(compiled_ratio - 1.0) * 100.0:.1f}%"

        summary_rows.append(
            [
                str(num_envs),
                compiled_vs_eager,
                compiled_gain_pct,
                f"{connectors_vs_actuators_only:.2f}x",
                connector_gain_pct,
                connectors_vs_eager,
                connectors_vs_compiled,
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
        description="Benchmark MJW full-step throughput and snapshot-based connector-step overhead.",
    )
    parser.add_argument(
        "--num-envs",
        nargs="+",
        type=int,
        default=[256, 512, 1024],
        help="Vector-env sizes to benchmark.",
    )
    parser.add_argument("--warmup-steps", type=int, default=64, help="Warmup steps per mode.")
    parser.add_argument("--steps", type=int, default=256, help="Measured steps per mode.")
    parser.add_argument("--action-pool-size", type=int, default=16, help="Number of prebuilt random actions to cycle.")
    parser.add_argument(
        "--episode-length",
        type=int,
        default=100_000,
        help="Episode length to avoid truncation noise in full-step mode.",
    )
    parser.add_argument(
        "--connector-prob",
        type=float,
        default=0.15,
        help="Connector activation probability in full-step mode action pool.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Base RNG seed.")
    parser.add_argument(
        "--compile-reward-kernel",
        action="store_true",
        help="Benchmark compiled reward and MJW environment tensor operations instead of the eager full step.",
    )
    parser.add_argument(
        "--benchmark-both-full-step-modes",
        action="store_true",
        help="Benchmark both eager and compiled full-step modes in the same run.",
    )
    parser.add_argument(
        "--reward-kernel-compile-mode",
        type=str,
        default="default",
        help=(
            "torch.compile mode for reward and MJW environment tensor operations in "
            "compiled full-step mode."
        ),
    )
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT, help="Optional path for machine-readable output.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if any(num_envs <= 0 for num_envs in args.num_envs):
        raise ValueError("--num-envs values must be positive.")
    if args.warmup_steps < 0 or args.steps <= 0 or args.action_pool_size <= 0:
        raise ValueError("warmup/steps/action-pool-size must be positive, with warmup >= 0.")
    if not 0.0 <= args.connector_prob <= 1.0:
        raise ValueError("--connector-prob must be in [0, 1].")
    needs_compiled_full_step = args.compile_reward_kernel or args.benchmark_both_full_step_modes
    if needs_compiled_full_step and not hasattr(torch, "compile"):
        raise ValueError("Compiled full-step benchmarking requires torch.compile support.")
    if needs_compiled_full_step and importlib.util.find_spec("triton") is None:
        raise ValueError(
            "Compiled full-step benchmarking requires a working Triton install. "
            "On native Windows that usually means installing triton-windows, not triton."
        )


def benchmark_full_step_variant(
    *,
    config: BenchmarkConfig,
    mode: str,
    compile_reward_kernel: bool,
) -> ModeResult:
    env = create_env(replace(config, compile_reward_kernel=compile_reward_kernel))
    try:
        action_pool = build_action_pool(env, config)
        return measure_full_step(
            env=env,
            mode=mode,
            action_pool=action_pool,
            warmup_steps=config.warmup_steps,
            measured_steps=config.measured_steps,
            seed=config.seed,
        )
    finally:
        env.close()


def benchmark_connector_snapshot_branches(config: BenchmarkConfig) -> list[ModeResult]:
    source_env = create_env(replace(config, compile_reward_kernel=False))
    branch_env = create_env(replace(config, compile_reward_kernel=False))
    try:
        action_pool = build_action_pool(source_env, config)
        return measure_connector_branch_steps(
            source_env=source_env,
            branch_env=branch_env,
            action_pool=action_pool,
            warmup_steps=config.warmup_steps,
            measured_steps=config.measured_steps,
            seed=config.seed + 77_000,
        )
    finally:
        branch_env.close()
        source_env.close()


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
    results: list[ModeResult] = []
    for num_envs in args.num_envs:
        config = BenchmarkConfig(
            num_envs=num_envs,
            warmup_steps=args.warmup_steps,
            measured_steps=args.steps,
            action_pool_size=args.action_pool_size,
            episode_length=args.episode_length,
            connector_prob=args.connector_prob,
            seed=args.seed,
            compile_reward_kernel=bool(args.compile_reward_kernel),
            reward_kernel_compile_mode=str(args.reward_kernel_compile_mode),
        )
        full_step_variants: list[tuple[str, bool]]
        if args.benchmark_both_full_step_modes:
            full_step_variants = [
                ("full_step_eager", False),
                ("full_step_compiled", True),
            ]
        elif config.compile_reward_kernel:
            full_step_variants = [("full_step_compiled", True)]
        else:
            full_step_variants = [("full_step_eager", False)]

        logger.info(
            f"Benchmarking MJW overhead at num_envs={config.num_envs}, "
            f"warmup_steps={config.warmup_steps}, measured_steps={config.measured_steps}, "
            f"full_step_variants={[mode for mode, _ in full_step_variants]}, "
            f"reward_kernel_compile_mode={config.reward_kernel_compile_mode!r}"
        )
        for mode, compile_reward_kernel in full_step_variants:
            results.append(
                benchmark_full_step_variant(
                    config=config,
                    mode=mode,
                    compile_reward_kernel=compile_reward_kernel,
                )
            )
        results.extend(benchmark_connector_snapshot_branches(config))

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


