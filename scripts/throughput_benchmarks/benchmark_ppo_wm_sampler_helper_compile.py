from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.throughput_benchmarks.throughput_benchmark_paths import default_throughput_benchmark_json_out

DEFAULT_JSON_OUT = default_throughput_benchmark_json_out(__file__)

from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPOEpisodeSegment
from swarmbots.learn.algos.world_modeling.wm_sampler_helper import (
    WMEpisodeWindows,
    build_wm_episode_windows_batch,
    ensure_wm_window_helper_compile_available,
    reset_wm_window_helper_compile_cache,
)
from swarmbots.learn.torch_logging import enable_torch_compile_logging


# Defaults aligned with scripts/run_mat_qcs_nop_wall_mjw.py.
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_NUM_LENGTH_BUCKETS = 4
DEFAULT_TOTAL_EPISODES = 1024
DEFAULT_EPISODES_PER_LENGTH = DEFAULT_TOTAL_EPISODES // DEFAULT_NUM_LENGTH_BUCKETS
DEFAULT_NUM_NEXT_STEPS = 3
DEFAULT_N_AGENTS = 5
DEFAULT_LOCAL_OBS_DIM = 154
DEFAULT_GLOBAL_OBS_DIM = 0
DEFAULT_HIDDEN_LOCAL_VARS_DIM = 4
DEFAULT_HIDDEN_GLOBAL_VARS_DIM = 3
DEFAULT_ACTION_DIM = 12


@dataclass(frozen=True)
class BenchmarkConfig:
    device: str
    episodes_per_length: int
    num_next_steps: int
    n_agents: int
    local_obs_dim: int
    global_obs_dim: int
    hidden_local_vars_dim: int
    hidden_global_vars_dim: int
    action_dim: int
    warmup_iters: int
    measured_iters: int
    compile_mode: str
    seed: int


@dataclass(frozen=True)
class EpisodeBatch:
    episodes: list[PPOEpisodeSegment]
    num_steps: int
    batch_size: int
    with_agent_mask: bool


@dataclass(frozen=True)
class FixedLengthResult:
    length: int
    with_agent_mask: bool
    batch_size: int
    eager_seconds: float
    eager_calls_per_second: float
    compiled_first_call_seconds: float
    compiled_steady_seconds: float
    compiled_steady_calls_per_second: float
    steady_speedup: float


@dataclass(frozen=True)
class MixedLengthsResult:
    name: str
    lengths: list[int]
    with_agent_mask: bool
    total_calls_per_iteration: int
    total_steps_per_iteration: int
    eager_seconds: float
    eager_iterations_per_second: float
    compiled_first_iteration_seconds: float
    compiled_steady_seconds: float
    compiled_steady_iterations_per_second: float
    steady_speedup: float
    first_iteration_speedup: float


def maybe_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def require_compile_support(config: BenchmarkConfig) -> None:
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but torch.cuda.is_available() is False.")
    ensure_wm_window_helper_compile_available(compile_mode=config.compile_mode)


def make_episode_batch(
        config: BenchmarkConfig,
        *,
        num_steps: int,
        with_agent_mask: bool,
        seed_offset: int,
) -> EpisodeBatch:
    device = torch.device(config.device)
    generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    generator.manual_seed(config.seed + seed_offset)

    episodes: list[PPOEpisodeSegment] = []
    for _ in range(config.episodes_per_length):
        local_obs = torch.randn(
            (num_steps, config.n_agents, config.local_obs_dim),
            generator=generator,
            device=device,
        )
        global_obs = torch.randn(
            (num_steps, config.global_obs_dim),
            generator=generator,
            device=device,
        )
        hidden_local_vars = torch.randn(
            (num_steps, config.n_agents, config.hidden_local_vars_dim),
            generator=generator,
            device=device,
        )
        hidden_global_vars = torch.randn(
            (num_steps, config.hidden_global_vars_dim),
            generator=generator,
            device=device,
        )
        actions = torch.randn(
            (num_steps, config.n_agents, config.action_dim),
            generator=generator,
            device=device,
        )

        agent_mask = None
        final_agent_mask = None
        if with_agent_mask:
            agent_mask = torch.rand(
                (num_steps, config.n_agents),
                generator=generator,
                device=device,
            ) > 0.2
            agent_mask[:, 0] = True
            final_agent_mask = torch.rand(
                (config.n_agents,),
                generator=generator,
                device=device,
            ) > 0.2
            final_agent_mask[0] = True

        episodes.append(
            PPOEpisodeSegment(
                local_obs=local_obs,
                global_obs=global_obs,
                hidden_local_vars=hidden_local_vars,
                hidden_global_vars=hidden_global_vars,
                agent_mask=agent_mask,
                actions=actions,
                rewards=torch.randn((num_steps,), generator=generator, device=device),
                log_probs=torch.randn((num_steps, config.n_agents), generator=generator, device=device),
                values=torch.randn((num_steps,), generator=generator, device=device),
                final_local_obs=torch.randn(
                    (config.n_agents, config.local_obs_dim),
                    generator=generator,
                    device=device,
                ),
                final_global_obs=torch.randn(
                    (config.global_obs_dim,),
                    generator=generator,
                    device=device,
                ),
                final_hidden_local_vars=torch.randn(
                    (config.n_agents, config.hidden_local_vars_dim),
                    generator=generator,
                    device=device,
                ),
                final_hidden_global_vars=torch.randn(
                    (config.hidden_global_vars_dim,),
                    generator=generator,
                    device=device,
                ),
                final_agent_mask=final_agent_mask,
                final_value=torch.randn((), generator=generator, device=device),
                initial_previous_actions=torch.randn(
                    (config.n_agents, config.action_dim),
                    generator=generator,
                    device=device,
                ),
                returns=torch.randn((num_steps,), generator=generator, device=device),
                advantages=torch.randn((num_steps,), generator=generator, device=device),
            )
        )

    return EpisodeBatch(
        episodes=episodes,
        num_steps=num_steps,
        batch_size=len(episodes),
        with_agent_mask=with_agent_mask,
    )


def build_windows(
        batch: EpisodeBatch,
        *,
        num_next_steps: int,
        compile_modules: bool,
        compile_mode: str,
) -> WMEpisodeWindows:
    return build_wm_episode_windows_batch(
        batch.episodes,
        num_next_steps=num_next_steps,
        compile_modules=compile_modules,
        compile_mode=compile_mode,
    )


def assert_windows_equal(eager_windows: WMEpisodeWindows, compiled_windows: WMEpisodeWindows) -> None:
    for field in fields(WMEpisodeWindows):
        eager_value = getattr(eager_windows, field.name)
        compiled_value = getattr(compiled_windows, field.name)
        if eager_value is None:
            if compiled_value is not None:
                raise AssertionError(f"{field.name} expected None")
            continue
        if not torch.equal(eager_value, compiled_value):
            raise AssertionError(f"{field.name} mismatch")


def benchmark_batch(
        batch: EpisodeBatch,
        *,
        config: BenchmarkConfig,
        compile_modules: bool,
        measured_iters: int,
        warmup_iters: int,
) -> float:
    device = batch.episodes[0].local_obs.device
    for _ in range(warmup_iters):
        build_windows(
            batch,
            num_next_steps=config.num_next_steps,
            compile_modules=compile_modules,
            compile_mode=config.compile_mode,
        )
    maybe_synchronize(device)

    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        start = time.perf_counter()
        for _ in range(measured_iters):
            build_windows(
                batch,
                num_next_steps=config.num_next_steps,
                compile_modules=compile_modules,
                compile_mode=config.compile_mode,
            )
        maybe_synchronize(device)
        return time.perf_counter() - start
    finally:
        if gc_was_enabled:
            gc.enable()


def benchmark_batch_sequence(
        batches: list[EpisodeBatch],
        *,
        config: BenchmarkConfig,
        compile_modules: bool,
        measured_iters: int,
        warmup_iters: int,
) -> float:
    device = batches[0].episodes[0].local_obs.device
    for _ in range(warmup_iters):
        for batch in batches:
            build_windows(
                batch,
                num_next_steps=config.num_next_steps,
                compile_modules=compile_modules,
                compile_mode=config.compile_mode,
            )
    maybe_synchronize(device)

    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        start = time.perf_counter()
        for _ in range(measured_iters):
            for batch in batches:
                build_windows(
                    batch,
                    num_next_steps=config.num_next_steps,
                    compile_modules=compile_modules,
                    compile_mode=config.compile_mode,
                )
        maybe_synchronize(device)
        return time.perf_counter() - start
    finally:
        if gc_was_enabled:
            gc.enable()


def benchmark_fixed_length(
        config: BenchmarkConfig,
        *,
        num_steps: int,
        with_agent_mask: bool,
) -> FixedLengthResult:
    batch = make_episode_batch(
        config,
        num_steps=num_steps,
        with_agent_mask=with_agent_mask,
        seed_offset=1000 + num_steps + (100 if with_agent_mask else 0),
    )
    eager_seconds = benchmark_batch(
        batch,
        config=config,
        compile_modules=False,
        measured_iters=config.measured_iters,
        warmup_iters=config.warmup_iters,
    )

    reset_wm_window_helper_compile_cache()
    eager_windows = build_windows(
        batch,
        num_next_steps=config.num_next_steps,
        compile_modules=False,
        compile_mode=config.compile_mode,
    )
    gc.collect()
    start = time.perf_counter()
    compiled_windows = build_windows(
        batch,
        num_next_steps=config.num_next_steps,
        compile_modules=True,
        compile_mode=config.compile_mode,
    )
    maybe_synchronize(batch.episodes[0].local_obs.device)
    compiled_first_call_seconds = time.perf_counter() - start
    assert_windows_equal(eager_windows, compiled_windows)

    compiled_steady_seconds = benchmark_batch(
        batch,
        config=config,
        compile_modules=True,
        measured_iters=config.measured_iters,
        warmup_iters=config.warmup_iters,
    )

    return FixedLengthResult(
        length=num_steps,
        with_agent_mask=with_agent_mask,
        batch_size=batch.batch_size,
        eager_seconds=eager_seconds,
        eager_calls_per_second=config.measured_iters / eager_seconds,
        compiled_first_call_seconds=compiled_first_call_seconds,
        compiled_steady_seconds=compiled_steady_seconds,
        compiled_steady_calls_per_second=config.measured_iters / compiled_steady_seconds,
        steady_speedup=eager_seconds / compiled_steady_seconds,
    )


def benchmark_mixed_lengths(
        config: BenchmarkConfig,
        *,
        lengths: list[int],
        with_agent_mask: bool,
        name: str,
) -> MixedLengthsResult:
    batches = [
        make_episode_batch(
            config,
            num_steps=length,
            with_agent_mask=with_agent_mask,
            seed_offset=2000 + idx + (100 if with_agent_mask else 0),
        )
        for idx, length in enumerate(lengths)
    ]

    eager_seconds = benchmark_batch_sequence(
        batches,
        config=config,
        compile_modules=False,
        measured_iters=config.measured_iters,
        warmup_iters=config.warmup_iters,
    )

    reset_wm_window_helper_compile_cache()
    eager_windows_sequence = [
        build_windows(
            batch,
            num_next_steps=config.num_next_steps,
            compile_modules=False,
            compile_mode=config.compile_mode,
        )
        for batch in batches
    ]
    gc.collect()
    start = time.perf_counter()
    for batch, eager_windows in zip(batches, eager_windows_sequence):
        compiled_windows = build_windows(
            batch,
            num_next_steps=config.num_next_steps,
            compile_modules=True,
            compile_mode=config.compile_mode,
        )
        assert_windows_equal(eager_windows, compiled_windows)
    maybe_synchronize(batches[0].episodes[0].local_obs.device)
    compiled_first_iteration_seconds = time.perf_counter() - start

    compiled_steady_seconds = benchmark_batch_sequence(
        batches,
        config=config,
        compile_modules=True,
        measured_iters=config.measured_iters,
        warmup_iters=config.warmup_iters,
    )

    total_steps_per_iteration = sum(batch.batch_size * batch.num_steps for batch in batches)
    return MixedLengthsResult(
        name=name,
        lengths=lengths,
        with_agent_mask=with_agent_mask,
        total_calls_per_iteration=len(batches),
        total_steps_per_iteration=total_steps_per_iteration,
        eager_seconds=eager_seconds,
        eager_iterations_per_second=config.measured_iters / eager_seconds,
        compiled_first_iteration_seconds=compiled_first_iteration_seconds,
        compiled_steady_seconds=compiled_steady_seconds,
        compiled_steady_iterations_per_second=config.measured_iters / compiled_steady_seconds,
        steady_speedup=eager_seconds / compiled_steady_seconds,
        first_iteration_speedup=(eager_seconds / config.measured_iters) / compiled_first_iteration_seconds,
    )


def parse_args() -> tuple[BenchmarkConfig, Path | None]:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the actual WM helper public path "
            "build_wm_episode_windows_batch(..., compile_modules=False/True) "
            f"for fixed lengths and mixed 1..{DEFAULT_NUM_LENGTH_BUCKETS} length sequences."
        )
    )
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--episodes-per-length", type=int, default=DEFAULT_EPISODES_PER_LENGTH)
    parser.add_argument("--num-next-steps", type=int, default=DEFAULT_NUM_NEXT_STEPS)
    parser.add_argument("--n-agents", type=int, default=DEFAULT_N_AGENTS)
    parser.add_argument("--local-obs-dim", type=int, default=DEFAULT_LOCAL_OBS_DIM)
    parser.add_argument("--global-obs-dim", type=int, default=DEFAULT_GLOBAL_OBS_DIM)
    parser.add_argument("--hidden-local-vars-dim", type=int, default=DEFAULT_HIDDEN_LOCAL_VARS_DIM)
    parser.add_argument("--hidden-global-vars-dim", type=int, default=DEFAULT_HIDDEN_GLOBAL_VARS_DIM)
    parser.add_argument("--action-dim", type=int, default=DEFAULT_ACTION_DIM)
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--measured-iters", type=int, default=100)
    parser.add_argument("--compile-mode", type=str, default="default")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    args = parser.parse_args()
    return BenchmarkConfig(
        device=args.device,
        episodes_per_length=args.episodes_per_length,
        num_next_steps=args.num_next_steps,
        n_agents=args.n_agents,
        local_obs_dim=args.local_obs_dim,
        global_obs_dim=args.global_obs_dim,
        hidden_local_vars_dim=args.hidden_local_vars_dim,
        hidden_global_vars_dim=args.hidden_global_vars_dim,
        action_dim=args.action_dim,
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
        compile_mode=args.compile_mode,
        seed=args.seed,
    ), args.json_out


def main() -> None:
    config, json_out = parse_args()
    require_compile_support(config)
    enable_torch_compile_logging(verbose=True)

    logger.info("Running PPO WM helper compile benchmark with config: {}", config)

    fixed_results: list[FixedLengthResult] = []
    mixed_results: list[MixedLengthsResult] = []

    for with_agent_mask in (False, True):
        for length in range(1, DEFAULT_NUM_LENGTH_BUCKETS + 1):
            logger.info("Benchmarking fixed length={}, with_agent_mask={}", length, with_agent_mask)
            fixed_results.append(
                benchmark_fixed_length(
                    config,
                    num_steps=length,
                    with_agent_mask=with_agent_mask,
                )
            )

        lengths = list(range(1, DEFAULT_NUM_LENGTH_BUCKETS + 1))
        logger.info("Benchmarking mixed lengths={}, with_agent_mask={}", lengths, with_agent_mask)
        mixed_results.append(
            benchmark_mixed_lengths(
                config,
                lengths=lengths,
                with_agent_mask=with_agent_mask,
                name=f"mixed_len_1_to_{DEFAULT_NUM_LENGTH_BUCKETS}_{'mask' if with_agent_mask else 'no_mask'}",
            )
        )

    logger.info("Fixed-length summary:")
    for result in fixed_results:
        logger.info(
            "len={} mask={}: steady_speedup={:.2f}x, eager={:.2f} call/s, compiled={:.2f} call/s, first_call={:.4f}s",
            result.length,
            result.with_agent_mask,
            result.steady_speedup,
            result.eager_calls_per_second,
            result.compiled_steady_calls_per_second,
            result.compiled_first_call_seconds,
        )

    logger.info("Mixed-length summary:")
    for result in mixed_results:
        logger.info(
            "{}: steady_speedup={:.2f}x, first_iter_speedup={:.4f}x, eager={:.2f} iter/s, compiled={:.2f} iter/s, first_iter={:.4f}s",
            result.name,
            result.steady_speedup,
            result.first_iteration_speedup,
            result.eager_iterations_per_second,
            result.compiled_steady_iterations_per_second,
            result.compiled_first_iteration_seconds,
        )

    if json_out is not None:
        json_out.write_text(
            json.dumps(
                {
                    "fixed_results": [asdict(result) for result in fixed_results],
                    "mixed_results": [asdict(result) for result in mixed_results],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Wrote benchmark JSON to {}", json_out)


if __name__ == "__main__":
    main()


