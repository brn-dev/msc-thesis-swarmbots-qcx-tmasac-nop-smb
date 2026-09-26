from __future__ import annotations

import argparse
import gc
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

from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPOEpisodeSegment
from swarmbots.learn.algos.ppo.ppo_sampler import PPOSampler
from swarmbots.learn.algos.world_modeling.base_wm_sampler import BaseWMSampler
from swarmbots.learn.algos.world_modeling.ppo_wm_sampler import PPOWMSampler, PPOWMSamplerConfig
from swarmbots.learn.algos.world_modeling.wm_sampler_helper import build_wm_episode_windows

# Default workload aligned with scripts/run_mat_qcs_nop_wall_mjw.py.
# That training setup uses StepsRolloutMode(4096) with n_envs=1024, so the flat sampler
# mostly sees PPOEpisodeSegment lengths of 4 rollout steps per env, not true episode_length=512.
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_NUM_EPISODES = 1024
DEFAULT_MAX_SEGMENT_LENGTH = 4
DEFAULT_NUM_NEXT_STEPS = 3
DEFAULT_BATCH_SIZE = 4096
DEFAULT_N_AGENTS = 5
DEFAULT_LOCAL_OBS_DIM = 154
DEFAULT_GLOBAL_OBS_DIM = 0
DEFAULT_HIDDEN_LOCAL_VARS_DIM = 4
DEFAULT_HIDDEN_GLOBAL_VARS_DIM = 3
DEFAULT_ACTION_DIM = 12


@dataclass(frozen=True)
class BenchmarkConfig:
    device: str
    num_episodes: int
    max_episode_length: int
    num_next_steps: int
    batch_size: int
    compile_wm_window_helper: bool
    wm_window_helper_compile_mode: str
    warmup_iters: int
    measured_iters: int
    n_agents: int
    local_obs_dim: int
    global_obs_dim: int
    hidden_local_vars_dim: int
    hidden_global_vars_dim: int
    action_dim: int
    seed: int


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    length_mode: str
    with_agent_mask: bool


@dataclass(frozen=True)
class BenchmarkResult:
    case: str
    length_mode: str
    with_agent_mask: bool
    num_episodes: int
    total_steps: int
    reference_seconds: float
    current_seconds: float
    reference_iters_per_second: float
    current_iters_per_second: float
    speedup: float


ALL_CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase(name="fixed_no_mask", length_mode="fixed", with_agent_mask=False),
    BenchmarkCase(name="fixed_mask", length_mode="fixed", with_agent_mask=True),
    BenchmarkCase(name="bucketed_no_mask", length_mode="bucketed", with_agent_mask=False),
    BenchmarkCase(name="bucketed_mask", length_mode="bucketed", with_agent_mask=True),
    BenchmarkCase(name="ragged_no_mask", length_mode="ragged", with_agent_mask=False),
    BenchmarkCase(name="ragged_mask", length_mode="ragged", with_agent_mask=True),
)


class ReferencePPOWMSampler(
    PPOSampler,
    BaseWMSampler,
):
    def __init__(
            self,
            episodes: list[PPOEpisodeSegment],
            config: PPOWMSamplerConfig,
            requires_previous_actions: bool = False,
    ) -> None:
        num_next_steps = config.num_next_steps
        if num_next_steps < 1:
            raise ValueError(f"num_next_steps must be >= 1, got {num_next_steps}")
        super().__init__(
            episodes=episodes,
            config=config,
            requires_previous_actions=requires_previous_actions,
        )

        multi_step_actions_list = []
        next_local_obs_list = []
        wm_target_time_mask_list = []
        next_global_obs_list = []
        wm_agent_mask_list: list[torch.Tensor] = []
        wm_loss_agent_mask_list: list[torch.Tensor] = []
        has_agent_mask = self.agent_mask is not None

        for episode in episodes:
            episode_windows = build_wm_episode_windows(
                episode,
                num_next_steps=num_next_steps,
            )
            multi_step_actions_list.append(episode_windows.multi_step_actions)
            next_local_obs_list.append(episode_windows.next_local_obs)
            wm_target_time_mask_list.append(episode_windows.wm_target_time_mask)
            next_global_obs_list.append(episode_windows.next_global_obs)
            if has_agent_mask:
                if episode_windows.wm_agent_mask is None:
                    raise ValueError("wm_agent_mask must be set when agent_mask is enabled")
                if episode_windows.wm_loss_agent_mask is None:
                    raise ValueError("wm_loss_agent_mask must be set when agent_mask is enabled")
                wm_agent_mask_list.append(episode_windows.wm_agent_mask)
                wm_loss_agent_mask_list.append(episode_windows.wm_loss_agent_mask)

        self.multi_step_actions = torch.cat(multi_step_actions_list, dim=0).contiguous()
        self.next_local_obs = torch.cat(next_local_obs_list, dim=0).contiguous()
        self.wm_target_time_mask = torch.cat(wm_target_time_mask_list, dim=0).contiguous()
        self.next_global_obs = torch.cat(next_global_obs_list, dim=0).contiguous()
        self.wm_agent_mask = (
            torch.cat(wm_agent_mask_list, dim=0).contiguous() if has_agent_mask else None
        )
        self.wm_loss_agent_mask = (
            torch.cat(wm_loss_agent_mask_list, dim=0).contiguous() if has_agent_mask else None
        )


def make_episode_lengths(
        config: BenchmarkConfig,
        case: BenchmarkCase,
        *,
        generator: torch.Generator,
) -> list[int]:
    if case.length_mode == "fixed":
        return [config.max_episode_length] * config.num_episodes
    if case.length_mode == "bucketed":
        bucket_lengths = torch.tensor(
            [
                max(1, config.max_episode_length // 4),
                max(1, config.max_episode_length // 2),
                max(1, (3 * config.max_episode_length) // 4),
                config.max_episode_length,
            ],
            dtype=torch.long,
        )
        bucket_indices = torch.randint(
            low=0,
            high=len(bucket_lengths),
            size=(config.num_episodes,),
            generator=generator,
        )
        return bucket_lengths[bucket_indices].tolist()
    if case.length_mode == "ragged":
        return torch.randint(
            low=1,
            high=config.max_episode_length + 1,
            size=(config.num_episodes,),
            generator=generator,
        ).tolist()
    raise ValueError(f"Unknown length_mode: {case.length_mode}")


def make_episodes(
        config: BenchmarkConfig,
        case: BenchmarkCase,
) -> list[PPOEpisodeSegment]:
    device = torch.device(config.device)
    length_generator = torch.Generator(device="cpu")
    length_generator.manual_seed(config.seed)
    tensor_generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    tensor_generator.manual_seed(config.seed)
    episode_lengths = make_episode_lengths(config, case, generator=length_generator)

    episodes: list[PPOEpisodeSegment] = []
    for episode_idx, num_steps in enumerate(episode_lengths):
        local_obs = torch.randn(
            (num_steps, config.n_agents, config.local_obs_dim),
            generator=tensor_generator,
            device=device,
        )
        global_obs = torch.randn(
            (num_steps, config.global_obs_dim),
            generator=tensor_generator,
            device=device,
        )
        hidden_local_vars = torch.randn(
            (num_steps, config.n_agents, config.hidden_local_vars_dim),
            generator=tensor_generator,
            device=device,
        )
        hidden_global_vars = torch.randn(
            (num_steps, config.hidden_global_vars_dim),
            generator=tensor_generator,
            device=device,
        )
        actions = torch.randn(
            (num_steps, config.n_agents, config.action_dim),
            generator=tensor_generator,
            device=device,
        )
        rewards = torch.randn((num_steps,), generator=tensor_generator, device=device)
        log_probs = torch.randn((num_steps, config.n_agents), generator=tensor_generator, device=device)
        values = torch.randn((num_steps,), generator=tensor_generator, device=device)
        returns = torch.randn((num_steps,), generator=tensor_generator, device=device)
        advantages = torch.randn((num_steps,), generator=tensor_generator, device=device)
        initial_previous_actions = torch.randn(
            (config.n_agents, config.action_dim),
            generator=tensor_generator,
            device=device,
        )

        agent_mask = None
        final_agent_mask = None
        if case.with_agent_mask:
            agent_mask = torch.rand(
                (num_steps, config.n_agents),
                generator=tensor_generator,
                device=device,
            ) > 0.2
            agent_mask[:, 0] = True
            final_agent_mask = torch.rand(
                (config.n_agents,),
                generator=tensor_generator,
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
                rewards=rewards,
                log_probs=log_probs,
                values=values,
                final_local_obs=torch.randn(
                    (config.n_agents, config.local_obs_dim),
                    generator=tensor_generator,
                    device=device,
                ),
                final_global_obs=torch.randn(
                    (config.global_obs_dim,),
                    generator=tensor_generator,
                    device=device,
                ),
                final_hidden_local_vars=torch.randn(
                    (config.n_agents, config.hidden_local_vars_dim),
                    generator=tensor_generator,
                    device=device,
                ),
                final_hidden_global_vars=torch.randn(
                    (config.hidden_global_vars_dim,),
                    generator=tensor_generator,
                    device=device,
                ),
                final_agent_mask=final_agent_mask,
                final_value=torch.randn((), generator=tensor_generator, device=device),
                initial_previous_actions=initial_previous_actions,
                is_true_episode_start=(episode_idx % 3) != 1,
                returns=returns,
                advantages=advantages,
            )
        )
    return episodes


def maybe_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_constructor(
        constructor,
        *,
        episodes: list[PPOEpisodeSegment],
        config: PPOWMSamplerConfig,
        measured_iters: int,
        warmup_iters: int,
) -> float:
    device = episodes[0].local_obs.device

    for _ in range(warmup_iters):
        constructor(episodes=episodes, config=config, requires_previous_actions=True)
    maybe_synchronize(device)

    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        start = time.perf_counter()
        for _ in range(measured_iters):
            constructor(episodes=episodes, config=config, requires_previous_actions=True)
        maybe_synchronize(device)
        elapsed = time.perf_counter() - start
    finally:
        if gc_was_enabled:
            gc.enable()
    return elapsed


def assert_equivalent(
        reference_sampler: ReferencePPOWMSampler,
        current_sampler: PPOWMSampler,
) -> None:
    tensor_fields = (
        "local_obs",
        "global_obs",
        "hidden_local_vars",
        "hidden_global_vars",
        "previous_actions",
        "actions",
        "log_probs",
        "values",
        "returns",
        "advantages",
        "multi_step_actions",
        "next_local_obs",
        "wm_target_time_mask",
        "next_global_obs",
        "wm_agent_mask",
        "wm_loss_agent_mask",
    )
    for field_name in tensor_fields:
        reference_value = getattr(reference_sampler, field_name)
        current_value = getattr(current_sampler, field_name)
        if reference_value is None:
            if current_value is not None:
                raise AssertionError(f"{field_name} mismatch: expected None")
            continue
        if not torch.equal(reference_value, current_value):
            raise AssertionError(f"{field_name} mismatch")


def benchmark_case(
        config: BenchmarkConfig,
        case: BenchmarkCase,
) -> BenchmarkResult:
    episodes = make_episodes(config, case)
    sampler_config = PPOWMSamplerConfig(
        batch_size=config.batch_size,
        num_next_steps=config.num_next_steps,
        compile_wm_window_helper=config.compile_wm_window_helper,
        wm_window_helper_compile_mode=config.wm_window_helper_compile_mode,
    )
    reference_sampler = ReferencePPOWMSampler(
        episodes=episodes,
        config=sampler_config,
        requires_previous_actions=True,
    )
    current_sampler = PPOWMSampler(
        episodes=episodes,
        config=sampler_config,
        requires_previous_actions=True,
    )
    assert_equivalent(reference_sampler, current_sampler)

    reference_seconds = benchmark_constructor(
        ReferencePPOWMSampler,
        episodes=episodes,
        config=sampler_config,
        measured_iters=config.measured_iters,
        warmup_iters=config.warmup_iters,
    )
    current_seconds = benchmark_constructor(
        PPOWMSampler,
        episodes=episodes,
        config=sampler_config,
        measured_iters=config.measured_iters,
        warmup_iters=config.warmup_iters,
    )

    total_steps = sum(int(episode.local_obs.shape[0]) for episode in episodes)
    return BenchmarkResult(
        case=case.name,
        length_mode=case.length_mode,
        with_agent_mask=case.with_agent_mask,
        num_episodes=len(episodes),
        total_steps=total_steps,
        reference_seconds=reference_seconds,
        current_seconds=current_seconds,
        reference_iters_per_second=config.measured_iters / reference_seconds,
        current_iters_per_second=config.measured_iters / current_seconds,
        speedup=reference_seconds / current_seconds,
    )


def parse_args() -> tuple[BenchmarkConfig, list[str], Path | None]:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark PPO WM sampler construction speed for the old serial per-episode "
            "window builder versus the current grouped batched implementation."
        )
    )
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--num-episodes", type=int, default=DEFAULT_NUM_EPISODES)
    parser.add_argument(
        "--max-episode-length",
        type=int,
        default=DEFAULT_MAX_SEGMENT_LENGTH,
        help=(
            "Maximum PPOEpisodeSegment length fed into the sampler. "
            "The default matches run_mat_qcs_nop_wall_mjw.py step-rollout segments, not the true env episode length."
        ),
    )
    parser.add_argument("--num-next-steps", type=int, default=DEFAULT_NUM_NEXT_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--compile-wm-window-helper", action="store_true")
    parser.add_argument("--wm-window-helper-compile-mode", type=str, default="default")
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--measured-iters", type=int, default=100)
    parser.add_argument("--n-agents", type=int, default=DEFAULT_N_AGENTS)
    parser.add_argument("--local-obs-dim", type=int, default=DEFAULT_LOCAL_OBS_DIM)
    parser.add_argument("--global-obs-dim", type=int, default=DEFAULT_GLOBAL_OBS_DIM)
    parser.add_argument("--hidden-local-vars-dim", type=int, default=DEFAULT_HIDDEN_LOCAL_VARS_DIM)
    parser.add_argument("--hidden-global-vars-dim", type=int, default=DEFAULT_HIDDEN_GLOBAL_VARS_DIM)
    parser.add_argument("--action-dim", type=int, default=DEFAULT_ACTION_DIM)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=[case.name for case in ALL_CASES],
        default=[case.name for case in ALL_CASES],
        help="Benchmark case subset to run.",
    )
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    args = parser.parse_args()
    return BenchmarkConfig(
        device=args.device,
        num_episodes=args.num_episodes,
        max_episode_length=args.max_episode_length,
        num_next_steps=args.num_next_steps,
        batch_size=args.batch_size,
        compile_wm_window_helper=args.compile_wm_window_helper,
        wm_window_helper_compile_mode=args.wm_window_helper_compile_mode,
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
        n_agents=args.n_agents,
        local_obs_dim=args.local_obs_dim,
        global_obs_dim=args.global_obs_dim,
        hidden_local_vars_dim=args.hidden_local_vars_dim,
        hidden_global_vars_dim=args.hidden_global_vars_dim,
        action_dim=args.action_dim,
        seed=args.seed,
    ), args.cases if args.cases is not None else [case.name for case in ALL_CASES], args.json_out


def main() -> None:
    config, selected_case_names, json_out = parse_args()
    logger.info("Running PPO WM sampler benchmark with config: {}", config)
    selected_cases = [case for case in ALL_CASES if case.name in selected_case_names]

    results: list[BenchmarkResult] = []
    for case in selected_cases:
        logger.info(
            "Benchmarking case={}, length_mode={}, with_agent_mask={}",
            case.name,
            case.length_mode,
            case.with_agent_mask,
        )
        results.append(benchmark_case(config, case))

    logger.info("Benchmark summary:")
    for result in results:
        logger.info(
            "{}: speedup={:.2f}x, current={:.2f} it/s, reference={:.2f} it/s, total_steps={}",
            result.case,
            result.speedup,
            result.current_iters_per_second,
            result.reference_iters_per_second,
            result.total_steps,
        )

    if json_out is not None:
        json_out.write_text(
            json.dumps([asdict(result) for result in results], indent=2),
            encoding="utf-8",
        )
        logger.info("Wrote benchmark JSON to {}", json_out)


if __name__ == "__main__":
    main()


