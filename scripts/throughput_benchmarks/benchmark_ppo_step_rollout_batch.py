import argparse
import gc
import time
from dataclasses import replace
from typing import Callable

import torch

from swarmbots.learn.algos.ppo.ppo_rollout_batch import (
    PPORolloutBatch,
    PPORolloutBatchBuilder,
    compute_step_rollout_gae,
)
from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPOEpisodeSegment
from swarmbots.learn.algos.ppo.ppo_sampler import PPOBatchSampler, PPOSampler, PPOSamplerConfig
from swarmbots.learn.algos.world_modeling.ppo_wm_sampler import (
    PPOWMBatchSampler,
    PPOWMSampler,
    PPOWMSamplerConfig,
)


def make_rollout_batch(
        *,
        n_envs: int,
        n_steps: int,
        n_agents: int,
        local_obs_dim: int,
        global_obs_dim: int,
        hidden_local_dim: int,
        hidden_global_dim: int,
        action_dim: int,
        done_probability: float,
        with_agent_mask: bool,
        device: torch.device,
) -> PPORolloutBatch:
    generator = torch.Generator(device=device).manual_seed(12345)
    local_obs = torch.randn(n_envs, n_steps, n_agents, local_obs_dim, generator=generator, device=device)
    global_obs = torch.randn(n_envs, n_steps, global_obs_dim, generator=generator, device=device)
    hidden_local_vars = torch.randn(n_envs, n_steps, n_agents, hidden_local_dim, generator=generator, device=device)
    hidden_global_vars = torch.randn(n_envs, n_steps, hidden_global_dim, generator=generator, device=device)
    previous_actions = torch.randn(n_envs, n_steps, n_agents, action_dim, generator=generator, device=device)
    actions = torch.randn(n_envs, n_steps, n_agents, action_dim, generator=generator, device=device)
    rewards = torch.randn(n_envs, n_steps, generator=generator, device=device)
    log_probs = torch.randn(n_envs, n_steps, n_agents, generator=generator, device=device)
    values = torch.randn(n_envs, n_steps, generator=generator, device=device)

    dones = torch.rand(n_envs, n_steps, generator=generator, device=device) < done_probability
    terminations = dones & (torch.rand(n_envs, n_steps, generator=generator, device=device) < 0.7)
    truncations = dones & ~terminations
    bootstrap_values = torch.randn(n_envs, n_steps, generator=generator, device=device).masked_fill(terminations, 0.0)

    episode_start_mask = torch.zeros_like(dones)
    episode_start_mask[:, 0] = True
    episode_start_mask[:, 1:] = dones[:, :-1]
    agent_mask = _make_agent_mask(
        with_agent_mask=with_agent_mask,
        n_envs=n_envs,
        n_steps=n_steps,
        n_agents=n_agents,
        generator=generator,
        device=device,
    )

    bootstrap_local_obs = torch.randn(
        n_envs,
        n_steps,
        n_agents,
        local_obs_dim,
        generator=generator,
        device=device,
    )
    bootstrap_global_obs = torch.randn(n_envs, n_steps, global_obs_dim, generator=generator, device=device)
    bootstrap_hidden_local_vars = torch.randn(
        n_envs,
        n_steps,
        n_agents,
        hidden_local_dim,
        generator=generator,
        device=device,
    )
    bootstrap_hidden_global_vars = torch.randn(n_envs, n_steps, hidden_global_dim, generator=generator, device=device)
    nonterminal_with_next_step = ~dones[:, :-1]
    bootstrap_local_obs[:, :-1] = torch.where(
        nonterminal_with_next_step.view(n_envs, n_steps - 1, 1, 1),
        local_obs[:, 1:],
        bootstrap_local_obs[:, :-1],
    )
    bootstrap_global_obs[:, :-1] = torch.where(
        nonterminal_with_next_step.view(n_envs, n_steps - 1, 1),
        global_obs[:, 1:],
        bootstrap_global_obs[:, :-1],
    )
    bootstrap_hidden_local_vars[:, :-1] = torch.where(
        nonterminal_with_next_step.view(n_envs, n_steps - 1, 1, 1),
        hidden_local_vars[:, 1:],
        bootstrap_hidden_local_vars[:, :-1],
    )
    bootstrap_hidden_global_vars[:, :-1] = torch.where(
        nonterminal_with_next_step.view(n_envs, n_steps - 1, 1),
        hidden_global_vars[:, 1:],
        bootstrap_hidden_global_vars[:, :-1],
    )

    advantages = compute_step_rollout_gae(
        rewards=rewards,
        values=values,
        bootstrap_values=bootstrap_values,
        dones=dones,
        gamma=0.99,
        gae_lambda=0.95,
    )
    return PPORolloutBatch(
        local_obs=local_obs,
        global_obs=global_obs,
        hidden_local_vars=hidden_local_vars,
        hidden_global_vars=hidden_global_vars,
        agent_mask=agent_mask,
        previous_actions=previous_actions,
        actions=actions,
        rewards=rewards,
        log_probs=log_probs,
        values=values,
        bootstrap_local_obs=bootstrap_local_obs,
        bootstrap_global_obs=bootstrap_global_obs,
        bootstrap_hidden_local_vars=bootstrap_hidden_local_vars,
        bootstrap_hidden_global_vars=bootstrap_hidden_global_vars,
        bootstrap_agent_mask=_make_bootstrap_agent_mask(
            agent_mask=agent_mask,
            dones=dones,
            n_envs=n_envs,
            n_steps=n_steps,
            generator=generator,
            device=device,
        ),
        bootstrap_values=bootstrap_values,
        terminations=terminations,
        truncations=truncations,
        dones=dones,
        episode_start_mask=episode_start_mask,
        returns=advantages + values,
        advantages=advantages,
    )


def _make_agent_mask(
        *,
        with_agent_mask: bool,
        n_envs: int,
        n_steps: int,
        n_agents: int,
        generator: torch.Generator,
        device: torch.device,
) -> torch.Tensor | None:
    if not with_agent_mask:
        return None
    agent_mask = torch.rand(n_envs, n_steps, n_agents, generator=generator, device=device) > 0.15
    agent_mask[..., 0] = True
    return agent_mask


def _make_bootstrap_agent_mask(
        *,
        agent_mask: torch.Tensor | None,
        dones: torch.Tensor,
        n_envs: int,
        n_steps: int,
        generator: torch.Generator,
        device: torch.device,
) -> torch.Tensor | None:
    if agent_mask is None:
        return None
    bootstrap_agent_mask = torch.rand(
        n_envs,
        n_steps,
        agent_mask.shape[-1],
        generator=generator,
        device=device,
    ) > 0.15
    bootstrap_agent_mask[..., 0] = True
    nonterminal_with_next_step = ~dones[:, :-1]
    bootstrap_agent_mask[:, :-1] = torch.where(
        nonterminal_with_next_step.view(n_envs, n_steps - 1, 1),
        agent_mask[:, 1:],
        bootstrap_agent_mask[:, :-1],
    )
    return bootstrap_agent_mask


def make_reference_episode_sampler(
        *,
        rollout_batch: PPORolloutBatch,
        config: PPOSamplerConfig,
        requires_previous_actions: bool,
) -> PPOSampler:
    return PPOSampler(
        episodes=_split_rollout_batch_like_previous_step_path(rollout_batch),
        config=config,
        requires_previous_actions=requires_previous_actions,
    )


def make_reference_wm_sampler(
        *,
        rollout_batch: PPORolloutBatch,
        config: PPOWMSamplerConfig,
        requires_previous_actions: bool,
) -> PPOWMSampler:
    return PPOWMSampler(
        episodes=_split_rollout_batch_like_previous_step_path(rollout_batch),
        config=config,
        requires_previous_actions=requires_previous_actions,
    )


def make_batch_sampler(
        *,
        rollout_batch: PPORolloutBatch,
        config: PPOSamplerConfig,
        requires_previous_actions: bool,
) -> PPOBatchSampler:
    batch = _rollout_batch_with_recomputed_gae(rollout_batch)
    return PPOBatchSampler(
        rollout_batch=batch,
        config=config,
        requires_previous_actions=requires_previous_actions,
    )


def make_wm_batch_sampler(
        *,
        rollout_batch: PPORolloutBatch,
        config: PPOWMSamplerConfig,
        requires_previous_actions: bool,
) -> PPOWMBatchSampler:
    batch = _rollout_batch_with_recomputed_gae(rollout_batch)
    return PPOWMBatchSampler(
        rollout_batch=batch,
        config=config,
        requires_previous_actions=requires_previous_actions,
    )


def make_rollout_batch_builder(
        *,
        rollout_batch: PPORolloutBatch,
        config: PPOSamplerConfig,
        requires_previous_actions: bool,
) -> PPORolloutBatchBuilder:
    _ = config
    return PPORolloutBatchBuilder(
        n_envs=rollout_batch.n_envs,
        n_steps=rollout_batch.n_steps,
        n_agents=int(rollout_batch.local_obs.shape[2]),
        agent_obs_shape=tuple(rollout_batch.local_obs.shape[3:]),
        global_obs_shape=tuple(rollout_batch.global_obs.shape[2:]),
        hidden_local_vars_shape=tuple(rollout_batch.hidden_local_vars.shape[3:]),
        hidden_global_vars_shape=tuple(rollout_batch.hidden_global_vars.shape[2:]),
        n_agent_actions=int(rollout_batch.actions.shape[-1]),
        has_agent_mask=rollout_batch.agent_mask is not None,
        has_previous_actions=requires_previous_actions,
        gamma=0.99,
        gae_lambda=0.95,
        storage_device=rollout_batch.local_obs.device,
        storage_dtype=rollout_batch.local_obs.dtype,
    )


def _rollout_batch_with_recomputed_gae(rollout_batch: PPORolloutBatch) -> PPORolloutBatch:
    advantages = compute_step_rollout_gae(
        rewards=rollout_batch.rewards,
        values=rollout_batch.values,
        bootstrap_values=rollout_batch.bootstrap_values,
        dones=rollout_batch.dones,
        gamma=0.99,
        gae_lambda=0.95,
    )
    return replace(rollout_batch, advantages=advantages, returns=advantages + rollout_batch.values)


def _split_rollout_batch_like_previous_step_path(
        rollout_batch: PPORolloutBatch,
) -> list[PPOEpisodeSegment]:
    episodes: list[PPOEpisodeSegment] = []
    for env_idx in range(rollout_batch.n_envs):
        start_idx = 0
        for step_idx in range(rollout_batch.n_steps):
            is_segment_end = bool(rollout_batch.dones[env_idx, step_idx]) or step_idx == rollout_batch.n_steps - 1
            if not is_segment_end:
                continue
            end_idx = step_idx + 1
            episode = PPOEpisodeSegment(
                local_obs=rollout_batch.local_obs[env_idx, start_idx:end_idx].clone(),
                global_obs=rollout_batch.global_obs[env_idx, start_idx:end_idx].clone(),
                hidden_local_vars=rollout_batch.hidden_local_vars[env_idx, start_idx:end_idx].clone(),
                hidden_global_vars=rollout_batch.hidden_global_vars[env_idx, start_idx:end_idx].clone(),
                agent_mask=(
                    None
                    if rollout_batch.agent_mask is None
                    else rollout_batch.agent_mask[env_idx, start_idx:end_idx].clone()
                ),
                actions=rollout_batch.actions[env_idx, start_idx:end_idx].clone(),
                rewards=rollout_batch.rewards[env_idx, start_idx:end_idx].clone(),
                log_probs=rollout_batch.log_probs[env_idx, start_idx:end_idx].clone(),
                values=rollout_batch.values[env_idx, start_idx:end_idx].clone(),
                final_local_obs=rollout_batch.bootstrap_local_obs[env_idx, step_idx].clone(),
                final_global_obs=rollout_batch.bootstrap_global_obs[env_idx, step_idx].clone(),
                final_hidden_local_vars=rollout_batch.bootstrap_hidden_local_vars[env_idx, step_idx].clone(),
                final_hidden_global_vars=rollout_batch.bootstrap_hidden_global_vars[env_idx, step_idx].clone(),
                final_agent_mask=(
                    None
                    if rollout_batch.bootstrap_agent_mask is None
                    else rollout_batch.bootstrap_agent_mask[env_idx, step_idx].clone()
                ),
                final_value=rollout_batch.bootstrap_values[env_idx, step_idx].clone(),
                initial_previous_actions=rollout_batch.previous_actions[env_idx, start_idx].clone(),
                is_true_episode_start=bool(rollout_batch.episode_start_mask[env_idx, start_idx]),
            )
            episode.compute_gae(gamma=0.99, gae_lambda=0.95)
            episodes.append(episode)
            start_idx = end_idx
    return episodes


def benchmark(
        fn: Callable[..., object],
        *,
        rollout_batch: PPORolloutBatch,
        config: PPOSamplerConfig,
        requires_previous_actions: bool,
        warmup_iters: int,
        measured_iters: int,
) -> float:
    for _ in range(warmup_iters):
        fn(
            rollout_batch=rollout_batch,
            config=config,
            requires_previous_actions=requires_previous_actions,
        )
    _maybe_synchronize(rollout_batch.local_obs.device)

    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        start = time.perf_counter()
        for _ in range(measured_iters):
            fn(
                rollout_batch=rollout_batch,
                config=config,
                requires_previous_actions=requires_previous_actions,
            )
        _maybe_synchronize(rollout_batch.local_obs.device)
        return time.perf_counter() - start
    finally:
        if gc_was_enabled:
            gc.enable()


def _maybe_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-envs", type=int, default=1024)
    parser.add_argument("--n-steps", type=int, default=4)
    parser.add_argument("--n-agents", type=int, default=8)
    parser.add_argument("--local-obs-dim", type=int, default=64)
    parser.add_argument("--global-obs-dim", type=int, default=64)
    parser.add_argument("--hidden-local-dim", type=int, default=8)
    parser.add_argument("--hidden-global-dim", type=int, default=8)
    parser.add_argument("--action-dim", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-next-steps", type=int, default=3)
    parser.add_argument("--done-probability", type=float, default=0.01)
    parser.add_argument("--with-agent-mask", action="store_true")
    parser.add_argument("--plain-ppo", action="store_true")
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--measured-iters", type=int, default=50)
    args = parser.parse_args()

    device = torch.device(args.device)
    rollout_batch = make_rollout_batch(
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        n_agents=args.n_agents,
        local_obs_dim=args.local_obs_dim,
        global_obs_dim=args.global_obs_dim,
        hidden_local_dim=args.hidden_local_dim,
        hidden_global_dim=args.hidden_global_dim,
        action_dim=args.action_dim,
        done_probability=args.done_probability,
        with_agent_mask=args.with_agent_mask,
        device=device,
    )

    if args.plain_ppo:
        config = PPOSamplerConfig(batch_size=args.batch_size)
        reference_fn = make_reference_episode_sampler
        batch_fn = make_batch_sampler
    else:
        config = PPOWMSamplerConfig(batch_size=args.batch_size, num_next_steps=args.num_next_steps)
        reference_fn = make_reference_wm_sampler
        batch_fn = make_wm_batch_sampler

    reference_seconds = benchmark(
        reference_fn,
        rollout_batch=rollout_batch,
        config=config,
        requires_previous_actions=True,
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    builder_init_seconds = benchmark(
        make_rollout_batch_builder,
        rollout_batch=rollout_batch,
        config=config,
        requires_previous_actions=True,
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    batch_seconds = benchmark(
        batch_fn,
        rollout_batch=rollout_batch,
        config=config,
        requires_previous_actions=True,
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )

    print(f"device: {device}")
    print(f"samples: {rollout_batch.n_samples}")
    print(f"reference_episode_path_seconds: {reference_seconds:.6f}")
    print(f"rollout_batch_builder_init_seconds: {builder_init_seconds:.6f}")
    print(f"rollout_batch_path_seconds: {batch_seconds:.6f}")
    print(f"rollout_batch_path_with_builder_seconds: {batch_seconds + builder_init_seconds:.6f}")
    print(f"speedup: {reference_seconds / batch_seconds:.2f}x")
    print(f"speedup_including_builder: {reference_seconds / (batch_seconds + builder_init_seconds):.2f}x")
    print(f"reference_iters_per_second: {args.measured_iters / reference_seconds:.2f}")
    print(f"rollout_batch_builder_init_iters_per_second: {args.measured_iters / builder_init_seconds:.2f}")
    print(f"rollout_batch_iters_per_second: {args.measured_iters / batch_seconds:.2f}")


if __name__ == "__main__":
    main()
