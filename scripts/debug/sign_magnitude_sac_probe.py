from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from swarmbots.learn.action_dists.action_dist import ActionDist
from swarmbots.learn.algos.off_policy.replay_buffer import OffPolicyReplayBatch, OffPolicyReplayEpisodeSegmentBatch
from swarmbots.learn.algos.sac import BaseSACPolicy, SAC
from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from swarmbots.learn.testing_env import TestingSwarmBotsEnv

ActionDistFactory = Callable[[int, int, float | None], ActionDist]


@dataclass(frozen=True)
class TargetCase:
    name: str
    target_action: float
    expected_sign: int


@dataclass(frozen=True)
class CaseResult:
    name: str
    mode_mean: float
    sample_mean: float
    sample_abs_mean: float
    positive_fraction: float
    negative_fraction: float
    near_zero_fraction: float
    final_actor_loss: float
    final_q_pi: float
    action_histogram_path: Path | None


@dataclass(frozen=True)
class PassThresholds:
    sign_fraction: float
    signed_sample_mean: float
    zero_mode_abs: float
    zero_sample_abs_mean: float


@dataclass(frozen=True)
class ActionHistogramConfig:
    output_dir: Path
    samples_per_update: int
    bins: int


class MockCriticSignMagnitudeSACPolicy(BaseSACPolicy):
    def __init__(
            self,
            *,
            n_agents: int,
            action_dim: int,
            target_action: float,
            q_scale: float,
            action_dist_factory: ActionDistFactory,
            gumbel_temperature: float | None,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.target_action = float(target_action)
        self.q_scale = float(q_scale)
        self.action_dist = action_dist_factory(1, action_dim, gumbel_temperature)
        self.critic_bias = torch.nn.Parameter(torch.tensor(0.0))

    def action_log_prob(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
            deterministic: bool = False,
            use_rsample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = (global_obs, hidden_local_vars, hidden_global_vars, previous_actions)
        latent = local_obs.new_zeros((*local_obs.shape[:2], 1))
        actions, log_probs = self.action_dist.get_actions_with_log_probs(
            latent,
            deterministic=deterministic,
            use_rsample=use_rsample,
        )
        if agent_mask is not None:
            actions = actions.masked_fill(~agent_mask.unsqueeze(-1), 0.0)
            log_probs = log_probs.masked_fill(~agent_mask, 0.0)
        return actions, log_probs

    def q_values(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            actions: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = (local_obs, global_obs, hidden_local_vars, hidden_global_vars)
        target = torch.full_like(actions, self.target_action)
        q_per_action = -self.q_scale * (actions - target).square()
        if agent_mask is not None:
            q_per_action = q_per_action * agent_mask.unsqueeze(-1).to(dtype=q_per_action.dtype)
        q_value = q_per_action.sum(dim=(1, 2)) + self.critic_bias
        return q_value, q_value

    def target_q_values(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            actions: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = (global_obs, actions, hidden_local_vars, hidden_global_vars, agent_mask)
        target_q = local_obs.new_zeros(local_obs.shape[0])
        return target_q, target_q

    def actor_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.action_dist.parameters())

    def critic_parameters(self) -> list[torch.nn.Parameter]:
        return [self.critic_bias]

    def compute_actor_nop_loss(
            self,
            batch: OffPolicyReplayBatch | OffPolicyReplayEpisodeSegmentBatch,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        _ = batch
        return None, {}

    def compute_critic_nop_loss(
            self,
            batch: OffPolicyReplayBatch | OffPolicyReplayEpisodeSegmentBatch,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        _ = batch
        return None, {}

    def polyak_update_targets(self, tau: float) -> None:
        _ = tau

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            "target_action": self.target_action,
            "q_scale": self.q_scale,
            "action_dist": self.action_dist.get_hyper_parameters(),
        }

    def get_grad_norms(self) -> dict[str, float]:
        return {
            "actor": self._grad_norm_from_parameters(self.actor_parameters()),
            "critic": self._parameter_grad_norm(self.critic_bias),
        }

    def update_loss_weights(self, **weights: float) -> None:
        if weights:
            raise ValueError(f"Unknown weights given: {weights}")


def make_env() -> SwarmBotsLearnEnvWrapper:
    vector_env = SyncVectorEnv(
        [
            lambda: TestingSwarmBotsEnv(
                n_agents=1,
                n_local_obs=1,
                n_global_obs=1,
                actuators_dim=1,
                connectors_dim=1,
                n_hidden_local_vars=1,
                n_hidden_global_vars=1,
                continuous_connector_actions=True,
            )
        ],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    return SwarmBotsLearnEnvWrapper(vector_env)


def make_batch(env: SwarmBotsLearnEnvWrapper, *, batch_size: int, device: torch.device) -> OffPolicyReplayBatch:
    n_agents = env.n_agents
    action_dim = env.action_space.total_agent_action_dim
    return OffPolicyReplayBatch(
        local_obs=torch.zeros(batch_size, n_agents, env.local_obs_dim, device=device),
        global_obs=torch.zeros(batch_size, env.global_obs_dim, device=device),
        hidden_local_vars=torch.zeros(batch_size, n_agents, env.hidden_local_vars_dim, device=device),
        hidden_global_vars=torch.zeros(batch_size, env.hidden_global_vars_dim, device=device),
        agent_mask=None,
        actions=torch.zeros(batch_size, n_agents, action_dim, device=device),
        rewards=torch.zeros(batch_size, device=device),
        terminations=torch.ones(batch_size, dtype=torch.bool, device=device),
        truncations=torch.zeros(batch_size, dtype=torch.bool, device=device),
        previous_actions=None,
        next_local_obs=torch.zeros(batch_size, n_agents, env.local_obs_dim, device=device),
        next_global_obs=torch.zeros(batch_size, env.global_obs_dim, device=device),
        next_hidden_local_vars=torch.zeros(batch_size, n_agents, env.hidden_local_vars_dim, device=device),
        next_hidden_global_vars=torch.zeros(batch_size, env.hidden_global_vars_dim, device=device),
        next_agent_mask=None,
    )


def sample_policy_actions(
        policy: MockCriticSignMagnitudeSACPolicy,
        *,
        n_agents: int,
        samples: int,
        device: torch.device,
) -> torch.Tensor:
    with torch.no_grad():
        latent = torch.zeros(samples, n_agents, 1, device=device)
        policy.action_dist.update_latent_features(latent)
        return policy.action_dist.sample().detach().flatten().cpu()


def compute_action_histogram(actions: torch.Tensor, *, bins: int) -> torch.Tensor:
    histogram = torch.histc(actions.to(dtype=torch.float32), bins=bins, min=-1.0, max=1.0)
    total = histogram.sum().clamp_min(1.0)
    return histogram / total


def plot_action_histograms(
        histogram_rows: list[torch.Tensor],
        *,
        target_case: TargetCase,
        output_dir: Path,
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Action histogram plotting requires matplotlib. Install the plot extra, e.g. `uv sync --extra plot`."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    histogram_matrix = torch.stack(histogram_rows).numpy()
    output_path = output_dir / f"{target_case.name}_action_histograms.png"

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    image = ax.imshow(
        histogram_matrix,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=(-1.0, 1.0, 1, len(histogram_rows)),
    )
    ax.axvline(target_case.target_action, color="white", linestyle="--", linewidth=1.0)
    ax.set_title(f"{target_case.name} action distribution per update")
    ax.set_xlabel("action")
    ax.set_ylabel("update")
    fig.colorbar(image, ax=ax, label="fraction")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def train_case(
        target_case: TargetCase,
        *,
        steps: int,
        batch_size: int,
        learning_rate: float,
        q_scale: float,
        eval_samples: int,
        action_histogram_config: ActionHistogramConfig | None,
        action_dist_factory: ActionDistFactory,
        gumbel_temperature: float | None,
        device: torch.device,
) -> CaseResult:
    env = make_env()
    try:
        policy = MockCriticSignMagnitudeSACPolicy(
            n_agents=env.n_agents,
            action_dim=env.action_space.total_agent_action_dim,
            target_action=target_case.target_action,
            q_scale=q_scale,
            action_dist_factory=action_dist_factory,
            gumbel_temperature=gumbel_temperature,
        )
        algo = SAC(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_capacity_per_env=batch_size,
            learning_starts=0,
            batch_size=batch_size,
            gamma=0.0,
            ent_coef=0.0,
            target_entropy=0.0,
            max_grad_norm=None,
            train_device=device,
            rollout_device=device,
            replay_storage_device=device,
            learning_rate_warmup_updates=0,
        )
        batch = make_batch(env, batch_size=batch_size, device=device)
        last_metrics: dict[str, float] = {}
        action_histogram_rows: list[torch.Tensor] = []
        for update_idx in range(steps):
            last_metrics, _actor_grad_norm, _critic_grad_norm = algo._train_step(
                batch,
                global_update_idx=update_idx,
            )
            if action_histogram_config is not None:
                actions = sample_policy_actions(
                    policy,
                    n_agents=env.n_agents,
                    samples=action_histogram_config.samples_per_update,
                    device=device,
                )
                action_histogram_rows.append(
                    compute_action_histogram(actions, bins=action_histogram_config.bins)
                )

        with torch.no_grad():
            eval_latent = torch.zeros(eval_samples, env.n_agents, 1, device=device)
            policy.action_dist.update_latent_features(eval_latent)
            mode = policy.action_dist.mode()
            samples = policy.action_dist.sample()
        action_histogram_path = (
            None
            if action_histogram_config is None
            else plot_action_histograms(
                action_histogram_rows,
                target_case=target_case,
                output_dir=action_histogram_config.output_dir,
            )
        )

        return CaseResult(
            name=target_case.name,
            mode_mean=mode.mean().item(),
            sample_mean=samples.mean().item(),
            sample_abs_mean=samples.abs().mean().item(),
            positive_fraction=(samples > 0.0).to(torch.float32).mean().item(),
            negative_fraction=(samples < 0.0).to(torch.float32).mean().item(),
            near_zero_fraction=(samples.abs() < 0.05).to(torch.float32).mean().item(),
            final_actor_loss=last_metrics["actor_loss"],
            final_q_pi=last_metrics["q_pi"],
            action_histogram_path=action_histogram_path,
        )
    finally:
        env.close()


def assert_case_passed(result: CaseResult, target_case: TargetCase, thresholds: PassThresholds) -> None:
    if target_case.expected_sign > 0:
        if (
                result.positive_fraction < thresholds.sign_fraction
                or result.sample_mean < thresholds.signed_sample_mean
        ):
            raise AssertionError(f"{result.name} did not learn the positive side: {result}")
        return
    if target_case.expected_sign < 0:
        if (
                result.negative_fraction < thresholds.sign_fraction
                or result.sample_mean > -thresholds.signed_sample_mean
        ):
            raise AssertionError(f"{result.name} did not learn the negative side: {result}")
        return
    if (
            abs(result.mode_mean) > thresholds.zero_mode_abs
            or result.sample_abs_mean > thresholds.zero_sample_abs_mean
    ):
        raise AssertionError(f"{result.name} did not learn near-zero actions: {result}")


def parse_args(
        *,
        description: str,
        default_action_hist_dir: Path,
        supports_gumbel_temperature: bool,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--q-scale", type=float, default=20.0)
    parser.add_argument("--eval-samples", type=int, default=10_000)
    parser.add_argument("--sign-fraction-threshold", type=float, default=0.9)
    parser.add_argument("--signed-sample-mean-threshold", type=float, default=0.35)
    parser.add_argument("--zero-mode-abs-threshold", type=float, default=0.13)
    parser.add_argument("--zero-sample-abs-mean-threshold", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-assert", action="store_true")
    parser.add_argument("--no-action-hist", action="store_true")
    parser.add_argument("--action-hist-dir", type=Path, default=default_action_hist_dir)
    parser.add_argument("--action-hist-samples", type=int, default=2048)
    parser.add_argument("--action-hist-bins", type=int, default=80)
    if supports_gumbel_temperature:
        parser.add_argument("--gumbel-temperature", type=float, default=1.0)
    return parser.parse_args()


def run_probe(
        *,
        description: str,
        default_action_hist_dir: Path,
        action_dist_factory: ActionDistFactory,
        supports_gumbel_temperature: bool = False,
) -> None:
    args = parse_args(
        description=description,
        default_action_hist_dir=default_action_hist_dir,
        supports_gumbel_temperature=supports_gumbel_temperature,
    )
    if args.action_hist_samples <= 0:
        raise ValueError(f"--action-hist-samples must be positive, got {args.action_hist_samples}")
    if args.action_hist_bins <= 0:
        raise ValueError(f"--action-hist-bins must be positive, got {args.action_hist_bins}")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    action_histogram_config = (
        None
        if args.no_action_hist
        else ActionHistogramConfig(
            output_dir=args.action_hist_dir,
            samples_per_update=args.action_hist_samples,
            bins=args.action_hist_bins,
        )
    )
    target_cases = [
        TargetCase(name="positive", target_action=0.75, expected_sign=1),
        TargetCase(name="negative", target_action=-0.75, expected_sign=-1),
        TargetCase(name="zero", target_action=0.0, expected_sign=0),
        TargetCase(name="-0.1", target_action=-0.1, expected_sign=-1),
        TargetCase(name="-0.05", target_action=-0.05, expected_sign=-1),
        TargetCase(name="-0.02", target_action=-0.02, expected_sign=-1),
        TargetCase(name="-0.01", target_action=-0.01, expected_sign=-1),
    ]
    thresholds = PassThresholds(
        sign_fraction=args.sign_fraction_threshold,
        signed_sample_mean=args.signed_sample_mean_threshold,
        zero_mode_abs=args.zero_mode_abs_threshold,
        zero_sample_abs_mean=args.zero_sample_abs_mean_threshold,
    )
    gumbel_temperature = args.gumbel_temperature if supports_gumbel_temperature else None

    for target_case in target_cases:
        result = train_case(
            target_case,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            q_scale=args.q_scale,
            eval_samples=args.eval_samples,
            action_histogram_config=action_histogram_config,
            action_dist_factory=action_dist_factory,
            gumbel_temperature=gumbel_temperature,
            device=device,
        )
        print(
            f"{result.name:>8}: "
            f"mode_mean={result.mode_mean:+.4f}, "
            f"sample_mean={result.sample_mean:+.4f}, "
            f"abs_mean={result.sample_abs_mean:.4f}, "
            f"pos_frac={result.positive_fraction:.3f}, "
            f"neg_frac={result.negative_fraction:.3f}, "
            f"near_zero_frac={result.near_zero_fraction:.3f}, "
            f"actor_loss={result.final_actor_loss:+.4f}, "
            f"q_pi={result.final_q_pi:+.4f}"
        )
        if result.action_histogram_path is not None:
            print(f"{result.name:>8} action histogram: {result.action_histogram_path}")
        if not args.no_assert:
            assert_case_passed(result, target_case, thresholds)
