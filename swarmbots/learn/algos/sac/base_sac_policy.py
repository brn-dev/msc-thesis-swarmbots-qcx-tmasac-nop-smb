import abc
from typing import Any

import torch

from swarmbots.learn.action_dists.action_sampling import ActionSampleStrategy, expand_action_sample_batch
from swarmbots.learn.action_dists.hybrid_action_dist import HybridActionDistribution
from swarmbots.learn.algos.off_policy.replay_buffer import OffPolicyReplayBatch, OffPolicyReplayEpisodeSegmentBatch
from swarmbots.learn.base_policy import BasePolicy


class BaseSACPolicy(BasePolicy, abc.ABC):
    action_dist: HybridActionDistribution

    @property
    def gsde_enabled(self) -> bool:
        action_dist = getattr(self, "action_dist", None)
        return bool(getattr(action_dist, "has_gsde", False))

    @abc.abstractmethod
    def action_log_prob(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            scenario_ids: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
            deterministic: bool = False,
            use_rsample: bool = True,
            num_action_samples: int = 1,
            action_sample_strategy: ActionSampleStrategy = "iid",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ordinary action/log-prob shapes for one sample, otherwise prepend K."""
        raise NotImplementedError()

    @abc.abstractmethod
    def q_values(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            actions: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            scenario_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def q_values_with_nop_latents(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            actions: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            scenario_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        q_value_kwargs = dict(
            local_obs=local_obs,
            global_obs=global_obs,
            actions=actions,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
        )
        if scenario_ids is not None:
            q_value_kwargs["scenario_ids"] = scenario_ids
        q1, q2 = self.q_values(**q_value_kwargs)
        return q1, q2, None

    @abc.abstractmethod
    def target_q_values(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            actions: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            scenario_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def q_values_samples(
            self,
            *,
            actions: torch.Tensor,
            num_action_samples: int,
            target: bool = False,
            **observations: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate candidates with the same observations, preserving the leading K axis."""
        q_values = self.target_q_values if target else self.q_values
        if num_action_samples == 1:
            return q_values(actions=actions, **observations)
        flat_observations = {
            name: expand_action_sample_batch(value, num_action_samples)
            for name, value in observations.items()
        }
        q1, q2 = q_values(actions=actions.flatten(0, 1), **flat_observations)
        return q1.unflatten(0, (num_action_samples, -1)), q2.unflatten(0, (num_action_samples, -1))

    @abc.abstractmethod
    def actor_parameters(self) -> list[torch.nn.Parameter]:
        raise NotImplementedError()

    @abc.abstractmethod
    def critic_parameters(self) -> list[torch.nn.Parameter]:
        raise NotImplementedError()

    @abc.abstractmethod
    def compute_actor_nop_loss(
            self,
            batch: OffPolicyReplayBatch | OffPolicyReplayEpisodeSegmentBatch,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        raise NotImplementedError()

    @abc.abstractmethod
    def compute_critic_nop_loss(
            self,
            batch: OffPolicyReplayBatch | OffPolicyReplayEpisodeSegmentBatch,
            *,
            source_latents: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        raise NotImplementedError()

    @abc.abstractmethod
    def polyak_update_targets(self, tau: float) -> None:
        raise NotImplementedError()

    def act(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            scenario_ids: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
            deterministic: bool = False,
    ) -> torch.Tensor:
        action_kwargs = dict(
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            deterministic=deterministic,
            use_rsample=False,
        )
        if scenario_ids is not None:
            action_kwargs["scenario_ids"] = scenario_ids
        actions, _log_probs = self.action_log_prob(**action_kwargs)
        if agent_mask is not None:
            actions = actions.masked_fill(~agent_mask.unsqueeze(-1), 0.0)
        return actions

    def requires_previous_actions(self) -> bool:
        action_dist = getattr(self, "action_dist", None)
        requires_previous_actions = getattr(action_dist, "requires_previous_actions", None)
        return bool(callable(requires_previous_actions) and requires_previous_actions())

    def requires_recurrent_training(self) -> bool:
        return False

    def has_nop_loss(self) -> bool:
        return False

    def get_nop_num_next_steps(self) -> int:
        return 1

    def after_optimizer_step(self) -> None:
        pass
