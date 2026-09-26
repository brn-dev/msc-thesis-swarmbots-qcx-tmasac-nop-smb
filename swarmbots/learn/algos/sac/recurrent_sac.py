from dataclasses import dataclass, replace
from typing import Any

import torch
from loguru import logger

from swarmbots.learn.algos.off_policy.replay_buffer import (
    NoEpisodeSegmentCandidatesError,
    OffPolicyReplayBatch,
    OffPolicyReplayBuffer,
    OffPolicyReplayEpisodeSegmentBatch,
)
from swarmbots.learn.algos.sac.recurrent_tmasac_policy import (
    RecurrentCriticState,
    RecurrentTMASACPolicy,
)
from swarmbots.learn.algos.sac.sac import SAC, TrainStepResult
from swarmbots.learn.algos.sac.sac_nop import SACNOPSequenceBatch
from swarmbots.learn.algos.sac.segment_tmasac_policy import SegmentTMASACPolicy
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.temporal_state import (
    concatenate_temporal_states,
    detach_temporal_state,
)


class RecurrentSAC(SAC):
    policy: RecurrentTMASACPolicy | SegmentTMASACPolicy
    supports_recurrent_training = True

    def __init__(
            self,
            policy: RecurrentTMASACPolicy | SegmentTMASACPolicy,
            env: BaseLearnEnvWrapper,
            *,
            burn_in_steps: int = 32,
            learning_steps: int = 64,
            temporal_state_store_interval: int = 32,
            temporal_state_storage_dtype: torch.dtype | None = None,
            max_truncations_per_segment: int = 1,
            **kwargs: Any,
    ) -> None:
        if not isinstance(policy, (RecurrentTMASACPolicy, SegmentTMASACPolicy)):
            raise TypeError(
                "RecurrentSAC requires RecurrentTMASACPolicy or SegmentTMASACPolicy, got "
                f"{type(policy).__name__}"
            )
        self.burn_in_steps = int(burn_in_steps)
        self.learning_steps = int(learning_steps)
        self.temporal_state_store_interval = int(temporal_state_store_interval)
        self.temporal_state_storage_dtype = temporal_state_storage_dtype
        self.max_truncations_per_segment = int(max_truncations_per_segment)
        super().__init__(policy=policy, env=env, **kwargs)
        self.policy.configure_actor_compilation(
            encoder_only_sequence_lengths=(self.burn_in_steps,) if self.burn_in_steps > 0 else (),
            action_sequence_lengths=(1,),
            action_sequence_with_selected_states_lengths=(self.learning_steps,),
        )

    @property
    def replay_fill_target(self) -> int:
        sequence_fill_target = (
            (self.burn_in_steps + self.learning_steps)
            * self.env.action_space.n_envs
        )
        return max(super().replay_fill_target, sequence_fill_target)

    @property
    def _critic_uses_temporal_state(self) -> bool:
        return self.policy.recurrent_critic or getattr(
            self.policy,
            "uses_recurrent_shared_encoder",
            False,
        )

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            **super().get_hyper_parameters(),
            "burn_in_steps": self.burn_in_steps,
            "learning_steps": self.learning_steps,
            "compiled_actor_encoder_sequence_lengths": sorted(
                self.policy.compiled_actor_encoder_sequence_lengths
            ),
            "compiled_actor_sequence_lengths": sorted(self.policy.compiled_actor_sequence_lengths),
            "compiled_actor_selected_state_sequence_lengths": sorted(
                self.policy.compiled_actor_selected_state_sequence_lengths
            ),
            "max_truncations_per_segment": self.max_truncations_per_segment,
            "temporal_state_store_interval": self.temporal_state_store_interval,
            "temporal_state_storage_dtype": str(
                self.replay_buffer.temporal_state_storage_dtype
            ),
        }

    def train(self, *, gradient_steps: int) -> dict[str, Any]:
        try:
            return super().train(gradient_steps=gradient_steps)
        except NoEpisodeSegmentCandidatesError:
            logger.warning(
                "Skipping recurrent SAC updates because replay has no temporal-state-anchored "
                "stream segment of the requested length."
            )
            return {
                "updates": 0,
                "total_updates": self.n_total_updates,
                "replay_size": len(self.replay_buffer),
                "training_skipped": True,
            }

    def _build_replay_buffer(self) -> OffPolicyReplayBuffer:
        return OffPolicyReplayBuffer(
            capacity_per_env=self.buffer_capacity_per_env,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            store_previous_actions=self.policy.requires_previous_actions(),
            temporal_state_store_interval=self.temporal_state_store_interval,
            temporal_state_storage_dtype=self.temporal_state_storage_dtype,
            storage_device=self.replay_storage_device,
            storage_dtype=torch.float32,
            storage_pin_memory=self.replay_storage_pin_memory,
            train_device=self.train_device,
            train_dtype=torch.float32,
            compile_tensor_operations=self.replay_compile_tensor_operations,
        )

    def _sample_training_batches(
            self,
    ) -> tuple[
        OffPolicyReplayEpisodeSegmentBatch,
        OffPolicyReplayEpisodeSegmentBatch | None,
        bool,
    ]:
        batch = self.replay_buffer.sample_episode_segments(
            self.batch_size,
            segment_length=self.learning_steps,
            burn_in_steps=self.burn_in_steps,
            require_initial_temporal_state=True,
            allow_episode_boundaries=True,
            max_train_truncations=self.max_truncations_per_segment,
        )
        return batch, None, False

    def _train_step(
            self,
            batch: OffPolicyReplayEpisodeSegmentBatch,
            *,
            nop_batch: OffPolicyReplayEpisodeSegmentBatch | None = None,
            reuse_critic_nop_latents: bool = False,
            global_update_idx: int,
            materialize_metrics: bool = True,
    ) -> TrainStepResult:
        self._mark_cuda_graph_train_step_begin()
        _ = nop_batch
        _ = reuse_critic_nop_latents
        actor_critic_lr = self._apply_actor_critic_learning_rate_for_update(global_update_idx)
        learning_batch = _slice_segment(batch, self.burn_in_steps, batch.sequence_length)
        flat_batch = _flatten_segment(learning_batch)
        actor_state, critic_state, target_critic_state = self._burn_in_states(batch)
        truncation_indices, state_output_mask = self._padded_truncation_indices(
            learning_batch.truncations,
        )

        self._reset_train_gsde_noise(learning_batch.local_obs)
        (
            actions_pi,
            log_prob_pi,
            actor_latents,
            learning_next_actor_state,
            truncation_actor_states,
            last_layer_actor_state_sequence,
            current_shared_latents,
        ) = self._learning_sequence_actions(
            batch=learning_batch,
            initial_state=actor_state,
            state_output_indices=truncation_indices,
        )
        current_actor_state = None
        if getattr(self.policy, "uses_actor_state_critic_input", False):
            current_actor_state = self.policy.actor_last_layer_state_critic_input(
                last_layer_actor_state_sequence,
            )
        actor_action_dist_losses, actor_action_dist_metrics = self._compute_actor_action_dist_extra_losses(
            agent_mask=learning_batch.agent_mask,
        )
        actor_action_dist_losses = {
            name: _flatten_sequence_tensor(
                value,
                batch_size=learning_batch.actions.shape[0],
                sequence_length=learning_batch.actions.shape[1],
            )
            for name, value in actor_action_dist_losses.items()
        }
        reduced_actor_action_dist_losses = self._reduce_actor_action_dist_extra_losses(
            batch=flat_batch,
            extra_losses=actor_action_dist_losses,
        )

        log_prob_pi_mean = self._mean_agent_log_probs(log_prob_pi, learning_batch.agent_mask)
        ent_coef, ent_coef_loss = self._update_entropy_coefficient(
            log_prob_mean=self._mean_action_samples(
                log_prob_pi_mean, self.actor_action_samples,
            ).reshape(-1),
            batch=flat_batch,
        )
        target_entropy = self._target_entropy(
            batch=flat_batch,
            dtype=log_prob_pi_mean.dtype,
            device=log_prob_pi_mean.device,
        )

        with torch.no_grad():
            bootstrap_batch = self._mask_terminal_next_observations(learning_batch)
            next_actions, next_log_probs, next_actor_state = self._next_policy_actions(
                batch=bootstrap_batch,
                actions_pi=actions_pi,
                log_prob_pi=log_prob_pi,
                current_actor_state=current_actor_state,
                next_actor_state=detach_temporal_state(learning_next_actor_state),
                truncation_actor_states=detach_temporal_state(truncation_actor_states),
                truncation_indices=truncation_indices,
                truncation_mask=state_output_mask,
                return_actor_state=True,
                actor_latents=(
                    actor_latents.detach()
                    if max(self.actor_action_samples, self.target_action_samples) > 1
                    else None
                ),
            )
            next_log_prob_mean = self._mean_agent_log_probs(
                next_log_probs, bootstrap_batch.next_agent_mask,
            )
            target_q1, target_q2 = self._target_next_q_values(
                batch=bootstrap_batch,
                next_actions=next_actions,
                initial_state=target_critic_state,
                actor_state=next_actor_state,
            )
            target_q = self._tensor_operations.bellman_target(
                learning_batch.rewards,
                learning_batch.terminal_mask,
                target_q1,
                target_q2,
                next_log_prob_mean,
                ent_coef,
                self.gamma,
            )
            target_q = self._mean_action_samples(target_q, self.target_action_samples)

        current_q1, current_q2, critic_nop_latents, _next_critic_state = self.policy.q_values_sequence(
            local_obs=learning_batch.local_obs,
            global_obs=learning_batch.global_obs,
            actions=learning_batch.actions,
            hidden_local_vars=learning_batch.hidden_local_vars,
            hidden_global_vars=learning_batch.hidden_global_vars,
            agent_mask=learning_batch.agent_mask,
            scenario_ids=learning_batch.scenario_ids,
            target=False,
            initial_state=critic_state,
            time_mask=learning_batch.train_mask,
            reset_mask=learning_batch.episode_start_mask,
            **({} if current_actor_state is None else {"actor_state": current_actor_state}),
            **(
                {}
                if current_shared_latents is None
                else {
                    "precomputed_shared_latents": current_shared_latents,
                    "precomputed_shared_state": learning_next_actor_state,
                }
            ),
        )
        critic_loss = self._tensor_operations.critic_loss(
            current_q1,
            current_q2,
            target_q,
        )
        nop_training_batch = self._build_nop_training_batch(
            batch=learning_batch,
            actor_latents=actor_latents,
            critic_latents=critic_nop_latents,
        )
        if nop_training_batch is None:
            critic_nop_loss, critic_nop_metrics = None, {}
        else:
            critic_nop_loss, critic_nop_metrics = self.policy.compute_critic_nop_loss_from_latents(
                source_latents=nop_training_batch.critic_source_latents,
                batch=nop_training_batch.batch,
            )
        critic_total_loss = critic_loss if critic_nop_loss is None else critic_loss + critic_nop_loss

        self.critic_optimizer.zero_grad()
        critic_total_loss.backward()
        critic_grad_norm = self._clip_grad_norm(self.policy.critic_parameters())
        self._step_actor_or_critic_optimizer(
            optimizer=self.critic_optimizer,
            compiled_step=self._critic_optimizer_step,
            global_update_idx=global_update_idx,
        )

        actor_critic_state = critic_state
        if self._critic_uses_temporal_state:
            actor_critic_state = self._burn_in_critic_state(batch, target=False)

        critic_parameters = self.policy.critic_parameters()
        self._set_requires_grad(critic_parameters, False)
        try:
            q1_pi, q2_pi = self._actor_q_values(
                batch=learning_batch,
                actions_pi=actions_pi,
                initial_state=actor_critic_state,
                actor_state=current_actor_state,
            )
            actor_loss = self._tensor_operations.actor_loss(
                q1_pi,
                q2_pi,
                log_prob_pi_mean.reshape_as(q1_pi),
                ent_coef,
            )
        finally:
            self._set_requires_grad(critic_parameters, True)

        if (
                nop_training_batch is None
                or nop_training_batch.actor_source_latents is None
        ):
            actor_nop_loss, actor_nop_metrics = None, {}
        else:
            actor_nop_loss, actor_nop_metrics = self.policy.compute_actor_nop_loss_from_latents(
                source_latents=nop_training_batch.actor_source_latents,
                batch=nop_training_batch.batch,
            )
        actor_total_loss = actor_loss if actor_nop_loss is None else actor_loss + actor_nop_loss
        if reduced_actor_action_dist_losses:
            actor_total_loss = actor_total_loss + torch.stack(
                tuple(reduced_actor_action_dist_losses.values())
            ).sum()

        self.actor_optimizer.zero_grad()
        actor_total_loss.backward()
        actor_grad_norm = self._clip_grad_norm(self.policy.actor_parameters())
        self._step_actor_or_critic_optimizer(
            optimizer=self.actor_optimizer,
            compiled_step=self._actor_optimizer_step,
            global_update_idx=global_update_idx,
        )

        if global_update_idx % self.target_update_interval == 0:
            self.policy.polyak_update_targets(self.tau)
        self.policy.after_optimizer_step()

        metrics = {
            "critic_loss": critic_loss.detach(),
            "critic_total_loss": critic_total_loss.detach(),
            "actor_loss": actor_loss.detach(),
            "actor_total_loss": actor_total_loss.detach(),
            "target_q": target_q.mean().detach(),
            "current_q1": current_q1.mean().detach(),
            "current_q2": current_q2.mean().detach(),
            "q_pi": torch.minimum(q1_pi, q2_pi).mean().detach(),
            "log_prob": log_prob_pi_mean.mean().detach(),
            "entropy": (-log_prob_pi_mean).mean().detach(),
            "target_entropy": target_entropy.mean().detach(),
            "ent_coef": ent_coef.detach(),
            "actor_critic_learning_rate": actor_critic_lr,
        }
        if ent_coef_loss is not None:
            metrics["ent_coef_loss"] = ent_coef_loss.detach()
            metrics["ent_coef_learning_rate"] = self._resolved_ent_coef_learning_rate()
        metrics.update({
            f"actor_action_dist_{name}_loss_scaled": value.detach()
            for name, value in reduced_actor_action_dist_losses.items()
        })
        metrics.update({
            f"actor_action_dist_{name}": value
            for name, value in actor_action_dist_metrics.items()
        })
        metrics.update(actor_nop_metrics)
        metrics.update(critic_nop_metrics)
        result = (metrics, actor_grad_norm, critic_grad_norm)
        if not materialize_metrics:
            return self._preserve_train_step_result(result)
        return self._materialize_train_step_results([result])[0]

    def _learning_sequence_actions(
            self,
            *,
            batch: OffPolicyReplayEpisodeSegmentBatch,
            initial_state: Any,
            state_output_indices: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Any,
        Any,
        Any | None,
        torch.Tensor | None,
    ]:
        if getattr(self.policy, "uses_recurrent_shared_encoder", False):
            assert isinstance(self.policy, RecurrentTMASACPolicy)
            (
                actions,
                log_probs,
                actor_latents,
                next_state,
                selected_states,
                shared_latents,
            ) = self.policy.action_log_prob_sequence_with_selected_states_and_shared_latents(
                local_obs=batch.local_obs,
                global_obs=batch.global_obs,
                agent_mask=batch.agent_mask,
                scenario_ids=batch.scenario_ids,
                previous_actions=batch.previous_actions,
                deterministic=False,
                use_rsample=True,
                num_action_samples=self.actor_action_samples,
                action_sample_strategy=self.action_sample_strategy,
                initial_state=initial_state,
                state_output_indices=state_output_indices,
                time_mask=batch.train_mask,
                reset_mask=batch.episode_start_mask,
            )
            return (
                actions,
                log_probs,
                actor_latents,
                next_state,
                selected_states,
                None,
                shared_latents,
            )

        result = self.policy.action_log_prob_sequence_with_selected_states(
            local_obs=batch.local_obs,
            global_obs=batch.global_obs,
            agent_mask=batch.agent_mask,
            scenario_ids=batch.scenario_ids,
            previous_actions=batch.previous_actions,
            deterministic=False,
            use_rsample=True,
            num_action_samples=self.actor_action_samples,
            action_sample_strategy=self.action_sample_strategy,
            initial_state=initial_state,
            state_output_indices=state_output_indices,
            time_mask=batch.train_mask,
            reset_mask=batch.episode_start_mask,
        )
        return (*result, None)

    def _burn_in_states(
            self,
            batch: OffPolicyReplayEpisodeSegmentBatch,
    ) -> tuple[Any, RecurrentCriticState | None, RecurrentCriticState | None]:
        actor_state = batch.initial_temporal_state
        critic_state = None
        target_critic_state = None
        if self._critic_uses_temporal_state:
            critic_state = self._burn_in_critic_state(batch, target=False)
            target_critic_state = self._burn_in_critic_state(batch, target=True)
        if self.burn_in_steps == 0:
            return actor_state, critic_state, target_critic_state

        if not self.policy.uses_temporal_actor_state:
            return actor_state, critic_state, target_critic_state

        burn_in_batch = _slice_segment(batch, 0, self.burn_in_steps)
        with torch.no_grad():
            _latents, actor_state = self.policy.encode_actor_sequence(
                local_obs=burn_in_batch.local_obs,
                global_obs=burn_in_batch.global_obs,
                agent_mask=burn_in_batch.agent_mask,
                scenario_ids=burn_in_batch.scenario_ids,
                initial_state=actor_state,
                time_mask=torch.ones_like(burn_in_batch.train_mask),
                reset_mask=burn_in_batch.episode_start_mask,
            )
        actor_state = detach_temporal_state(actor_state)
        if getattr(self.policy, "uses_recurrent_shared_encoder", False):
            critic_state = actor_state
        return actor_state, critic_state, target_critic_state

    def _burn_in_critic_state(
            self,
            batch: OffPolicyReplayEpisodeSegmentBatch,
            *,
            target: bool,
    ) -> RecurrentCriticState | None:
        if getattr(self.policy, "uses_recurrent_shared_encoder", False):
            critic_state = batch.initial_temporal_state
        else:
            critic_state = self.policy.initial_critic_state(
                batch_size=batch.actions.shape[0],
                n_agents=batch.actions.shape[2],
                device=batch.actions.device,
                dtype=batch.actions.dtype,
                target=target,
            )
        if not self._critic_uses_temporal_state or self.burn_in_steps == 0:
            return critic_state
        burn_in_batch = _slice_segment(batch, 0, self.burn_in_steps)
        with torch.no_grad():
            _q1, _q2, _latents, critic_state = self.policy.q_values_sequence(
                local_obs=burn_in_batch.local_obs,
                global_obs=burn_in_batch.global_obs,
                actions=burn_in_batch.actions,
                hidden_local_vars=burn_in_batch.hidden_local_vars,
                hidden_global_vars=burn_in_batch.hidden_global_vars,
                agent_mask=burn_in_batch.agent_mask,
                scenario_ids=burn_in_batch.scenario_ids,
                target=target,
                initial_state=critic_state,
                time_mask=torch.ones_like(burn_in_batch.train_mask),
                reset_mask=burn_in_batch.episode_start_mask,
            )
        return detach_temporal_state(critic_state)

    def _next_policy_actions(
            self,
            *,
            batch: OffPolicyReplayEpisodeSegmentBatch,
            actions_pi: torch.Tensor,
            log_prob_pi: torch.Tensor,
            next_actor_state: Any,
            truncation_actor_states: Any,
            truncation_indices: torch.Tensor,
            truncation_mask: torch.Tensor,
            current_actor_state: torch.Tensor | None = None,
            return_actor_state: bool = False,
            actor_latents: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        target_inputs = _final_and_truncation_actor_inputs(
            batch=batch,
            next_actor_state=next_actor_state,
            truncation_actor_states=truncation_actor_states,
            truncation_indices=truncation_indices,
        )

        self._reset_train_gsde_noise(target_inputs["local_obs"])
        if actor_latents is not None:
            target_latents, target_next_actor_state = self.policy.encode_actor_sequence(**target_inputs)
            next_latents = _successor_sequence(
                actor_latents, target_latents, truncation_indices, truncation_mask,
            )
            next_actions, next_log_probs = self.policy.action_log_prob_from_latents(
                actor_latents=next_latents,
                agent_mask=batch.next_agent_mask,
                previous_actions=batch.actions,
                use_rsample=False,
                num_action_samples=self.target_action_samples,
                action_sample_strategy=self.action_sample_strategy,
            )
        else:
            target_actions, target_log_probs, _latents, target_next_actor_state = self.policy.action_log_prob_sequence(
                **target_inputs,
                previous_actions=_final_and_truncation_rows(batch.actions, truncation_indices),
                deterministic=False,
                use_rsample=False,
            )
            next_actions = _successor_sequence(
                actions_pi.detach(), target_actions, truncation_indices, truncation_mask,
            )
            next_log_probs = _successor_sequence(
                log_prob_pi.detach(), target_log_probs, truncation_indices, truncation_mask,
            )
        if not return_actor_state:
            return next_actions, next_log_probs

        next_actor_state_input = None
        if current_actor_state is not None:
            next_actor_state_input = _successor_sequence(
                current_actor_state,
                self.policy.actor_state_critic_input(target_next_actor_state),
                truncation_indices,
                truncation_mask,
            )
        return next_actions, next_log_probs, next_actor_state_input

    def _padded_truncation_indices(
            self,
            truncations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = truncations.shape[0]
        batch_indices = torch.arange(batch_size, dtype=torch.long, device=truncations.device)
        time_indices = truncations.to(dtype=torch.long).argmax(dim=1)
        indices = torch.stack((batch_indices, time_indices), dim=1)
        return indices, truncations.any(dim=1)

    def _target_next_q_values(
            self,
            *,
            batch: OffPolicyReplayEpisodeSegmentBatch,
            next_actions: torch.Tensor,
            initial_state: RecurrentCriticState | None,
            actor_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_action_samples = next_actions.shape[0] if next_actions.ndim == batch.actions.ndim + 1 else 1
        if not self._critic_uses_temporal_state:
            q1, q2, _latents, _state = self.policy.q_values_sequence(
                num_action_samples=num_action_samples,
                local_obs=batch.next_local_obs,
                global_obs=batch.next_global_obs,
                actions=next_actions,
                hidden_local_vars=batch.next_hidden_local_vars,
                hidden_global_vars=batch.next_hidden_global_vars,
                agent_mask=batch.next_agent_mask,
                scenario_ids=batch.next_scenario_ids,
                target=True,
                **({} if actor_state is None else {"actor_state": actor_state}),
            )
            return q1, q2

        history_state = initial_state
        q1_values: list[torch.Tensor] = []
        q2_values: list[torch.Tensor] = []
        for time_idx in range(batch.sequence_length):
            _q1, _q2, _latents, history_state = self.policy.q_values_sequence(
                local_obs=batch.local_obs[:, time_idx],
                global_obs=batch.global_obs[:, time_idx],
                actions=batch.actions[:, time_idx],
                hidden_local_vars=batch.hidden_local_vars[:, time_idx],
                hidden_global_vars=batch.hidden_global_vars[:, time_idx],
                agent_mask=None if batch.agent_mask is None else batch.agent_mask[:, time_idx],
                scenario_ids=(
                    None
                    if batch.scenario_ids is None
                    else batch.scenario_ids[:, time_idx]
                ),
                target=True,
                initial_state=history_state,
                reset_mask=(
                    None
                    if batch.episode_start_mask is None
                    else batch.episode_start_mask[:, time_idx]
                ),
            )
            q1, q2, _latents, _branch_state = self.policy.q_values_sequence(
                num_action_samples=num_action_samples,
                local_obs=batch.next_local_obs[:, time_idx],
                global_obs=batch.next_global_obs[:, time_idx],
                actions=next_actions[:, :, time_idx] if num_action_samples > 1 else next_actions[:, time_idx],
                hidden_local_vars=batch.next_hidden_local_vars[:, time_idx],
                hidden_global_vars=batch.next_hidden_global_vars[:, time_idx],
                agent_mask=(
                    None
                    if batch.next_agent_mask is None
                    else batch.next_agent_mask[:, time_idx]
                ),
                scenario_ids=(
                    None
                    if batch.next_scenario_ids is None
                    else batch.next_scenario_ids[:, time_idx]
                ),
                target=True,
                initial_state=history_state,
            )
            q1_values.append(q1)
            q2_values.append(q2)
        time_dim = 2 if num_action_samples > 1 else 1
        return torch.stack(q1_values, dim=time_dim), torch.stack(q2_values, dim=time_dim)

    def _actor_q_values(
            self,
            *,
            batch: OffPolicyReplayEpisodeSegmentBatch,
            actions_pi: torch.Tensor,
            initial_state: RecurrentCriticState | None,
            actor_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_action_samples = actions_pi.shape[0] if actions_pi.ndim == batch.actions.ndim + 1 else 1
        if not self._critic_uses_temporal_state:
            q1, q2, _latents, _state = self.policy.q_values_sequence(
                num_action_samples=num_action_samples,
                local_obs=batch.local_obs,
                global_obs=batch.global_obs,
                actions=actions_pi,
                hidden_local_vars=batch.hidden_local_vars,
                hidden_global_vars=batch.hidden_global_vars,
                agent_mask=batch.agent_mask,
                scenario_ids=batch.scenario_ids,
                target=False,
                **({} if actor_state is None else {"actor_state": actor_state}),
            )
            return q1, q2

        history_state = initial_state
        q1_values: list[torch.Tensor] = []
        q2_values: list[torch.Tensor] = []
        for time_idx in range(batch.sequence_length):
            reset_mask = (
                None
                if batch.episode_start_mask is None
                else batch.episode_start_mask[:, time_idx]
            )
            q1, q2, _latents, branch_state = self.policy.q_values_sequence(
                num_action_samples=num_action_samples,
                local_obs=batch.local_obs[:, time_idx],
                global_obs=batch.global_obs[:, time_idx],
                actions=actions_pi[:, :, time_idx] if num_action_samples > 1 else actions_pi[:, time_idx],
                hidden_local_vars=batch.hidden_local_vars[:, time_idx],
                hidden_global_vars=batch.hidden_global_vars[:, time_idx],
                agent_mask=None if batch.agent_mask is None else batch.agent_mask[:, time_idx],
                scenario_ids=(
                    None
                    if batch.scenario_ids is None
                    else batch.scenario_ids[:, time_idx]
                ),
                target=False,
                initial_state=history_state,
                reset_mask=reset_mask,
            )
            if getattr(self.policy, "uses_recurrent_shared_encoder", False):
                history_state = branch_state
            else:
                _q1, _q2, _latents, history_state = self.policy.q_values_sequence(
                    local_obs=batch.local_obs[:, time_idx],
                    global_obs=batch.global_obs[:, time_idx],
                    actions=batch.actions[:, time_idx],
                    hidden_local_vars=batch.hidden_local_vars[:, time_idx],
                    hidden_global_vars=batch.hidden_global_vars[:, time_idx],
                    agent_mask=None if batch.agent_mask is None else batch.agent_mask[:, time_idx],
                    scenario_ids=(
                        None
                        if batch.scenario_ids is None
                        else batch.scenario_ids[:, time_idx]
                    ),
                    target=False,
                    initial_state=history_state,
                    reset_mask=reset_mask,
                )
            q1_values.append(q1)
            q2_values.append(q2)
        time_dim = 2 if num_action_samples > 1 else 1
        return torch.stack(q1_values, dim=time_dim), torch.stack(q2_values, dim=time_dim)

    def _build_nop_training_batch(
            self,
            *,
            batch: OffPolicyReplayEpisodeSegmentBatch,
            actor_latents: torch.Tensor,
            critic_latents: torch.Tensor | None,
    ) -> "_RecurrentNOPTrainingBatch | None":
        if not self.policy.has_nop_loss():
            return None
        num_next_steps = self.policy.get_nop_num_next_steps()
        num_origins = batch.sequence_length - num_next_steps + 1
        nop_modules = (self.policy.actor_nop, self.policy.critic_nop)
        nop_batch = _parallel_nop_windows(
            batch,
            window_length=num_next_steps,
            include_global_obs=any(
                module is not None and module.has_global_next_obs_pred_targets
                for module in nop_modules
            ),
        )
        return _RecurrentNOPTrainingBatch(
            batch=nop_batch,
            actor_source_latents=(
                None
                if self.policy.actor_nop is None
                else actor_latents[:, :num_origins]
            ),
            critic_source_latents=(
                None
                if critic_latents is None
                else critic_latents[:, :num_origins]
            ),
        )

    def _mask_terminal_next_observations(
            self,
            batch: OffPolicyReplayEpisodeSegmentBatch,
    ) -> OffPolicyReplayEpisodeSegmentBatch:
        (
            next_local_obs,
            next_global_obs,
            next_hidden_local_vars,
            next_hidden_global_vars,
            next_agent_mask,
        ) = self._tensor_operations.mask_terminal_observations(
            batch.terminal_mask,
            batch.next_local_obs,
            batch.next_global_obs,
            batch.next_hidden_local_vars,
            batch.next_hidden_global_vars,
            batch.next_agent_mask,
        )
        return replace(
            batch,
            next_local_obs=next_local_obs,
            next_global_obs=next_global_obs,
            next_hidden_local_vars=next_hidden_local_vars,
            next_hidden_global_vars=next_hidden_global_vars,
            next_agent_mask=next_agent_mask,
        )

    def _validate_hyper_parameters(self) -> None:
        super()._validate_hyper_parameters()
        if self.burn_in_steps < 0:
            raise ValueError(f"burn_in_steps must be >= 0, got {self.burn_in_steps}")
        if self.learning_steps <= 0:
            raise ValueError(f"learning_steps must be > 0, got {self.learning_steps}")
        if self.temporal_state_store_interval <= 0:
            raise ValueError(
                "temporal_state_store_interval must be > 0, got "
                f"{self.temporal_state_store_interval}"
            )
        if self.learning_steps < self.temporal_state_store_interval:
            raise ValueError(
                "learning_steps must be >= temporal_state_store_interval so checkpoint-anchored "
                "segments cover every replay transition; "
                f"got learning_steps={self.learning_steps}, "
                f"temporal_state_store_interval={self.temporal_state_store_interval}"
            )
        if self.max_truncations_per_segment != 1:
            raise ValueError(
                "Only one truncation per recurrent segment is supported; "
                "max_truncations_per_segment must be 1, got "
                f"{self.max_truncations_per_segment}"
            )
        minimum_buffer_capacity_per_env = (
            self.burn_in_steps
            + self.learning_steps
            + self.temporal_state_store_interval
            - 1
        )
        if self.buffer_capacity_per_env < minimum_buffer_capacity_per_env:
            raise ValueError(
                "buffer_capacity_per_env must be at least burn_in_steps + learning_steps + "
                "temporal_state_store_interval - 1 so a full replay ring always contains a "
                "checkpoint-anchored training segment; got "
                f"{self.buffer_capacity_per_env}, need at least {minimum_buffer_capacity_per_env}"
            )
        if self.independent_nop_sampling:
            raise ValueError("RecurrentSAC does not support independent_nop_sampling=True.")
        if (
                self.policy.has_nop_loss()
                and self.policy.get_nop_num_next_steps() > self.learning_steps
        ):
            raise ValueError(
                "Recurrent TMASAC NOP num_next_steps cannot exceed learning_steps; "
                "NOP reuses the sampled learning sequence."
            )


def _final_and_truncation_rows(
        tensor: torch.Tensor | None,
        truncation_indices: torch.Tensor,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    rows, times = truncation_indices.unbind(-1)
    return torch.cat((tensor[:, -1], tensor[rows, times]))


def _final_and_truncation_actor_inputs(
        *,
        batch: OffPolicyReplayEpisodeSegmentBatch,
        next_actor_state: Any,
        truncation_actor_states: Any,
        truncation_indices: torch.Tensor,
) -> dict[str, Any]:
    """Pack the final successor and one padded truncation successor per replay row."""
    return {
        "local_obs": _final_and_truncation_rows(batch.next_local_obs, truncation_indices),
        "global_obs": _final_and_truncation_rows(batch.next_global_obs, truncation_indices),
        "agent_mask": _final_and_truncation_rows(batch.next_agent_mask, truncation_indices),
        "scenario_ids": _final_and_truncation_rows(batch.next_scenario_ids, truncation_indices),
        "initial_state": concatenate_temporal_states((next_actor_state, truncation_actor_states)),
    }


def _successor_sequence(
        current: torch.Tensor,
        final_and_truncation: torch.Tensor,
        truncation_indices: torch.Tensor,
        truncation_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size = current.shape[0]
    successor = torch.cat((current[:, 1:], final_and_truncation[:batch_size].unsqueeze(1)), dim=1)
    rows, times = truncation_indices.unbind(-1)
    mask = truncation_mask.reshape(batch_size, *((1,) * (current.ndim - 2)))
    successor[rows, times] = torch.where(mask, final_and_truncation[batch_size:], successor[rows, times])
    return successor


def _slice_segment(
        batch: OffPolicyReplayEpisodeSegmentBatch,
        start: int,
        end: int,
) -> OffPolicyReplayEpisodeSegmentBatch:
    def slice_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
        return None if tensor is None else tensor[:, start:end]

    return OffPolicyReplayEpisodeSegmentBatch(
        local_obs=batch.local_obs[:, start:end],
        global_obs=batch.global_obs[:, start:end],
        hidden_local_vars=batch.hidden_local_vars[:, start:end],
        hidden_global_vars=batch.hidden_global_vars[:, start:end],
        agent_mask=slice_tensor(batch.agent_mask),
        actions=batch.actions[:, start:end],
        rewards=batch.rewards[:, start:end],
        terminations=batch.terminations[:, start:end],
        truncations=batch.truncations[:, start:end],
        previous_actions=slice_tensor(batch.previous_actions),
        next_local_obs=batch.next_local_obs[:, start:end],
        next_global_obs=batch.next_global_obs[:, start:end],
        next_hidden_local_vars=batch.next_hidden_local_vars[:, start:end],
        next_hidden_global_vars=batch.next_hidden_global_vars[:, start:end],
        next_agent_mask=slice_tensor(batch.next_agent_mask),
        episode_start_mask=slice_tensor(batch.episode_start_mask),
        train_mask=batch.train_mask[:, start:end],
        initial_temporal_state=batch.initial_temporal_state,
        burn_in_steps=0,
        scenario_ids=slice_tensor(batch.scenario_ids),
        next_scenario_ids=slice_tensor(batch.next_scenario_ids),
    )


@dataclass(frozen=True, slots=True)
class _RecurrentNOPTrainingBatch:
    batch: SACNOPSequenceBatch
    actor_source_latents: torch.Tensor | None
    critic_source_latents: torch.Tensor | None


def _parallel_nop_windows(
        batch: OffPolicyReplayEpisodeSegmentBatch,
        *,
        window_length: int,
        include_global_obs: bool,
) -> SACNOPSequenceBatch:
    sequence_length = batch.sequence_length
    num_origins = sequence_length - window_length + 1

    def parallel_windows(tensor: torch.Tensor) -> torch.Tensor:
        unfolded = tensor.unfold(dimension=1, size=window_length, step=1)
        time_dimension = unfolded.ndim - 1
        dimension_order = (0, 1, time_dimension, *range(2, time_dimension))
        return unfolded.permute(dimension_order)

    def optional_parallel_windows(tensor: torch.Tensor | None) -> torch.Tensor | None:
        return None if tensor is None else parallel_windows(tensor)

    episode_end_windows = parallel_windows(batch.episode_ends)
    prior_episode_ends = torch.cat((
        torch.zeros_like(episode_end_windows[..., :1]),
        episode_end_windows[..., :-1],
    ), dim=-1)
    within_origin_episode = prior_episode_ends.to(dtype=torch.long).cumsum(dim=-1) == 0

    return SACNOPSequenceBatch(
        local_obs=batch.local_obs[:, :num_origins],
        global_obs=(
            batch.global_obs[:, :num_origins]
            if include_global_obs
            else None
        ),
        agent_mask=optional_parallel_windows(batch.agent_mask),
        actions=parallel_windows(batch.actions),
        next_local_obs=parallel_windows(batch.next_local_obs),
        next_global_obs=(
            parallel_windows(batch.next_global_obs)
            if include_global_obs
            else None
        ),
        next_agent_mask=optional_parallel_windows(batch.next_agent_mask),
        train_mask=parallel_windows(batch.train_mask) & within_origin_episode,
    )


def _flatten_segment(batch: OffPolicyReplayEpisodeSegmentBatch) -> OffPolicyReplayBatch:
    batch_size, sequence_length = batch.actions.shape[:2]

    def flatten(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        return tensor.reshape(batch_size * sequence_length, *tensor.shape[2:])

    return OffPolicyReplayBatch(
        local_obs=flatten(batch.local_obs),
        global_obs=flatten(batch.global_obs),
        hidden_local_vars=flatten(batch.hidden_local_vars),
        hidden_global_vars=flatten(batch.hidden_global_vars),
        agent_mask=flatten(batch.agent_mask),
        actions=flatten(batch.actions),
        rewards=flatten(batch.rewards),
        terminations=flatten(batch.terminations),
        truncations=flatten(batch.truncations),
        previous_actions=flatten(batch.previous_actions),
        next_local_obs=flatten(batch.next_local_obs),
        next_global_obs=flatten(batch.next_global_obs),
        next_hidden_local_vars=flatten(batch.next_hidden_local_vars),
        next_hidden_global_vars=flatten(batch.next_hidden_global_vars),
        next_agent_mask=flatten(batch.next_agent_mask),
        episode_start_mask=flatten(batch.episode_start_mask),
        scenario_ids=flatten(batch.scenario_ids),
        next_scenario_ids=flatten(batch.next_scenario_ids),
    )


def _flatten_sequence_tensor(
        tensor: torch.Tensor,
        *,
        batch_size: int,
        sequence_length: int,
) -> torch.Tensor:
    if tensor.ndim < 2 or tensor.shape[:2] != (batch_size, sequence_length):
        return tensor
    return tensor.reshape(batch_size * sequence_length, *tensor.shape[2:])
