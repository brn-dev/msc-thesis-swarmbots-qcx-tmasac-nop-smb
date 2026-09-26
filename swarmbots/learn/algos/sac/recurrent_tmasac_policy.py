from collections.abc import Collection
from dataclasses import dataclass, field, replace
from typing import Any, Callable, TypeVar, cast

import torch
from torch import nn
from torch._dynamo import config as torch_dynamo_config

from swarmbots.learn.algos.mat.mat_encoder import MATEncoder, MATEncoderConfig
from swarmbots.learn.algos.off_policy.replay_buffer import (
    OffPolicyReplayBatch,
    OffPolicyReplayEpisodeSegmentBatch,
)
from swarmbots.learn.algos.r_mat.r_mat_encoder import RMATEncoder, RMATEncoderConfig, RMATEncoderState
from swarmbots.learn.algos.r_mat.temporal_sequence_model import LSTMTemporalSequenceModel
from swarmbots.learn.algos.sac.sac_nop import SACNOPLatentSource, SACNOPSequenceBatch
from swarmbots.learn.action_dists.action_sampling import ActionSampleStrategy, expand_action_sample_batch
from swarmbots.learn.algos.sac.tmasac_policy import (
    TMASACCriticConfig,
    TMASACObservationActionEncoder,
    TMASACPolicy,
    TMASACPolicyConfig,
    TMASACTwinCritic,
)
from swarmbots.learn.algos.xlstm.slstm import SLSTMTemporalSequenceModel
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.nn_components.feed_forward import MLP, MLPConfig
from swarmbots.learn.nn_components.nn_init import make_init_linear_orthogonal
from swarmbots.learn.serialization_utils import serialize_dataclass
from swarmbots.learn.temporal_state import concatenate_temporal_states


RecurrentCriticState = RMATEncoderState | tuple[RMATEncoderState, RMATEncoderState]
ActorEncoderCallable = Callable[
    ...,
    tuple[torch.Tensor, RMATEncoderState]
    | tuple[torch.Tensor, RMATEncoderState, RMATEncoderState]
    | tuple[torch.Tensor, RMATEncoderState, RMATEncoderState, Any],
]
ActorActionSequenceCallable = Callable[
    ...,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, RMATEncoderState],
]
ActorActionSequenceWithSelectedStatesCallable = Callable[
    ...,
    tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        RMATEncoderState,
        RMATEncoderState,
        Any | None,
    ],
]
ActorActionsAndLogProbsCallable = Callable[
    ...,
    tuple[torch.Tensor, torch.Tensor],
]
ActorEntryPointT = TypeVar("ActorEntryPointT", bound=Callable[..., Any])


@dataclass(frozen=True)
class ActorStateCriticInputConfig:
    projection_dim: int | None = None
    projection_hidden_dims: tuple[int, ...] | None = None
    include_slstm_memory_strength: bool = False
    init_gain: float = 1.0
    output_init_gain: float = 1.0


@dataclass(frozen=True)
class RecurrentTMASACPolicyConfig(TMASACPolicyConfig):
    actor_encoder_config: MATEncoderConfig = field(default_factory=RMATEncoderConfig)
    recurrent_critic: bool = False
    recurrent_shared_encoder: bool = False
    actor_state_critic_input_config: ActorStateCriticInputConfig | None = None
    experimental_compile_lstm: bool = False


class ActorStateTMASACTwinCritic(TMASACTwinCritic):
    def __init__(
            self,
            *,
            actor_state_input_dim: int,
            actor_state_config: ActorStateCriticInputConfig,
            actor_state_default_projection_dim: int,
            **kwargs: Any,
    ) -> None:
        projection_dim = (
            actor_state_default_projection_dim
            if actor_state_config.projection_dim is None
            else int(actor_state_config.projection_dim)
        )
        hidden_dims = (
            (projection_dim,)
            if actor_state_config.projection_hidden_dims is None
            else actor_state_config.projection_hidden_dims
        )
        local_input_dim = int(kwargs.pop("local_input_dim"))
        super().__init__(local_input_dim=local_input_dim + projection_dim, **kwargs)
        self.actor_state_encoder = MLP(
            input_dim=actor_state_input_dim,
            hidden_dims=[*hidden_dims, projection_dim],
            end_with_act_fn=False,
            linear_init=make_init_linear_orthogonal(actor_state_config.init_gain),
            final_linear_init=make_init_linear_orthogonal(actor_state_config.output_init_gain),
            act_fn_cls=kwargs["act_fn_cls"],
        )

    def encode(
            self,
            *,
            actor_state: torch.Tensor,
            local_inputs: torch.Tensor,
            **kwargs: Any,
    ) -> torch.Tensor:
        return super().encode(
            local_inputs=self._append_actor_state(local_inputs, actor_state),
            **kwargs,
        )

    def forward(
            self,
            *,
            actor_state: torch.Tensor,
            local_obs: torch.Tensor,
            **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return super().forward(
            local_obs=self._append_actor_state(local_obs, actor_state),
            **kwargs,
        )

    def _append_actor_state(self, local_inputs: torch.Tensor, actor_state: torch.Tensor) -> torch.Tensor:
        projected_state = self.actor_state_encoder(actor_state.detach())
        return torch.cat((local_inputs, projected_state), dim=-1)


class RecurrentTMASACTwinCritic(TMASACTwinCritic):
    def __init__(
            self,
            *,
            n_agents: int,
            max_agents: int,
            local_input_dim: int,
            global_input_dim: int,
            hidden_local_vars_dim: int,
            hidden_global_vars_dim: int,
            action_dim: int,
            encoder_config: RMATEncoderConfig,
            critic_config: TMASACCriticConfig,
            dropout: float,
    ) -> None:
        nn.Module.__init__(self)
        self.n_agents = int(n_agents)
        self.hidden_local_vars_dim = int(hidden_local_vars_dim)
        self.hidden_global_vars_dim = int(hidden_global_vars_dim)
        self.action_dim = int(action_dim)
        self.d_model = int(encoder_config.d_model)
        self.encoder_config = replace(
            encoder_config,
            dropout=dropout,
            local_obs_encoder_config=MLPConfig(
                hidden_dims=(
                    [self.d_model]
                    if critic_config.action_coembed_hidden_dims is None
                    else [*critic_config.action_coembed_hidden_dims]
                )
            ),
            linear_init_gain=critic_config.action_coembed_init_gain,
            linear_projection_init_gain=critic_config.action_coembed_output_init_gain,
        )
        local_observation_input_dim = local_input_dim + self.hidden_local_vars_dim
        global_encoder_input_dim = global_input_dim + self.hidden_global_vars_dim

        def build_observation_action_encoder() -> TMASACObservationActionEncoder:
            return TMASACObservationActionEncoder(
                observation_input_dim=local_observation_input_dim,
                action_dim=self.action_dim,
                d_model=self.d_model,
                action_encoder_dim=critic_config.action_encoder_dim,
                linear_init_gain=critic_config.action_coembed_init_gain,
                act_fn_cls=encoder_config.act_fn_cls,
            )

        self.observation_action_encoder = (
            build_observation_action_encoder()
            if critic_config.separate_observation_action_encoders
            else None
        )
        local_encoder_input_dim = (
            local_observation_input_dim + self.action_dim
            if self.observation_action_encoder is None
            else self.observation_action_encoder.output_dim
        )

        def build_encoder() -> RMATEncoder:
            return RMATEncoder(
                config=self.encoder_config,
                max_agents=max_agents,
                local_obs_dim=local_encoder_input_dim,
                global_obs_dim=global_encoder_input_dim,
            )

        self.encoder = build_encoder()
        self.encoder2 = build_encoder() if critic_config.independent_encoders else None
        self.observation_action_encoder2 = (
            build_observation_action_encoder()
            if self.observation_action_encoder is not None and self.encoder2 is not None
            else None
        )
        self.nop_source_latent_dim = self.d_model * (2 if self.encoder2 is not None else 1)
        self.q1 = self._build_q_network(
            q_local_dim=self.d_model,
            critic_config=critic_config,
            act_fn_cls=encoder_config.act_fn_cls,
        )
        self.q2 = self._build_q_network(
            q_local_dim=self.d_model,
            critic_config=critic_config,
            act_fn_cls=encoder_config.act_fn_cls,
        )

    def initial_state(
            self,
            batch_size: int,
            n_agents: int,
            *,
            device: torch.device,
            dtype: torch.dtype,
    ) -> RecurrentCriticState:
        primary_state = self.encoder.initial_state(
            batch_size=batch_size,
            n_agents=n_agents,
            device=device,
            dtype=dtype,
        )
        if self.encoder2 is None:
            return primary_state
        return primary_state, self.encoder2.initial_state(
            batch_size=batch_size,
            n_agents=n_agents,
            device=device,
            dtype=dtype,
        )

    def encode(
            self,
            *,
            local_inputs: torch.Tensor,
            global_inputs: torch.Tensor,
            actions: torch.Tensor,
            hidden_local_vars: torch.Tensor | None,
            hidden_global_vars: torch.Tensor | None,
            agent_mask: torch.Tensor | None,
            time_mask: torch.Tensor | None = None,
            initial_state: RecurrentCriticState | None = None,
            reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, RecurrentCriticState]:
        primary_state, secondary_state = self._split_initial_state(initial_state)
        local_encoder_inputs, global_encoder_inputs = self._encoder_inputs(
            local_inputs=local_inputs,
            global_inputs=global_inputs,
            actions=actions,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            observation_action_encoder=self.observation_action_encoder,
        )
        primary_latents, next_primary_state = self.encoder(
            local_encoder_inputs,
            global_encoder_inputs,
            agent_mask=agent_mask,
            time_mask=time_mask,
            initial_state=primary_state,
            reset_mask=reset_mask,
        )
        if self.encoder2 is None:
            return primary_latents, next_primary_state
        secondary_local_encoder_inputs = local_encoder_inputs
        if self.observation_action_encoder2 is not None:
            secondary_local_encoder_inputs, _secondary_global_encoder_inputs = self._encoder_inputs(
                local_inputs=local_inputs,
                global_inputs=global_inputs,
                actions=actions,
                hidden_local_vars=hidden_local_vars,
                hidden_global_vars=hidden_global_vars,
                observation_action_encoder=self.observation_action_encoder2,
            )
        secondary_latents, next_secondary_state = self.encoder2(
            secondary_local_encoder_inputs,
            global_encoder_inputs,
            agent_mask=agent_mask,
            time_mask=time_mask,
            initial_state=secondary_state,
            reset_mask=reset_mask,
        )
        return torch.cat((primary_latents, secondary_latents), dim=-1), (
            next_primary_state,
            next_secondary_state,
        )

    def forward(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            actions: torch.Tensor,
            hidden_local_vars: torch.Tensor | None,
            hidden_global_vars: torch.Tensor | None,
            agent_mask: torch.Tensor | None,
            time_mask: torch.Tensor | None = None,
            initial_state: RecurrentCriticState | None = None,
            reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, RecurrentCriticState]:
        primary_state, secondary_state = self._split_initial_state(initial_state)
        local_encoder_inputs, global_encoder_inputs = self._encoder_inputs(
            local_inputs=local_obs,
            global_inputs=global_obs,
            actions=actions,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            observation_action_encoder=self.observation_action_encoder,
        )
        q1_latents, next_primary_state = self.encoder(
            local_encoder_inputs,
            global_encoder_inputs,
            agent_mask=agent_mask,
            time_mask=time_mask,
            initial_state=primary_state,
            reset_mask=reset_mask,
        )
        if self.encoder2 is None:
            q2_latents = q1_latents
            next_state: RecurrentCriticState = next_primary_state
        else:
            secondary_local_encoder_inputs = local_encoder_inputs
            if self.observation_action_encoder2 is not None:
                secondary_local_encoder_inputs, _secondary_global_encoder_inputs = self._encoder_inputs(
                    local_inputs=local_obs,
                    global_inputs=global_obs,
                    actions=actions,
                    hidden_local_vars=hidden_local_vars,
                    hidden_global_vars=hidden_global_vars,
                    observation_action_encoder=self.observation_action_encoder2,
                )
            q2_latents, next_secondary_state = self.encoder2(
                secondary_local_encoder_inputs,
                global_encoder_inputs,
                agent_mask=agent_mask,
                time_mask=time_mask,
                initial_state=secondary_state,
                reset_mask=reset_mask,
            )
            next_state = next_primary_state, next_secondary_state
        return (
            self._evaluate_q_head(self.q1, q1_latents, agent_mask),
            self._evaluate_q_head(self.q2, q2_latents, agent_mask),
            q1_latents if self.encoder2 is None else torch.cat((q1_latents, q2_latents), dim=-1),
            next_state,
        )

    def _encoder_inputs(
            self,
            *,
            local_inputs: torch.Tensor,
            global_inputs: torch.Tensor,
            actions: torch.Tensor,
            hidden_local_vars: torch.Tensor | None,
            hidden_global_vars: torch.Tensor | None,
            observation_action_encoder: TMASACObservationActionEncoder | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_observation_parts = [local_inputs]
        if self.hidden_local_vars_dim > 0:
            if hidden_local_vars is None:
                raise ValueError("hidden_local_vars must be provided when hidden_local_vars_dim > 0")
            local_observation_parts.append(hidden_local_vars)
        local_observation_inputs = (
            local_observation_parts[0]
            if len(local_observation_parts) == 1
            else torch.cat(local_observation_parts, dim=-1)
        )
        local_encoder_inputs = (
            torch.cat((local_observation_inputs, actions), dim=-1)
            if observation_action_encoder is None
            else observation_action_encoder(local_observation_inputs, actions)
        )

        global_parts = [global_inputs] if global_inputs.shape[-1] > 0 else []
        if self.hidden_global_vars_dim > 0:
            if hidden_global_vars is None:
                raise ValueError("hidden_global_vars must be provided when hidden_global_vars_dim > 0")
            global_parts.append(hidden_global_vars)
        if global_parts:
            global_encoder_inputs = global_parts[0] if len(global_parts) == 1 else torch.cat(global_parts, dim=-1)
        else:
            global_encoder_inputs = global_inputs
        return local_encoder_inputs, global_encoder_inputs

    def _load_from_state_dict(
            self,
            state_dict: dict[str, Any],
            prefix: str,
            local_metadata: dict[str, Any],
            strict: bool,
            missing_keys: list[str],
            unexpected_keys: list[str],
            error_msgs: list[str],
    ) -> None:
        if self.observation_action_encoder2 is not None:
            primary_prefix = f"{prefix}observation_action_encoder."
            secondary_prefix = f"{prefix}observation_action_encoder2."
            has_primary_encoder = any(key.startswith(primary_prefix) for key in state_dict)
            has_secondary_encoder = any(key.startswith(secondary_prefix) for key in state_dict)
            if has_primary_encoder and not has_secondary_encoder:
                for key, value in list(state_dict.items()):
                    if key.startswith(primary_prefix):
                        state_dict[f"{secondary_prefix}{key.removeprefix(primary_prefix)}"] = value
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _split_initial_state(
            self,
        state: RecurrentCriticState | None,
    ) -> tuple[RMATEncoderState | None, RMATEncoderState | None]:
        if self.encoder2 is None:
            return cast(RMATEncoderState | None, state), None
        if state is None:
            return None, None
        if not isinstance(state, tuple) or len(state) != 2:
            raise ValueError("Independent recurrent critic encoders require a pair of temporal states.")
        return state

    @staticmethod
    def _evaluate_q_head(
            q_head: nn.Module,
            latents: torch.Tensor,
            agent_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if latents.ndim == 3:
            return q_head(latents, agent_mask=agent_mask)
        batch_size, sequence_length, n_agents, latent_dim = latents.shape
        flat_mask = None if agent_mask is None else agent_mask.reshape(batch_size * sequence_length, n_agents)
        values = q_head(
            latents.reshape(batch_size * sequence_length, n_agents, latent_dim),
            agent_mask=flat_mask,
        )
        return values.reshape(batch_size, sequence_length)


class RecurrentTMASACPolicy(TMASACPolicy):
    config: RecurrentTMASACPolicyConfig
    _compiled_actor_encoders: dict[int, ActorEncoderCallable]
    _compiled_actor_action_sequences: dict[int, ActorActionSequenceCallable]
    _compiled_actor_action_sequences_with_selected_states: dict[
        int,
        ActorActionSequenceWithSelectedStatesCallable,
    ]
    _actor_encoder_compilation_enabled: bool
    _actor_end_to_end_compilation_enabled: bool
    _actor_encoder_uses_lstm: bool

    def __init__(
            self,
            env: BaseLearnEnvWrapper,
            config: RecurrentTMASACPolicyConfig = RecurrentTMASACPolicyConfig(),
    ) -> None:
        if config.recurrent_shared_encoder:
            if not isinstance(config.shared_encoder_config, RMATEncoderConfig):
                raise TypeError(
                    "recurrent_shared_encoder=True requires an RMATEncoderConfig for "
                    "shared_encoder_config."
                )
            if isinstance(config.actor_encoder_config, RMATEncoderConfig):
                raise TypeError(
                    "recurrent_shared_encoder=True requires a non-recurrent MATEncoderConfig "
                    "for actor_encoder_config."
                )
            if config.recurrent_critic:
                raise ValueError(
                    "recurrent_shared_encoder=True cannot be combined with recurrent_critic=True."
                )
            if config.actor_state_critic_input_config is not None:
                raise ValueError(
                    "The recurrent shared latent replaces actor_state_critic_input_config."
                )
        else:
            if not isinstance(config.actor_encoder_config, RMATEncoderConfig):
                raise TypeError(
                    "RecurrentTMASACPolicy requires an RMATEncoderConfig for actor_encoder_config."
                )
            if config.shared_encoder_config is not None or config.share_observation_encoder:
                raise ValueError(
                    "RecurrentTMASACPolicy only supports a shared observation encoder when "
                    "recurrent_shared_encoder=True."
                )
        if config.recurrent_critic and not isinstance(config.critic_encoder_config, RMATEncoderConfig):
            raise TypeError("recurrent_critic=True requires an RMATEncoderConfig for critic_encoder_config.")
        actor_state_config = config.actor_state_critic_input_config
        if actor_state_config is not None:
            if config.recurrent_critic:
                raise ValueError("actor_state_critic_input_config is not supported with recurrent_critic=True.")
            if actor_state_config.projection_dim is not None and actor_state_config.projection_dim < 1:
                raise ValueError("actor-state critic projection_dim must be >= 1.")
            if (
                    actor_state_config.projection_hidden_dims is not None
                    and any(hidden_dim < 1 for hidden_dim in actor_state_config.projection_hidden_dims)
            ):
                raise ValueError("actor-state critic projection_hidden_dims must all be >= 1.")
        object.__setattr__(self, "_compiled_actor_encoders", {})
        object.__setattr__(self, "_compiled_actor_action_sequences", {})
        object.__setattr__(self, "_compiled_actor_action_sequences_with_selected_states", {})
        object.__setattr__(self, "_actor_encoder_compilation_enabled", False)
        object.__setattr__(self, "_actor_end_to_end_compilation_enabled", False)
        object.__setattr__(self, "_actor_encoder_uses_lstm", False)
        super().__init__(env=env, config=config)

    def _apply_optional_compile(self) -> None:
        if not self.config.compile_modules:
            return
        if not hasattr(torch, "compile") or not callable(torch.compile):
            raise RuntimeError("RecurrentTMASACPolicyConfig.compile_modules=True requires torch.compile support.")
        if not self.config.compile_mode:
            raise ValueError(
                "RecurrentTMASACPolicyConfig.compile_mode must be a non-empty string when compile_modules=True."
            )

        recurrent_encoder = (
            self.shared_observation_encoder
            if self.uses_recurrent_shared_encoder
            else self._actor_encoder
        )
        assert recurrent_encoder is not None
        self._actor_encoder_uses_lstm = any(
            isinstance(module, nn.LSTM)
            for module in recurrent_encoder.modules()
        )
        self._actor_encoder_compilation_enabled = (
            not self._actor_encoder_uses_lstm
            or self.config.experimental_compile_lstm
        )
        self._actor_end_to_end_compilation_enabled = (
            self._actor_encoder_compilation_enabled
            and self.action_dist.compile_friendly
            and not self.uses_recurrent_shared_encoder
        )
        self.configure_actor_compilation(
            encoder_only_sequence_lengths=(),
            action_sequence_lengths=(1,),
            action_sequence_with_selected_states_lengths=(),
        )
        if self.uses_recurrent_shared_encoder:
            self._actor_encoder = self._compile_module(self._actor_encoder)
        if not self._actor_end_to_end_compilation_enabled:
            if self.action_dist.compile_friendly:
                self._compiled_actor_actions_and_log_probs = self._compile_actor_tail_callable(
                    self._actor_actions_and_log_probs_impl,
                )
            elif self.actor_head.supports_standalone_compile:
                self.actor_head = self._compile_module(self.actor_head)
        if not self.recurrent_critic:
            self.critic = self._compile_module(self.critic)
            self.critic_target = self._compile_module(self.critic_target)

    def _compile_actor_callable(
            self,
            fn: ActorEntryPointT,
            *,
            fullgraph: bool,
    ) -> ActorEntryPointT:
        def compile_callable() -> Callable[..., Any]:
            return torch.compile(
                fn,
                mode=self.config.compile_mode,
                fullgraph=fullgraph,
                dynamic=False,
            )

        if self._actor_encoder_uses_lstm and self.config.experimental_compile_lstm:
            with torch_dynamo_config.patch("allow_rnn", True):
                return cast(ActorEntryPointT, compile_callable())
        return cast(ActorEntryPointT, compile_callable())

    def _compile_actor_tail_callable(
            self,
            fn: ActorActionsAndLogProbsCallable,
    ) -> ActorActionsAndLogProbsCallable:
        return self._compile_actor_callable(
            fn,
            fullgraph=True,
        )

    def configure_actor_compilation(
            self,
            *,
            encoder_only_sequence_lengths: Collection[int],
            action_sequence_lengths: Collection[int],
            action_sequence_with_selected_states_lengths: Collection[int],
    ) -> None:
        encoder_only_lengths = self._normalize_compiled_sequence_lengths(
            encoder_only_sequence_lengths,
        )
        action_lengths = self._normalize_compiled_sequence_lengths(action_sequence_lengths)
        selected_state_action_lengths = self._normalize_compiled_sequence_lengths(
            action_sequence_with_selected_states_lengths,
        )

        if not self.config.compile_modules:
            self._clear_compiled_actor_entry_points()
            return

        # Multi-sample calls encode once before sampling, even when single-sample
        # calls use an end-to-end compiled entry point for the same length.
        encoder_lengths = encoder_only_lengths | action_lengths | selected_state_action_lengths
        if self._actor_encoder_compilation_enabled:
            actor_encoder_entry_point = (
                self._encode_recurrent_shared_sequence_impl
                if self.uses_recurrent_shared_encoder
                else self._actor_encoder
            )
            self._configure_compiled_entry_points(
                self._compiled_actor_encoders,
                actor_encoder_entry_point,
                sequence_lengths=encoder_lengths,
                fullgraph=True,
            )
        else:
            self._compiled_actor_encoders.clear()

        if not self._actor_end_to_end_compilation_enabled:
            self._compiled_actor_action_sequences.clear()
            self._compiled_actor_action_sequences_with_selected_states.clear()
            return

        self._configure_compiled_actor_action_entry_points(
            action_sequence_lengths=action_lengths,
            action_sequence_with_selected_states_lengths=selected_state_action_lengths,
        )

    @staticmethod
    def _normalize_compiled_sequence_lengths(
            sequence_lengths: Collection[int],
    ) -> frozenset[int]:
        normalized_lengths = frozenset(int(length) for length in sequence_lengths)
        if any(length <= 0 for length in normalized_lengths):
            raise ValueError(
                f"Compiled actor sequence lengths must be positive, got {sorted(normalized_lengths)}"
            )
        return normalized_lengths

    def _configure_compiled_actor_action_entry_points(
            self,
            *,
            action_sequence_lengths: frozenset[int],
            action_sequence_with_selected_states_lengths: frozenset[int],
    ) -> None:
        self._configure_compiled_entry_points(
            self._compiled_actor_action_sequences,
            self._action_log_prob_sequence_impl,
            sequence_lengths=action_sequence_lengths,
            fullgraph=True,
        )
        self._configure_compiled_entry_points(
            self._compiled_actor_action_sequences_with_selected_states,
            self._action_log_prob_sequence_with_selected_states_impl,
            sequence_lengths=action_sequence_with_selected_states_lengths,
            fullgraph=True,
        )

    def _configure_compiled_entry_points(
            self,
            compiled_entry_points: dict[int, ActorEntryPointT],
            entry_point: ActorEntryPointT,
            *,
            sequence_lengths: frozenset[int],
            fullgraph: bool,
    ) -> None:
        existing_lengths = frozenset(compiled_entry_points)
        for sequence_length in sequence_lengths - existing_lengths:
            compiled_entry_points[sequence_length] = self._compile_actor_callable(
                entry_point,
                fullgraph=fullgraph,
            )
        for sequence_length in existing_lengths - sequence_lengths:
            del compiled_entry_points[sequence_length]

    def _clear_compiled_actor_entry_points(self) -> None:
        self._compiled_actor_encoders.clear()
        self._compiled_actor_action_sequences.clear()
        self._compiled_actor_action_sequences_with_selected_states.clear()

    @property
    def compiled_actor_sequence_lengths(self) -> frozenset[int]:
        return frozenset(self._compiled_actor_action_sequences)

    @property
    def compiled_actor_selected_state_sequence_lengths(self) -> frozenset[int]:
        return frozenset(self._compiled_actor_action_sequences_with_selected_states)

    @property
    def compiled_actor_encoder_sequence_lengths(self) -> frozenset[int]:
        return frozenset(self._compiled_actor_encoders)

    @property
    def actor_encoder_compilation_enabled(self) -> bool:
        return self._actor_encoder_compilation_enabled

    @property
    def actor_end_to_end_compilation_enabled(self) -> bool:
        return self._actor_end_to_end_compilation_enabled

    @property
    def recurrent_critic(self) -> bool:
        return self.config.recurrent_critic

    @property
    def uses_recurrent_shared_encoder(self) -> bool:
        return self.config.recurrent_shared_encoder

    @property
    def critic_uses_temporal_state(self) -> bool:
        return self.recurrent_critic or self.uses_recurrent_shared_encoder

    @property
    def uses_actor_state_critic_input(self) -> bool:
        return self.config.actor_state_critic_input_config is not None

    @property
    def uses_temporal_actor_state(self) -> bool:
        return True

    def requires_recurrent_training(self) -> bool:
        return True

    def initial_temporal_state(
            self,
            batch_size: int,
            n_agents: int,
            *,
            device: torch.device,
            dtype: torch.dtype,
    ) -> RMATEncoderState:
        encoder = (
            self._shared_recurrent_encoder(target=False)
            if self.uses_recurrent_shared_encoder
            else self._actor_encoder
        )
        assert isinstance(encoder, RMATEncoder)
        return encoder.initial_state(
            batch_size=batch_size,
            n_agents=n_agents,
            device=device,
            dtype=dtype,
        )

    def act_with_temporal_state(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            scenario_ids: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
            deterministic: bool = False,
            *,
            temporal_state: RMATEncoderState | None = None,
            episode_start_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, RMATEncoderState]:
        _ = hidden_local_vars
        _ = hidden_global_vars
        actions, _log_probs, _latents, next_state = self.action_log_prob_sequence(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            scenario_ids=scenario_ids,
            previous_actions=previous_actions,
            deterministic=deterministic,
            use_rsample=False,
            initial_state=temporal_state,
            reset_mask=episode_start_mask,
        )
        return actions, next_state

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
        """Sample from a reset recurrent state; sequence methods accept replay state."""
        _ = (hidden_local_vars, hidden_global_vars)
        actions, log_probs, _latents, _state = self.action_log_prob_sequence(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            scenario_ids=scenario_ids,
            previous_actions=previous_actions,
            deterministic=deterministic,
            use_rsample=use_rsample,
            initial_state=None,
            num_action_samples=num_action_samples,
            action_sample_strategy=action_sample_strategy,
        )
        return actions, log_probs

    def encode_actor(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            agent_mask: torch.Tensor | None,
            scenario_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = local_obs, global_obs, agent_mask, scenario_ids
        raise NotImplementedError(
            "RecurrentTMASACPolicy does not support state-free actor encoding; "
            "use encode_actor_sequence with an explicit recurrent state."
        )

    def action_log_prob_sequence(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            agent_mask: torch.Tensor | None,
            scenario_ids: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None,
            deterministic: bool,
            use_rsample: bool,
            initial_state: RMATEncoderState | None,
            time_mask: torch.Tensor | None = None,
            reset_mask: torch.Tensor | None = None,
            num_action_samples: int = 1,
            action_sample_strategy: ActionSampleStrategy = "iid",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, RMATEncoderState]:
        if self._actor_end_to_end_compilation_enabled and num_action_samples == 1:
            sequence_length = self._actor_sequence_length(local_obs)
            actor_entry_point = self._compiled_actor_action_sequences.get(sequence_length)
            if actor_entry_point is None:
                raise RuntimeError(
                    f"No compiled actor action entry point for sequence length {sequence_length}; "
                    f"configured lengths are {sorted(self._compiled_actor_action_sequences)}."
                )
            return self._run_actor_action_entry_point(
                actor_entry_point,
                local_obs=local_obs,
                global_obs=global_obs,
                agent_mask=agent_mask,
                scenario_ids=scenario_ids,
                previous_actions=previous_actions,
                deterministic=deterministic,
                use_rsample=use_rsample,
                initial_state=initial_state,
                time_mask=time_mask,
                reset_mask=reset_mask,
            )

        actor_latents, next_state = self.encode_actor_sequence(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            scenario_ids=scenario_ids,
            time_mask=time_mask,
            initial_state=initial_state,
            reset_mask=reset_mask,
        )
        actions, log_probs = self.action_log_prob_from_latents(
            num_action_samples=num_action_samples,
            action_sample_strategy=action_sample_strategy,
            actor_latents=actor_latents,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            deterministic=deterministic,
            use_rsample=use_rsample,
        )
        return actions, log_probs, actor_latents, next_state

    def action_log_prob_sequence_with_selected_states(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            agent_mask: torch.Tensor | None,
            scenario_ids: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None,
            deterministic: bool,
            use_rsample: bool,
            initial_state: RMATEncoderState | None,
            state_output_indices: torch.Tensor,
            time_mask: torch.Tensor | None = None,
            reset_mask: torch.Tensor | None = None,
            num_action_samples: int = 1,
            action_sample_strategy: ActionSampleStrategy = "iid",
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        RMATEncoderState,
        RMATEncoderState,
        Any | None,
    ]:
        if self._actor_end_to_end_compilation_enabled and num_action_samples == 1:
            sequence_length = self._actor_sequence_length(local_obs)
            actor_entry_point = self._compiled_actor_action_sequences_with_selected_states.get(
                sequence_length,
            )
            if actor_entry_point is None:
                raise RuntimeError(
                    "No compiled actor action-with-selected-states entry point for sequence length "
                    f"{sequence_length}; configured lengths are "
                    f"{sorted(self._compiled_actor_action_sequences_with_selected_states)}."
                )
            return self._run_actor_action_entry_point(
                actor_entry_point,
                local_obs=local_obs,
                global_obs=global_obs,
                agent_mask=agent_mask,
                scenario_ids=scenario_ids,
                previous_actions=previous_actions,
                deterministic=deterministic,
                use_rsample=use_rsample,
                initial_state=initial_state,
                state_output_indices=state_output_indices,
                time_mask=time_mask,
                reset_mask=reset_mask,
            )

        actor_encoder_result = self.encode_actor_sequence_with_selected_states(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            scenario_ids=scenario_ids,
            time_mask=time_mask,
            initial_state=initial_state,
            reset_mask=reset_mask,
            state_output_indices=state_output_indices,
        )
        if self.uses_actor_state_critic_input:
            actor_latents, next_state, selected_states, last_layer_state_sequence = actor_encoder_result
        else:
            actor_latents, next_state, selected_states = actor_encoder_result
            last_layer_state_sequence = None
        actions, log_probs = self.action_log_prob_from_latents(
            num_action_samples=num_action_samples,
            action_sample_strategy=action_sample_strategy,
            actor_latents=actor_latents,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            deterministic=deterministic,
            use_rsample=use_rsample,
        )
        return (
            actions,
            log_probs,
            actor_latents,
            next_state,
            selected_states,
            last_layer_state_sequence,
        )

    def action_log_prob_sequence_with_selected_states_and_shared_latents(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            agent_mask: torch.Tensor | None,
            scenario_ids: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None,
            deterministic: bool,
            use_rsample: bool,
            initial_state: RMATEncoderState | None,
            state_output_indices: torch.Tensor,
            time_mask: torch.Tensor | None = None,
            reset_mask: torch.Tensor | None = None,
            num_action_samples: int = 1,
            action_sample_strategy: ActionSampleStrategy = "iid",
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        RMATEncoderState,
        RMATEncoderState,
        torch.Tensor,
    ]:
        if not self.uses_recurrent_shared_encoder:
            raise RuntimeError("Recurrent shared observation encoding is not configured.")
        result = self._run_actor_encoder(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            scenario_ids=scenario_ids,
            time_mask=time_mask,
            initial_state=initial_state,
            reset_mask=reset_mask,
            state_output_indices=state_output_indices,
        )
        actor_latents, next_state, selected_states, shared_latents = result
        actions, log_probs = self.action_log_prob_from_latents(
            num_action_samples=num_action_samples,
            action_sample_strategy=action_sample_strategy,
            actor_latents=actor_latents,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            deterministic=deterministic,
            use_rsample=use_rsample,
        )
        return (
            actions,
            log_probs,
            actor_latents,
            next_state,
            selected_states,
            shared_latents,
        )

    def _action_log_prob_sequence_impl(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            agent_mask: torch.Tensor | None,
            scenario_ids: torch.Tensor | None,
            previous_actions: torch.Tensor | None,
            deterministic: bool,
            use_rsample: bool,
            initial_state: RMATEncoderState | None,
            time_mask: torch.Tensor | None,
            reset_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, RMATEncoderState]:
        actor_latents, next_state = self._encode_actor_sequence_impl(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            scenario_ids=scenario_ids,
            time_mask=time_mask,
            initial_state=initial_state,
            reset_mask=reset_mask,
        )
        actions, log_probs = self._actor_actions_and_log_probs_impl(
            actor_latents=actor_latents,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            deterministic=deterministic,
            use_rsample=use_rsample,
        )
        return actions, log_probs, actor_latents, next_state

    def _action_log_prob_sequence_with_selected_states_impl(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            agent_mask: torch.Tensor | None,
            scenario_ids: torch.Tensor | None,
            previous_actions: torch.Tensor | None,
            deterministic: bool,
            use_rsample: bool,
            initial_state: RMATEncoderState | None,
            state_output_indices: torch.Tensor,
            time_mask: torch.Tensor | None,
            reset_mask: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        RMATEncoderState,
        RMATEncoderState,
        Any | None,
    ]:
        actor_encoder_result = self._encode_actor_sequence_impl(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            scenario_ids=scenario_ids,
            time_mask=time_mask,
            initial_state=initial_state,
            reset_mask=reset_mask,
            state_output_indices=state_output_indices,
            return_last_layer_state_sequence=self.uses_actor_state_critic_input,
        )
        if self.uses_actor_state_critic_input:
            actor_latents, next_state, selected_states, last_layer_state_sequence = actor_encoder_result
        else:
            actor_latents, next_state, selected_states = actor_encoder_result
            last_layer_state_sequence = None
        actions, log_probs = self._actor_actions_and_log_probs_impl(
            actor_latents=actor_latents,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            deterministic=deterministic,
            use_rsample=use_rsample,
        )
        return (
            actions,
            log_probs,
            actor_latents,
            next_state,
            selected_states,
            last_layer_state_sequence,
        )

    def _actor_actions_and_log_probs_impl(
            self,
            *,
            actor_latents: torch.Tensor,
            agent_mask: torch.Tensor | None,
            previous_actions: torch.Tensor | None,
            deterministic: bool,
            use_rsample: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return super()._actor_actions_and_log_probs_impl(
            actor_latents=actor_latents,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            deterministic=deterministic,
            use_rsample=use_rsample,
        )

    def encode_actor_sequence(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            agent_mask: torch.Tensor | None,
            initial_state: RMATEncoderState | None,
            scenario_ids: torch.Tensor | None = None,
            time_mask: torch.Tensor | None = None,
            reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, RMATEncoderState]:
        result = self._run_actor_encoder(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            scenario_ids=scenario_ids,
            time_mask=time_mask,
            initial_state=initial_state,
            reset_mask=reset_mask,
        )
        actor_latents, next_state = result
        return actor_latents, next_state

    def encode_actor_sequence_with_selected_states(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            agent_mask: torch.Tensor | None,
            initial_state: RMATEncoderState | None,
            state_output_indices: torch.Tensor,
            scenario_ids: torch.Tensor | None = None,
            time_mask: torch.Tensor | None = None,
            reset_mask: torch.Tensor | None = None,
    ) -> (
        tuple[torch.Tensor, RMATEncoderState, RMATEncoderState]
        | tuple[torch.Tensor, RMATEncoderState, RMATEncoderState, Any]
    ):
        result = self._run_actor_encoder(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            scenario_ids=scenario_ids,
            time_mask=time_mask,
            initial_state=initial_state,
            reset_mask=reset_mask,
            state_output_indices=state_output_indices,
            return_last_layer_state_sequence=self.uses_actor_state_critic_input,
        )
        if self.uses_actor_state_critic_input:
            actor_latents, next_state, selected_states, last_layer_state_sequence = result
            return actor_latents, next_state, selected_states, last_layer_state_sequence
        if self.uses_recurrent_shared_encoder:
            actor_latents, next_state, selected_states, _shared_latents = result
            return actor_latents, next_state, selected_states
        actor_latents, next_state, selected_states = result
        return actor_latents, next_state, selected_states

    def actor_state_critic_input(self, state: RMATEncoderState) -> torch.Tensor:
        return self.actor_last_layer_state_critic_input(state[-1])

    def actor_last_layer_state_critic_input(self, last_layer_state: Any) -> torch.Tensor:
        if not self.uses_actor_state_critic_input:
            raise RuntimeError("Actor-state critic input is not configured.")
        temporal_model = self._actor_encoder.layers[-1].temporal_model
        if isinstance(temporal_model, LSTMTemporalSequenceModel):
            hidden_state, cell_state = last_layer_state
            return torch.cat((hidden_state[..., -1, :], cell_state[..., -1, :]), dim=-1).detach()
        if isinstance(temporal_model, SLSTMTemporalSequenceModel):
            hidden_state, cell_state, normalizer_state, stabilizer_state = last_layer_state
            safe_normalizer_state = normalizer_state.clamp_min(1.0)
            normalized_memory = cell_state / safe_normalizer_state
            actor_state_config = self.config.actor_state_critic_input_config
            assert actor_state_config is not None
            if not actor_state_config.include_slstm_memory_strength:
                return torch.cat((hidden_state, normalized_memory), dim=-1).detach()
            log_memory_strength = torch.log(safe_normalizer_state) + stabilizer_state
            bounded_memory_strength = torch.nn.functional.softsign(log_memory_strength)
            return torch.cat(
                (hidden_state, normalized_memory, bounded_memory_strength),
                dim=-1,
            ).detach()
        raise TypeError(
            "Actor-state critic input supports only LSTMTemporalSequenceModel and "
            f"SLSTMTemporalSequenceModel, got {type(temporal_model).__name__}."
        )

    def _run_actor_encoder(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            agent_mask: torch.Tensor | None,
            scenario_ids: torch.Tensor | None,
            time_mask: torch.Tensor | None = None,
            initial_state: RMATEncoderState | None = None,
            reset_mask: torch.Tensor | None = None,
            state_output_indices: torch.Tensor | None = None,
            return_last_layer_state_sequence: bool = False,
    ) -> (
        tuple[torch.Tensor, RMATEncoderState]
        | tuple[torch.Tensor, RMATEncoderState, RMATEncoderState]
        | tuple[torch.Tensor, RMATEncoderState, RMATEncoderState, Any]
    ):
        sequence_length = self._actor_sequence_length(local_obs)
        if self.uses_recurrent_shared_encoder:
            shared_encoder = (
                self._encode_recurrent_shared_sequence_impl
                if not self._actor_encoder_compilation_enabled
                else self._compiled_actor_encoders.get(sequence_length)
            )
            if shared_encoder is None:
                raise RuntimeError(
                    f"No compiled shared encoder entry point for sequence length {sequence_length}; "
                    f"configured lengths are {sorted(self._compiled_actor_encoders)}."
                )
            shared_result = shared_encoder(
                local_obs=local_obs,
                global_obs=global_obs,
                agent_mask=agent_mask,
                time_mask=time_mask,
                initial_state=initial_state,
                reset_mask=reset_mask,
                state_output_indices=state_output_indices,
            )
            if state_output_indices is None:
                shared_latents, next_state = shared_result
                actor_latents = self._encode_downstream_actor(
                    shared_latents.detach(),
                    global_obs=global_obs,
                    agent_mask=agent_mask,
                )
                return actor_latents, next_state
            shared_latents, next_state, selected_states = shared_result
            actor_latents = self._encode_downstream_actor(
                shared_latents.detach(),
                global_obs=global_obs,
                agent_mask=agent_mask,
            )
            return actor_latents, next_state, selected_states, shared_latents

        actor_local_inputs, actor_global_inputs = self._actor_observation_inputs(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            scenario_ids=scenario_ids,
        )
        if not self._actor_encoder_compilation_enabled:
            actor_encoder = self._actor_encoder
        else:
            actor_encoder = self._compiled_actor_encoders.get(sequence_length)
            if actor_encoder is None:
                raise RuntimeError(
                    f"No compiled actor encoder entry point for sequence length {sequence_length}; "
                    f"configured lengths are {sorted(self._compiled_actor_encoders)}."
                )

        def run_actor_encoder() -> (
            tuple[torch.Tensor, RMATEncoderState]
            | tuple[torch.Tensor, RMATEncoderState, RMATEncoderState]
            | tuple[torch.Tensor, RMATEncoderState, RMATEncoderState, Any]
        ):
            return actor_encoder(
                actor_local_inputs,
                actor_global_inputs,
                agent_mask=agent_mask,
                time_mask=time_mask,
                initial_state=initial_state,
                reset_mask=reset_mask,
                state_output_indices=state_output_indices,
                return_last_layer_state_sequence=return_last_layer_state_sequence,
            )

        return self._run_with_optional_lstm_compilation(run_actor_encoder)

    def _encode_recurrent_shared_sequence_impl(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            agent_mask: torch.Tensor | None,
            time_mask: torch.Tensor | None,
            initial_state: RMATEncoderState | None,
            reset_mask: torch.Tensor | None,
            state_output_indices: torch.Tensor | None = None,
    ) -> (
        tuple[torch.Tensor, RMATEncoderState]
        | tuple[torch.Tensor, RMATEncoderState, RMATEncoderState]
    ):
        return self._shared_recurrent_encoder(target=False)(
            local_obs,
            global_obs,
            agent_mask=agent_mask,
            time_mask=time_mask,
            initial_state=initial_state,
            reset_mask=reset_mask,
            state_output_indices=state_output_indices,
        )

    def _encode_actor_sequence_impl(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            agent_mask: torch.Tensor | None,
            scenario_ids: torch.Tensor | None,
            time_mask: torch.Tensor | None,
            initial_state: RMATEncoderState | None,
            reset_mask: torch.Tensor | None,
            state_output_indices: torch.Tensor | None = None,
            return_last_layer_state_sequence: bool = False,
    ) -> (
        tuple[torch.Tensor, RMATEncoderState]
        | tuple[torch.Tensor, RMATEncoderState, RMATEncoderState]
        | tuple[torch.Tensor, RMATEncoderState, RMATEncoderState, Any]
    ):
        if not self.uses_recurrent_shared_encoder:
            actor_local_inputs, actor_global_inputs = self._actor_observation_inputs(
                local_obs=local_obs,
                global_obs=global_obs,
                agent_mask=agent_mask,
                scenario_ids=scenario_ids,
            )
            return self._actor_encoder(
                actor_local_inputs,
                actor_global_inputs,
                agent_mask=agent_mask,
                time_mask=time_mask,
                initial_state=initial_state,
                reset_mask=reset_mask,
                state_output_indices=state_output_indices,
                return_last_layer_state_sequence=return_last_layer_state_sequence,
            )

        if return_last_layer_state_sequence:
            raise ValueError(
                "The recurrent shared encoder does not expose actor-state critic summaries."
            )
        shared_result = self._encode_recurrent_shared_sequence_impl(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            time_mask=time_mask,
            initial_state=initial_state,
            reset_mask=reset_mask,
            state_output_indices=state_output_indices,
        )
        if state_output_indices is None:
            shared_latents, next_state = shared_result
            actor_latents = self._encode_downstream_actor(
                shared_latents.detach(),
                global_obs=global_obs,
                agent_mask=agent_mask,
            )
            return actor_latents, next_state
        shared_latents, next_state, selected_states = shared_result
        actor_latents = self._encode_downstream_actor(
            shared_latents.detach(),
            global_obs=global_obs,
            agent_mask=agent_mask,
        )
        return actor_latents, next_state, selected_states, shared_latents

    def _encode_downstream_actor(
            self,
            shared_latents: torch.Tensor,
            *,
            global_obs: torch.Tensor,
            agent_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if shared_latents.ndim == 3:
            return self._actor_encoder(
                shared_latents,
                global_obs,
                agent_mask=agent_mask,
            )
        batch_size, sequence_length, n_agents, latent_dim = shared_latents.shape
        flat_mask = (
            None
            if agent_mask is None
            else agent_mask.reshape(batch_size * sequence_length, n_agents)
        )
        flat_actor_latents = self._actor_encoder(
            shared_latents.reshape(batch_size * sequence_length, n_agents, latent_dim),
            global_obs.reshape(batch_size * sequence_length, *global_obs.shape[2:]),
            agent_mask=flat_mask,
        )
        return flat_actor_latents.reshape(
            batch_size,
            sequence_length,
            n_agents,
            flat_actor_latents.shape[-1],
        )

    def _run_actor_action_entry_point(
            self,
            entry_point: Callable[..., Any],
            **kwargs: Any,
    ) -> Any:
        return self._run_with_optional_lstm_compilation(lambda: entry_point(**kwargs))

    def _run_with_optional_lstm_compilation(self, fn: Callable[[], Any]) -> Any:
        if self._actor_encoder_uses_lstm and self.config.experimental_compile_lstm:
            with torch_dynamo_config.patch("allow_rnn", True):
                return fn()
        return fn()

    @staticmethod
    def _actor_sequence_length(local_obs: torch.Tensor) -> int:
        return 1 if local_obs.ndim == 3 else int(local_obs.shape[1])

    def q_values_sequence(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            actions: torch.Tensor,
            hidden_local_vars: torch.Tensor | None,
            hidden_global_vars: torch.Tensor | None,
            agent_mask: torch.Tensor | None,
            target: bool,
            scenario_ids: torch.Tensor | None = None,
            initial_state: RecurrentCriticState | None = None,
            time_mask: torch.Tensor | None = None,
            reset_mask: torch.Tensor | None = None,
            actor_state: torch.Tensor | None = None,
            precomputed_shared_latents: torch.Tensor | None = None,
            precomputed_shared_state: RMATEncoderState | None = None,
            num_action_samples: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, RecurrentCriticState | None]:
        """Evaluate replay actions or K candidates branching from the same history.

        Candidates return no NOP latents. Only observation-only recurrence can
        return a common next state; action-conditioned branches have distinct states.
        """
        if self.uses_recurrent_shared_encoder:
            return self._shared_encoder_q_values_sequence(
                local_obs=local_obs,
                global_obs=global_obs,
                actions=actions,
                hidden_local_vars=hidden_local_vars,
                hidden_global_vars=hidden_global_vars,
                agent_mask=agent_mask,
                target=target,
                scenario_ids=scenario_ids,
                initial_state=initial_state,
                time_mask=time_mask,
                reset_mask=reset_mask,
                precomputed_shared_latents=precomputed_shared_latents,
                precomputed_shared_state=precomputed_shared_state,
                num_action_samples=num_action_samples,
            )
        if precomputed_shared_latents is not None or precomputed_shared_state is not None:
            raise ValueError(
                "Precomputed shared latents require recurrent shared observation encoding."
            )
        if num_action_samples > 1:
            return self._q_values_sequence_candidates(
                actions=actions,
                num_action_samples=num_action_samples,
                target=target,
                initial_state=initial_state,
                local_obs=local_obs,
                global_obs=global_obs,
                hidden_local_vars=hidden_local_vars,
                hidden_global_vars=hidden_global_vars,
                agent_mask=agent_mask,
                scenario_ids=scenario_ids,
                time_mask=time_mask,
                reset_mask=reset_mask,
                actor_state=actor_state,
            )
        if self.recurrent_critic:
            critic = self.critic_target if target else self.critic
            assert isinstance(critic, RecurrentTMASACTwinCritic)
            (
                critic_local_inputs,
                critic_global_inputs,
                hidden_local_vars,
                hidden_global_vars,
            ) = self._critic_observation_inputs(
                local_obs=local_obs,
                global_obs=global_obs,
                hidden_local_vars=hidden_local_vars,
                hidden_global_vars=hidden_global_vars,
                agent_mask=agent_mask,
                scenario_ids=scenario_ids,
                target=target,
            )
            q1, q2, latents, next_state = critic(
                local_obs=critic_local_inputs,
                global_obs=critic_global_inputs,
                actions=actions,
                hidden_local_vars=hidden_local_vars,
                hidden_global_vars=hidden_global_vars,
                agent_mask=agent_mask,
                time_mask=time_mask,
                initial_state=initial_state,
                reset_mask=reset_mask,
            )
            return q1, q2, None if target or self.critic_nop is None else latents, next_state

        if self.uses_actor_state_critic_input:
            if actor_state is None:
                raise ValueError("actor_state must be provided when actor-state critic input is configured.")
            return self._actor_state_q_values_sequence_impl(
                local_obs=local_obs,
                global_obs=global_obs,
                actions=actions,
                hidden_local_vars=hidden_local_vars,
                hidden_global_vars=hidden_global_vars,
                agent_mask=agent_mask,
                scenario_ids=scenario_ids,
                target=target,
                actor_state=actor_state,
            )

        flat_inputs, batch_size, sequence_length = self._flatten_sequence_inputs(
            local_obs=local_obs,
            global_obs=global_obs,
            actions=actions,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            scenario_ids=scenario_ids,
        )
        if target:
            q1, q2 = self.target_q_values(**flat_inputs)
            latents = None
        else:
            q1, q2, latents = self.q_values_with_nop_latents(**flat_inputs)
        if local_obs.ndim == 3:
            return q1, q2, latents, None
        q1 = q1.reshape(batch_size, sequence_length)
        q2 = q2.reshape(batch_size, sequence_length)
        if latents is not None:
            latents = latents.reshape(batch_size, sequence_length, *latents.shape[1:])
        return q1, q2, latents, None

    def _q_values_sequence_candidates(
            self,
            *,
            actions: torch.Tensor,
            num_action_samples: int,
            target: bool,
            initial_state: RecurrentCriticState | None,
            **observations: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        flat_observations = {
            name: expand_action_sample_batch(value, num_action_samples)
            for name, value in observations.items()
        }
        q1, q2, _latents, _state = self.q_values_sequence(
            actions=actions.flatten(0, 1),
            target=target,
            initial_state=concatenate_temporal_states([initial_state] * num_action_samples),
            **flat_observations,
        )
        return (
            q1.unflatten(0, (num_action_samples, -1)),
            q2.unflatten(0, (num_action_samples, -1)),
            None,
            None,
        )

    def _shared_encoder_q_values_sequence(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            actions: torch.Tensor,
            hidden_local_vars: torch.Tensor | None,
            hidden_global_vars: torch.Tensor | None,
            agent_mask: torch.Tensor | None,
            target: bool,
            scenario_ids: torch.Tensor | None,
            initial_state: RecurrentCriticState | None,
            time_mask: torch.Tensor | None,
            reset_mask: torch.Tensor | None,
            precomputed_shared_latents: torch.Tensor | None,
            precomputed_shared_state: RMATEncoderState | None,
            num_action_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, RMATEncoderState]:
        if scenario_ids is not None:
            raise ValueError(
                "Recurrent shared TMASAC does not support scenario-specific observation encoding."
            )
        if precomputed_shared_latents is None:
            if precomputed_shared_state is not None:
                raise ValueError(
                    "precomputed_shared_state requires precomputed_shared_latents."
                )
            shared_latents, next_state = self._shared_recurrent_encoder(target=target)(
                local_obs,
                global_obs,
                agent_mask=agent_mask,
                time_mask=time_mask,
                initial_state=cast(RMATEncoderState | None, initial_state),
                reset_mask=reset_mask,
            )
        else:
            if target:
                raise ValueError("Target Q values cannot reuse online shared latents.")
            if precomputed_shared_state is None:
                raise ValueError(
                    "precomputed_shared_latents requires precomputed_shared_state."
                )
            shared_latents = precomputed_shared_latents
            next_state = precomputed_shared_state
        flat_inputs, batch_size, sequence_length = self._flatten_sequence_inputs(
            local_obs=shared_latents,
            global_obs=global_obs,
            actions=actions if num_action_samples == 1 else None,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            scenario_ids=None,
        )
        if num_action_samples > 1:
            flat_inputs = {
                name: expand_action_sample_batch(value, num_action_samples)
                for name, value in flat_inputs.items()
            }
            flat_inputs["actions"] = actions.reshape(-1, *actions.shape[-2:])
        critic = self.critic_target if target else self.critic
        q1, q2, critic_latents = critic(
            local_obs=cast(torch.Tensor, flat_inputs["local_obs"]),
            global_obs=cast(torch.Tensor, flat_inputs["global_obs"]),
            actions=cast(torch.Tensor, flat_inputs["actions"]),
            hidden_local_vars=flat_inputs["hidden_local_vars"],
            hidden_global_vars=flat_inputs["hidden_global_vars"],
            agent_mask=flat_inputs["agent_mask"],
        )
        if num_action_samples > 1:
            # Candidate forwards share the observation recurrence. NOP trains on
            # executed replay actions, so its latents are not needed on this path.
            value_shape = (num_action_samples, *local_obs.shape[:-2])
            return q1.reshape(value_shape), q2.reshape(value_shape), None, next_state
        nop_latents = (
            shared_latents
            if self.nop_latent_source is SACNOPLatentSource.SHARED_ENCODER
            else critic_latents
        )
        if local_obs.ndim == 3:
            return (
                q1,
                q2,
                None if target or self.critic_nop is None else nop_latents,
                next_state,
            )
        q1 = q1.reshape(batch_size, sequence_length)
        q2 = q2.reshape(batch_size, sequence_length)
        if target or self.critic_nop is None:
            return q1, q2, None, next_state
        if nop_latents.ndim == 3:
            nop_latents = nop_latents.reshape(
                batch_size,
                sequence_length,
                *nop_latents.shape[1:],
            )
        return q1, q2, nop_latents, next_state

    def _actor_state_q_values_sequence_impl(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            actions: torch.Tensor,
            hidden_local_vars: torch.Tensor | None,
            hidden_global_vars: torch.Tensor | None,
            agent_mask: torch.Tensor | None,
            scenario_ids: torch.Tensor | None,
            target: bool,
            actor_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, RecurrentCriticState | None]:
        flat_inputs, batch_size, sequence_length = self._flatten_sequence_inputs(
            local_obs=local_obs,
            global_obs=global_obs,
            actions=actions,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            scenario_ids=scenario_ids,
        )
        flat_scenario_ids = flat_inputs.pop("scenario_ids")
        (
            flat_inputs["local_obs"],
            flat_inputs["global_obs"],
            flat_inputs["hidden_local_vars"],
            flat_inputs["hidden_global_vars"],
        ) = self._critic_observation_inputs(
            local_obs=cast(torch.Tensor, flat_inputs["local_obs"]),
            global_obs=cast(torch.Tensor, flat_inputs["global_obs"]),
            hidden_local_vars=flat_inputs["hidden_local_vars"],
            hidden_global_vars=flat_inputs["hidden_global_vars"],
            agent_mask=flat_inputs["agent_mask"],
            scenario_ids=flat_scenario_ids,
            target=target,
        )
        flat_actor_state = actor_state.reshape(-1, *actor_state.shape[-2:])
        critic = self.critic_target if target else self.critic
        q1, q2, latents = critic(actor_state=flat_actor_state, **flat_inputs)
        if local_obs.ndim == 3:
            return q1, q2, None if target or self.critic_nop is None else latents, None
        q1 = q1.reshape(batch_size, sequence_length)
        q2 = q2.reshape(batch_size, sequence_length)
        if target or self.critic_nop is None:
            return q1, q2, None, None
        return q1, q2, latents.reshape(batch_size, sequence_length, *latents.shape[1:]), None

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
        if self.uses_recurrent_shared_encoder:
            q1, q2, _latents, _state = self.q_values_sequence(
                local_obs=local_obs,
                global_obs=global_obs,
                actions=actions,
                hidden_local_vars=hidden_local_vars,
                hidden_global_vars=hidden_global_vars,
                agent_mask=agent_mask,
                scenario_ids=scenario_ids,
                target=False,
            )
            return q1, q2
        if self.uses_actor_state_critic_input:
            q1, q2, _latents, _state = self._stateless_actor_state_q_values(
                local_obs=local_obs,
                global_obs=global_obs,
                actions=actions,
                hidden_local_vars=hidden_local_vars,
                hidden_global_vars=hidden_global_vars,
                agent_mask=agent_mask,
                scenario_ids=scenario_ids,
                target=False,
            )
            return q1, q2
        if not self.recurrent_critic:
            return super().q_values(
                local_obs=local_obs,
                global_obs=global_obs,
                actions=actions,
                hidden_local_vars=hidden_local_vars,
                hidden_global_vars=hidden_global_vars,
                agent_mask=agent_mask,
                scenario_ids=scenario_ids,
            )
        q1, q2, _latents, _state = self.q_values_sequence(
            local_obs=local_obs,
            global_obs=global_obs,
            actions=actions,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            scenario_ids=scenario_ids,
            target=False,
        )
        return q1, q2

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
        if self.uses_recurrent_shared_encoder:
            q1, q2, latents, _state = self.q_values_sequence(
                local_obs=local_obs,
                global_obs=global_obs,
                actions=actions,
                hidden_local_vars=hidden_local_vars,
                hidden_global_vars=hidden_global_vars,
                agent_mask=agent_mask,
                scenario_ids=scenario_ids,
                target=False,
            )
            return q1, q2, latents
        if self.uses_actor_state_critic_input:
            q1, q2, latents, _state = self._stateless_actor_state_q_values(
                local_obs=local_obs,
                global_obs=global_obs,
                actions=actions,
                hidden_local_vars=hidden_local_vars,
                hidden_global_vars=hidden_global_vars,
                agent_mask=agent_mask,
                scenario_ids=scenario_ids,
                target=False,
            )
            return q1, q2, latents
        if not self.recurrent_critic:
            return super().q_values_with_nop_latents(
                local_obs=local_obs,
                global_obs=global_obs,
                actions=actions,
                hidden_local_vars=hidden_local_vars,
                hidden_global_vars=hidden_global_vars,
                agent_mask=agent_mask,
                scenario_ids=scenario_ids,
            )
        q1, q2, latents, _state = self.q_values_sequence(
            local_obs=local_obs,
            global_obs=global_obs,
            actions=actions,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            scenario_ids=scenario_ids,
            target=False,
        )
        return q1, q2, latents

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
        if self.uses_recurrent_shared_encoder:
            q1, q2, _latents, _state = self.q_values_sequence(
                local_obs=local_obs,
                global_obs=global_obs,
                actions=actions,
                hidden_local_vars=hidden_local_vars,
                hidden_global_vars=hidden_global_vars,
                agent_mask=agent_mask,
                scenario_ids=scenario_ids,
                target=True,
            )
            return q1, q2
        if self.uses_actor_state_critic_input:
            q1, q2, _latents, _state = self._stateless_actor_state_q_values(
                local_obs=local_obs,
                global_obs=global_obs,
                actions=actions,
                hidden_local_vars=hidden_local_vars,
                hidden_global_vars=hidden_global_vars,
                agent_mask=agent_mask,
                scenario_ids=scenario_ids,
                target=True,
            )
            return q1, q2
        if not self.recurrent_critic:
            return super().target_q_values(
                local_obs=local_obs,
                global_obs=global_obs,
                actions=actions,
                hidden_local_vars=hidden_local_vars,
                hidden_global_vars=hidden_global_vars,
                agent_mask=agent_mask,
                scenario_ids=scenario_ids,
            )
        q1, q2, _latents, _state = self.q_values_sequence(
            local_obs=local_obs,
            global_obs=global_obs,
            actions=actions,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            scenario_ids=scenario_ids,
            target=True,
        )
        return q1, q2

    def encode_critic(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            actions: torch.Tensor,
            hidden_local_vars: torch.Tensor | None,
            hidden_global_vars: torch.Tensor | None,
            agent_mask: torch.Tensor | None,
            scenario_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.uses_recurrent_shared_encoder:
            shared_latents, _state = self._shared_recurrent_encoder(target=False)(
                local_obs,
                global_obs,
                agent_mask=agent_mask,
            )
            flat_inputs, batch_size, sequence_length = self._flatten_sequence_inputs(
                local_obs=shared_latents,
                global_obs=global_obs,
                actions=actions,
                hidden_local_vars=hidden_local_vars,
                hidden_global_vars=hidden_global_vars,
                agent_mask=agent_mask,
                scenario_ids=scenario_ids,
            )
            latents = self._critic_module().encode(
                local_inputs=cast(torch.Tensor, flat_inputs["local_obs"]),
                global_inputs=cast(torch.Tensor, flat_inputs["global_obs"]),
                actions=cast(torch.Tensor, flat_inputs["actions"]),
                hidden_local_vars=flat_inputs["hidden_local_vars"],
                hidden_global_vars=flat_inputs["hidden_global_vars"],
                agent_mask=flat_inputs["agent_mask"],
            )
            if local_obs.ndim == 3:
                return latents
            return latents.reshape(
                batch_size,
                sequence_length,
                *latents.shape[1:],
            )
        if self.uses_actor_state_critic_input:
            actor_state = self._stateless_actor_state_critic_input(
                local_obs=local_obs,
                global_obs=global_obs,
                agent_mask=agent_mask,
                scenario_ids=scenario_ids,
            )
            critic = self._critic_module()
            assert isinstance(critic, ActorStateTMASACTwinCritic)
            (
                critic_local_inputs,
                critic_global_inputs,
                hidden_local_vars,
                hidden_global_vars,
            ) = self._critic_observation_inputs(
                local_obs=local_obs,
                global_obs=global_obs,
                hidden_local_vars=hidden_local_vars,
                hidden_global_vars=hidden_global_vars,
                agent_mask=agent_mask,
                scenario_ids=scenario_ids,
                target=False,
            )
            return critic.encode(
                actor_state=actor_state,
                local_inputs=critic_local_inputs,
                global_inputs=critic_global_inputs,
                actions=actions,
                hidden_local_vars=hidden_local_vars,
                hidden_global_vars=hidden_global_vars,
                agent_mask=agent_mask,
            )
        if not self.recurrent_critic:
            return super().encode_critic(
                local_obs=local_obs,
                global_obs=global_obs,
                actions=actions,
                hidden_local_vars=hidden_local_vars,
                hidden_global_vars=hidden_global_vars,
                agent_mask=agent_mask,
                scenario_ids=scenario_ids,
            )
        critic = self._critic_module()
        assert isinstance(critic, RecurrentTMASACTwinCritic)
        (
            critic_local_inputs,
            critic_global_inputs,
            hidden_local_vars,
            hidden_global_vars,
        ) = self._critic_observation_inputs(
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            scenario_ids=scenario_ids,
            target=False,
        )
        latents, _state = critic.encode(
            local_inputs=critic_local_inputs,
            global_inputs=critic_global_inputs,
            actions=actions,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
        )
        return latents

    def _stateless_actor_state_q_values(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            actions: torch.Tensor,
            hidden_local_vars: torch.Tensor | None,
            hidden_global_vars: torch.Tensor | None,
            agent_mask: torch.Tensor | None,
            scenario_ids: torch.Tensor | None,
            target: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, RecurrentCriticState | None]:
        actor_state = self._stateless_actor_state_critic_input(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            scenario_ids=scenario_ids,
        )
        return self.q_values_sequence(
            local_obs=local_obs,
            global_obs=global_obs,
            actions=actions,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            scenario_ids=scenario_ids,
            target=target,
            actor_state=actor_state,
        )

    def _stateless_actor_state_critic_input(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            agent_mask: torch.Tensor | None,
            scenario_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if local_obs.ndim != 3:
            raise ValueError(
                "State-free critic methods expect local_obs shape (B, A, F); "
                "use q_values_sequence with explicit actor_state for sequences."
            )
        with torch.no_grad():
            actor_local_inputs, actor_global_inputs = self._actor_observation_inputs(
                local_obs=local_obs,
                global_obs=global_obs,
                agent_mask=agent_mask,
                scenario_ids=scenario_ids,
            )
            _actor_latents, actor_state = self._actor_encoder(
                actor_local_inputs,
                actor_global_inputs,
                agent_mask=agent_mask,
            )
        return self.actor_state_critic_input(actor_state)

    def initial_critic_state(
            self,
            batch_size: int,
            n_agents: int,
            *,
            device: torch.device,
            dtype: torch.dtype,
            target: bool,
    ) -> RecurrentCriticState | None:
        if self.uses_recurrent_shared_encoder:
            return self._shared_recurrent_encoder(target=target).initial_state(
                batch_size=batch_size,
                n_agents=n_agents,
                device=device,
                dtype=dtype,
            )
        if not self.recurrent_critic:
            return None
        critic = self.critic_target if target else self.critic
        assert isinstance(critic, RecurrentTMASACTwinCritic)
        return critic.initial_state(
            batch_size=batch_size,
            n_agents=n_agents,
            device=device,
            dtype=dtype,
        )

    def compute_actor_nop_loss_from_latents(
            self,
            *,
            source_latents: torch.Tensor,
            batch: SACNOPSequenceBatch,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        if self.actor_nop is None:
            return None, {}
        return self.actor_nop.compute_loss(source_latents=source_latents, batch=batch)

    def compute_actor_nop_loss(
            self,
            batch: OffPolicyReplayBatch | OffPolicyReplayEpisodeSegmentBatch,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        _ = batch
        raise NotImplementedError(
            "RecurrentTMASACPolicy NOP loss requires sequence-aware actor latents; "
            "RecurrentSAC uses compute_actor_nop_loss_from_latents."
        )

    def compute_critic_nop_loss_from_latents(
            self,
            *,
            source_latents: torch.Tensor | None,
            batch: SACNOPSequenceBatch,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        if self.critic_nop is None:
            return None, {}
        if source_latents is None:
            raise ValueError("Critic NOP requires critic source latents.")
        return self.critic_nop.compute_loss(source_latents=source_latents, batch=batch)

    def get_hyper_parameters(self) -> dict[str, Any]:
        hyper_parameters = super().get_hyper_parameters()
        hyper_parameters["tmasac_policy_config"]["recurrent_critic"] = self.recurrent_critic
        hyper_parameters["tmasac_policy_config"]["recurrent_shared_encoder"] = (
            self.uses_recurrent_shared_encoder
        )
        hyper_parameters["tmasac_policy_config"]["actor_state_critic_input_config"] = (
            None
            if self.config.actor_state_critic_input_config is None
            else serialize_dataclass(self.config.actor_state_critic_input_config)
        )
        hyper_parameters["tmasac_policy_config"]["actor_encoder_compilation_enabled"] = (
            self.actor_encoder_compilation_enabled
        )
        hyper_parameters["tmasac_policy_config"]["actor_end_to_end_compilation_enabled"] = (
            self.actor_end_to_end_compilation_enabled
        )
        hyper_parameters["tmasac_policy_config"]["compiled_actor_encoder_sequence_lengths"] = sorted(
            self.compiled_actor_encoder_sequence_lengths
        )
        hyper_parameters["tmasac_policy_config"]["compiled_actor_sequence_lengths"] = sorted(
            self.compiled_actor_sequence_lengths
        )
        hyper_parameters["tmasac_policy_config"][
            "compiled_actor_selected_state_sequence_lengths"
        ] = sorted(self.compiled_actor_selected_state_sequence_lengths)
        return hyper_parameters

    def _build_actor_encoder(
            self,
            *,
            local_obs_dim: int,
            global_obs_dim: int,
    ) -> MATEncoder | RMATEncoder:
        if self.config.recurrent_shared_encoder:
            return MATEncoder(
                config=self.actor_encoder_config,
                max_agents=self.max_agents,
                local_obs_dim=local_obs_dim,
                global_obs_dim=global_obs_dim,
            )
        return RMATEncoder(
            config=cast(RMATEncoderConfig, self.actor_encoder_config),
            max_agents=self.max_agents,
            local_obs_dim=local_obs_dim,
            global_obs_dim=global_obs_dim,
        )

    def _build_observation_encoder(
            self,
            config: MATEncoderConfig,
            *,
            local_obs_dim: int | None = None,
            global_obs_dim: int | None = None,
    ) -> MATEncoder | RMATEncoder:
        if self.config.recurrent_shared_encoder:
            if not isinstance(config, RMATEncoderConfig):
                raise TypeError(
                    "The recurrent shared observation encoder requires RMATEncoderConfig."
                )
            return RMATEncoder(
                config=config,
                max_agents=self.max_agents,
                local_obs_dim=(
                    self.local_obs_dim
                    if local_obs_dim is None
                    else int(local_obs_dim)
                ),
                global_obs_dim=(
                    self.global_obs_dim
                    if global_obs_dim is None
                    else int(global_obs_dim)
                ),
            )
        return super()._build_observation_encoder(
            config,
            local_obs_dim=local_obs_dim,
            global_obs_dim=global_obs_dim,
        )

    def encode_shared_observations(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            agent_mask: torch.Tensor | None,
            target: bool,
    ) -> torch.Tensor:
        if not self.uses_recurrent_shared_encoder:
            return super().encode_shared_observations(
                local_obs=local_obs,
                global_obs=global_obs,
                agent_mask=agent_mask,
                target=target,
            )
        latents, _state = self._shared_recurrent_encoder(target=target)(
            local_obs,
            global_obs,
            agent_mask=agent_mask,
        )
        return latents

    def _shared_recurrent_encoder(self, *, target: bool) -> RMATEncoder:
        encoder = (
            self.shared_observation_encoder_target
            if target
            else self.shared_observation_encoder
        )
        if isinstance(encoder, RMATEncoder):
            return encoder
        orig_mod = getattr(encoder, "_orig_mod", None)
        if isinstance(orig_mod, RMATEncoder):
            return orig_mod
        raise RuntimeError("Recurrent TMASAC has no recurrent shared encoder configured.")

    def _build_critic(
            self,
            *,
            local_input_dim: int,
            global_input_dim: int,
    ) -> TMASACTwinCritic:
        actor_state_config = self.config.actor_state_critic_input_config
        if actor_state_config is not None:
            temporal_model = self._actor_encoder.layers[-1].temporal_model
            if isinstance(temporal_model, LSTMTemporalSequenceModel):
                actor_state_input_dim = 2 * self.actor_encoder_config.d_model
            elif isinstance(temporal_model, SLSTMTemporalSequenceModel):
                actor_state_input_dim = (
                    3 if actor_state_config.include_slstm_memory_strength else 2
                ) * self.actor_encoder_config.d_model
            else:
                raise TypeError(
                    "Actor-state critic input supports only LSTMTemporalSequenceModel and "
                    f"SLSTMTemporalSequenceModel, got {type(temporal_model).__name__}."
                )
            return ActorStateTMASACTwinCritic(
                n_agents=self.n_agents,
                max_agents=self.max_agents,
                local_input_dim=local_input_dim,
                global_input_dim=global_input_dim,
                hidden_local_vars_dim=self.critic_hidden_local_vars_dim,
                hidden_global_vars_dim=self.critic_hidden_global_vars_dim,
                action_dim=self.agent_action_dim,
                encoder_config=self.critic_encoder_config,
                critic_config=self.config.critic_config,
                act_fn_cls=self.config.act_fn_cls,
                dropout=self.dropout,
                actor_state_input_dim=actor_state_input_dim,
                actor_state_config=actor_state_config,
                actor_state_default_projection_dim=self.actor_encoder_config.d_model,
            )
        if not self.config.recurrent_critic:
            return super()._build_critic(
                local_input_dim=local_input_dim,
                global_input_dim=global_input_dim,
            )
        encoder_config = cast(RMATEncoderConfig, self.critic_encoder_config)
        return RecurrentTMASACTwinCritic(
            n_agents=self.n_agents,
            max_agents=self.max_agents,
            local_input_dim=local_input_dim,
            global_input_dim=global_input_dim,
            hidden_local_vars_dim=self.critic_hidden_local_vars_dim,
            hidden_global_vars_dim=self.critic_hidden_global_vars_dim,
            action_dim=self.agent_action_dim,
            encoder_config=encoder_config,
            critic_config=self.config.critic_config,
            dropout=self.dropout,
        )

    def _critic_module(self) -> TMASACTwinCritic:
        if isinstance(self.critic, TMASACTwinCritic):
            return self.critic
        orig_mod = getattr(self.critic, "_orig_mod", None)
        if isinstance(orig_mod, TMASACTwinCritic):
            return orig_mod
        raise RuntimeError(f"Cannot resolve TMASAC critic from {type(self.critic).__name__}")

    @staticmethod
    def _flatten_sequence_inputs(
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            actions: torch.Tensor,
            hidden_local_vars: torch.Tensor | None,
            hidden_global_vars: torch.Tensor | None,
            agent_mask: torch.Tensor | None,
            scenario_ids: torch.Tensor | None,
    ) -> tuple[dict[str, torch.Tensor | None], int, int]:
        if local_obs.ndim == 3:
            return {
                "local_obs": local_obs,
                "global_obs": global_obs,
                "actions": actions,
                "hidden_local_vars": hidden_local_vars,
                "hidden_global_vars": hidden_global_vars,
                "agent_mask": agent_mask,
                "scenario_ids": scenario_ids,
            }, local_obs.shape[0], 1
        batch_size, sequence_length = local_obs.shape[:2]

        def flatten(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None:
                return None
            return tensor.reshape(batch_size * sequence_length, *tensor.shape[2:])

        return {
            "local_obs": flatten(local_obs),
            "global_obs": flatten(global_obs),
            "actions": flatten(actions),
            "hidden_local_vars": flatten(hidden_local_vars),
            "hidden_global_vars": flatten(hidden_global_vars),
            "agent_mask": flatten(agent_mask),
            "scenario_ids": flatten(scenario_ids),
        }, batch_size, sequence_length
