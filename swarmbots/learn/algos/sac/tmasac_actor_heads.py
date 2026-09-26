from dataclasses import dataclass, field, replace
from enum import Enum

import torch
from torch import nn

from swarmbots.learn.action_dists.hybrid_action_dist import HybridActionDistribution
from swarmbots.learn.algos.mat_qc_base_policy import MATQCBasePolicy
from swarmbots.learn.algos.mat_qcx.mat_qcx_decoder import MATQCXDecoder, MATQCXDecoderConfig
from swarmbots.learn.nn_components.activations import ActivationFactory
from swarmbots.learn.nn_components.feed_forward import MLP
from swarmbots.learn.nn_components.nn_init import make_init_linear_orthogonal


class TMASACActorHeadKind(Enum):
    INDEPENDENT = "independent"
    QCX = "qcx"
    DECENTRALIZED = "decentralized"


@dataclass(frozen=True)
class TMASACActorHeadConfig:
    kind: TMASACActorHeadKind | str = TMASACActorHeadKind.INDEPENDENT
    hidden_dims: list[int] | None = None
    normalize_input: bool = False
    init_gain: float = 1.0
    qcx_decoder_config: MATQCXDecoderConfig = field(default_factory=MATQCXDecoderConfig)


class TMASACActorHead(nn.Module):
    latent_dim: int
    supports_standalone_compile: bool = True

    def actions_and_log_probs(
            self,
            *,
            actor_latents: torch.Tensor,
            action_dist: HybridActionDistribution,
            agent_mask: torch.Tensor | None,
            previous_actions: torch.Tensor | None,
            deterministic: bool,
            use_rsample: bool,
            stratified_sample_dim: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @staticmethod
    def _mask_actions(
            actions: torch.Tensor,
            agent_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if agent_mask is None:
            return actions
        return actions.masked_fill(~agent_mask.unsqueeze(-1), 0.0)

    @staticmethod
    def _mask_log_probs(
            log_probs: torch.Tensor,
            agent_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if agent_mask is None:
            return log_probs
        return log_probs.masked_fill(~agent_mask, 0.0)


class TMASACIndependentActorHead(TMASACActorHead):
    def __init__(
            self,
            *,
            d_model: int,
            config: TMASACActorHeadConfig,
            act_fn_cls: ActivationFactory,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.config = config
        self.input_norm = nn.LayerNorm(d_model) if config.normalize_input else nn.Identity()
        if config.hidden_dims:
            self.head = MLP(
                input_dim=d_model,
                hidden_dims=[*config.hidden_dims],
                end_with_act_fn=True,
                linear_init=make_init_linear_orthogonal(config.init_gain),
                act_fn_cls=act_fn_cls,
            )
            self.latent_dim = int(config.hidden_dims[-1])
        else:
            self.head = nn.Identity()
            self.latent_dim = int(d_model)

    def forward(self, actor_latents: torch.Tensor, agent_mask: torch.Tensor | None = None) -> torch.Tensor:
        _ = agent_mask
        return self.head(self.input_norm(actor_latents)).contiguous()

    def actions_and_log_probs(
            self,
            *,
            actor_latents: torch.Tensor,
            action_dist: HybridActionDistribution,
            agent_mask: torch.Tensor | None,
            previous_actions: torch.Tensor | None,
            deterministic: bool,
            use_rsample: bool,
            stratified_sample_dim: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent_pi = self(actor_latents, agent_mask=agent_mask)
        actions, log_probs = action_dist.get_actions_with_log_probs(
            latent_pi,
            deterministic=deterministic,
            previous_actions=previous_actions,
            use_rsample=use_rsample,
            stratified_sample_dim=stratified_sample_dim,
        )
        return (
            self._mask_actions(actions, agent_mask),
            self._mask_log_probs(log_probs, agent_mask),
        )


class TMASACQCXActorHead(TMASACActorHead):
    supports_standalone_compile = False

    def __init__(
            self,
            *,
            actor_latent_dim: int,
            action_dim: int,
            n_agents: int,
            max_agents: int,
            config: TMASACActorHeadConfig,
            act_fn_cls: ActivationFactory,
            dropout: float,
    ) -> None:
        super().__init__()
        decoder_d_model = (
            actor_latent_dim
            if config.qcx_decoder_config.d_model is None
            else int(config.qcx_decoder_config.d_model)
        )
        decoder_config = replace(
            config.qcx_decoder_config,
            d_model=decoder_d_model,
            act_fn_cls=act_fn_cls,
            dropout=dropout,
        )
        if decoder_config.actor_head_hidden_dims is not None:
            raise ValueError(
                "TMASAC QCX uses TMASACActorHeadConfig.hidden_dims; "
                "qcx_decoder_config.actor_head_hidden_dims must be None."
            )
        if decoder_config.normalize_actor_head_input:
            raise ValueError(
                "TMASAC QCX uses TMASACActorHeadConfig.normalize_input; "
                "qcx_decoder_config.normalize_actor_head_input must be False."
            )

        self.decoder_config = decoder_config
        self.action_dim = int(action_dim)
        self.n_agents = int(n_agents)
        self.action_input_norm = (
            nn.LayerNorm(action_dim)
            if decoder_config.normalize_action_input
            else nn.Identity()
        )
        action_encoder_dims = decoder_config.action_encoder_dims
        if action_encoder_dims is None:
            self.action_encoder = MATQCBasePolicy._build_token_encoder(
                input_dim=action_dim,
                output_dim=decoder_d_model,
                hidden_dims=None,
                act_fn_cls=act_fn_cls,
                linear_init_gain=decoder_config.token_encoder_init_gain,
                projection_init_gain=decoder_config.token_encoder_projection_init_gain,
                end_with_act_fn=decoder_config.action_encoder_end_with_act_fn,
            )
            action_token_dim = decoder_d_model
        elif len(action_encoder_dims) == 0:
            self.action_encoder = nn.Identity()
            action_token_dim = action_dim
        else:
            self.action_encoder = MATQCBasePolicy._build_encoder_from_dims(
                input_dim=action_dim,
                dims=action_encoder_dims,
                act_fn_cls=act_fn_cls,
                linear_init_gain=decoder_config.token_encoder_init_gain,
                projection_init_gain=decoder_config.token_encoder_projection_init_gain,
                end_with_act_fn=decoder_config.action_encoder_end_with_act_fn,
            )
            action_token_dim = action_encoder_dims[-1]
        self.action_token_norm = (
            nn.LayerNorm(action_token_dim)
            if decoder_config.normalize_action_tokens
            else nn.Identity()
        )

        self.memory_input_norm = (
            nn.LayerNorm(actor_latent_dim)
            if decoder_config.normalize_memory_input
            else nn.Identity()
        )
        memory_dims = decoder_config.memory_dims
        if memory_dims is None:
            self.memory_encoder = nn.Identity()
            memory_dim = actor_latent_dim
        elif len(memory_dims) == 0:
            raise ValueError("qcx_decoder_config.memory_dims must be None or contain at least one dimension")
        else:
            self.memory_encoder = MATQCBasePolicy._build_encoder_from_dims(
                input_dim=actor_latent_dim,
                dims=memory_dims,
                act_fn_cls=act_fn_cls,
                linear_init_gain=decoder_config.token_encoder_init_gain,
                projection_init_gain=decoder_config.token_encoder_projection_init_gain,
                end_with_act_fn=decoder_config.memory_encoder_end_with_act_fn,
            )
            memory_dim = memory_dims[-1]
        self.memory_token_norm = (
            nn.LayerNorm(memory_dim)
            if decoder_config.normalize_memory_tokens
            else nn.Identity()
        )
        self.decoder = MATQCXDecoder(
            config=decoder_config,
            max_agents=max_agents,
            input_d_model=actor_latent_dim,
            action_d_model=action_token_dim,
            memory_d_model=memory_dim,
        )
        self.output_head = TMASACIndependentActorHead(
            d_model=decoder_d_model,
            config=config,
            act_fn_cls=act_fn_cls,
        )
        self.latent_dim = self.output_head.latent_dim
        self.action_token_dim = action_token_dim

    def actions_and_log_probs(
            self,
            *,
            actor_latents: torch.Tensor,
            action_dist: HybridActionDistribution,
            agent_mask: torch.Tensor | None,
            previous_actions: torch.Tensor | None,
            deterministic: bool,
            use_rsample: bool,
            stratified_sample_dim: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        leading_shape = actor_latents.shape[:-2]
        flat_actor_latents = actor_latents.reshape(-1, self.n_agents, actor_latents.shape[-1])
        flat_agent_mask = None if agent_mask is None else agent_mask.reshape(-1, self.n_agents)
        memory_tokens = self.encode_memory_tokens(flat_actor_latents)
        action_tokens = flat_actor_latents.new_zeros(
            (flat_actor_latents.shape[0], 0, self.action_token_dim)
        )
        actions_by_agent: list[torch.Tensor] = []
        log_probs_by_agent: list[torch.Tensor] = []
        actor_outputs_by_agent: list[torch.Tensor] = []

        for agent_idx in range(self.n_agents):
            flat_latent_pi = self.forward_step(
                actor_latents=flat_actor_latents,
                memory_tokens=memory_tokens,
                query_agent_idx=agent_idx,
                action_tokens=action_tokens,
                agent_mask=flat_agent_mask,
            )
            latent_pi = flat_latent_pi.reshape(*leading_shape, 1, self.latent_dim)
            sample_agent_idx = agent_idx if action_dist.sampling_depends_on_agent else None
            previous_action = (
                None
                if previous_actions is None
                else previous_actions[..., agent_idx:agent_idx + 1, :]
            )
            action, log_prob = action_dist.get_actions_with_log_probs(
                latent_pi,
                deterministic=deterministic,
                agent=sample_agent_idx,
                previous_actions=previous_action,
                use_rsample=use_rsample,
                stratified_sample_dim=stratified_sample_dim,
            )
            if agent_mask is not None:
                active_agent = agent_mask[..., agent_idx:agent_idx + 1]
                action = action.masked_fill(~active_agent.unsqueeze(-1), 0.0)
                log_prob = log_prob.masked_fill(~active_agent, 0.0)
            actions_by_agent.append(action)
            log_probs_by_agent.append(log_prob)
            actor_outputs_by_agent.append(latent_pi)
            action_tokens = torch.cat(
                (
                    action_tokens,
                    self.encode_action_tokens(
                        action.reshape(-1, 1, self.action_dim)
                    ),
                ),
                dim=1,
            )

        actions = torch.cat(actions_by_agent, dim=-2)
        log_probs = torch.cat(log_probs_by_agent, dim=-1)
        actor_outputs = torch.cat(actor_outputs_by_agent, dim=-2)
        action_dist.update_latent_features(actor_outputs)
        return actions, log_probs

    def forward(
            self,
            actor_latents: torch.Tensor,
            actions: torch.Tensor,
            agent_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        flat_latents, leading_shape = self._flatten_agent_tensor(actor_latents)
        flat_actions, _ = self._flatten_agent_tensor(actions)
        flat_agent_mask = self._flatten_agent_mask(agent_mask)
        action_tokens = self.encode_action_tokens(flat_actions[:, :-1, :])
        decoder_output = self.decoder(
            query_tokens=flat_latents,
            action_tokens=action_tokens,
            memory_tokens=self.encode_memory_tokens(flat_latents),
            agent_mask=flat_agent_mask,
            memory_mask=flat_agent_mask,
        )
        latent_pi = self.output_head(decoder_output)
        return latent_pi.reshape(*leading_shape, actor_latents.shape[-2], self.latent_dim)

    def forward_step(
            self,
            *,
            actor_latents: torch.Tensor,
            memory_tokens: torch.Tensor,
            query_agent_idx: int,
            action_tokens: torch.Tensor,
            agent_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        decoder_output = self.decoder.forward_step(
            action_tokens=action_tokens,
            query_token=actor_latents[:, query_agent_idx:query_agent_idx + 1, :],
            memory_tokens=memory_tokens,
            query_prefix_tokens=actor_latents[:, :query_agent_idx, :],
            context_mask=None if agent_mask is None else agent_mask[:, :query_agent_idx],
            query_prefix_mask=None if agent_mask is None else agent_mask[:, :query_agent_idx],
            query_mask=None if agent_mask is None else agent_mask[:, query_agent_idx],
            memory_mask=agent_mask,
        )
        return self.output_head(decoder_output)

    def encode_action_tokens(self, actions: torch.Tensor) -> torch.Tensor:
        return self.action_token_norm(self.action_encoder(self.action_input_norm(actions)))

    def encode_memory_tokens(self, actor_latents: torch.Tensor) -> torch.Tensor:
        return self.memory_token_norm(self.memory_encoder(self.memory_input_norm(actor_latents)))

    @staticmethod
    def _flatten_agent_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
        leading_shape = tensor.shape[:-2]
        return tensor.reshape(-1, tensor.shape[-2], tensor.shape[-1]), leading_shape

    @staticmethod
    def _flatten_agent_mask(agent_mask: torch.Tensor | None) -> torch.Tensor | None:
        if agent_mask is None:
            return None
        return agent_mask.reshape(-1, agent_mask.shape[-1])
