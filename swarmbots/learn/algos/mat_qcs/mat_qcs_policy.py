from dataclasses import dataclass, field, replace
from typing import Any

import torch
from torch import nn

from swarmbots.learn.algos.mat_qc_base_policy import (
    MATQCBasePolicy,
    MATQCBasePolicyConfig,
    MATQCCriticConfig,
    _ensure_torch_compile_available,
)
from swarmbots.learn.algos.mat_qcs.mat_qcs_decoder import MATQCSDecoder, MATQCSDecoderConfig
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper


MATQCSCriticConfig = MATQCCriticConfig


@dataclass(frozen=True)
class MATQCSPolicyConfig(MATQCBasePolicyConfig):
    decoder_config: MATQCSDecoderConfig = field(default_factory=MATQCSDecoderConfig)
    critic_config: MATQCSCriticConfig = field(default_factory=MATQCSCriticConfig)


class MATQCSPolicy(MATQCBasePolicy):

    def __init__(
            self,
            env: BaseLearnEnvWrapper,
            config: MATQCSPolicyConfig = MATQCSPolicyConfig(),
    ) -> None:
        super().__init__(env=env, config=config)

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            "mat_qcs_policy_config": self._get_hyper_parameters_payload(),
        }

    def _build_query_input_norm(self) -> nn.Module:
        return (
            nn.LayerNorm(self.d_model_encoder)
            if self.config.decoder_config.normalize_query_input
            else nn.Identity()
        )

    def _build_query_encoder(self) -> nn.Module:
        return self._build_token_encoder(
            input_dim=self.d_model_encoder,
            output_dim=self.d_model_decoder,
            hidden_dims=self.config.decoder_config.query_encoder_hidden_dims,
            act_fn_cls=self.config.act_fn_cls,
            linear_init_gain=self.config.decoder_config.token_encoder_init_gain,
            projection_init_gain=self.config.decoder_config.token_encoder_projection_init_gain,
            end_with_act_fn=self.config.decoder_config.token_encoder_end_with_act_fn,
        )

    def _build_query_token_norm(self) -> nn.Module:
        return (
            nn.LayerNorm(self.d_model_decoder)
            if self.config.decoder_config.normalize_query_tokens
            else nn.Identity()
        )

    def _build_context_input_norm(self) -> nn.Module:
        return (
            nn.LayerNorm(self.d_model_encoder + self.agent_action_dim)
            if self.config.decoder_config.normalize_context_input
            else nn.Identity()
        )

    def _build_context_encoder(self) -> nn.Module:
        return self._build_token_encoder(
            input_dim=self.d_model_encoder + self.agent_action_dim,
            output_dim=self.d_model_decoder,
            hidden_dims=self.config.decoder_config.context_encoder_hidden_dims,
            act_fn_cls=self.config.act_fn_cls,
            linear_init_gain=self.config.decoder_config.token_encoder_init_gain,
            projection_init_gain=self.config.decoder_config.token_encoder_projection_init_gain,
            end_with_act_fn=self.config.decoder_config.token_encoder_end_with_act_fn,
        )

    def _build_context_token_norm(self) -> nn.Module:
        return (
            nn.LayerNorm(self.d_model_decoder)
            if self.config.decoder_config.normalize_context_tokens
            else nn.Identity()
        )

    def _build_decoder_config(self) -> MATQCSDecoderConfig:
        return replace(
            self.config.decoder_config,
            d_model=self.d_model_decoder,
            act_fn_cls=self.config.act_fn_cls,
            dropout=self.config.dropout,
        )

    def _build_decoder(
            self,
            decoder_config: MATQCSDecoderConfig,
    ) -> nn.Module:
        return MATQCSDecoder(
            config=decoder_config,
            max_agents=self.max_agents,
            memory_d_model=self.memory_d_model,
        )

    def _encode_query_tokens(
            self,
            augmented_observations: torch.Tensor,
            *,
            agent_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = self.query_encoder(self.query_input_norm(augmented_observations))
        if agent_embeddings is not None:
            return self.query_token_norm(tokens + agent_embeddings)
        if self.agent_embeddings_decoder is not None:
            tokens = tokens + self.agent_embeddings_decoder[:, :tokens.shape[1], :]

        return self.query_token_norm(tokens)

    def _encode_decoder_context_tokens(
            self,
            augmented_observations: torch.Tensor,
            actions: torch.Tensor,
            *,
            agent_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context_input = torch.cat((augmented_observations, actions), dim=-1)
        tokens = self.context_encoder(self.context_input_norm(context_input))
        if agent_embeddings is not None:
            return self.context_token_norm(tokens + agent_embeddings)
        if self.agent_embeddings_decoder is not None:
            tokens = tokens + self.agent_embeddings_decoder[:, :tokens.shape[1], :]

        return self.context_token_norm(tokens)

    def _parallel_decoder_context_inputs(
            self,
            augmented_observations: torch.Tensor,
            actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return augmented_observations[:, :-1, :], actions[:, :-1, :]

    def _initial_decoder_context_token_dim(self) -> int:
        return self.d_model_decoder

    def _decode_step(
            self,
            *,
            decoder_context_tokens: torch.Tensor,
            query_token: torch.Tensor,
            memory_tokens: torch.Tensor,
            query_prefix_tokens: torch.Tensor,
            context_mask: torch.Tensor | None,
            query_prefix_mask: torch.Tensor | None,
            query_mask: torch.Tensor | None,
            memory_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.decoder.forward_step(
            context_tokens=decoder_context_tokens,
            query_token=query_token,
            memory_tokens=memory_tokens,
            query_prefix_tokens=query_prefix_tokens,
            context_mask=context_mask,
            query_prefix_mask=query_prefix_mask,
            query_mask=query_mask,
            memory_mask=memory_mask,
        )

    def _decode_parallel(
            self,
            *,
            query_tokens: torch.Tensor,
            decoder_context_tokens: torch.Tensor,
            memory_tokens: torch.Tensor,
            agent_mask: torch.Tensor | None,
            memory_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.decoder(
            query_tokens=query_tokens,
            context_tokens=decoder_context_tokens,
            memory_tokens=memory_tokens,
            agent_mask=agent_mask,
            memory_mask=memory_mask,
        )
