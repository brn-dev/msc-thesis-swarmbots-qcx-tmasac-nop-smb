from dataclasses import dataclass, field, replace
from typing import Any

import torch
from torch import nn

from swarmbots.learn.algos.mat_qc_base_policy import MATQCBasePolicy, MATQCBasePolicyConfig
from swarmbots.learn.algos.mat_qcx.mat_qcx_decoder import MATQCXDecoder, MATQCXDecoderConfig
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.serialization_utils import serialize_dataclass


@dataclass(frozen=True)
class MATQCXPolicyConfig(MATQCBasePolicyConfig):
    decoder_config: MATQCXDecoderConfig = field(default_factory=MATQCXDecoderConfig)


class MATQCXPolicy(MATQCBasePolicy):

    def __init__(
            self,
            env: BaseLearnEnvWrapper,
            config: MATQCXPolicyConfig = MATQCXPolicyConfig(),
    ) -> None:
        super().__init__(env=env, config=config)

        self.action_input_norm = (
            nn.LayerNorm(self.agent_action_dim)
            if config.decoder_config.normalize_action_input
            else nn.Identity()
        )
        action_encoder_dims = config.decoder_config.action_encoder_dims
        if action_encoder_dims is None:
            self.action_encoder = self._build_token_encoder(
                input_dim=self.agent_action_dim,
                output_dim=self.d_model_decoder,
                hidden_dims=None,
                act_fn_cls=config.act_fn_cls,
                linear_init_gain=config.decoder_config.token_encoder_init_gain,
                projection_init_gain=config.decoder_config.token_encoder_projection_init_gain,
                end_with_act_fn=config.decoder_config.action_encoder_end_with_act_fn,
            )
        elif len(action_encoder_dims) == 0:
            self.action_encoder = nn.Identity()
        else:
            self.action_encoder = self._build_encoder_from_dims(
                input_dim=self.agent_action_dim,
                dims=action_encoder_dims,
                act_fn_cls=config.act_fn_cls,
                linear_init_gain=config.decoder_config.token_encoder_init_gain,
                projection_init_gain=config.decoder_config.token_encoder_projection_init_gain,
                end_with_act_fn=config.decoder_config.action_encoder_end_with_act_fn,
            )
        self.action_token_norm = (
            nn.LayerNorm(self._initial_decoder_context_token_dim())
            if config.decoder_config.normalize_action_tokens
            else nn.Identity()
        )

        if self.config.compile_modules:
            self.action_input_norm = self._compile_module(self.action_input_norm)
            self.action_encoder = self._compile_module(self.action_encoder)
            self.action_token_norm = self._compile_module(self.action_token_norm)

    def get_hyper_parameters(self) -> dict[str, Any]:
        hyper_parameters = self._get_hyper_parameters_payload()
        hyper_parameters["decoder_config"] = serialize_dataclass(self.decoder_config)
        return {
            "mat_qcx_policy_config": hyper_parameters,
        }

    def _build_decoder_config(self) -> MATQCXDecoderConfig:
        return replace(
            self.config.decoder_config,
            d_model=self.d_model_decoder,
            act_fn_cls=self.config.act_fn_cls,
            dropout=self.config.dropout,
        )

    def _build_query_input_norm(self) -> nn.Module:
        return nn.Identity()

    def _build_query_encoder(self) -> nn.Module:
        return nn.Identity()

    def _build_query_token_norm(self) -> nn.Module:
        return nn.Identity()

    def _build_context_input_norm(self) -> nn.Module:
        return nn.Identity()

    def _build_context_encoder(self) -> nn.Module:
        return nn.Identity()

    def _build_context_token_norm(self) -> nn.Module:
        return nn.Identity()

    def _build_memory_encoder(self) -> tuple[nn.Module, int]:
        memory_dims = self.config.decoder_config.memory_dims
        if memory_dims is None:
            return nn.Identity(), self.d_model_encoder
        if len(memory_dims) == 0:
            raise ValueError("decoder_config.memory_dims must be None or contain at least one dimension")
        return (
            self._build_encoder_from_dims(
                input_dim=self.d_model_encoder,
                dims=memory_dims,
                act_fn_cls=self.config.act_fn_cls,
                linear_init_gain=self.config.decoder_config.token_encoder_init_gain,
                projection_init_gain=self.config.decoder_config.token_encoder_projection_init_gain,
                end_with_act_fn=self.config.decoder_config.memory_encoder_end_with_act_fn,
            ),
            memory_dims[-1],
        )

    def _initial_decoder_context_token_dim(self) -> int:
        action_encoder_dims = self.config.decoder_config.action_encoder_dims
        if action_encoder_dims is None:
            return self.d_model_decoder
        if len(action_encoder_dims) == 0:
            return self.agent_action_dim
        return action_encoder_dims[-1]

    def _build_decoder(
            self,
            decoder_config: MATQCXDecoderConfig,
    ) -> nn.Module:
        return MATQCXDecoder(
            config=decoder_config,
            max_agents=self.max_agents,
            input_d_model=self.d_model_encoder,
            action_d_model=self._initial_decoder_context_token_dim(),
            memory_d_model=self.memory_d_model,
        )

    def _encode_query_tokens(
            self,
            augmented_observations: torch.Tensor,
            *,
            agent_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = agent_embeddings
        return augmented_observations

    def _encode_decoder_context_tokens(
            self,
            augmented_observations: torch.Tensor,
            actions: torch.Tensor,
            *,
            agent_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = augmented_observations, agent_embeddings
        return self.action_token_norm(self.action_encoder(self.action_input_norm(actions)))

    def _parallel_decoder_context_inputs(
            self,
            augmented_observations: torch.Tensor,
            actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return augmented_observations[:, :-1, :], actions[:, :-1, :]

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
            action_tokens=decoder_context_tokens,
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
            action_tokens=decoder_context_tokens,
            memory_tokens=memory_tokens,
            agent_mask=agent_mask,
            memory_mask=memory_mask,
        )

    def get_grad_norms(self) -> dict[str, float]:
        return {
            "encoder": self._module_grad_norm(self.encoder),
            "action_input_norm": self._module_grad_norm(self.action_input_norm),
            "action_encoder": self._module_grad_norm(self.action_encoder),
            "action_token_norm": self._module_grad_norm(self.action_token_norm),
            "memory_input_norm": self._module_grad_norm(self.memory_input_norm),
            "memory_encoder": self._module_grad_norm(self.memory_encoder),
            "memory_token_norm": self._module_grad_norm(self.memory_token_norm),
            "decoder": self._module_grad_norm(self.decoder),
            "actor_head_input_norm": self._module_grad_norm(self.actor_head_input_norm),
            "actor_head": self._module_grad_norm(self.actor_head),
            "action_dist": self._module_grad_norm(self.action_dist),
            "critic": self._module_grad_norm(self.critic),
            "total": self._module_grad_norm(self),
        }
