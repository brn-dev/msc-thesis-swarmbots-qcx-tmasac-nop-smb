from dataclasses import dataclass, replace
from typing import Any

import torch

from swarmbots.learn.algos.mat.mat_encoder import MATEncoder, MATEncoderConfig
from swarmbots.learn.algos.mat.mat_ind_policy import MATIndPolicy, MATIndPolicyConfig
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import (
    BaseLearnEnvWrapper,
)
from swarmbots.learn.serialization_utils import serialize_dataclass


@dataclass(frozen=True)
class MATDecPolicyConfig(MATIndPolicyConfig):
    actor_encoder_config: MATEncoderConfig | None = None


class MATDecPolicy(MATIndPolicy):

    def __init__(
            self,
            env: BaseLearnEnvWrapper,
            config: MATDecPolicyConfig | None = None,
    ) -> None:
        config = MATDecPolicyConfig() if config is None else config
        super().__init__(env=env, config=replace(config, compile_modules=False))
        self.config = config
        self.actor_encoder_config = self._resolve_actor_encoder_config()
        self.actor_encoder = MATEncoder(
            config=self.actor_encoder_config,
            max_agents=self.max_agents,
            local_obs_dim=self.local_obs_dim,
            global_obs_dim=self.global_obs_dim,
        )
        self._apply_optional_compile()

    def _actor_input_dim(self) -> int:
        return self._resolve_actor_encoder_config().d_model

    def _resolve_actor_encoder_config(self) -> MATEncoderConfig:
        source_config = self.config.actor_encoder_config or self.config.encoder_config
        return replace(
            source_config,
            act_fn_cls=self.config.act_fn_cls,
            dropout=self.config.dropout,
            use_agent_attention=False,
        )

    def _actor_observations(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            augmented_observations: torch.Tensor,
    ) -> torch.Tensor:
        _ = augmented_observations
        return self._encode_actor_observations(local_obs, global_obs)

    def _encode_actor_observations(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
    ) -> torch.Tensor:
        return self.actor_encoder(local_obs, global_obs)

    def get_hyper_parameters(self) -> dict[str, Any]:
        hyper_parameters = super().get_hyper_parameters()
        policy_config = hyper_parameters["mat_ind_policy_config"]
        policy_config["actor_encoder_config"] = serialize_dataclass(self.actor_encoder_config)
        return {
            "mat_dec_policy_config": policy_config,
        }

    def act(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
            deterministic: bool = False,
    ) -> torch.Tensor:
        _ = hidden_local_vars
        _ = hidden_global_vars
        actor_observations = self._encode_actor_observations(local_obs, global_obs)
        actions, _ = self._generate_actions(
            augmented_observations=actor_observations,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            deterministic=deterministic,
            return_log_probs=False,
        )
        return actions

    def get_grad_norms(self) -> dict[str, float]:
        grad_norms = super().get_grad_norms()
        grad_norms["actor_encoder"] = self._module_grad_norm(self.actor_encoder)
        return grad_norms

    def _apply_optional_compile(self) -> None:
        super()._apply_optional_compile()
        if self.config.compile_modules:
            self.actor_encoder = self._compile_module(self.actor_encoder)
