from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from swarmbots.learn.action_dists.action_dist import ActionMetricsSplitterInput
from swarmbots.learn.action_dists.hybrid_action_dist import HybridActionDistribution
from swarmbots.learn.algos.ppo.base_ppo_policy import BasePPOPolicy, DelegatingPPOPolicyTemporalStateMixin
from swarmbots.learn.algos.ppo.ppo_rollout_batch import PPORolloutBatch
from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPOEpisodeSegment
from swarmbots.learn.algos.ppo.ppo_sampler import PPOSamples, PPOSamplerConfig
from swarmbots.learn.algos.r_mat.r_mat_policy_mixin import RMATPolicyMixin
from swarmbots.learn.algos.world_modeling.base_wm_sampler import BaseWMSampler
from swarmbots.learn.algos.world_modeling.ppo_wm_sampler import (
    PPOWMBatchSampler,
    PPOWMSampler,
    PPOWMSamples,
    PPOWMSamplerConfig,
)
from swarmbots.learn.algos.world_modeling.spr_mixin import SPRMixin
from swarmbots.learn.algos.world_modeling.transformer_transition_model import (
    TransformerTransitionModel,
    TransformerTransitionModelConfig,
)
from swarmbots.learn.algos.world_modeling.wm_recurrent_batch import build_wm_target_time_mask
from swarmbots.learn.losses import LossDict, LossMetrics
from swarmbots.learn.nn_components.activations import ActivationFactory
from swarmbots.learn.nn_components.feed_forward import MLP


@dataclass(frozen=True)
class SPRWorldModelConfig:
    n_agents: int
    local_latent_dim: int
    action_dim: int
    world_model_loss_coef: float = 1.0
    world_model_target_tau: float | None = None
    act_fn_cls: ActivationFactory = nn.GELU
    transition_model_dropout: float = 0.0
    online_encoder_attr: str = "encoder"
    d_model_transition_model: int = 128
    nhead_transition_model: int = 4
    num_layers_transition_model: int = 2
    dim_feedforward_transition_model: int = 256
    add_agent_embeddings_transition_model: bool = False
    transition_model_coembed_hidden_dims: list[int] | None = None
    transition_model_head_hidden_dims: list[int] | None = None
    spr_projection_dims: list[int] | None = None
    spr_predictor_hidden_dims: list[int] | None = None
    residual_predictor: bool = True
    spr_loss_weight: float = 1.0


class SPRWrapper(
    DelegatingPPOPolicyTemporalStateMixin,
    BasePPOPolicy[PPOWMSamples, PPOWMSamplerConfig],
    SPRMixin,
):

    def __init__(
            self,
            policy: BasePPOPolicy[PPOSamples, PPOSamplerConfig],
            *,
            world_model_config: SPRWorldModelConfig,
    ) -> None:
        super().__init__()
        if world_model_config.n_agents < 1:
            raise ValueError(f"n_agents must be >= 1, got {world_model_config.n_agents}")
        if world_model_config.local_latent_dim < 1:
            raise ValueError(f"local_latent_dim must be >= 1, got {world_model_config.local_latent_dim}")
        if world_model_config.action_dim < 1:
            raise ValueError(f"action_dim must be >= 1, got {world_model_config.action_dim}")
        if world_model_config.world_model_loss_coef < 0:
            raise ValueError(f"world_model_loss_coef must be >= 0, got {world_model_config.world_model_loss_coef}")
        if (
                world_model_config.world_model_target_tau is not None
                and not (0.0 < world_model_config.world_model_target_tau <= 1.0)
        ):
            raise ValueError(
                f"world_model_target_tau must be in (0, 1], got {world_model_config.world_model_target_tau}"
            )
        if not (0.0 <= world_model_config.transition_model_dropout < 1.0):
            raise ValueError(
                "transition_model_dropout must be in [0, 1), "
                f"got {world_model_config.transition_model_dropout}"
            )
        if not world_model_config.online_encoder_attr:
            raise ValueError("online_encoder_attr must be a non-empty attribute path")
        if world_model_config.spr_loss_weight < 0:
            raise ValueError(f"spr_loss_weight must be >= 0, got {world_model_config.spr_loss_weight}")
        if _contains_rmat_policy(policy):
            raise ValueError(
                "SPRWrapper does not support RMAT policies. "
                "SPR target latents currently ignore recurrent history, so use NextObsPredWrapper with RMAT instead."
            )
        self.policy = policy
        self._online_encoder_attr = world_model_config.online_encoder_attr
        _ = self.online_encoder
        self._setup_world_model_from_config(world_model_config)

    @property
    def action_dist(self) -> HybridActionDistribution:
        return self.policy.action_dist

    @property
    def has_popart(self) -> bool:
        return self.policy.has_popart

    def update_value_normalizer(self, targets: torch.Tensor) -> None:
        self.policy.update_value_normalizer(targets)

    def normalize_values(self, values: torch.Tensor) -> torch.Tensor:
        return self.policy.normalize_values(values)

    def get_value_normalizer_metrics(self) -> dict[str, float]:
        return self.policy.get_value_normalizer_metrics()

    @property
    def online_encoder(self) -> nn.Module:
        current: Any = self.policy
        for attr_name in self._online_encoder_attr.split("."):
            if not hasattr(current, attr_name):
                raise ValueError(
                    f"Could not resolve online encoder attribute path "
                    f"{self._online_encoder_attr!r} on policy {type(self.policy).__name__}"
                )
            current = getattr(current, attr_name)
        if not isinstance(current, nn.Module):
            raise TypeError(
                f"Expected online encoder at {self._online_encoder_attr!r} to be nn.Module, got {type(current).__name__}"
            )
        return current

    def forward(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
            deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.policy(
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            deterministic=deterministic,
        )

    def predict_values(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.policy.predict_values(
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
        )

    def _evaluate_actions(
            self,
            batch: PPOWMSamples,
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> tuple[torch.Tensor, torch.Tensor, LossDict, LossMetrics, torch.Tensor]:
        return self.policy._evaluate_actions(
            batch=batch,
            action_splitter=action_splitter,
        )

    def evaluate_actions(
            self,
            batch: PPOWMSamples,
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        LossDict,
        LossMetrics,
    ]:
        log_probs, values, extra_losses, extra_loss_metrics, local_latents = self._evaluate_actions(
            batch=batch,
            action_splitter=action_splitter,
        )
        wm_target_time_mask = build_wm_target_time_mask(
            wm_target_time_mask=batch.wm_target_time_mask,
            sequence_time_mask=getattr(batch, "time_mask", None),
        )
        spr_loss = self.compute_spr_loss(
            online_local_latents=local_latents,
            next_local_obs=batch.next_local_obs,
            next_global_obs=batch.next_global_obs,
            actions=batch.wm_actions,
            agent_mask=batch.wm_agent_mask,
            loss_agent_mask=batch.wm_loss_agent_mask,
            time_mask=wm_target_time_mask,
        )
        spr_loss_weighted = self.spr_loss_weight * spr_loss
        world_model_loss_scaled = self.world_model_loss_coef * spr_loss_weighted
        merged_extra_losses = dict(extra_losses)
        merged_extra_losses["world_model"] = world_model_loss_scaled
        merged_extra_loss_metrics = dict(extra_loss_metrics)
        merged_extra_loss_metrics["spr_loss"] = spr_loss.item()
        merged_extra_loss_metrics["spr_loss_weighted"] = spr_loss_weighted.item()
        merged_extra_loss_metrics["wm_loss"] = spr_loss_weighted.item()
        merged_extra_loss_metrics["wm_loss_scaled"] = world_model_loss_scaled.item()
        return (
            log_probs,
            values,
            merged_extra_losses,
            merged_extra_loss_metrics,
        )

    def after_optimizer_step(self) -> None:
        self.policy.after_optimizer_step()
        tau = self.world_model_target_tau
        if tau is None:
            return
        self.update_spr_targets(tau)

    def make_sampler(
            self,
            episodes: list[PPOEpisodeSegment],
            config: PPOWMSamplerConfig,
    ) -> BaseWMSampler[Any, PPOWMSamplerConfig]:
        if type(config) is PPOWMSamplerConfig:
            return PPOWMSampler(
                episodes=episodes,
                config=config,
                requires_previous_actions=self.requires_previous_actions(),
            )
        sampler = self.policy.make_sampler(episodes=episodes, config=config)
        if not isinstance(sampler, BaseWMSampler):
            raise ValueError("Policy does not create a WM Sampler!")
        return sampler

    def supports_rollout_batch_sampler(self, config: PPOWMSamplerConfig) -> bool:
        if type(config) is PPOWMSamplerConfig:
            return True
        return self.policy.supports_rollout_batch_sampler(config)

    def make_rollout_batch_sampler(
            self,
            rollout_batch: PPORolloutBatch,
            config: PPOWMSamplerConfig,
    ) -> BaseWMSampler[Any, PPOWMSamplerConfig]:
        if type(config) is PPOWMSamplerConfig:
            return PPOWMBatchSampler(
                rollout_batch=rollout_batch,
                config=config,
                requires_previous_actions=self.requires_previous_actions(),
            )
        sampler = self.policy.make_rollout_batch_sampler(rollout_batch=rollout_batch, config=config)
        if not isinstance(sampler, BaseWMSampler):
            raise ValueError("Policy does not create a WM Sampler!")
        return sampler

    def get_hyper_parameters(self) -> dict[str, Any]:
        projection_dims = [] if self._spr_projection_dims is None else self._spr_projection_dims
        predictor_hidden_dims = [] if self._spr_predictor_hidden_dims is None else self._spr_predictor_hidden_dims
        residual_predictor = False if self._spr_residual_predictor is None else self._spr_residual_predictor
        return {
            **self.policy.get_hyper_parameters(),
            "spr_wrapper": {
                "world_model_loss_coef": self.world_model_loss_coef,
                "world_model_target_tau": self.world_model_target_tau,
                "n_agents": self._wm_n_agents,
                "local_latent_dim": self._wm_local_latent_dim,
                "action_dim": self._wm_action_dim,
                "act_fn_cls": str(self._wm_act_fn_cls),
                "transition_model_dropout": self._wm_transition_model_dropout,
                "online_encoder_attr": self._online_encoder_attr,
                "world_model_config": self.get_spr_hyper_parameters(
                    projection_dims=projection_dims,
                    predictor_hidden_dims=predictor_hidden_dims,
                    residual_predictor=residual_predictor,
                ),
            },
        }

    def get_grad_norms(self) -> dict[str, float]:
        grad_norms = self.policy.get_grad_norms()
        grad_norms.update(
            {
                "wm_transition_model": self._module_grad_norm(self.transition_model),
                "wm_online_projection": self._module_grad_norm(self.online_projection),
                "wm_predictor": self._module_grad_norm(self.predictor),
                "wm_target_encoder": self._module_grad_norm(self.target_encoder),
                "wm_target_projection": self._module_grad_norm(self.target_projection),
            }
        )
        return grad_norms

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
        return self.policy.act(
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            deterministic=deterministic,
        )

    def requires_previous_actions(self) -> bool:
        return self.policy.requires_previous_actions()

    def update_loss_weights(self, **weights: float) -> None:
        if not weights:
            return

        remaining_weights = dict(weights)
        spr_weight = self._pop_loss_weight_alias(
            remaining_weights,
            aliases=("spr_loss_weight", "spr_loss", "spr"),
        )
        if spr_weight is not None:
            alias, value = spr_weight
            if value < 0:
                raise ValueError(f"{alias} must be >= 0, got {value}")
            self.spr_loss_weight = value
        world_model_weight = self._pop_loss_weight_alias(
            remaining_weights,
            aliases=("world_model_loss_coef", "wm_loss_coef"),
        )
        if world_model_weight is not None:
            alias, value = world_model_weight
            if value < 0:
                raise ValueError(f"{alias} must be >= 0, got {value}")
            self.world_model_loss_coef = value

        self.policy.update_loss_weights(**remaining_weights)

    def _setup_world_model_from_config(self, world_model_config: SPRWorldModelConfig) -> None:
        self.world_model_loss_coef = float(world_model_config.world_model_loss_coef)
        self.world_model_target_tau = world_model_config.world_model_target_tau
        self.spr_loss_weight = float(world_model_config.spr_loss_weight)
        self._wm_n_agents = int(world_model_config.n_agents)
        self._wm_local_latent_dim = int(world_model_config.local_latent_dim)
        self._wm_action_dim = int(world_model_config.action_dim)
        self._wm_act_fn_cls = world_model_config.act_fn_cls
        self._wm_transition_model_dropout = float(world_model_config.transition_model_dropout)

        if world_model_config.spr_projection_dims is None:
            projection_dims = [self._wm_local_latent_dim]
        else:
            projection_dims = list(world_model_config.spr_projection_dims)

        if world_model_config.spr_predictor_hidden_dims is None:
            predictor_hidden_dims = [projection_dims[-1]]
        else:
            predictor_hidden_dims = [*world_model_config.spr_predictor_hidden_dims, projection_dims[-1]]

        self._spr_projection_dims = list(projection_dims)
        self._spr_predictor_hidden_dims = list(predictor_hidden_dims)
        self._spr_residual_predictor = world_model_config.residual_predictor

        self.setup_spr(
            transition_model=TransformerTransitionModel(config=TransformerTransitionModelConfig(
                n_agents=self._wm_n_agents,
                latent_dim=self._wm_local_latent_dim,
                action_dim=self._wm_action_dim,
                d_model=world_model_config.d_model_transition_model,
                nhead=world_model_config.nhead_transition_model,
                num_layers=world_model_config.num_layers_transition_model,
                dim_feedforward=world_model_config.dim_feedforward_transition_model,
                dropout=self._wm_transition_model_dropout,
                act_fn_cls=self._wm_act_fn_cls,
                add_agent_embeddings=world_model_config.add_agent_embeddings_transition_model,
                predict_delta=True,
                coembed_mlp_hidden_dims=world_model_config.transition_model_coembed_hidden_dims,
                head_mlp_hidden_dims=world_model_config.transition_model_head_hidden_dims,
            )),
            projection=MLP(
                input_dim=self._wm_local_latent_dim,
                hidden_dims=[*projection_dims],
                end_with_act_fn=False,
                act_fn_cls=self._wm_act_fn_cls,
            ),
            predictor=MLP(
                input_dim=projection_dims[-1],
                hidden_dims=[*predictor_hidden_dims],
                end_with_act_fn=False,
                act_fn_cls=self._wm_act_fn_cls,
            ),
            residual_predictor=world_model_config.residual_predictor,
        )


def _contains_rmat_policy(policy: BasePPOPolicy[Any, Any]) -> bool:
    current: Any = policy
    seen_ids: set[int] = set()
    while True:
        if isinstance(current, RMATPolicyMixin):
            return True
        current_id = id(current)
        if current_id in seen_ids or not hasattr(current, "policy"):
            return False
        seen_ids.add(current_id)
        current = current.policy
