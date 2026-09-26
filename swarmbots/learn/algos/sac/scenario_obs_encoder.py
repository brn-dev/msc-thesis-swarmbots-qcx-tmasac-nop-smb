from dataclasses import dataclass

import torch
from torch import nn

from swarmbots.learn.nn_components.activations import ActivationFactory


@dataclass(frozen=True)
class ScenarioObservationSpec:
    scenario_id: int
    name: str
    global_obs_dim: int
    hidden_local_vars_dim: int
    hidden_global_vars_dim: int


@dataclass(frozen=True)
class ScenarioFieldEncoderConfig:
    output_dim: int
    hidden_dims: tuple[int, ...] = ()
    normalize_input: bool = False
    init_gain: float = 1.0


@dataclass(frozen=True)
class TMASACScenarioEncoderConfig:
    scenarios: tuple[ScenarioObservationSpec, ...]
    global_obs: ScenarioFieldEncoderConfig
    hidden_local_vars: ScenarioFieldEncoderConfig
    hidden_global_vars: ScenarioFieldEncoderConfig
    scenario_embedding_dim: int = 16


class ScenarioBatchedLinear(nn.Module):
    def __init__(
            self,
            *,
            num_scenarios: int,
            input_dim: int,
            output_dim: int,
            init_gain: float,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_scenarios, output_dim, input_dim))
        self.bias = nn.Parameter(torch.zeros(num_scenarios, output_dim))
        if input_dim == 0:
            return
        with torch.no_grad():
            for scenario_weight in self.weight:
                nn.init.orthogonal_(scenario_weight, gain=init_gain)

    def forward(self, inputs: torch.Tensor, scenario_ids: torch.Tensor) -> torch.Tensor:
        selected_weight = self.weight[scenario_ids]
        selected_bias = self.bias[scenario_ids]
        return torch.einsum("...oi,...i->...o", selected_weight, inputs) + selected_bias


class ScenarioFieldEncoder(nn.Module):
    def __init__(
            self,
            *,
            input_dims: tuple[int, ...],
            padded_input_dim: int,
            config: ScenarioFieldEncoderConfig,
            act_fn_cls: ActivationFactory,
    ) -> None:
        super().__init__()
        self.num_scenarios = len(input_dims)
        self.padded_input_dim = int(padded_input_dim)
        self.output_dim = int(config.output_dim)
        encoded_scenario_ids = tuple(
            scenario_id
            for scenario_id, input_dim in enumerate(input_dims)
            if input_dim > 0
        )
        empty_scenario_ids = tuple(
            scenario_id
            for scenario_id, input_dim in enumerate(input_dims)
            if input_dim == 0
        )
        scenario_encoder_indices = torch.zeros(self.num_scenarios, dtype=torch.long)
        scenario_vector_indices = torch.zeros(self.num_scenarios, dtype=torch.long)
        for encoder_index, scenario_id in enumerate(encoded_scenario_ids):
            scenario_encoder_indices[scenario_id] = encoder_index
        for vector_index, scenario_id in enumerate(empty_scenario_ids):
            scenario_vector_indices[scenario_id] = vector_index
        empty_scenario_mask = torch.zeros(self.num_scenarios, dtype=torch.bool)
        if empty_scenario_ids:
            empty_scenario_mask[list(empty_scenario_ids)] = True
        self.register_buffer(
            "scenario_encoder_indices",
            scenario_encoder_indices,
            persistent=False,
        )
        self.register_buffer(
            "scenario_vector_indices",
            scenario_vector_indices,
            persistent=False,
        )
        self.register_buffer(
            "empty_scenario_mask",
            empty_scenario_mask,
            persistent=False,
        )
        self.empty_scenario_vectors = (
            nn.Parameter(torch.zeros(len(empty_scenario_ids), self.output_dim))
            if empty_scenario_ids
            else None
        )

        feature_indices = torch.arange(self.padded_input_dim).unsqueeze(0)
        encoded_input_dims = tuple(input_dims[scenario_id] for scenario_id in encoded_scenario_ids)
        input_dim_tensor = torch.tensor(encoded_input_dims, dtype=torch.long).unsqueeze(1)
        self.register_buffer("input_mask", feature_indices < input_dim_tensor, persistent=False)

        self.input_norm_weight = (
            nn.Parameter(torch.ones(len(encoded_scenario_ids), self.padded_input_dim))
            if config.normalize_input and encoded_scenario_ids
            else None
        )
        self.input_norm_bias = (
            nn.Parameter(torch.zeros(len(encoded_scenario_ids), self.padded_input_dim))
            if config.normalize_input and encoded_scenario_ids
            else None
        )
        layer_dims = (self.padded_input_dim, *config.hidden_dims, self.output_dim)
        self.layers = nn.ModuleList([
            ScenarioBatchedLinear(
                num_scenarios=len(encoded_scenario_ids),
                input_dim=input_dim,
                output_dim=output_dim,
                init_gain=config.init_gain,
            )
            for input_dim, output_dim in zip(layer_dims[:-1], layer_dims[1:], strict=True)
        ]) if encoded_scenario_ids else nn.ModuleList()
        self.activations = nn.ModuleList([
            act_fn_cls()
            for _ in config.hidden_dims
        ]) if encoded_scenario_ids else nn.ModuleList()

    def forward(self, inputs: torch.Tensor, scenario_ids: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(inputs=inputs, scenario_ids=scenario_ids)
        if not self.layers:
            assert self.empty_scenario_vectors is not None
            return self.empty_scenario_vectors[self.scenario_vector_indices[scenario_ids]]

        encoder_indices = self.scenario_encoder_indices[scenario_ids]
        selected_empty_mask = self.empty_scenario_mask[scenario_ids]
        selected_input_mask = (
            self.input_mask[encoder_indices]
            & ~selected_empty_mask.unsqueeze(-1)
        )
        input_mask = selected_input_mask.to(dtype=inputs.dtype)
        hidden = inputs.masked_fill(~selected_input_mask, 0.0)
        if self.input_norm_weight is not None and self.input_norm_bias is not None:
            active_features = input_mask.sum(dim=-1, keepdim=True)
            safe_active_features = active_features.clamp_min(1.0)
            mean = hidden.sum(dim=-1, keepdim=True) / safe_active_features
            centered = (hidden - mean) * input_mask
            variance = centered.square().sum(dim=-1, keepdim=True) / safe_active_features
            standardized = centered * torch.rsqrt(variance + 1e-5)
            normalized = torch.where(active_features > 1.0, standardized, hidden)
            hidden = (
                normalized * self.input_norm_weight[encoder_indices]
                + self.input_norm_bias[encoder_indices] * input_mask
            )
        for layer_idx, layer in enumerate(self.layers):
            hidden = layer(hidden, encoder_indices)
            if layer_idx < len(self.activations):
                hidden = self.activations[layer_idx](hidden)
        if self.empty_scenario_vectors is None:
            return hidden
        learned_vectors = self.empty_scenario_vectors[
            self.scenario_vector_indices[scenario_ids]
        ]
        return torch.where(
            selected_empty_mask.unsqueeze(-1),
            learned_vectors,
            hidden,
        )

    def _validate_inputs(self, *, inputs: torch.Tensor, scenario_ids: torch.Tensor) -> None:
        if inputs.shape[:-1] != scenario_ids.shape:
            raise ValueError(
                "Expected scenario_ids to match observation leading dimensions, got "
                f"inputs={tuple(inputs.shape)}, scenario_ids={tuple(scenario_ids.shape)}."
            )
        if inputs.shape[-1] != self.padded_input_dim:
            raise ValueError(
                f"Expected padded input dimension {self.padded_input_dim}, got {inputs.shape[-1]}."
            )
        if scenario_ids.dtype != torch.long:
            raise ValueError(f"Expected scenario_ids dtype torch.long, got {scenario_ids.dtype}.")
        torch._assert_async(
            torch.all((scenario_ids >= 0) & (scenario_ids < self.num_scenarios)),
            f"scenario_ids must be in [0, {self.num_scenarios}).",
        )


class TMASACScenarioObservationEncoder(nn.Module):
    def __init__(
            self,
            *,
            config: TMASACScenarioEncoderConfig,
            global_obs_dim: int,
            hidden_local_vars_dim: int,
            hidden_global_vars_dim: int,
            act_fn_cls: ActivationFactory,
            include_hidden_fields: bool,
    ) -> None:
        super().__init__()
        _validate_config(
            config,
            global_obs_dim=global_obs_dim,
            hidden_local_vars_dim=hidden_local_vars_dim,
            hidden_global_vars_dim=hidden_global_vars_dim,
        )
        self.config = config
        self.num_scenarios = len(config.scenarios)
        self.scenario_names = tuple(spec.name for spec in config.scenarios)
        self.global_obs_encoder = ScenarioFieldEncoder(
            input_dims=tuple(spec.global_obs_dim for spec in config.scenarios),
            padded_input_dim=global_obs_dim,
            config=config.global_obs,
            act_fn_cls=act_fn_cls,
        )
        self.scenario_embedding = (
            nn.Embedding(self.num_scenarios, config.scenario_embedding_dim)
            if config.scenario_embedding_dim > 0
            else None
        )
        self.hidden_local_vars_encoder = (
            ScenarioFieldEncoder(
                input_dims=tuple(spec.hidden_local_vars_dim for spec in config.scenarios),
                padded_input_dim=hidden_local_vars_dim,
                config=config.hidden_local_vars,
                act_fn_cls=act_fn_cls,
            )
            if include_hidden_fields
            else None
        )
        self.hidden_global_vars_encoder = (
            ScenarioFieldEncoder(
                input_dims=tuple(spec.hidden_global_vars_dim for spec in config.scenarios),
                padded_input_dim=hidden_global_vars_dim,
                config=config.hidden_global_vars,
                act_fn_cls=act_fn_cls,
            )
            if include_hidden_fields
            else None
        )

    @property
    def global_output_dim(self) -> int:
        return self.config.global_obs.output_dim + self.config.scenario_embedding_dim

    @property
    def hidden_local_output_dim(self) -> int:
        return self.config.hidden_local_vars.output_dim

    @property
    def hidden_global_output_dim(self) -> int:
        return self.config.hidden_global_vars.output_dim

    def encode_global_obs(self, global_obs: torch.Tensor, scenario_ids: torch.Tensor) -> torch.Tensor:
        encoded = self.global_obs_encoder(global_obs, scenario_ids)
        if self.scenario_embedding is None:
            return encoded
        return torch.cat((encoded, self.scenario_embedding(scenario_ids)), dim=-1)

    def encode_hidden_local_vars(
            self,
            hidden_local_vars: torch.Tensor,
            scenario_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.hidden_local_vars_encoder is None:
            raise RuntimeError("This scenario encoder does not own a hidden-local encoder.")
        agent_scenario_ids = scenario_ids.unsqueeze(-1).expand(*scenario_ids.shape, hidden_local_vars.shape[-2])
        return self.hidden_local_vars_encoder(hidden_local_vars, agent_scenario_ids)

    def encode_hidden_global_vars(
            self,
            hidden_global_vars: torch.Tensor,
            scenario_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.hidden_global_vars_encoder is None:
            raise RuntimeError("This scenario encoder does not own a hidden-global encoder.")
        return self.hidden_global_vars_encoder(hidden_global_vars, scenario_ids)


def _validate_config(
        config: TMASACScenarioEncoderConfig,
        *,
        global_obs_dim: int,
        hidden_local_vars_dim: int,
        hidden_global_vars_dim: int,
) -> None:
    if not config.scenarios:
        raise ValueError("TMASACScenarioEncoderConfig.scenarios must not be empty.")
    expected_ids = tuple(range(len(config.scenarios)))
    scenario_ids = tuple(spec.scenario_id for spec in config.scenarios)
    if scenario_ids != expected_ids:
        raise ValueError(f"Scenario IDs must be contiguous and ordered as {expected_ids}, got {scenario_ids}.")
    scenario_names = tuple(spec.name for spec in config.scenarios)
    if len(set(scenario_names)) != len(scenario_names):
        raise ValueError(f"Scenario names must be unique, got {scenario_names}.")
    for field_name, padded_dim, input_dims in (
        ("global_obs", global_obs_dim, tuple(spec.global_obs_dim for spec in config.scenarios)),
        (
            "hidden_local_vars",
            hidden_local_vars_dim,
            tuple(spec.hidden_local_vars_dim for spec in config.scenarios),
        ),
        (
            "hidden_global_vars",
            hidden_global_vars_dim,
            tuple(spec.hidden_global_vars_dim for spec in config.scenarios),
        ),
    ):
        if any(input_dim < 0 for input_dim in input_dims):
            raise ValueError(f"{field_name} input dimensions must be non-negative, got {input_dims}.")
        if max(input_dims) != padded_dim:
            raise ValueError(
                f"Expected padded {field_name} dimension {max(input_dims)}, got environment dimension {padded_dim}."
            )
    for field_name, field_config in (
        ("global_obs", config.global_obs),
        ("hidden_local_vars", config.hidden_local_vars),
        ("hidden_global_vars", config.hidden_global_vars),
    ):
        if field_config.output_dim <= 0:
            raise ValueError(f"{field_name}.output_dim must be positive, got {field_config.output_dim}.")
        if any(hidden_dim <= 0 for hidden_dim in field_config.hidden_dims):
            raise ValueError(f"{field_name}.hidden_dims must contain only positive values.")
    if config.scenario_embedding_dim < 0:
        raise ValueError(
            f"scenario_embedding_dim must be non-negative, got {config.scenario_embedding_dim}."
        )
