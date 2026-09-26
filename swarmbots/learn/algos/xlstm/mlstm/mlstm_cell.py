import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from swarmbots.learn.algos.xlstm.head_utils import (
    MultiHeadLayerNorm,
    init_linspace_bias_,
    require_heads_divide_hidden_dim,
)

MLSTMCellState = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class MLSTMCellConfig:
    num_heads: int = 4
    eps: float = 1e-6
    gate_bias_init_start: float = 3.0
    gate_bias_init_end: float = 6.0


class MLSTMCell(nn.Module):

    def __init__(self, *, hidden_dim: int, config: MLSTMCellConfig = MLSTMCellConfig()) -> None:
        super().__init__()
        if config.num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {config.num_heads}")
        require_heads_divide_hidden_dim(hidden_dim=hidden_dim, num_heads=config.num_heads)

        self.hidden_dim = hidden_dim
        self.config = config
        self.head_dim = hidden_dim // config.num_heads
        self.input_gate = nn.Linear(3 * hidden_dim, config.num_heads)
        self.forget_gate = nn.Linear(3 * hidden_dim, config.num_heads)
        self.output_norm = MultiHeadLayerNorm(
            hidden_dim=hidden_dim,
            num_heads=config.num_heads,
            bias=False,
            residual_weight=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.output_norm.reset_parameters()
        nn.init.zeros_(self.input_gate.weight)
        nn.init.normal_(self.input_gate.bias, mean=0.0, std=0.1)
        nn.init.zeros_(self.forget_gate.weight)
        init_linspace_bias_(
            self.forget_gate.bias,
            start=self.config.gate_bias_init_start,
            end=self.config.gate_bias_init_end,
        )

    def initial_state(
            self,
            batch_size: int,
            *,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ) -> MLSTMCellState:
        c_state = torch.zeros(
            (batch_size, self.config.num_heads, self.head_dim, self.head_dim),
            device=device,
            dtype=dtype,
        )
        n_state = torch.zeros((batch_size, self.config.num_heads, self.head_dim), device=device, dtype=dtype)
        m_state = torch.zeros((batch_size, self.config.num_heads), device=device, dtype=dtype)
        return c_state, n_state, m_state

    def forward(
            self,
            queries: torch.Tensor,
            keys: torch.Tensor,
            values: torch.Tensor,
            state: MLSTMCellState,
    ) -> tuple[torch.Tensor, MLSTMCellState]:
        if queries.ndim != 2:
            raise ValueError(f"Expected queries shape (B, H), got {tuple(queries.shape)}")

        batch_size = queries.shape[0]
        q = queries.reshape(batch_size, self.config.num_heads, self.head_dim)
        k = keys.reshape(batch_size, self.config.num_heads, self.head_dim)
        v = values.reshape(batch_size, self.config.num_heads, self.head_dim)

        gate_inputs = torch.cat([queries, keys, values], dim=-1)
        input_preact = self.input_gate(gate_inputs)
        forget_preact = self.forget_gate(gate_inputs)

        c_state, n_state, m_state = state
        log_forget = F.logsigmoid(forget_preact)
        m_new = torch.maximum(log_forget + m_state, input_preact)
        forget_gate = torch.exp(log_forget + m_state - m_new)
        input_gate = torch.exp(input_preact - m_new)

        k_scaled = k / math.sqrt(self.head_dim)
        c_new = (
            forget_gate[..., None, None] * c_state
            + input_gate[..., None, None] * (k_scaled.unsqueeze(-1) @ v.unsqueeze(-2))
        )
        n_new = forget_gate[..., None] * n_state + input_gate[..., None] * k_scaled

        hidden_numerator = (q.unsqueeze(-2) @ c_new).squeeze(-2)
        normalizer = torch.maximum(
            (q * n_new).sum(dim=-1, keepdim=True).abs(),
            torch.exp(-m_new).unsqueeze(-1),
        )
        hidden = hidden_numerator / (normalizer + self.config.eps)
        hidden = self.output_norm(hidden.unsqueeze(2)).squeeze(2).reshape(batch_size, self.hidden_dim)
        return hidden, (c_new, n_new, m_new)

    def forward_sequence(
            self,
            queries: torch.Tensor,
            keys: torch.Tensor,
            values: torch.Tensor,
            state: MLSTMCellState,
            *,
            valid_mask: torch.Tensor | None = None,
            reset_mask: torch.Tensor | None = None,
            state_output_indices: torch.Tensor | None = None,
    ) -> (
        tuple[torch.Tensor, MLSTMCellState]
        | tuple[torch.Tensor, MLSTMCellState, MLSTMCellState]
    ):
        if queries.ndim != 3:
            raise ValueError(f"Expected queries shape (B, T, H), got {tuple(queries.shape)}")

        batch_size, sequence_length, _ = queries.shape
        if valid_mask is not None:
            expected_mask_shape = (batch_size, sequence_length)
            if valid_mask.shape != expected_mask_shape:
                raise ValueError(f"Expected valid_mask shape {expected_mask_shape}, got {tuple(valid_mask.shape)}")
            if valid_mask.dtype != torch.bool:
                raise ValueError(f"Expected valid_mask dtype torch.bool, got {valid_mask.dtype}")
        if reset_mask is not None:
            expected_mask_shape = (batch_size, sequence_length)
            if reset_mask.shape != expected_mask_shape:
                raise ValueError(f"Expected reset_mask shape {expected_mask_shape}, got {tuple(reset_mask.shape)}")
            if reset_mask.dtype != torch.bool:
                raise ValueError(f"Expected reset_mask dtype torch.bool, got {reset_mask.dtype}")

        if valid_mask is not None:
            valid_feature_mask = valid_mask.unsqueeze(-1)
            queries = queries.masked_fill(~valid_feature_mask, 0.0)
            keys = keys.masked_fill(~valid_feature_mask, 0.0)
            values = values.masked_fill(~valid_feature_mask, 0.0)

        q = queries.reshape(batch_size, sequence_length, self.config.num_heads, self.head_dim).transpose(1, 2)
        k = keys.reshape(batch_size, sequence_length, self.config.num_heads, self.head_dim).transpose(1, 2)
        v = values.reshape(batch_size, sequence_length, self.config.num_heads, self.head_dim).transpose(1, 2)

        gate_inputs = torch.cat([queries, keys, values], dim=-1)
        input_preact = self.input_gate(gate_inputs).transpose(1, 2)
        forget_preact = self.forget_gate(gate_inputs).transpose(1, 2)
        log_forget = F.logsigmoid(forget_preact)
        if valid_mask is not None:
            valid_mask_heads = valid_mask.unsqueeze(1)
            log_forget = torch.where(valid_mask_heads, log_forget, torch.zeros_like(log_forget))
            input_preact = torch.where(valid_mask_heads, input_preact, torch.full_like(input_preact, -torch.inf))
        log_forget_cumsum = torch.cumsum(log_forget, dim=-1)

        c_state, n_state, m_state = state
        state_log_scale = m_state.unsqueeze(-1) + log_forget_cumsum
        normalizer_state_log_scale = state_log_scale
        state_survives = None

        row_log_forget_cumsum = log_forget_cumsum.unsqueeze(-1)
        col_log_forget_cumsum = log_forget_cumsum.unsqueeze(-2)
        input_log_scale = input_preact.unsqueeze(-2) + row_log_forget_cumsum - col_log_forget_cumsum
        causal_mask = torch.tril(
            torch.ones((sequence_length, sequence_length), dtype=torch.bool, device=queries.device),
        )
        input_log_scale = input_log_scale.masked_fill(~causal_mask, -torch.inf)
        if reset_mask is not None:
            reset_groups = torch.cumsum(reset_mask.to(torch.int64), dim=-1)
            same_reset_group = reset_groups.unsqueeze(-1) == reset_groups.unsqueeze(-2)
            same_reset_group = same_reset_group.unsqueeze(1)
            input_log_scale = input_log_scale.masked_fill(~same_reset_group, -torch.inf)
            state_survives = reset_groups == 0
            state_log_scale = state_log_scale.masked_fill(~state_survives.unsqueeze(1), -torch.inf)
            segment_log_forget = log_forget.unsqueeze(-2).masked_fill(
                ~(same_reset_group & causal_mask.view(1, 1, sequence_length, sequence_length)),
                0.0,
            ).sum(dim=-1)
            normalizer_state_log_scale = torch.where(
                state_survives.unsqueeze(1),
                normalizer_state_log_scale,
                segment_log_forget,
            )

        normalizer_log_scale = torch.maximum(
            normalizer_state_log_scale,
            input_log_scale.max(dim=-1).values,
        )
        no_contribution = torch.isneginf(normalizer_log_scale)
        safe_normalizer_log_scale = normalizer_log_scale.masked_fill(no_contribution, 0.0)
        state_weights = torch.exp(state_log_scale - safe_normalizer_log_scale).masked_fill(no_contribution, 0.0)
        input_weights = torch.exp(input_log_scale - safe_normalizer_log_scale.unsqueeze(-1))
        input_weights = input_weights.masked_fill(no_contribution.unsqueeze(-1), 0.0)

        k_scaled = k / math.sqrt(self.head_dim)
        c_from_inputs = torch.einsum("bnts,bnsd,bnse->bntde", input_weights, k_scaled, v)
        expanded_c_state = c_state.unsqueeze(2)
        expanded_n_state = n_state.unsqueeze(2)
        if state_survives is not None:
            expanded_c_state = expanded_c_state.masked_fill(
                ~state_survives[:, None, :, None, None],
                0.0,
            )
            expanded_n_state = expanded_n_state.masked_fill(
                ~state_survives[:, None, :, None],
                0.0,
            )
        c_sequence = c_from_inputs + state_weights[..., None, None] * expanded_c_state
        n_sequence = torch.einsum("bnts,bnsd->bntd", input_weights, k_scaled)
        n_sequence = n_sequence + state_weights[..., None] * expanded_n_state

        hidden_numerator = torch.einsum("bntd,bntde->bnte", q, c_sequence)
        denominator = torch.maximum(
            (q * n_sequence).sum(dim=-1, keepdim=True).abs(),
            torch.exp(-safe_normalizer_log_scale).unsqueeze(-1),
        )
        hidden = hidden_numerator / (denominator + self.config.eps)
        hidden = hidden.masked_fill(no_contribution.unsqueeze(-1), 0.0)
        hidden = self.output_norm(hidden).transpose(1, 2).reshape(batch_size, sequence_length, self.hidden_dim)
        if valid_mask is not None:
            hidden = hidden.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        final_state = (
            c_sequence[:, :, -1].contiguous(),
            n_sequence[:, :, -1].contiguous(),
            safe_normalizer_log_scale[:, :, -1].contiguous(),
        )
        if state_output_indices is not None:
            batch_indices = state_output_indices[:, 0]
            time_indices = state_output_indices[:, 1]
            selected_states = (
                c_sequence[batch_indices, :, time_indices].contiguous(),
                n_sequence[batch_indices, :, time_indices].contiguous(),
                safe_normalizer_log_scale[batch_indices, :, time_indices].contiguous(),
            )
            return hidden, final_state, selected_states
        return hidden, final_state
