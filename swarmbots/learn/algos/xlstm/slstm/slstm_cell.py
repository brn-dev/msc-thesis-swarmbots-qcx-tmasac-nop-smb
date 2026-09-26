import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from swarmbots.learn.algos.xlstm.head_utils import require_heads_divide_hidden_dim

SLSTMCellState = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class SLSTMCellConfig:
    num_heads: int = 4
    recurrent_weight_init: Literal["zeros", "standard"] = "zeros"
    bias_init: Literal["powerlaw", "small_init", "zeros", "standard"] = "powerlaw"
    forget_bias_init_start: float = 3.0
    forget_bias_init_end: float = 6.0


class SLSTMCell(nn.Module):

    def __init__(self, *, hidden_dim: int, config: SLSTMCellConfig = SLSTMCellConfig()) -> None:
        super().__init__()
        if config.num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {config.num_heads}")
        require_heads_divide_hidden_dim(hidden_dim=hidden_dim, num_heads=config.num_heads)

        self.hidden_dim = hidden_dim
        self.config = config
        self.head_dim = hidden_dim // config.num_heads
        self.recurrent_kernel = nn.Parameter(torch.empty(config.num_heads, self.head_dim, 4, self.head_dim))
        self.bias = nn.Parameter(torch.empty(config.num_heads, 4, self.head_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.config.recurrent_weight_init == "zeros":
            nn.init.zeros_(self.recurrent_kernel)
        elif self.config.recurrent_weight_init == "standard":
            bound = 1.0 / math.sqrt(self.hidden_dim)
            nn.init.uniform_(self.recurrent_kernel, -bound, bound)
        else:
            raise ValueError(f"Unsupported recurrent_weight_init={self.config.recurrent_weight_init}")

        if self.config.bias_init == "powerlaw":
            nn.init.zeros_(self.bias)
            with torch.no_grad():
                if self.head_dim == 1:
                    forget_bias = torch.full(
                        (self.config.num_heads, self.head_dim),
                        5.0,
                        device=self.bias.device,
                        dtype=self.bias.dtype,
                    )
                else:
                    head_positions = torch.arange(self.head_dim, device=self.bias.device, dtype=self.bias.dtype)
                    forget_bias = 5.0 - 12.0 * (head_positions / (self.head_dim - 1)) ** 0.3
                    forget_bias = forget_bias.unsqueeze(0).expand(self.config.num_heads, -1)
                self.bias[:, 1, :].copy_(forget_bias)
        elif self.config.bias_init == "small_init":
            nn.init.zeros_(self.bias)
            with torch.no_grad():
                forget_bias = torch.linspace(
                    self.config.forget_bias_init_start,
                    self.config.forget_bias_init_end,
                    self.config.num_heads * self.head_dim,
                    device=self.bias.device,
                    dtype=self.bias.dtype,
                ).reshape(self.config.num_heads, self.head_dim)
                self.bias[:, 1, :].copy_(forget_bias)
        elif self.config.bias_init == "zeros":
            nn.init.zeros_(self.bias)
        elif self.config.bias_init == "standard":
            bound = 1.0 / math.sqrt(self.hidden_dim)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            raise ValueError(f"Unsupported bias_init={self.config.bias_init}")

    def initial_state(
            self,
            batch_size: int,
            *,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ) -> SLSTMCellState:
        shape = (batch_size, self.hidden_dim)
        hidden_state = torch.zeros(shape, device=device, dtype=dtype)
        cell_state = torch.zeros(shape, device=device, dtype=dtype)
        normalizer_state = torch.zeros(shape, device=device, dtype=dtype)
        stabilizer_state = torch.zeros(shape, device=device, dtype=dtype)
        return hidden_state, cell_state, normalizer_state, stabilizer_state

    def forward(self, gate_inputs: torch.Tensor, state: SLSTMCellState) -> tuple[torch.Tensor, SLSTMCellState]:
        if gate_inputs.ndim != 2 or gate_inputs.shape[-1] != 4 * self.hidden_dim:
            raise ValueError(f"Expected gate_inputs shape (B, {4 * self.hidden_dim}), got {tuple(gate_inputs.shape)}")

        hidden_state, cell_state, normalizer_state, stabilizer_state = state
        batch_size = gate_inputs.shape[0]

        recurrent_gates = torch.einsum(
            "bnh,nhgd->bngd",
            hidden_state.reshape(batch_size, self.config.num_heads, self.head_dim),
            self.recurrent_kernel,
        )
        gates = gate_inputs.reshape(batch_size, 4, self.config.num_heads, self.head_dim).permute(0, 2, 1, 3)
        gates = gates + recurrent_gates + self.bias.unsqueeze(0)
        input_preact, forget_preact, cell_preact, output_preact = gates.unbind(dim=2)

        cell_heads = cell_state.reshape(batch_size, self.config.num_heads, self.head_dim)
        normalizer_heads = normalizer_state.reshape(batch_size, self.config.num_heads, self.head_dim)
        stabilizer_heads = stabilizer_state.reshape(batch_size, self.config.num_heads, self.head_dim)

        log_forget = F.logsigmoid(forget_preact)
        m_candidate = torch.maximum(input_preact, stabilizer_heads + log_forget)
        # The NXAI reference special-cases an all-zero initial normalizer. Applying it per row
        # preserves the same stabilization when individual RL episodes reset inside a batch.
        m_new = torch.where(normalizer_heads == 0.0, input_preact, m_candidate)
        input_gate = torch.minimum(torch.exp(input_preact - m_new), torch.ones_like(input_preact))
        forget_gate = torch.minimum(torch.exp(stabilizer_heads + log_forget - m_new), torch.ones_like(forget_preact))

        cell_new = forget_gate * cell_heads + input_gate * torch.tanh(cell_preact)
        normalizer_new = torch.maximum(
            forget_gate * normalizer_heads + input_gate,
            torch.ones_like(normalizer_heads),
        )
        hidden_new = torch.sigmoid(output_preact) * cell_new / normalizer_new

        hidden_new = hidden_new.reshape(batch_size, self.hidden_dim)
        return hidden_new, (
            hidden_new,
            cell_new.reshape(batch_size, self.hidden_dim),
            normalizer_new.reshape(batch_size, self.hidden_dim),
            m_new.reshape(batch_size, self.hidden_dim),
        )
