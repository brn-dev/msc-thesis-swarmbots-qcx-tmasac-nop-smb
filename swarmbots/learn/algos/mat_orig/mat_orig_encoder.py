from dataclasses import dataclass

import math
import torch
from torch import nn


def _init_linear(linear: nn.Linear, *, activate: bool = False) -> nn.Linear:
    gain = nn.init.calculate_gain("relu") if activate else 0.01
    nn.init.orthogonal_(linear.weight, gain=gain)
    nn.init.zeros_(linear.bias)
    return linear


class MATOrigSelfAttention(nn.Module):

    def __init__(
            self,
            d_model: int,
            nhead: int,
            *,
            max_agents: int,
            masked: bool,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model must be divisible by nhead, got {d_model=} and {nhead=}")

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.masked = masked

        self.key = _init_linear(nn.Linear(d_model, d_model))
        self.query = _init_linear(nn.Linear(d_model, d_model))
        self.value = _init_linear(nn.Linear(d_model, d_model))
        self.proj = _init_linear(nn.Linear(d_model, d_model))

        causal_mask = torch.tril(torch.ones(max_agents, max_agents, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, max_agents, max_agents))

    def forward(
            self,
            key: torch.Tensor,
            value: torch.Tensor,
            query: torch.Tensor,
            *,
            key_padding_mask: torch.Tensor | None = None,
            query_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, query_len, _ = query.shape
        key_len = key.shape[1]
        if value.shape[:2] != (batch_size, key_len):
            raise ValueError(
                "Expected key and value to match in batch/sequence dims, "
                f"got {tuple(key.shape)} and {tuple(value.shape)}"
            )
        if key.shape[-1] != self.d_model or value.shape[-1] != self.d_model or query.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected key/value/query last dim {self.d_model}, got "
                f"{key.shape[-1]}, {value.shape[-1]}, {query.shape[-1]}"
            )

        k = self.key(key).view(batch_size, key_len, self.nhead, self.head_dim).transpose(1, 2)
        q = self.query(query).view(batch_size, query_len, self.nhead, self.head_dim).transpose(1, 2)
        v = self.value(value).view(batch_size, key_len, self.nhead, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        if self.masked:
            attn = attn.masked_fill(~self.causal_mask[:, :, :query_len, :key_len], float("-inf"))

        if key_padding_mask is not None:
            if key_padding_mask.dtype != torch.bool:
                raise ValueError(f"Expected key_padding_mask dtype bool, got {key_padding_mask.dtype}")
            if key_padding_mask.shape != (batch_size, key_len):
                raise ValueError(
                    f"Expected key_padding_mask shape {(batch_size, key_len)}, got {tuple(key_padding_mask.shape)}"
                )
            attn = attn.masked_fill(~key_padding_mask[:, None, None, :], float("-inf"))

        attn = torch.softmax(attn, dim=-1)
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(batch_size, query_len, self.d_model)
        y = self.proj(y)

        if query_padding_mask is not None:
            if query_padding_mask.dtype != torch.bool:
                raise ValueError(f"Expected query_padding_mask dtype bool, got {query_padding_mask.dtype}")
            if query_padding_mask.shape != (batch_size, query_len):
                raise ValueError(
                    f"Expected query_padding_mask shape {(batch_size, query_len)}, got {tuple(query_padding_mask.shape)}"
                )
            y = y.masked_fill(~query_padding_mask.unsqueeze(-1), 0.0)

        return y


class MATOrigEncodeBlock(nn.Module):

    def __init__(
            self,
            d_model: int,
            nhead: int,
            *,
            max_agents: int,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = MATOrigSelfAttention(d_model, nhead, max_agents=max_agents, masked=False)
        self.mlp = nn.Sequential(
            _init_linear(nn.Linear(d_model, d_model), activate=True),
            nn.GELU(),
            _init_linear(nn.Linear(d_model, d_model)),
        )

    def forward(
            self,
            x: torch.Tensor,
            *,
            agent_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.ln1(
            x + self.attn(
                x,
                x,
                x,
                key_padding_mask=agent_mask,
                query_padding_mask=agent_mask,
            )
        )
        x = self.ln2(x + self.mlp(x))
        if agent_mask is not None:
            x = x.masked_fill(~agent_mask.unsqueeze(-1), 0.0)
        return x


@dataclass(frozen=True)
class MATOrigEncoderConfig:
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2


class MATOrigEncoder(nn.Module):

    def __init__(
            self,
            config: MATOrigEncoderConfig,
            *,
            max_agents: int,
            local_obs_dim: int,
            global_obs_dim: int,
    ) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.max_agents = max_agents
        self.local_obs_dim = local_obs_dim
        self.global_obs_dim = global_obs_dim
        self.obs_input_dim = local_obs_dim + global_obs_dim

        self.obs_encoder = nn.Sequential(
            nn.LayerNorm(self.obs_input_dim),
            _init_linear(nn.Linear(self.obs_input_dim, config.d_model), activate=True),
            nn.GELU(),
        )
        self.ln = nn.LayerNorm(config.d_model)
        self.blocks = nn.ModuleList([
            MATOrigEncodeBlock(config.d_model, config.nhead, max_agents=max_agents)
            for _ in range(config.num_layers)
        ])

    def forward(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            *,
            agent_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n_agents = local_obs.shape[1]
        if n_agents > self.max_agents:
            raise ValueError(f"Expected local_obs second dim <= {self.max_agents}, got {n_agents}")

        if self.global_obs_dim > 0:
            global_obs_expanded = global_obs.unsqueeze(1).expand(-1, n_agents, -1)
            obs_input = torch.cat((local_obs, global_obs_expanded), dim=-1)
        else:
            obs_input = local_obs

        x = self.obs_encoder(obs_input)
        if agent_mask is not None:
            x = x.masked_fill(~agent_mask.unsqueeze(-1), 0.0)
        x = self.ln(x)
        if agent_mask is not None:
            x = x.masked_fill(~agent_mask.unsqueeze(-1), 0.0)

        for block in self.blocks:
            x = block(x, agent_mask=agent_mask)
        return x
