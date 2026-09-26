from dataclasses import dataclass

import torch
from torch import nn

from swarmbots.learn.nn_components.activations import ActivationFactory, make_activation
from swarmbots.learn.nn_components.nn_init import init_transformer_feedforward, reinitialize_multihead_attention


@dataclass(frozen=True)
class MATQCCDecoderConfig:
    d_model: int | None = None
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.0
    act_fn_cls: ActivationFactory = nn.GELU
    norm_first: bool = True
    layer_norm_eps: float = 1e-5
    bias: bool = True
    add_agent_embeddings: bool = False
    token_encoder_init_gain: float = 1.0
    token_encoder_projection_init_gain: float | None = 1.0
    actor_head_init_gain: float = 1.0
    transformer_ff_init_gain: float | None = 1.0
    query_encoder_hidden_dims: list[int] | None = None
    context_encoder_hidden_dims: list[int] | None = None
    memory_dims: list[int] | None = None
    actor_head_hidden_dims: list[int] | None = None
    token_encoder_end_with_act_fn: bool = False
    normalize_query_input: bool = False
    normalize_context_input: bool = False
    normalize_memory_input: bool = False
    normalize_query_tokens: bool = False
    normalize_context_tokens: bool = False
    normalize_memory_tokens: bool = False
    normalize_actor_head_input: bool = False
    assume_agent_mask_is_active_prefix: bool = True
    tie_query_context_and_context_self_attention: bool = True


class MATQCCDecoderLayer(nn.Module):

    def __init__(
            self,
            *,
            d_model: int,
            memory_d_model: int,
            nhead: int,
            dim_feedforward: int,
            dropout: float,
            activation: nn.Module,
            layer_norm_eps: float,
            norm_first: bool,
            bias: bool,
            tie_query_context_and_context_self_attention: bool,
    ) -> None:
        super().__init__()
        self.query_context_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
            bias=bias,
        )
        self.context_self_attn = (
            None
            if tie_query_context_and_context_self_attention
            else nn.MultiheadAttention(
                d_model,
                nhead,
                dropout=dropout,
                batch_first=True,
                bias=bias,
            )
        )
        self.memory_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            kdim=memory_d_model,
            vdim=memory_d_model,
            batch_first=True,
            bias=bias,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation

    def forward(
            self,
            query_tokens: torch.Tensor,
            context_tokens: torch.Tensor,
            memory_tokens: torch.Tensor,
            *,
            query_context_attention_mask: torch.Tensor | None,
            query_has_visible_context: torch.Tensor | None,
            query_context_key_padding_mask: torch.Tensor | None,
            context_self_attention_mask: torch.Tensor | None,
            context_has_visible_context: torch.Tensor | None,
            context_self_key_padding_mask: torch.Tensor | None,
            memory_key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        new_query_tokens = self._forward_tokens(
            target_tokens=query_tokens,
            context_tokens=context_tokens,
            memory_tokens=memory_tokens,
            context_attention_mask=query_context_attention_mask,
            has_visible_context=query_has_visible_context,
            context_key_padding_mask=query_context_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            context_attn=self.query_context_attn,
            skip_first_token_without_context=True,
        )
        new_context_tokens = self._forward_tokens(
            target_tokens=context_tokens,
            context_tokens=context_tokens,
            memory_tokens=memory_tokens,
            context_attention_mask=context_self_attention_mask,
            has_visible_context=context_has_visible_context,
            context_key_padding_mask=context_self_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            context_attn=self._context_self_attention(),
            skip_first_token_without_context=False,
        )
        return new_query_tokens, new_context_tokens

    def _context_self_attention(self) -> nn.MultiheadAttention:
        if self.context_self_attn is None:
            return self.query_context_attn
        return self.context_self_attn

    def _forward_tokens(
            self,
            *,
            target_tokens: torch.Tensor,
            context_tokens: torch.Tensor,
            memory_tokens: torch.Tensor,
            context_attention_mask: torch.Tensor | None,
            has_visible_context: torch.Tensor | None,
            context_key_padding_mask: torch.Tensor | None,
            memory_key_padding_mask: torch.Tensor | None,
            context_attn: nn.MultiheadAttention,
            skip_first_token_without_context: bool,
    ) -> torch.Tensor:
        if target_tokens.shape[1] == 0:
            return target_tokens

        x = target_tokens
        if self.norm_first:
            x = x + self._context_attn_block(
                self.norm1(x),
                context_tokens,
                context_attention_mask=context_attention_mask,
                has_visible_context=has_visible_context,
                context_key_padding_mask=context_key_padding_mask,
                context_attn=context_attn,
                skip_first_token_without_context=skip_first_token_without_context,
            )
            x = x + self._memory_attn_block(
                self.norm2(x),
                memory_tokens,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            return x + self._ff_block(self.norm3(x))

        x = self.norm1(
            x + self._context_attn_block(
                x,
                context_tokens,
                context_attention_mask=context_attention_mask,
                has_visible_context=has_visible_context,
                context_key_padding_mask=context_key_padding_mask,
                context_attn=context_attn,
                skip_first_token_without_context=skip_first_token_without_context,
            )
        )
        x = self.norm2(
            x + self._memory_attn_block(
                x,
                memory_tokens,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        )
        return self.norm3(x + self._ff_block(x))

    def _context_attn_block(
            self,
            query_tokens: torch.Tensor,
            context_tokens: torch.Tensor,
            *,
            context_attention_mask: torch.Tensor | None,
            has_visible_context: torch.Tensor | None,
            context_key_padding_mask: torch.Tensor | None,
            context_attn: nn.MultiheadAttention,
            skip_first_token_without_context: bool,
    ) -> torch.Tensor:
        if context_tokens.shape[1] == 0:
            return torch.zeros_like(query_tokens)

        if (
                skip_first_token_without_context
                and has_visible_context is not None
                and query_tokens.shape[1] > 1
                and context_attention_mask is not None
                and not has_visible_context[:, 0].any()
        ):
            sliced_attention_mask = self._slice_attention_mask_first_query(context_attention_mask, has_visible_context)
            attended_output = context_attn(
                query=query_tokens[:, 1:, :],
                key=context_tokens,
                value=context_tokens,
                attn_mask=sliced_attention_mask,
                key_padding_mask=context_key_padding_mask,
                need_weights=False,
            )[0]
            attended_output = attended_output.masked_fill(~has_visible_context[:, 1:].unsqueeze(-1), 0.0)
            first_output = torch.zeros_like(query_tokens[:, :1, :])
            return self.dropout1(torch.cat((first_output, attended_output), dim=1))

        output = context_attn(
            query=query_tokens,
            key=context_tokens,
            value=context_tokens,
            attn_mask=context_attention_mask,
            key_padding_mask=context_key_padding_mask,
            need_weights=False,
        )[0]
        if has_visible_context is not None:
            output = output.masked_fill(~has_visible_context.unsqueeze(-1), 0.0)
        return self.dropout1(output)

    @staticmethod
    def _slice_attention_mask_first_query(
            attention_mask: torch.Tensor,
            has_visible_context: torch.Tensor,
    ) -> torch.Tensor:
        if attention_mask.dim() == 2:
            return attention_mask[1:, :]
        batch_size = has_visible_context.shape[0]
        heads_per_batch = attention_mask.shape[0] // batch_size
        return attention_mask.reshape(
            batch_size,
            heads_per_batch,
            attention_mask.shape[1],
            attention_mask.shape[2],
        )[:, :, 1:, :].reshape(
            attention_mask.shape[0],
            attention_mask.shape[1] - 1,
            attention_mask.shape[2],
        )

    def _memory_attn_block(
            self,
            query_tokens: torch.Tensor,
            memory_tokens: torch.Tensor,
            *,
            memory_key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.dropout2(
            self.memory_attn(
                query=query_tokens,
                key=memory_tokens,
                value=memory_tokens,
                key_padding_mask=memory_key_padding_mask,
                need_weights=False,
            )[0]
        )

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout3(self.linear2(self.dropout(self.activation(self.linear1(x)))))


class MATQCCDecoder(nn.Module):

    def __init__(
            self,
            config: MATQCCDecoderConfig,
            *,
            max_agents: int,
            memory_d_model: int,
    ) -> None:
        super().__init__()
        if config.d_model is None:
            raise ValueError("MATQCCDecoderConfig.d_model must be set before constructing MATQCCDecoder")
        if config.num_layers <= 0:
            raise ValueError(f"MATQCCDecoderConfig.num_layers must be > 0, got {config.num_layers}")

        self.max_agents = max_agents
        self.d_model = config.d_model
        self.memory_d_model = memory_d_model
        self.nhead = config.nhead
        self.assume_agent_mask_is_active_prefix = config.assume_agent_mask_is_active_prefix

        self.layers = nn.ModuleList(
            [
                MATQCCDecoderLayer(
                    d_model=self.d_model,
                    memory_d_model=memory_d_model,
                    nhead=config.nhead,
                    dim_feedforward=config.dim_feedforward,
                    dropout=config.dropout,
                    activation=make_activation(config.act_fn_cls, num_features=config.dim_feedforward),
                    layer_norm_eps=config.layer_norm_eps,
                    norm_first=config.norm_first,
                    bias=config.bias,
                    tie_query_context_and_context_self_attention=config.tie_query_context_and_context_self_attention,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(self.d_model, eps=config.layer_norm_eps, bias=config.bias)

        if config.transformer_ff_init_gain is not None:
            self._reinitialize_layers(feedforward_init_gain=config.transformer_ff_init_gain)

    def forward(
            self,
            query_tokens: torch.Tensor,
            context_tokens: torch.Tensor,
            memory_tokens: torch.Tensor,
            agent_mask: torch.Tensor | None = None,
            memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if memory_tokens.shape[1] > self.max_agents:
            raise ValueError(f"Expected memory_tokens second dim <= {self.max_agents}, got {memory_tokens.shape[1]}")
        n_queries = query_tokens.shape[1]
        n_contexts = context_tokens.shape[1]
        if n_contexts not in (n_queries, n_queries - 1):
            raise ValueError(
                f"Expected context_tokens second dim to be {n_queries} or {n_queries - 1}, got {n_contexts}"
            )

        context_attention_mask, has_visible_context, context_key_padding_mask = self._build_parallel_context_attention_mask(
            batch_size=query_tokens.shape[0],
            n_queries=n_queries,
            n_contexts=n_contexts,
            context_mask=agent_mask,
            device=query_tokens.device,
        )
        (
            context_self_attention_mask,
            context_has_visible_context,
            context_self_key_padding_mask,
        ) = self._build_context_self_attention_mask(
            batch_size=query_tokens.shape[0],
            n_contexts=context_tokens.shape[1],
            context_mask=agent_mask,
            device=query_tokens.device,
        )
        return self._decode(
            query_tokens=query_tokens,
            context_tokens=context_tokens,
            memory_tokens=memory_tokens,
            query_context_attention_mask=context_attention_mask,
            query_has_visible_context=has_visible_context,
            query_context_key_padding_mask=context_key_padding_mask,
            context_self_attention_mask=context_self_attention_mask,
            context_has_visible_context=context_has_visible_context,
            context_self_key_padding_mask=context_self_key_padding_mask,
            memory_mask=memory_mask,
        )

    def forward_step(
            self,
            context_tokens: torch.Tensor,
            query_token: torch.Tensor,
            memory_tokens: torch.Tensor,
            *,
            query_prefix_tokens: torch.Tensor | None = None,
            context_mask: torch.Tensor | None = None,
            query_prefix_mask: torch.Tensor | None = None,
            query_mask: torch.Tensor | None = None,
            memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = query_prefix_tokens, query_prefix_mask, query_mask
        if memory_tokens.shape[1] > self.max_agents:
            raise ValueError(
                f"Expected memory_tokens second dim <= {self.max_agents}, got {memory_tokens.shape[1]}"
            )

        context_attention_mask, has_visible_context, context_key_padding_mask = self._build_step_context_attention_mask(
            batch_size=query_token.shape[0],
            context_mask=context_mask,
            n_contexts=context_tokens.shape[1],
            device=query_token.device,
        )
        (
            context_self_attention_mask,
            context_has_visible_context,
            context_self_key_padding_mask,
        ) = self._build_context_self_attention_mask(
            batch_size=query_token.shape[0],
            n_contexts=context_tokens.shape[1],
            context_mask=context_mask,
            device=query_token.device,
        )
        return self._decode(
            query_tokens=query_token,
            context_tokens=context_tokens,
            memory_tokens=memory_tokens,
            query_context_attention_mask=context_attention_mask,
            query_has_visible_context=has_visible_context,
            query_context_key_padding_mask=context_key_padding_mask,
            context_self_attention_mask=context_self_attention_mask,
            context_has_visible_context=context_has_visible_context,
            context_self_key_padding_mask=context_self_key_padding_mask,
            memory_mask=memory_mask,
        )

    def _decode(
            self,
            *,
            query_tokens: torch.Tensor,
            context_tokens: torch.Tensor,
            memory_tokens: torch.Tensor,
            query_context_attention_mask: torch.Tensor | None,
            query_has_visible_context: torch.Tensor | None,
            query_context_key_padding_mask: torch.Tensor | None,
            context_self_attention_mask: torch.Tensor | None,
            context_has_visible_context: torch.Tensor | None,
            context_self_key_padding_mask: torch.Tensor | None,
            memory_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        memory_key_padding_mask = None if memory_mask is None else ~memory_mask
        x = query_tokens
        context_state = context_tokens
        for layer in self.layers:
            x, context_state = layer(
                query_tokens=x,
                context_tokens=context_state,
                memory_tokens=memory_tokens,
                query_context_attention_mask=query_context_attention_mask,
                query_has_visible_context=query_has_visible_context,
                query_context_key_padding_mask=query_context_key_padding_mask,
                context_self_attention_mask=context_self_attention_mask,
                context_has_visible_context=context_has_visible_context,
                context_self_key_padding_mask=context_self_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return self.norm(x)

    def _build_parallel_context_attention_mask(
            self,
            *,
            batch_size: int,
            n_queries: int,
            n_contexts: int,
            context_mask: torch.Tensor | None,
            device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if n_contexts == 0:
            return None, None, None

        if self._can_use_prefix_context_mask():
            has_visible_context = torch.arange(n_queries, device=device).unsqueeze(0) > 0
            has_visible_context = has_visible_context.expand(batch_size, n_queries)
            context_key_padding_mask = None
            if context_mask is not None:
                has_visible_context = has_visible_context & context_mask[:, :n_contexts].any(dim=1, keepdim=True)
                context_key_padding_mask = ~context_mask[:, :n_contexts]

            safe_visible_contexts = torch.tril(
                torch.ones(n_queries, n_contexts, dtype=torch.bool, device=device),
                diagonal=-1,
            )
            safe_visible_contexts[0, 0] = True
            return ~safe_visible_contexts, has_visible_context, context_key_padding_mask

        visible_contexts = torch.tril(
            torch.ones(n_queries, n_contexts, dtype=torch.bool, device=device),
            diagonal=-1,
        ).unsqueeze(0)
        if context_mask is not None:
            visible_contexts = visible_contexts & context_mask[:, None, :n_contexts]
        else:
            visible_contexts = visible_contexts.expand(batch_size, n_queries, n_contexts)

        attention_mask, has_visible_context = self._make_safe_context_attention_mask(visible_contexts)
        return attention_mask, has_visible_context, None

    def _build_step_context_attention_mask(
            self,
            *,
            batch_size: int,
            context_mask: torch.Tensor | None,
            n_contexts: int,
            device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if n_contexts == 0:
            return None, None, None

        if self._can_use_prefix_context_mask():
            context_key_padding_mask = None
            if context_mask is None:
                has_visible_context = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
            else:
                has_visible_context = context_mask[:, :n_contexts].any(dim=1, keepdim=True)
                context_key_padding_mask = ~context_mask[:, :n_contexts]
            return torch.zeros(1, n_contexts, dtype=torch.bool, device=device), has_visible_context, context_key_padding_mask

        if context_mask is None:
            visible_contexts = torch.ones(batch_size, 1, n_contexts, dtype=torch.bool, device=device)
        else:
            visible_contexts = context_mask[:, None, :n_contexts]

        attention_mask, has_visible_context = self._make_safe_context_attention_mask(visible_contexts)
        return attention_mask, has_visible_context, None

    def _build_context_self_attention_mask(
            self,
            *,
            batch_size: int,
            n_contexts: int,
            context_mask: torch.Tensor | None,
            device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if n_contexts == 0:
            return None, None, None

        if self._can_use_prefix_context_mask():
            context_key_padding_mask = None
            has_visible_context = torch.ones(batch_size, n_contexts, dtype=torch.bool, device=device)
            if context_mask is not None:
                has_visible_context = has_visible_context & context_mask[:, :n_contexts].any(dim=1, keepdim=True)
                context_key_padding_mask = ~context_mask[:, :n_contexts]
            attention_mask = ~torch.tril(torch.ones(n_contexts, n_contexts, dtype=torch.bool, device=device))
            return attention_mask, has_visible_context, context_key_padding_mask

        visible_contexts = torch.tril(
            torch.ones(n_contexts, n_contexts, dtype=torch.bool, device=device),
            diagonal=0,
        ).unsqueeze(0)
        if context_mask is not None:
            visible_contexts = visible_contexts & context_mask[:, None, :n_contexts]
        else:
            visible_contexts = visible_contexts.expand(batch_size, n_contexts, n_contexts)

        attention_mask, has_visible_context = self._make_safe_context_attention_mask(visible_contexts)
        return attention_mask, has_visible_context, None

    def _can_use_prefix_context_mask(self) -> bool:
        return self.assume_agent_mask_is_active_prefix

    def _make_safe_context_attention_mask(
            self,
            visible_contexts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        has_visible_context = visible_contexts.any(dim=-1)
        safe_visible_contexts = visible_contexts.clone()
        safe_visible_contexts[:, :, 0] = safe_visible_contexts[:, :, 0] | ~has_visible_context
        attention_mask = ~safe_visible_contexts
        return attention_mask.repeat_interleave(self.nhead, dim=0), has_visible_context

    def _reinitialize_layers(self, *, feedforward_init_gain: float) -> None:
        for layer in self.layers:
            reinitialize_multihead_attention(layer.query_context_attn)
            if layer.context_self_attn is not None:
                reinitialize_multihead_attention(layer.context_self_attn)
            reinitialize_multihead_attention(layer.memory_attn)
            init_transformer_feedforward(layer, gain=feedforward_init_gain)
