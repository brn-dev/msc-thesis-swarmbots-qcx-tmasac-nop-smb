from dataclasses import dataclass
from enum import Enum

import torch
from torch import nn

from swarmbots.learn.nn_components.custom_transformer_decoder_layer import CustomTransformerDecoderLayer
from swarmbots.learn.nn_components.activations import ActivationFactory, make_activation
from swarmbots.learn.nn_components.nn_init import reinitialize_transformer_stack


class MATQCSDecoderSelfAttentionMode(Enum):
    FULL_CAUSAL = 1
    PREVIOUS_AGENTS = 2  # c_i can not attend to q_i, only to q_<i and c_<i
    CONTEXT_TOKENS_ONLY = 3  # only context tokens can be attended to



@dataclass(frozen=True)
class MATQCSDecoderConfig:
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
    self_attention_mode: MATQCSDecoderSelfAttentionMode = MATQCSDecoderSelfAttentionMode.FULL_CAUSAL
    normalize_query_input: bool = False
    normalize_context_input: bool = False
    normalize_memory_input: bool = False
    normalize_query_tokens: bool = False
    normalize_context_tokens: bool = False
    normalize_memory_tokens: bool = False
    normalize_actor_head_input: bool = False
    assume_agent_mask_is_active_prefix: bool = True


class MATQCSDecoder(nn.Module):

    def __init__(
            self,
            config: MATQCSDecoderConfig,
            *,
            max_agents: int,
            memory_d_model: int,
    ) -> None:
        super().__init__()
        if config.d_model is None:
            raise ValueError("MATQCSDecoderConfig.d_model must be set before constructing MATQCSDecoder")
        if config.num_layers <= 0:
            raise ValueError(f"MATQCSDecoderConfig.num_layers must be > 0, got {config.num_layers}")
        self.max_agents = max_agents
        self.d_model = config.d_model
        self.memory_d_model = memory_d_model
        self.self_attention_mode = config.self_attention_mode
        self.nhead = config.nhead
        self.assume_agent_mask_is_active_prefix = config.assume_agent_mask_is_active_prefix

        self.decoder = nn.TransformerDecoder(
            decoder_layer=CustomTransformerDecoderLayer(
                d_model=self.d_model,
                memory_d_model=memory_d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                activation=make_activation(config.act_fn_cls, num_features=config.dim_feedforward),
                layer_norm_eps=config.layer_norm_eps,
                batch_first=True,
                norm_first=config.norm_first,
                bias=config.bias,
            ),
            num_layers=config.num_layers,
            norm=nn.LayerNorm(self.d_model),
        )
        if config.transformer_ff_init_gain is not None:
            reinitialize_transformer_stack(
                self.decoder,
                feedforward_init_gain=config.transformer_ff_init_gain,
            )

        self.register_buffer(
            "parallel_attention_mask",
            self._build_parallel_attention_mask(
                max_agents=self.max_agents,
                self_attention_mode=config.self_attention_mode,
            ),
        )
        self.parallel_attention_mask_is_causal = (
            config.self_attention_mode is MATQCSDecoderSelfAttentionMode.FULL_CAUSAL
        )
        causal_mask = torch.triu(
            torch.ones(self.max_agents + 1, self.max_agents + 1, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("step_causal_mask", causal_mask)

    @staticmethod
    def _build_parallel_attention_mask(
            *,
            max_agents: int,
            self_attention_mode: MATQCSDecoderSelfAttentionMode,
    ) -> torch.Tensor:
        if self_attention_mode is MATQCSDecoderSelfAttentionMode.FULL_CAUSAL:
            seq_len = max_agents * 2
            return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

        seq_len = max_agents * 2
        attention_mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
        for agent_idx in range(max_agents):
            query_idx = agent_idx * 2
            context_idx = query_idx + 1

            attention_mask[query_idx, query_idx] = False
            attention_mask[context_idx, context_idx] = False
            if agent_idx == 0:
                continue

            past_contexts = torch.arange(1, query_idx, 2)
            attention_mask[query_idx, past_contexts] = False
            attention_mask[context_idx, past_contexts] = False

            if self_attention_mode is MATQCSDecoderSelfAttentionMode.CONTEXT_TOKENS_ONLY:
                continue
            if self_attention_mode is MATQCSDecoderSelfAttentionMode.PREVIOUS_AGENTS:
                past_queries = torch.arange(0, query_idx, 2)
                attention_mask[query_idx, past_queries] = False
                attention_mask[context_idx, past_queries] = False
                continue
            raise ValueError(f"Unsupported self_attention_mode: {self_attention_mode}")

        if self_attention_mode is MATQCSDecoderSelfAttentionMode.PREVIOUS_AGENTS:
            context_rows = torch.arange(1, seq_len, 2)
            same_agent_query_cols = context_rows - 1
            attention_mask[context_rows, same_agent_query_cols] = True

        return attention_mask

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
        n_agents = query_tokens.shape[1]
        n_contexts = context_tokens.shape[1]
        if n_contexts not in (n_agents, n_agents - 1):
            raise ValueError(
                f"Expected context_tokens second dim to be {n_agents} or {n_agents - 1}, got {n_contexts}"
            )

        paired_tokens = torch.stack((query_tokens[:, :n_contexts, :], context_tokens), dim=2).reshape(
            query_tokens.shape[0], n_contexts * 2, query_tokens.shape[-1]
        )
        if n_contexts == n_agents:
            interleaved_tokens = paired_tokens
        else:
            interleaved_tokens = torch.cat((paired_tokens, query_tokens[:, -1:, :]), dim=1)
        target_seq_len = interleaved_tokens.shape[1]
        attention_mask = self.parallel_attention_mask[:target_seq_len, :target_seq_len]
        attention_mask_is_causal = self.parallel_attention_mask_is_causal

        target_key_padding_mask = None
        if agent_mask is not None:
            paired_key_padding_mask = torch.stack(
                (~agent_mask[:, :n_contexts], ~agent_mask[:, :n_contexts]), dim=2
            ).reshape(agent_mask.shape[0], n_contexts * 2)
            if n_contexts == n_agents:
                target_key_padding_mask = paired_key_padding_mask
            else:
                target_key_padding_mask = torch.cat((paired_key_padding_mask, ~agent_mask[:, -1:]), dim=1)
            if not self.assume_agent_mask_is_active_prefix:
                attention_mask = self._mask_inactive_target_keys(
                    attention_mask=attention_mask,
                    target_key_padding_mask=target_key_padding_mask,
                )
                target_key_padding_mask = None
                attention_mask_is_causal = False

        memory_key_padding_mask = None
        if memory_mask is not None:
            memory_key_padding_mask = ~memory_mask

        decoder_output = self.decoder(
            tgt=interleaved_tokens,
            memory=memory_tokens,
            tgt_mask=attention_mask,
            tgt_key_padding_mask=target_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_is_causal=attention_mask_is_causal,
        )
        return decoder_output[:, 0::2, :]

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
        if memory_tokens.shape[1] > self.max_agents:
            raise ValueError(
                f"Expected memory_tokens second dim <= {self.max_agents}, got {memory_tokens.shape[1]}"
            )

        use_interleaved_prefix = self.self_attention_mode is not MATQCSDecoderSelfAttentionMode.CONTEXT_TOKENS_ONLY
        if use_interleaved_prefix:
            if query_prefix_tokens is None:
                raise ValueError(
                    f"query_prefix_tokens must be provided for self_attention_mode {self.self_attention_mode}"
                )
            sequence_prefix = torch.stack((query_prefix_tokens, context_tokens), dim=2).reshape(
                query_token.shape[0], query_prefix_tokens.shape[1] * 2, query_token.shape[2]
            )
            sequence = torch.cat((sequence_prefix, query_token), dim=1)
            seq_len = sequence.shape[1]
            attention_mask = self.parallel_attention_mask[:seq_len, :seq_len]
        else:
            sequence = torch.cat((context_tokens, query_token), dim=1)
            seq_len = sequence.shape[1]
            attention_mask = self.step_causal_mask[:seq_len, :seq_len]
        attention_mask_is_causal = self.parallel_attention_mask_is_causal or not use_interleaved_prefix

        target_key_padding_mask = None
        if context_mask is not None or query_mask is not None:
            if context_mask is None:
                context_mask = torch.ones(
                    context_tokens.shape[:2], dtype=torch.bool, device=context_tokens.device
                )

            if use_interleaved_prefix:
                if query_prefix_mask is None:
                    query_prefix_mask = torch.ones(
                        context_tokens.shape[:2], dtype=torch.bool, device=context_tokens.device
                    )

            if query_mask is None:
                query_mask = torch.ones(query_token.shape[0], dtype=torch.bool, device=query_token.device)

            if use_interleaved_prefix:
                interleaved_prefix_mask = torch.stack((query_prefix_mask, context_mask), dim=2).reshape(
                    context_mask.shape[0], context_mask.shape[1] * 2
                )
                combined_mask = torch.cat((interleaved_prefix_mask, query_mask.unsqueeze(1)), dim=1)
            else:
                combined_mask = torch.cat((context_mask, query_mask.unsqueeze(1)), dim=1)
            target_key_padding_mask = ~combined_mask
            if not self.assume_agent_mask_is_active_prefix:
                target_key_padding_mask = self._ensure_step_has_target_key(target_key_padding_mask)

        memory_key_padding_mask = None
        if memory_mask is not None:
            memory_key_padding_mask = ~memory_mask

        decoder_output = self.decoder(
            tgt=sequence,
            memory=memory_tokens,
            tgt_mask=attention_mask,
            tgt_key_padding_mask=target_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_is_causal=attention_mask_is_causal,
        )
        return decoder_output[:, -1:, :]

    @staticmethod
    def _ensure_step_has_target_key(target_key_padding_mask: torch.Tensor) -> torch.Tensor:
        all_targets_masked = target_key_padding_mask.all(dim=1)
        safe_mask = target_key_padding_mask.clone()
        safe_mask[:, 0] = safe_mask[:, 0] & ~all_targets_masked
        return safe_mask

    def _mask_inactive_target_keys(
        self,
        *,
        attention_mask: torch.Tensor,
        target_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(target_key_padding_mask.shape[0])
        seq_len = int(target_key_padding_mask.shape[1])
        combined_mask = attention_mask.unsqueeze(0) | target_key_padding_mask.unsqueeze(1)
        all_keys_masked = combined_mask.all(dim=2)
        diagonal = torch.eye(seq_len, dtype=torch.bool, device=combined_mask.device).unsqueeze(0)
        combined_mask = combined_mask & ~(all_keys_masked.unsqueeze(2) & diagonal)
        return combined_mask.repeat_interleave(self.nhead, dim=0).reshape(batch_size * self.nhead, seq_len, seq_len)
