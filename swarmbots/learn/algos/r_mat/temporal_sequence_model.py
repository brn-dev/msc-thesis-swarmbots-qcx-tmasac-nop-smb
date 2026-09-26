import abc
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from swarmbots.learn.temporal_state import (
    index_temporal_state_batch_time,
    initialize_selected_temporal_state,
    stack_temporal_states,
    update_selected_temporal_state,
)

TemporalModelState = Any


class TemporalSequenceModel(nn.Module, abc.ABC):

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        self.hidden_dim = hidden_dim

    @abc.abstractmethod
    def initial_state(
            self,
            batch_size: int,
            *,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ) -> TemporalModelState:
        raise NotImplementedError

    @abc.abstractmethod
    def forward(
            self,
            inputs: torch.Tensor,
            *,
            valid_mask: torch.Tensor | None = None,
            initial_state: TemporalModelState | None = None,
            reset_mask: torch.Tensor | None = None,
            state_output_indices: torch.Tensor | None = None,
    ) -> (
        tuple[torch.Tensor, TemporalModelState]
        | tuple[torch.Tensor, TemporalModelState, TemporalModelState]
    ):
        raise NotImplementedError

    def forward_with_state_sequence(
            self,
            inputs: torch.Tensor,
            *,
            valid_mask: torch.Tensor | None = None,
            initial_state: TemporalModelState | None = None,
            reset_mask: torch.Tensor | None = None,
            state_output_indices: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        TemporalModelState,
        TemporalModelState | None,
        TemporalModelState,
    ]:
        raise NotImplementedError(
            f"{type(self).__name__} does not expose a post-step state sequence."
        )


LSTMTemporalModelState = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class LSTMTemporalSequenceModelConfig:
    num_layers: int = 1
    bias: bool = False
    dropout: float = 0.0


class LSTMTemporalSequenceModel(TemporalSequenceModel):

    def __init__(
            self,
            hidden_dim: int,
            config: LSTMTemporalSequenceModelConfig = LSTMTemporalSequenceModelConfig(),
    ) -> None:
        super().__init__(hidden_dim=hidden_dim)
        if config.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {config.num_layers}")
        if config.dropout < 0:
            raise ValueError(f"dropout must be >= 0, got {config.dropout}")

        self.config = config
        lstm_dropout = config.dropout if config.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=config.num_layers,
            bias=config.bias,
            dropout=lstm_dropout,
            batch_first=True,
        )

    def initial_state(
            self,
            batch_size: int,
            *,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ) -> LSTMTemporalModelState:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        parameter = next(self.parameters())
        device = parameter.device if device is None else device
        dtype = parameter.dtype if dtype is None else dtype
        shape = (batch_size, self.config.num_layers, self.hidden_dim)
        hidden_state = torch.zeros(shape, device=device, dtype=dtype)
        cell_state = torch.zeros(shape, device=device, dtype=dtype)
        return hidden_state, cell_state

    def forward(
            self,
            inputs: torch.Tensor,
            *,
            valid_mask: torch.Tensor | None = None,
            initial_state: LSTMTemporalModelState | None = None,
            reset_mask: torch.Tensor | None = None,
            state_output_indices: torch.Tensor | None = None,
    ) -> (
        tuple[torch.Tensor, LSTMTemporalModelState]
        | tuple[torch.Tensor, LSTMTemporalModelState, LSTMTemporalModelState]
    ):
        output_sequence, final_state, selected_states, _state_sequence = self._forward(
            inputs,
            valid_mask=valid_mask,
            initial_state=initial_state,
            reset_mask=reset_mask,
            state_output_indices=state_output_indices,
            return_state_sequence=False,
        )
        if state_output_indices is not None:
            return output_sequence, final_state, selected_states
        return output_sequence, final_state

    def forward_with_state_sequence(
            self,
            inputs: torch.Tensor,
            *,
            valid_mask: torch.Tensor | None = None,
            initial_state: LSTMTemporalModelState | None = None,
            reset_mask: torch.Tensor | None = None,
            state_output_indices: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        LSTMTemporalModelState,
        LSTMTemporalModelState | None,
        LSTMTemporalModelState,
    ]:
        output_sequence, final_state, selected_states, state_sequence = self._forward(
            inputs,
            valid_mask=valid_mask,
            initial_state=initial_state,
            reset_mask=reset_mask,
            state_output_indices=state_output_indices,
            return_state_sequence=True,
        )
        assert state_sequence is not None
        return output_sequence, final_state, selected_states, state_sequence

    def _forward(
            self,
            inputs: torch.Tensor,
            *,
            valid_mask: torch.Tensor | None,
            initial_state: LSTMTemporalModelState | None,
            reset_mask: torch.Tensor | None,
            state_output_indices: torch.Tensor | None,
            return_state_sequence: bool,
    ) -> tuple[
        torch.Tensor,
        LSTMTemporalModelState,
        LSTMTemporalModelState | None,
        LSTMTemporalModelState | None,
    ]:
        if inputs.ndim != 3:
            raise ValueError(f"Expected inputs shape (B, T, H), got {tuple(inputs.shape)}")

        batch_size, sequence_length, hidden_dim = inputs.shape

        valid_mask = self._normalize_mask(valid_mask, batch_size=batch_size, sequence_length=sequence_length, name="valid_mask")
        reset_mask = self._normalize_mask(reset_mask, batch_size=batch_size, sequence_length=sequence_length, name="reset_mask")

        if initial_state is None:
            hidden_state, cell_state = self.initial_state(
                batch_size=batch_size,
                device=inputs.device,
                dtype=inputs.dtype,
            )
        else:
            hidden_state, cell_state = initial_state

        outputs: list[torch.Tensor] = []
        selected_states = (
            initialize_selected_temporal_state((hidden_state, cell_state), state_output_indices)
            if state_output_indices is not None and not return_state_sequence
            else None
        )
        state_steps: list[LSTMTemporalModelState] = []
        zero_output = inputs.new_zeros((batch_size, 1, self.hidden_dim))

        for time_idx in range(sequence_length):
            if reset_mask is not None:
                reset_t = reset_mask[:, time_idx]
                reset_state_mask = reset_t.view(batch_size, 1, 1)
                hidden_state = hidden_state.masked_fill(reset_state_mask, 0.0)
                cell_state = cell_state.masked_fill(reset_state_mask, 0.0)

            step_output, (next_hidden_state, next_cell_state) = self.lstm(
                inputs[:, time_idx:time_idx + 1, :],
                (
                    hidden_state.transpose(0, 1).contiguous(),
                    cell_state.transpose(0, 1).contiguous(),
                ),
            )
            # Compiled nn.LSTM adds a singleton leading dimension to h_n and
            # c_n. Normalize to the documented (layers, batch, hidden) layout.
            lstm_state_shape = (self.config.num_layers, batch_size, self.hidden_dim)
            next_hidden_state = next_hidden_state.reshape(lstm_state_shape)
            next_cell_state = next_cell_state.reshape(lstm_state_shape)
            next_hidden_state = next_hidden_state.transpose(0, 1).contiguous()
            next_cell_state = next_cell_state.transpose(0, 1).contiguous()

            if valid_mask is None:
                outputs.append(step_output)
                hidden_state = next_hidden_state
                cell_state = next_cell_state
            else:
                valid_t = valid_mask[:, time_idx]
                valid_output_mask = valid_t.view(batch_size, 1, 1)
                valid_state_mask = valid_t.view(batch_size, 1, 1)
                outputs.append(torch.where(valid_output_mask, step_output, zero_output))
                hidden_state = torch.where(valid_state_mask, next_hidden_state, hidden_state)
                cell_state = torch.where(valid_state_mask, next_cell_state, cell_state)

            if return_state_sequence:
                state_steps.append((hidden_state, cell_state))
            elif state_output_indices is not None:
                selected_states = update_selected_temporal_state(
                    selected_states,
                    (hidden_state, cell_state),
                    state_output_indices,
                    time_idx,
                )

        output_sequence = torch.cat(outputs, dim=1).contiguous()
        final_state = (hidden_state, cell_state)
        state_sequence = stack_temporal_states(state_steps, dim=1) if return_state_sequence else None
        if return_state_sequence and state_output_indices is not None:
            selected_states = index_temporal_state_batch_time(state_sequence, state_output_indices)
        return output_sequence, final_state, selected_states, state_sequence

    @staticmethod
    def _normalize_mask(
            mask: torch.Tensor | None,
            *,
            batch_size: int,
            sequence_length: int,
            name: str,
    ) -> torch.Tensor | None:
        if mask is None:
            return None
        if mask.shape != (batch_size, sequence_length):
            raise ValueError(
                f"Expected {name} shape ({batch_size}, {sequence_length}), got {tuple(mask.shape)}"
            )
        if mask.dtype != torch.bool:
            raise ValueError(f"Expected {name} dtype torch.bool, got {mask.dtype}")
        return mask
