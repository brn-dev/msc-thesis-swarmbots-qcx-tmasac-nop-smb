from dataclasses import dataclass

import torch
from torch import nn

from swarmbots.learn.algos.r_mat.temporal_sequence_model import TemporalSequenceModel
from swarmbots.learn.algos.xlstm.head_utils import HeadwiseLinearProjection
from swarmbots.learn.algos.xlstm.temporal_utils import check_mask, check_sequence_inputs, reset_state, select_state
from swarmbots.learn.algos.xlstm.mlstm.mlstm_cell import MLSTMCell, MLSTMCellConfig, MLSTMCellState
from swarmbots.learn.temporal_state import initialize_selected_temporal_state, update_selected_temporal_state


@dataclass(frozen=True)
class MLSTMTemporalSequenceModelConfig:
    num_heads: int = 4
    bias: bool = False
    qkv_proj_blocksize: int = 4
    dense_qkv_proj: bool = False
    eps: float = 1e-6
    use_parallel_sequence: bool = True


class MLSTMTemporalSequenceModel(TemporalSequenceModel):

    def __init__(
            self,
            hidden_dim: int,
            config: MLSTMTemporalSequenceModelConfig = MLSTMTemporalSequenceModelConfig(),
    ) -> None:
        super().__init__(hidden_dim=hidden_dim)
        self.config = config
        if config.dense_qkv_proj:
            self.qkv_projection = nn.Linear(hidden_dim, 3 * hidden_dim, bias=config.bias)
        else:
            if config.qkv_proj_blocksize < 1:
                raise ValueError(f"qkv_proj_blocksize must be >= 1, got {config.qkv_proj_blocksize}")
            if hidden_dim % config.qkv_proj_blocksize != 0:
                raise ValueError(
                    f"hidden_dim={hidden_dim} must be divisible by qkv_proj_blocksize={config.qkv_proj_blocksize}",
                )
            self.qkv_projection = HeadwiseLinearProjection(
                hidden_dim=hidden_dim,
                num_heads=hidden_dim // config.qkv_proj_blocksize,
                num_projections=3,
                bias=config.bias,
            )
        self.cell = MLSTMCell(
            hidden_dim=hidden_dim,
            config=MLSTMCellConfig(num_heads=config.num_heads, eps=config.eps),
        )
        if config.dense_qkv_proj and self.qkv_projection.bias is not None:
            nn.init.zeros_(self.qkv_projection.bias)

    def initial_state(
            self,
            batch_size: int,
            *,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ) -> MLSTMCellState:
        parameter = next(self.parameters())
        return self.cell.initial_state(
            batch_size=batch_size,
            device=parameter.device if device is None else device,
            dtype=parameter.dtype if dtype is None else dtype,
        )

    def forward(
            self,
            inputs: torch.Tensor,
            *,
            valid_mask: torch.Tensor | None = None,
            initial_state: MLSTMCellState | None = None,
            reset_mask: torch.Tensor | None = None,
            state_output_indices: torch.Tensor | None = None,
    ) -> (
        tuple[torch.Tensor, MLSTMCellState]
        | tuple[torch.Tensor, MLSTMCellState, MLSTMCellState]
    ):
        batch_size, sequence_length, _ = check_sequence_inputs(inputs)
        check_mask(valid_mask, batch_size=batch_size, sequence_length=sequence_length, name="valid_mask")
        check_mask(reset_mask, batch_size=batch_size, sequence_length=sequence_length, name="reset_mask")

        state = (
            self.initial_state(batch_size=batch_size, device=inputs.device, dtype=inputs.dtype)
            if initial_state is None
            else initial_state
        )

        if self.config.use_parallel_sequence:
            return self._forward_parallel(
                inputs,
                valid_mask=valid_mask,
                initial_state=state,
                reset_mask=reset_mask,
                state_output_indices=state_output_indices,
            )

        return self._forward_recurrent(
            inputs,
            valid_mask=valid_mask,
            initial_state=state,
            reset_mask=reset_mask,
            state_output_indices=state_output_indices,
        )

    def _forward_recurrent(
            self,
            inputs: torch.Tensor,
            *,
            valid_mask: torch.Tensor | None,
            initial_state: MLSTMCellState,
            reset_mask: torch.Tensor | None,
            state_output_indices: torch.Tensor | None,
    ) -> (
        tuple[torch.Tensor, MLSTMCellState]
        | tuple[torch.Tensor, MLSTMCellState, MLSTMCellState]
    ):
        batch_size, sequence_length, _ = inputs.shape
        state = initial_state
        zero_output = inputs.new_zeros((batch_size, self.hidden_dim))
        outputs: list[torch.Tensor] = []
        selected_states = (
            initialize_selected_temporal_state(state, state_output_indices)
            if state_output_indices is not None
            else None
        )

        for time_idx in range(sequence_length):
            if reset_mask is not None:
                state = reset_state(state, reset_mask[:, time_idx])

            q, k, v = self.qkv_projection(inputs[:, time_idx]).chunk(3, dim=-1)
            step_output, next_state = self.cell(q, k, v, state)

            if valid_mask is None:
                outputs.append(step_output)
                state = next_state
            else:
                valid_t = valid_mask[:, time_idx]
                outputs.append(torch.where(valid_t.unsqueeze(-1), step_output, zero_output))
                state = select_state(next_state, state, valid_t)

            if state_output_indices is not None:
                selected_states = update_selected_temporal_state(
                    selected_states,
                    state,
                    state_output_indices,
                    time_idx,
                )

        output_sequence = torch.stack(outputs, dim=1).contiguous()
        if state_output_indices is not None:
            return output_sequence, state, selected_states
        return output_sequence, state

    def _forward_parallel(
            self,
            inputs: torch.Tensor,
            *,
            valid_mask: torch.Tensor | None,
            initial_state: MLSTMCellState,
            reset_mask: torch.Tensor | None,
            state_output_indices: torch.Tensor | None,
    ) -> (
        tuple[torch.Tensor, MLSTMCellState]
        | tuple[torch.Tensor, MLSTMCellState, MLSTMCellState]
    ):
        q, k, v = self.qkv_projection(inputs).chunk(3, dim=-1)
        result = self.cell.forward_sequence(
            q,
            k,
            v,
            initial_state,
            valid_mask=valid_mask,
            reset_mask=reset_mask,
            state_output_indices=state_output_indices,
        )
        if state_output_indices is not None:
            output, state, selected_states = result
            return output.contiguous(), state, selected_states
        output, state = result
        return output.contiguous(), state
