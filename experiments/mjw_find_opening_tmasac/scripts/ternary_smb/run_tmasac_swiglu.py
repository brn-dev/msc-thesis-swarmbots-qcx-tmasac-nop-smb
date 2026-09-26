from pathlib import Path

from common import PARAMETER_MATCHED_SWIGLU_HIDDEN_DIM, run_experiment
from torch import nn

from swarmbots.learn.nn_components.feed_forward import GLUStackConfig, SwiGLUConfig


def main() -> None:
    run_experiment(
        variant_name="tmasac_swiglu",
        temporal_model_variant="baseline",
        entrypoint_path=Path(__file__).resolve(),
        mat_encoder_transformer_ff_config=SwiGLUConfig(
            hidden_dim=PARAMETER_MATCHED_SWIGLU_HIDDEN_DIM,
            stacked=GLUStackConfig(
                n_layers=2,
                pre_norm=nn.LayerNorm,
                norm_first_layer=False,
                residual=True,
                residual_first_layer=False,
            ),
        ),
    )


if __name__ == "__main__":
    main()
