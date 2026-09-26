from pathlib import Path

from common import run_experiment
from torch import nn

from swarmbots.learn.nn_components.feed_forward import GLUStackConfig, SwiGLUConfig

PARAMETER_MATCHED_SWIGLU_HIDDEN_DIM = 344


def main() -> None:
    run_experiment(
        variant_name="tmasac_lr=5e-5_swiglu",
        sac_learning_rate=5e-5,
        sac_ent_coef_learning_rate=None,
        sac_ent_coef="auto_0.05",
        sac_target_entropy="auto_0.1",
        entrypoint_path=Path(__file__).resolve(),
        continuous_action_dist="gumbel_softmax_sign_magnitude_beta",
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
