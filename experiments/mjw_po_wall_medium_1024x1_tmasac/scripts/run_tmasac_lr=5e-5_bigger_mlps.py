from pathlib import Path

from common import run_experiment
from swarmbots.learn.nn_components.feed_forward import MLPConfig

def main() -> None:
    run_experiment(
        variant_name="tmasac_lr=5e-5_bigger_mlps",
        sac_learning_rate=5e-5,
        sac_ent_coef_learning_rate=None,
        sac_ent_coef="auto_0.05",
        sac_target_entropy="auto_0.1",
        entrypoint_path=Path(__file__).resolve(),
        continuous_action_dist="gumbel_softmax_sign_magnitude_beta",
        mat_encoder_transformer_ff_config=MLPConfig(hidden_dims=[512, 512]),
    )


if __name__ == "__main__":
    main()
