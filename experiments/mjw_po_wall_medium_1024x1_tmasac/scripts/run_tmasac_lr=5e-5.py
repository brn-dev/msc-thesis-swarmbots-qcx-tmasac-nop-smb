from pathlib import Path

from common import run_experiment


def main() -> None:
    run_experiment(
        variant_name="tmasac_lr=5e-5",
        sac_learning_rate=5e-5,
        sac_ent_coef_learning_rate=None,
        sac_ent_coef="auto_0.05",
        sac_target_entropy="auto_0.1",
        entrypoint_path=Path(__file__).resolve(),
        continuous_action_dist="gumbel_softmax_sign_magnitude_beta",
    )


if __name__ == "__main__":
    main()
