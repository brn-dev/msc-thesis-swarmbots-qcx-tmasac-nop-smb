from pathlib import Path

from common import run_experiment


def main() -> None:
    run_experiment(
        variant_name="mat_ind_rsmk_no_transition_obs",
        entrypoint_path=Path(__file__).resolve(),
        continuous_action_dist="reparameterized_sign_magnitude_kumaraswamy",
    )


if __name__ == "__main__":
    main()
