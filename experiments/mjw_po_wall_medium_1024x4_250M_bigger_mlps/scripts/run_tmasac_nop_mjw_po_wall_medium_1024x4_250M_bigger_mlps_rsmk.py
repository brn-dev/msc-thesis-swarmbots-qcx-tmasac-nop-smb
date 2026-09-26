from pathlib import Path

from common import run_experiment


def main() -> None:
    run_experiment(
        variant_name="tmasac_rsmk",
        entrypoint_path=Path(__file__).resolve(),
        policy_variant="tmasac",
        continuous_action_dist="reparameterized_sign_magnitude_kumaraswamy",
    )


if __name__ == "__main__":
    main()
