from pathlib import Path

from common import run_experiment


def main() -> None:
    run_experiment(
        variant_name="mat_qcx_predicted_std",
        entrypoint_path=Path(__file__).resolve(),
        policy_variant="mat_qcx",
        continuous_action_dist="predicted_std",
    )


if __name__ == "__main__":
    main()
