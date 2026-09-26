from pathlib import Path

from common import run_experiment


def main() -> None:
    run_experiment(
        variant_name="mat_dec_no_transition_obs",
        entrypoint_path=Path(__file__).resolve(),
        policy_variant="mat_dec",
        use_transition_obs=False,
    )


if __name__ == "__main__":
    main()
