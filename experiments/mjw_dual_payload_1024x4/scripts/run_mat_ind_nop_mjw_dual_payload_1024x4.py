from pathlib import Path

from common import run_experiment


def main() -> None:
    run_experiment(
        variant_name="mat_ind",
        entrypoint_path=Path(__file__).resolve(),
        policy_variant="mat_ind",
    )


if __name__ == "__main__":
    main()
