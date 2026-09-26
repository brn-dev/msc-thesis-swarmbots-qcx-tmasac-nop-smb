from pathlib import Path

from common import run_experiment


def main() -> None:
    run_experiment(
        variant_name="mat_orig",
        entrypoint_path=Path(__file__).resolve(),
        policy_variant="mat_orig",
    )


if __name__ == "__main__":
    main()
