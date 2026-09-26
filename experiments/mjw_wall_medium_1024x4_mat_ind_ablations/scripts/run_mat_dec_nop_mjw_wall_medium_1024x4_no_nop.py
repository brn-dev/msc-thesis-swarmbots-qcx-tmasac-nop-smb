from pathlib import Path

from common import run_experiment


def main() -> None:
    run_experiment(
        variant_name="mat_dec_no_nop",
        entrypoint_path=Path(__file__).resolve(),
        policy_variant="mat_dec",
        use_nop=False,
    )


if __name__ == "__main__":
    main()
