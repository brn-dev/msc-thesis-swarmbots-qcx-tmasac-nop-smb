from pathlib import Path

from common import run_experiment


def main() -> None:
    run_experiment(
        variant_name="mat_ind_no_nop",
        entrypoint_path=Path(__file__).resolve(),
        use_nop=False,
    )


if __name__ == "__main__":
    main()
