from pathlib import Path

from common import run_experiment


def main() -> None:
    run_experiment(
        variant_name="r_mat_qcc",
        entrypoint_path=Path(__file__).resolve(),
    )


if __name__ == "__main__":
    main()
