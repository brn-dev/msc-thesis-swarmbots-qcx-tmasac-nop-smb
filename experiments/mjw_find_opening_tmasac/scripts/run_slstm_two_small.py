from pathlib import Path

from common import run_experiment


def main() -> None:
    run_experiment(
        variant_name="slstm_two_small",
        temporal_model_variant="slstm",
        entrypoint_path=Path(__file__).resolve(),
    )


if __name__ == "__main__":
    main()
