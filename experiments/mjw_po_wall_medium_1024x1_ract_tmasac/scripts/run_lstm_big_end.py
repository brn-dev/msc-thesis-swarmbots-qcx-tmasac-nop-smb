from pathlib import Path

from common import run_experiment


if __name__ == "__main__":
    run_experiment(
        variant_name="lstm_big_end",
        temporal_model_variant="lstm",
        mlp_layout="big_end",
        entrypoint_path=Path(__file__),
    )
