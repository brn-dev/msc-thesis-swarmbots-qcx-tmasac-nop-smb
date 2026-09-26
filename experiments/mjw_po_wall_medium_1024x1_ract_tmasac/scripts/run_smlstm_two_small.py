from pathlib import Path

from common import run_experiment


if __name__ == "__main__":
    run_experiment(
        variant_name="smlstm_two_small",
        temporal_model_variant="smlstm",
        mlp_layout="two_small",
        entrypoint_path=Path(__file__),
    )
