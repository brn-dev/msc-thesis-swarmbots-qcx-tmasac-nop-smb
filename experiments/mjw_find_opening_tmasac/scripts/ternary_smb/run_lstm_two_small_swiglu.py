from pathlib import Path

from common import PARAMETER_MATCHED_SWIGLU_HIDDEN_DIM, run_experiment

from swarmbots.learn.nn_components.feed_forward import SwiGLUConfig


def main() -> None:
    swiglu_config = SwiGLUConfig(hidden_dim=PARAMETER_MATCHED_SWIGLU_HIDDEN_DIM)
    run_experiment(
        variant_name="lstm_two_small_swiglu",
        temporal_model_variant="lstm",
        entrypoint_path=Path(__file__).resolve(),
        mat_encoder_transformer_ff_config=swiglu_config,
        rmat_actor_transformer_ff_config=swiglu_config,
    )


if __name__ == "__main__":
    main()
