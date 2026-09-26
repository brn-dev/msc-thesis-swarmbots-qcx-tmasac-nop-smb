from pathlib import Path

from common import run_ablation
from swarmbots.learn.nn_components.activations import ParameterLearnMode, SignedSquaredLeakyReluFactory


def main() -> None:
    run_ablation(
        variant_name="sign_magnitude_beta_nop_signed_squared_leaky_relu_per_feature",
        entrypoint_path=Path(__file__).resolve(),
        act_fn_cls=SignedSquaredLeakyReluFactory(
            negative_slope_mode=ParameterLearnMode.PER_FEATURE,
        ),
    )


if __name__ == "__main__":
    main()
