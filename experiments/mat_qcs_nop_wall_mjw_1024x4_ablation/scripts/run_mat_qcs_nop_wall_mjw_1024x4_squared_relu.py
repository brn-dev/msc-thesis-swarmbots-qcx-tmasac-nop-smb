from pathlib import Path

from common import run_ablation
from swarmbots.learn.nn_components.activations import SquaredReLU


def main() -> None:
    run_ablation(
        variant_name="sign_magnitude_beta_nop_squared_relu",
        entrypoint_path=Path(__file__).resolve(),
        act_fn_cls=SquaredReLU,
    )


if __name__ == "__main__":
    main()
