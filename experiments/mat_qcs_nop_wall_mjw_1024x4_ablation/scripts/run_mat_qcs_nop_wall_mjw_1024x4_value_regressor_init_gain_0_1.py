from pathlib import Path

from common import MATInitGains, run_ablation


def main() -> None:
    run_ablation(
        variant_name="sign_magnitude_beta_nop_value_regressor_init_gain_0_1",
        entrypoint_path=Path(__file__).resolve(),
        mat_init_gains=MATInitGains(critic_value_regressor=0.1),
    )


if __name__ == "__main__":
    main()
