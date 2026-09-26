from pathlib import Path

from common import make_mat_hidden_init_gains, make_nop_hidden_init_gains, run_ablation


def main() -> None:
    run_ablation(
        variant_name="sign_magnitude_beta_nop_init_gain_0_1_projections_1",
        entrypoint_path=Path(__file__).resolve(),
        mat_init_gains=make_mat_hidden_init_gains(0.1, projection_gain=1.0),
        nop_init_gains=make_nop_hidden_init_gains(0.1, projection_gain=1.0),
    )


if __name__ == "__main__":
    main()
