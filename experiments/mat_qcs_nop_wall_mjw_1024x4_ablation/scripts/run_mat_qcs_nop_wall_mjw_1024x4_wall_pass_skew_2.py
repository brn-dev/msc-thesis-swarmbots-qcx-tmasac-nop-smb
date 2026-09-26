from pathlib import Path

from common import run_ablation


def main() -> None:
    run_ablation(
        variant_name="sign_magnitude_beta_nop_wall_pass_skew_2",
        entrypoint_path=Path(__file__).resolve(),
        scenario_kwargs={"wall_pass_reward_skew": 2.0},
    )


if __name__ == "__main__":
    main()
