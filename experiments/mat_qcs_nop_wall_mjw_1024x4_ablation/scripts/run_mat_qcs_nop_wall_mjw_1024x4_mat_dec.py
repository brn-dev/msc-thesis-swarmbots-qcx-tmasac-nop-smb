from pathlib import Path

from common import run_ablation


def main() -> None:
    run_ablation(
        variant_name="sign_magnitude_beta_nop_mat_dec",
        entrypoint_path=Path(__file__).resolve(),
        policy_variant="mat_dec",
    )


if __name__ == "__main__":
    main()
