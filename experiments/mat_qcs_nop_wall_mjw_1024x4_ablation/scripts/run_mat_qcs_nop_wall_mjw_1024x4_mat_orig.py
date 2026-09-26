from pathlib import Path

from common import run_ablation


def main() -> None:
    run_ablation(
        variant_name="sign_magnitude_beta_nop_mat_orig",
        entrypoint_path=Path(__file__).resolve(),
        policy_variant="mat_orig",
    )


if __name__ == "__main__":
    main()
