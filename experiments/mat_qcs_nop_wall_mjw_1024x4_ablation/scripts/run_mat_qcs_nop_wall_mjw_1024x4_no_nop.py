from pathlib import Path

from common import run_ablation


def main() -> None:
    run_ablation(
        variant_name="sign_magnitude_beta_no_nop",
        entrypoint_path=Path(__file__).resolve(),
        use_nop=False,
    )


if __name__ == "__main__":
    main()
