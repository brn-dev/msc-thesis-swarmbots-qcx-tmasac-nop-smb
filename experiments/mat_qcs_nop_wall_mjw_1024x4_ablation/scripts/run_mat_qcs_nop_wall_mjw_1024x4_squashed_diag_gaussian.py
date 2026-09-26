from pathlib import Path

from common import run_ablation


def main() -> None:
    run_ablation(
        variant_name="squashed_diag_gaussian_nop",
        entrypoint_path=Path(__file__).resolve(),
        continuous_action_dist="squashed_diag_gaussian",
    )


if __name__ == "__main__":
    main()
