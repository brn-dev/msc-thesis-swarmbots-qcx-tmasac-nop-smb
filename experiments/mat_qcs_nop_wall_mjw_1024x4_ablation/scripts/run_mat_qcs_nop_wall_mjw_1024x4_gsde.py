from pathlib import Path

from common import run_ablation


def main() -> None:
    run_ablation(
        variant_name="gsde_nop",
        entrypoint_path=Path(__file__).resolve(),
        continuous_action_dist="gsde",
    )


if __name__ == "__main__":
    main()
