from pathlib import Path

from common import run_ablation


def main() -> None:
    run_ablation(
        variant_name="sign_magnitude_beta_nop_shuffle_agents",
        entrypoint_path=Path(__file__).resolve(),
        shuffle_agents=True,
        preserve_inactive_prefix_structure=False,
    )


if __name__ == "__main__":
    main()
