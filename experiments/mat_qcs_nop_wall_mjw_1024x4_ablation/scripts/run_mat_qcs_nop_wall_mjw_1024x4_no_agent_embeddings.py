from pathlib import Path

from common import run_ablation


def main() -> None:
    run_ablation(
        variant_name="sign_magnitude_beta_nop_no_agent_embeddings",
        entrypoint_path=Path(__file__).resolve(),
        mat_add_agent_embeddings=False,
    )


if __name__ == "__main__":
    main()
