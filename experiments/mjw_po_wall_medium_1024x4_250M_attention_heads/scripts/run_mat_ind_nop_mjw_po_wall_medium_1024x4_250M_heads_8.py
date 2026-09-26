from pathlib import Path

from common import run_experiment


def main() -> None:
    run_experiment(
        variant_name="mat_ind_heads_8",
        entrypoint_path=Path(__file__).resolve(),
        policy_variant="mat_ind",
        encoder_attention_heads=8,
    )


if __name__ == "__main__":
    main()
