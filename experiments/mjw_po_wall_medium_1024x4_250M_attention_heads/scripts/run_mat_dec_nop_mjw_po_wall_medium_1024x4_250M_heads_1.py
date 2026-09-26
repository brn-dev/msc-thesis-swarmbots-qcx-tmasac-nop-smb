from pathlib import Path

from common import run_experiment


def main() -> None:
    run_experiment(
        variant_name="mat_dec_heads_1",
        entrypoint_path=Path(__file__).resolve(),
        policy_variant="mat_dec",
        encoder_attention_heads=1,
    )


if __name__ == "__main__":
    main()
