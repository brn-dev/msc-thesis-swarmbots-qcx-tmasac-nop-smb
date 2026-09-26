from pathlib import Path

from common import run_experiment


def main() -> None:
    run_experiment(
        variant_name="mat_qcs_context_tokens_only_gsde",
        entrypoint_path=Path(__file__).resolve(),
        continuous_action_dist="gsde",
    )


if __name__ == "__main__":
    main()
