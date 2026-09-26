from pathlib import Path

from common import run_experiment


def main() -> None:
    run_experiment(
        variant_name="mat_qcs_context_tokens_only_no_transition_obs",
        entrypoint_path=Path(__file__).resolve(),
        use_transition_obs=False,
    )


if __name__ == "__main__":
    main()
