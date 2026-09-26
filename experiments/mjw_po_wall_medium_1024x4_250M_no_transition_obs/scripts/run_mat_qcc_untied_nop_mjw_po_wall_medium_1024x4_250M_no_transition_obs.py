from pathlib import Path

from common import run_experiment


def main() -> None:
    run_experiment(
        variant_name="mat_qcc_untied",
        entrypoint_path=Path(__file__).resolve(),
        mat_qcc_tie_query_context_and_context_self_attention=False,
    )


if __name__ == "__main__":
    main()
