from pathlib import Path

from common import run_experiment


def main() -> None:
    run_experiment(
        variant_name="slstm_separate_observation_action_encoders",
        temporal_model_variant="slstm",
        separate_observation_action_encoders=True,
        entrypoint_path=Path(__file__).resolve(),
    )


if __name__ == "__main__":
    main()
