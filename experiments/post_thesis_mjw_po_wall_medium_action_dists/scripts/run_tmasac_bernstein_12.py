from pathlib import Path

from common import run_experiment

if __name__ == "__main__":
    run_experiment(
        continuous_action_dist="bernstein_12",
        entrypoint_path=Path(__file__).resolve(),
    )
