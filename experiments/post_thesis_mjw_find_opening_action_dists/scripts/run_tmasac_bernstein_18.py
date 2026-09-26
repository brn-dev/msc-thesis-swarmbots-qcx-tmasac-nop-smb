from pathlib import Path

from common import run_experiment

if __name__ == "__main__":
    run_experiment(
        continuous_action_dist="bernstein_18",
        entrypoint_path=Path(__file__).resolve(),
    )
