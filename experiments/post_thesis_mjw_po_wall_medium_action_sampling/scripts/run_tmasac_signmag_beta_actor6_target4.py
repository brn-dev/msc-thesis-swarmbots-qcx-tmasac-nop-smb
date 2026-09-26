from pathlib import Path

from common import run_experiment

if __name__ == "__main__":
    run_experiment(
        distribution="signmag_beta",
        actor_action_samples=6,
        entrypoint_path=Path(__file__).resolve(),
    )
