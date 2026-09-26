from pathlib import Path

from common import run_mat_experiment

if __name__ == "__main__":
    run_mat_experiment(
        policy_variant="mat_ind",
        continuous_action_dist="bernstein_6",
        entrypoint_path=Path(__file__).resolve(),
    )
