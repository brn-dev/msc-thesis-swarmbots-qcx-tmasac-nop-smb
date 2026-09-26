from pathlib import Path

from common_250m import run_experiment

if __name__ == "__main__":
    run_experiment(variant="mat_ind", entrypoint_path=Path(__file__).resolve())
