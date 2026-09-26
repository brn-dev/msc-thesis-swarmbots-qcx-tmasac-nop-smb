from pathlib import Path

from common import run_experiment

if __name__ == "__main__":
    run_experiment(
        variant="tmasac_swiglu",
        entrypoint_path=Path(__file__).resolve(),
    )
