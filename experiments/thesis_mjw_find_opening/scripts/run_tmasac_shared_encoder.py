from pathlib import Path

from common import run_experiment

if __name__ == "__main__":
    run_experiment(
        variant="tmasac_shared_encoder",
        entrypoint_path=Path(__file__).resolve(),
    )
