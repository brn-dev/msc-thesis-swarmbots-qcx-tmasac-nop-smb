from pathlib import Path

from common import run_slstm_tmasac_experiment

if __name__ == "__main__":
    run_slstm_tmasac_experiment(
        continuous_action_dist="rqs_4",
        entrypoint_path=Path(__file__).resolve(),
    )
