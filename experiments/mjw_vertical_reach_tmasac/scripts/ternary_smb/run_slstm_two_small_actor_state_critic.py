from pathlib import Path

from common import run_experiment

if __name__ == "__main__":
    run_experiment(
        variant="slstm_two_small_actor_state_critic",
        entrypoint_path=Path(__file__).resolve(),
    )
