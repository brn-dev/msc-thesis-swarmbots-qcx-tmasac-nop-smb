from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.mjw_experiment_common import run_experiment as run_mjw_wall_experiment

SCENARIO_KWARGS: dict[str, object] = {"continuous_connector_actions": False}


def run_experiment(
        *,
        num_envs: int,
        rollout_steps_per_env: int,
        variant_name: str,
        entrypoint_path: Path,
        virtual_mini_batches: int = 1,
        n_epochs: int = 8,
) -> None:
    run_mjw_wall_experiment(
        num_envs=num_envs,
        rollout_steps_per_env=rollout_steps_per_env,
        variant_name=variant_name,
        entrypoint_path=entrypoint_path,
        virtual_mini_batches=virtual_mini_batches,
        n_epochs=n_epochs,
        mat_add_agent_embeddings=True,
        nop_add_agent_embeddings_transition_model=True,
        scenario_kwargs=SCENARIO_KWARGS,
    )
