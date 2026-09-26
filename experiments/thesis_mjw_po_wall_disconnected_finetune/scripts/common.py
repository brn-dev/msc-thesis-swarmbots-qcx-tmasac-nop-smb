from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.tmasac_experiment_common import run_tmasac_experiment
from experiments.thesis_experiment_common import run_thesis_ppo_experiment
from swarmbots.mjw_env.swarm.mjw_homogeneous_swarm import (
    MJWPreConnectedUnitLocationsConfig,
)
from swarmbots.scenario_presets.scenario_presets_kwargs import (
    PO_WALL_MEDIUM_SCENARIO_KWARGS,
    make_scenario_kwargs,
)

EXPERIMENT_RUN_NAME = "thesis_mjw_po_wall_disconnected_finetune"
ADDITIONAL_TIMESTEPS = 20_000_000
EVALUATION_MILESTONES = (50.0, 100.0)
EVALUATION_RECORDING_EPISODES = 0
LIVE_RECORDING_SCHEDULE: dict[float, int] = {}
LONG_ADDITIONAL_TIMESTEPS = 50_000_000
LONG_EXPERIMENT_RUN_NAME = "thesis_mjw_po_wall_disconnected_finetune_50m"
HARD_WALL_HEIGHT = 0.4
HARD_WALL_EXPERIMENT_RUN_NAME = (
    "thesis_mjw_po_wall_disconnected_finetune_hard_wall_50m"
)


def _make_disconnected_pool(*, pool_seed_start: int) -> MJWPreConnectedUnitLocationsConfig:
    return MJWPreConnectedUnitLocationsConfig(
        num_units=5,
        num_unit_probs={4: 1.0, 5: 1.0},
        max_radius=1.5,
        unconnected_prob=1.0,
        z_pos=0.5,
        pool_seeds=tuple(range(pool_seed_start, pool_seed_start + 50)),
    )


SCENARIO_KWARGS = make_scenario_kwargs(
    PO_WALL_MEDIUM_SCENARIO_KWARGS,
    {
        "continuous_connector_actions": True,
        "unit_start_locations": _make_disconnected_pool(pool_seed_start=42_000),
    },
)
EVALUATION_SCENARIO_KWARGS = make_scenario_kwargs(
    PO_WALL_MEDIUM_SCENARIO_KWARGS,
    {
        "continuous_connector_actions": True,
        "unit_start_locations": _make_disconnected_pool(pool_seed_start=3_000_000),
    },
)
HARD_WALL_SCENARIO_KWARGS = make_scenario_kwargs(
    SCENARIO_KWARGS,
    {"wall_height": HARD_WALL_HEIGHT},
)
HARD_WALL_EVALUATION_SCENARIO_KWARGS = make_scenario_kwargs(
    EVALUATION_SCENARIO_KWARGS,
    {"wall_height": HARD_WALL_HEIGHT},
)


def run_experiment(*, checkpoint_path: Path, entrypoint_path: Path) -> None:
    checkpoint_path = _resolve_checkpoint(checkpoint_path)

    run_tmasac_experiment(
        experiment_run_name=EXPERIMENT_RUN_NAME,
        scenario_name="wall",
        scenario_kwargs=SCENARIO_KWARGS,
        evaluation_scenario_kwargs=EVALUATION_SCENARIO_KWARGS,
        evaluation_milestones=EVALUATION_MILESTONES,
        evaluation_recording_episodes=EVALUATION_RECORDING_EPISODES,
        live_recording_schedule=LIVE_RECORDING_SCHEDULE,
        variant="tmasac_baseline",
        variant_name="tmasac_disconnected_finetune",
        entrypoint_path=entrypoint_path,
        load_path=checkpoint_path,
        additional_timesteps=ADDITIONAL_TIMESTEPS,
    )


def run_tmasac_50m(*, checkpoint_path: Path, entrypoint_path: Path) -> None:
    _run_tmasac_50m_variant(
        checkpoint_path=checkpoint_path,
        entrypoint_path=entrypoint_path,
        experiment_run_name=LONG_EXPERIMENT_RUN_NAME,
        scenario_kwargs=SCENARIO_KWARGS,
        evaluation_scenario_kwargs=EVALUATION_SCENARIO_KWARGS,
        variant_name="tmasac_baseline",
    )


def run_tmasac_no_connectors_50m(
    *,
    checkpoint_path: Path,
    entrypoint_path: Path,
) -> None:
    _run_tmasac_50m_variant(
        checkpoint_path=checkpoint_path,
        entrypoint_path=entrypoint_path,
        experiment_run_name=LONG_EXPERIMENT_RUN_NAME,
        scenario_kwargs=SCENARIO_KWARGS,
        evaluation_scenario_kwargs=EVALUATION_SCENARIO_KWARGS,
        variant_name="tmasac_no_connectors",
        disable_connector_actions=True,
    )


def run_tmasac_hard_wall_50m(
    *,
    checkpoint_path: Path,
    entrypoint_path: Path,
) -> None:
    _run_tmasac_50m_variant(
        checkpoint_path=checkpoint_path,
        entrypoint_path=entrypoint_path,
        experiment_run_name=HARD_WALL_EXPERIMENT_RUN_NAME,
        scenario_kwargs=HARD_WALL_SCENARIO_KWARGS,
        evaluation_scenario_kwargs=HARD_WALL_EVALUATION_SCENARIO_KWARGS,
        variant_name="tmasac_baseline",
    )


def run_tmasac_no_connectors_hard_wall_50m(
    *,
    checkpoint_path: Path,
    entrypoint_path: Path,
) -> None:
    _run_tmasac_50m_variant(
        checkpoint_path=checkpoint_path,
        entrypoint_path=entrypoint_path,
        experiment_run_name=HARD_WALL_EXPERIMENT_RUN_NAME,
        scenario_kwargs=HARD_WALL_SCENARIO_KWARGS,
        evaluation_scenario_kwargs=HARD_WALL_EVALUATION_SCENARIO_KWARGS,
        variant_name="tmasac_no_connectors",
        disable_connector_actions=True,
    )


def _run_tmasac_50m_variant(
    *,
    checkpoint_path: Path,
    entrypoint_path: Path,
    experiment_run_name: str,
    scenario_kwargs: dict[str, object],
    evaluation_scenario_kwargs: dict[str, object],
    variant_name: str,
    disable_connector_actions: bool = False,
) -> None:
    checkpoint_path = _resolve_checkpoint(checkpoint_path)
    connector_ablation_kwargs: dict[str, object] = {}
    if disable_connector_actions:
        connector_ablation_kwargs = {
            "disable_connector_actions": True,
            "migrate_removed_connector_actions": True,
        }

    run_tmasac_experiment(
        experiment_run_name=experiment_run_name,
        scenario_name="wall",
        scenario_kwargs=scenario_kwargs,
        evaluation_scenario_kwargs=evaluation_scenario_kwargs,
        evaluation_milestones=EVALUATION_MILESTONES,
        evaluation_recording_episodes=EVALUATION_RECORDING_EPISODES,
        live_recording_schedule=LIVE_RECORDING_SCHEDULE,
        variant="tmasac_baseline",
        variant_name=variant_name,
        entrypoint_path=entrypoint_path,
        load_path=checkpoint_path,
        additional_timesteps=LONG_ADDITIONAL_TIMESTEPS,
        **connector_ablation_kwargs,
    )


def run_mat_qcx_50m(*, checkpoint_path: Path, entrypoint_path: Path) -> None:
    checkpoint_path = _resolve_checkpoint(checkpoint_path)
    run_thesis_ppo_experiment(
        experiment_run_name=LONG_EXPERIMENT_RUN_NAME,
        scenario_name="wall",
        scenario_kwargs=SCENARIO_KWARGS,
        evaluation_scenario_kwargs=EVALUATION_SCENARIO_KWARGS,
        evaluation_milestones=EVALUATION_MILESTONES,
        evaluation_recording_episodes=EVALUATION_RECORDING_EPISODES,
        live_recording_schedule=LIVE_RECORDING_SCHEDULE,
        variant="mat_qcx",
        variant_name="mat_qcx",
        entrypoint_path=entrypoint_path,
        load_path=checkpoint_path,
        additional_timesteps=LONG_ADDITIONAL_TIMESTEPS,
    )


def _resolve_checkpoint(checkpoint_path: Path) -> Path:
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    return checkpoint_path
