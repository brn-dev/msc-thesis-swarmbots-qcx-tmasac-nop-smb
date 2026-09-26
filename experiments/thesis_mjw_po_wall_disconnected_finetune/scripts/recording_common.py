from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.evaluate_thesis_mjw_unseen_morphologies import (
    discover_evaluation_checkpoints,
)
from experiments.record_thesis_mjw_unseen_morphologies import main as _record_main

EXPERIMENT_RUN_DIR = (
    REPO_ROOT
    / "runs"
    / "thesis_mjw_po_wall_disconnected_finetune_50m"
)
RECORDING_ROOT = EXPERIMENT_RUN_DIR / "recordings"
HARD_WALL_EXPERIMENT_RUN_DIR = (
    REPO_ROOT
    / "runs"
    / "thesis_mjw_po_wall_disconnected_finetune_hard_wall_50m"
)
HARD_WALL_RECORDING_ROOT = HARD_WALL_EXPERIMENT_RUN_DIR / "recordings"


def run_recording(
    argv: Sequence[str] | None = None,
    *,
    disable_policy_connector_actions: bool,
    variant_name: str,
    experiment_run_dir: Path = EXPERIMENT_RUN_DIR,
    recording_root: Path = RECORDING_ROOT,
    scenario_kwargs_overrides: Mapping[str, object] | None = None,
    morphology_description: str = "fully disconnected 4- and 5-unit PO-wall pools",
) -> int:
    recording_options = list(sys.argv[1:] if argv is None else argv)
    group_dir = experiment_run_dir / variant_name
    show_help = any(option in {"-h", "--help"} for option in recording_options)
    checkpoints = (
        []
        if show_help
        else discover_evaluation_checkpoints((group_dir,))
    )
    if not checkpoints and not show_help:
        raise FileNotFoundError(
            f"No final or best checkpoints found under {group_dir}"
        )

    checkpoint_options = [
        option
        for checkpoint in checkpoints
        for option in ("--po-wall-checkpoint", str(checkpoint))
    ]

    recorder_argv = [
        "--target",
        "po_wall_tmasac",
        "--unit-counts",
        "4",
        "5",
        "--pool-seed-base",
        "3000000",
        *checkpoint_options,
        *recording_options,
    ]
    return _record_main(
        recorder_argv,
        unconnected_prob=1.0,
        disable_policy_connector_actions=disable_policy_connector_actions,
        scenario_kwargs_overrides=scenario_kwargs_overrides,
        default_output_root=recording_root / variant_name,
        morphology_description=morphology_description,
    )
