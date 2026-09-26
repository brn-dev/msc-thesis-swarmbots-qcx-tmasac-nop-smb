from __future__ import annotations

from recording_common import (
    HARD_WALL_EXPERIMENT_RUN_DIR,
    HARD_WALL_RECORDING_ROOT,
    run_recording,
)


if __name__ == "__main__":
    raise SystemExit(
        run_recording(
            disable_policy_connector_actions=True,
            variant_name="tmasac_no_connectors",
            experiment_run_dir=HARD_WALL_EXPERIMENT_RUN_DIR,
            recording_root=HARD_WALL_RECORDING_ROOT,
            scenario_kwargs_overrides={"wall_height": 0.4},
            morphology_description=(
                "fully disconnected 4- and 5-unit PO-wall pools at wall height 0.4"
            ),
        )
    )
