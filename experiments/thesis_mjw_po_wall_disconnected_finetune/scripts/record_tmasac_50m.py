from __future__ import annotations

from recording_common import run_recording


if __name__ == "__main__":
    raise SystemExit(
        run_recording(
            disable_policy_connector_actions=False,
            variant_name="tmasac_baseline",
        )
    )
