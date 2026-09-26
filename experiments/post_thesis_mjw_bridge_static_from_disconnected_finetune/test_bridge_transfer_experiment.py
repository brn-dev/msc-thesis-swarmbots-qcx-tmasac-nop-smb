from pathlib import Path
from typing import Any

from experiments.post_thesis_mjw_bridge_static_from_disconnected_finetune.scripts import (
    common,
)


def test_transfer_experiment_uses_static_bridge_scenario(monkeypatch: Any) -> None:
    invocation: dict[str, object] = {}

    def fake_run_tmasac_experiment(**kwargs: object) -> None:
        invocation.update(kwargs)

    monkeypatch.setattr(common, "run_tmasac_experiment", fake_run_tmasac_experiment)
    common.run_experiment(
        checkpoint_path=Path(__file__),
        entrypoint_path=Path(__file__),
    )

    assert invocation["experiment_run_name"] == (
        "post_thesis_mjw_bridge_static_from_disconnected_finetune"
    )
    assert invocation["scenario_name"] == "bridge"
    assert invocation["scenario_kwargs"] == {"continuous_connector_actions": True}
    assert invocation["variant"] == "tmasac_baseline"
    assert invocation["variant_name"] == "tmasac_from_disconnected_finetune"
    assert invocation["additional_timesteps"] == 100_000_000
    assert invocation["transfer_checkpoint"] is True
