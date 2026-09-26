from pathlib import Path
from typing import Any

import pytest
import torch

import experiments.mjw_experiment_common as mjw_common
from experiments.post_thesis_mjw_multi_payload_goal_easy_from_disconnected_finetune.scripts import (
    common,
)


@pytest.mark.parametrize(
    ("task_variant", "expected_scenario_kwargs", "expected_run_variant_name"),
    [
        (
            "easy",
            {
                "max_payloads": 2,
                "active_payload_count_probs": {1: 0.5, 2: 0.5},
                "continuous_connector_actions": True,
            },
            "tmasac_from_disconnected_finetune",
        ),
        (
            "very_easy",
            {
                "max_payloads": 2,
                "active_payload_count_probs": {1: 0.9, 2: 0.1},
                "continuous_connector_actions": True,
            },
            "tmasac_from_disconnected_finetune_very_easy",
        ),
        (
            "single",
            {
                "max_payloads": 1,
                "active_payload_count_probs": {1: 1.0},
                "continuous_connector_actions": True,
            },
            "tmasac_from_disconnected_finetune_single",
        ),
    ],
)
def test_transfer_experiment_uses_requested_multi_payload_goal_variant(
    monkeypatch: Any,
    task_variant: common.TransferTaskVariant,
    expected_scenario_kwargs: dict[str, object],
    expected_run_variant_name: str,
) -> None:
    invocation: dict[str, object] = {}

    def fake_run_tmasac_experiment(**kwargs: object) -> None:
        invocation.update(kwargs)

    monkeypatch.setattr(common, "run_tmasac_experiment", fake_run_tmasac_experiment)
    common.run_experiment(
        checkpoint_path=Path(__file__),
        entrypoint_path=Path(__file__),
        task_variant=task_variant,
    )

    assert invocation["scenario_name"] == "multi_payload_goal"
    assert invocation["scenario_kwargs"] == expected_scenario_kwargs
    assert invocation["variant"] == "tmasac_baseline"
    assert invocation["variant_name"] == expected_run_variant_name
    assert invocation["additional_timesteps"] == 100_000_000
    assert invocation["transfer_checkpoint"] is True


def test_transfer_state_keeps_only_matching_names_and_shapes() -> None:
    matching = torch.ones(2, 3)
    source = {
        "matching": matching,
        "wrong_shape": torch.ones(3),
        "source_only": torch.ones(1),
        "critic.encoder.global_encoder.2.weight": torch.ones(2, 3),
    }
    target = {
        "matching": torch.zeros(2, 3),
        "wrong_shape": torch.zeros(4),
        "target_only": torch.zeros(1),
        "critic.encoder.global_encoder.2.weight": torch.zeros(2, 3),
    }

    transferred = mjw_common.retain_shape_compatible_policy_state(source, target)

    assert list(transferred) == ["matching"]
    assert transferred["matching"] is matching


def test_single_continuation_strictly_resumes_for_200m(monkeypatch: Any) -> None:
    invocation: dict[str, object] = {}

    def fake_run_tmasac_experiment(**kwargs: object) -> None:
        invocation.update(kwargs)

    monkeypatch.setattr(common, "run_tmasac_experiment", fake_run_tmasac_experiment)
    common.continue_single_experiment(
        checkpoint_path=Path(__file__),
        entrypoint_path=Path(__file__),
    )

    assert invocation["experiment_run_name"] == (
        "post_thesis_mjw_multi_payload_goal_single_continue_200m"
    )
    assert invocation["scenario_name"] == "multi_payload_goal"
    assert invocation["scenario_kwargs"] == {
        "max_payloads": 1,
        "active_payload_count_probs": {1: 1.0},
        "continuous_connector_actions": True,
    }
    assert invocation["variant_name"] == "tmasac_from_disconnected_finetune_single"
    assert invocation["additional_timesteps"] == 200_000_000
    assert invocation["transfer_checkpoint"] is False
