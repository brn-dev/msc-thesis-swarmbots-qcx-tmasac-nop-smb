from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from experiments.mjw_experiment_common import (
    ContinuousActionDistVariant,
    MJWScenarioName,
    PolicyVariant,
)
from experiments.mjw_experiment_common import run_experiment as run_mjw_experiment
from experiments.tmasac_experiment_common import (
    TMASACExperimentVariant,
    run_tmasac_experiment,
)

EXPERIMENT_TOTAL_TIMESTEPS = 100_000_000

ThesisAlgorithmVariant = Literal[
    "mappo",
    "mat_qcx",
    "mat_ind",
    "r_mat_ind",
    "mat_orig",
    "tmasac_baseline",
    "tmasac_shared_encoder",
    "slstm_two_small_actor_state_critic",
    "slstm_two_small_actor_state_critic_recurrent_skip",
    "slstm_shared_encoder",
    "lstm_two_small_actor_state_critic",
    "mat_qcx_gsde",
    "mat_qcx_no_nop",
    "tmasac_baseline_predicted_std",
    "tmasac_baseline_no_nop",
    "slstm_two_small_actor_state_critic_predicted_std",
    "slstm_two_small_actor_state_critic_no_nop",
    "slstm_two_small_actor_state_critic_no_memory_strength",
]


@dataclass(frozen=True)
class ThesisVariantConfig:
    continuous_action_dist: ContinuousActionDistVariant
    use_nop: bool
    include_slstm_memory_strength: bool = False
    policy_variant: PolicyVariant | None = None
    tmasac_variant: TMASACExperimentVariant | None = None


THESIS_VARIANT_CONFIGS: dict[ThesisAlgorithmVariant, ThesisVariantConfig] = {
    "mappo": ThesisVariantConfig(
        policy_variant="mat_ind",
        continuous_action_dist="sign_magnitude_beta",
        use_nop=True,
    ),
    "mat_qcx": ThesisVariantConfig(
        policy_variant="mat_qcx",
        continuous_action_dist="sign_magnitude_beta",
        use_nop=True,
    ),
    "mat_ind": ThesisVariantConfig(
        policy_variant="mat_ind",
        continuous_action_dist="sign_magnitude_beta",
        use_nop=True,
    ),
    "r_mat_ind": ThesisVariantConfig(
        policy_variant="r_mat_ind",
        continuous_action_dist="sign_magnitude_beta",
        use_nop=True,
    ),
    "mat_orig": ThesisVariantConfig(
        policy_variant="mat_orig",
        continuous_action_dist="sign_magnitude_beta",
        use_nop=True,
    ),
    "tmasac_baseline": ThesisVariantConfig(
        tmasac_variant="tmasac_baseline",
        continuous_action_dist="gumbel_softmax_sign_magnitude_beta",
        use_nop=True,
    ),
    "tmasac_shared_encoder": ThesisVariantConfig(
        tmasac_variant="tmasac_shared_encoder",
        continuous_action_dist="gumbel_softmax_sign_magnitude_beta",
        use_nop=True,
    ),
    "slstm_two_small_actor_state_critic": ThesisVariantConfig(
        tmasac_variant="slstm_two_small_actor_state_critic",
        continuous_action_dist="gumbel_softmax_sign_magnitude_beta",
        use_nop=True,
        include_slstm_memory_strength=False,
    ),
    "slstm_two_small_actor_state_critic_recurrent_skip": ThesisVariantConfig(
        tmasac_variant="slstm_two_small_actor_state_critic_recurrent_skip",
        continuous_action_dist="gumbel_softmax_sign_magnitude_beta",
        use_nop=True,
        include_slstm_memory_strength=False,
    ),
    "slstm_shared_encoder": ThesisVariantConfig(
        tmasac_variant="slstm_shared_encoder",
        continuous_action_dist="gumbel_softmax_sign_magnitude_beta",
        use_nop=True,
    ),
    "lstm_two_small_actor_state_critic": ThesisVariantConfig(
        tmasac_variant="lstm_two_small_actor_state_critic",
        continuous_action_dist="gumbel_softmax_sign_magnitude_beta",
        use_nop=True,
    ),
    "mat_qcx_gsde": ThesisVariantConfig(
        policy_variant="mat_qcx",
        continuous_action_dist="gsde",
        use_nop=True,
    ),
    "mat_qcx_no_nop": ThesisVariantConfig(
        policy_variant="mat_qcx",
        continuous_action_dist="sign_magnitude_beta",
        use_nop=False,
    ),
    "tmasac_baseline_predicted_std": ThesisVariantConfig(
        tmasac_variant="tmasac_baseline",
        continuous_action_dist="predicted_std",
        use_nop=True,
    ),
    "tmasac_baseline_no_nop": ThesisVariantConfig(
        tmasac_variant="tmasac_baseline",
        continuous_action_dist="gumbel_softmax_sign_magnitude_beta",
        use_nop=False,
    ),
    "slstm_two_small_actor_state_critic_predicted_std": ThesisVariantConfig(
        tmasac_variant="slstm_two_small_actor_state_critic",
        continuous_action_dist="predicted_std",
        use_nop=True,
        include_slstm_memory_strength=False,
    ),
    "slstm_two_small_actor_state_critic_no_nop": ThesisVariantConfig(
        tmasac_variant="slstm_two_small_actor_state_critic",
        continuous_action_dist="gumbel_softmax_sign_magnitude_beta",
        use_nop=False,
    ),
    "slstm_two_small_actor_state_critic_no_memory_strength": ThesisVariantConfig(
        tmasac_variant="slstm_two_small_actor_state_critic",
        continuous_action_dist="gumbel_softmax_sign_magnitude_beta",
        use_nop=True,
        include_slstm_memory_strength=False,
    ),
}


def run_thesis_experiment(
    *,
    experiment_run_name: str,
    scenario_name: MJWScenarioName,
    scenario_kwargs: dict[str, object],
    variant: ThesisAlgorithmVariant,
    entrypoint_path: Path,
) -> None:
    config = THESIS_VARIANT_CONFIGS[variant]
    if config.tmasac_variant is not None:
        run_tmasac_experiment(
            experiment_run_name=experiment_run_name,
            scenario_name=scenario_name,
            scenario_kwargs=scenario_kwargs,
            variant=config.tmasac_variant,
            entrypoint_path=entrypoint_path,
            continuous_action_dist=config.continuous_action_dist,
            use_nop=config.use_nop,
            include_slstm_memory_strength=config.include_slstm_memory_strength,
            variant_name=variant,
        )
        return

    run_thesis_ppo_experiment(
        experiment_run_name=experiment_run_name,
        scenario_name=scenario_name,
        scenario_kwargs=scenario_kwargs,
        variant=variant,
        entrypoint_path=entrypoint_path,
    )


def run_thesis_ppo_experiment(
    *,
    experiment_run_name: str,
    scenario_name: MJWScenarioName,
    scenario_kwargs: dict[str, object],
    variant: ThesisAlgorithmVariant,
    entrypoint_path: Path,
    num_envs: int = 1024,
    rollout_steps_per_env: int = 4,
    variant_name: str | None = None,
    total_timesteps: int = EXPERIMENT_TOTAL_TIMESTEPS,
    continuous_action_dist: ContinuousActionDistVariant | None = None,
    load_path: str | Path | None = None,
    additional_timesteps: int | None = None,
    evaluation_scenario_kwargs: dict[str, object] | None = None,
    evaluation_milestones: Sequence[float] | None = None,
    evaluation_recording_episodes: int | None = None,
    live_recording_schedule: Mapping[float, int] | None = None,
) -> None:
    config = THESIS_VARIANT_CONFIGS[variant]

    if config.policy_variant is None:
        raise ValueError(f"Thesis PPO variant has no policy variant: {variant!r}")

    continuation_kwargs: dict[str, object] = {}
    if load_path is not None:
        continuation_kwargs["load_path"] = load_path
    if additional_timesteps is not None:
        continuation_kwargs["additional_timesteps"] = additional_timesteps
    if evaluation_scenario_kwargs is not None:
        continuation_kwargs["evaluation_scenario_kwargs"] = evaluation_scenario_kwargs
    if evaluation_milestones is not None:
        continuation_kwargs["evaluation_milestones"] = evaluation_milestones
    if evaluation_recording_episodes is not None:
        continuation_kwargs["evaluation_recording_episodes"] = evaluation_recording_episodes
    if live_recording_schedule is not None:
        continuation_kwargs["live_recording_schedule"] = live_recording_schedule

    run_mjw_experiment(
        num_envs=num_envs,
        rollout_steps_per_env=rollout_steps_per_env,
        variant_name=variant if variant_name is None else variant_name,
        entrypoint_path=entrypoint_path,
        continuous_action_dist=(
            config.continuous_action_dist if continuous_action_dist is None else continuous_action_dist
        ),
        policy_variant=config.policy_variant,
        mat_add_agent_embeddings=False,
        mat_use_agent_attention=variant != "mappo",
        nop_add_agent_embeddings_transition_model=False,
        use_nop=config.use_nop,
        use_transition_obs=False,
        experiment_run_name=experiment_run_name,
        scenario_name=scenario_name,
        scenario_kwargs=scenario_kwargs,
        total_timesteps=total_timesteps,
        **continuation_kwargs,
    )
