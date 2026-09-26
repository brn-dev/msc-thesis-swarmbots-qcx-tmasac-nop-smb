from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from experiments.external_benchmarks.benchmark_experiment_common import (
    BenchmarkPolicyVariant,
    BenchmarkSuite,
    run_benchmark_experiment,
)

ExperimentVariant = Literal[
    "mat_qcx",
    "mat_qcx_nop",
    "mat_dec",
    "mat_dec_nop",
    "mat_ind",
    "mat_ind_nop",
    "tmasac",
    "tmasac_nop",
]


@dataclass(frozen=True)
class ScenarioExperimentConfig:
    suite: BenchmarkSuite
    scenario: str
    num_envs: int
    episode_length: int
    total_timesteps: int
    ppo_rollout_steps_per_env: int
    sac_rollout_steps_per_env: int
    mamujoco_agent_conf: str = "2x4"
    mamujoco_agent_obsk: int = 1
    mamujoco_num_workers: int | None = None


MAMUJOCO_DEFAULTS = {
    "num_envs": 16,
    "episode_length": 1_000,
    "total_timesteps": 10_000_000,
    "ppo_rollout_steps_per_env": 256,
    "sac_rollout_steps_per_env": 64,
    "mamujoco_num_workers": 8,
}
VMAS_DEFAULTS = {
    "num_envs": 1_024,
    "episode_length": 200,
    "total_timesteps": 100_000_000,
    "ppo_rollout_steps_per_env": 4,
    "sac_rollout_steps_per_env": 1,
}

EXPERIMENT_CONFIGS: dict[str, ScenarioExperimentConfig] = {
    "mamujoco_halfcheetah": ScenarioExperimentConfig(
        suite="mamujoco",
        scenario="HalfCheetah",
        mamujoco_agent_conf="2x3",
        **MAMUJOCO_DEFAULTS,
    ),
    "mamujoco_ant": ScenarioExperimentConfig(
        suite="mamujoco",
        scenario="Ant",
        mamujoco_agent_conf="2x4",
        **MAMUJOCO_DEFAULTS,
    ),
    "mamujoco_swimmer": ScenarioExperimentConfig(
        suite="mamujoco",
        scenario="Swimmer",
        mamujoco_agent_conf="2x1",
        **MAMUJOCO_DEFAULTS,
    ),
    "mamujoco_walker2d": ScenarioExperimentConfig(
        suite="mamujoco",
        scenario="Walker2d",
        mamujoco_agent_conf="2x3",
        **MAMUJOCO_DEFAULTS,
    ),
    "mamujoco_many_segment_swimmer": ScenarioExperimentConfig(
        suite="mamujoco",
        scenario="ManySegmentSwimmer",
        mamujoco_agent_conf="2x2",
        **MAMUJOCO_DEFAULTS,
    ),
    "vmas_balance": ScenarioExperimentConfig(
        suite="vmas", scenario="balance", **VMAS_DEFAULTS
    ),
    "vmas_navigation": ScenarioExperimentConfig(
        suite="vmas", scenario="navigation", **VMAS_DEFAULTS
    ),
    "vmas_transport": ScenarioExperimentConfig(
        suite="vmas", scenario="transport", **VMAS_DEFAULTS
    ),
    "vmas_sampling": ScenarioExperimentConfig(
        suite="vmas", scenario="sampling", **VMAS_DEFAULTS
    ),
    "vmas_discovery": ScenarioExperimentConfig(
        suite="vmas", scenario="discovery", **VMAS_DEFAULTS
    ),
    "vmas_flocking": ScenarioExperimentConfig(
        suite="vmas", scenario="flocking", **VMAS_DEFAULTS
    ),
    "vmas_give_way": ScenarioExperimentConfig(
        suite="vmas", scenario="give_way", **VMAS_DEFAULTS
    ),
    "vmas_football": ScenarioExperimentConfig(
        suite="vmas", scenario="football", **VMAS_DEFAULTS
    ),
}

VARIANT_CONFIGS: dict[ExperimentVariant, tuple[BenchmarkPolicyVariant, bool]] = {
    "mat_qcx": ("mat_qcx", False),
    "mat_qcx_nop": ("mat_qcx", True),
    "mat_dec": ("mat_dec", False),
    "mat_dec_nop": ("mat_dec", True),
    "mat_ind": ("mat_ind", False),
    "mat_ind_nop": ("mat_ind", True),
    "tmasac": ("tmasac", False),
    "tmasac_nop": ("tmasac", True),
}


def run_registered_experiment(
    *,
    experiment_name: str,
    entrypoint_path: Path,
) -> None:
    try:
        config = EXPERIMENT_CONFIGS[experiment_name]
    except KeyError as error:
        raise ValueError(
            f"Unknown external benchmark experiment: {experiment_name}"
        ) from error

    args = _parse_args(config)
    policy_variant, use_nop = VARIANT_CONFIGS[args.variant]
    rollout_steps_per_env = (
        args.rollout_steps_per_env
        if args.rollout_steps_per_env is not None
        else (
            config.sac_rollout_steps_per_env
            if policy_variant == "tmasac"
            else config.ppo_rollout_steps_per_env
        )
    )
    run_benchmark_experiment(
        suite=config.suite,
        policy_variant=policy_variant,
        use_nop=use_nop,
        variant_name=args.variant,
        entrypoint_path=entrypoint_path,
        experiment_definition_path=Path(__file__),
        scenario=config.scenario,
        num_envs=args.num_envs,
        rollout_steps_per_env=rollout_steps_per_env,
        episode_length=args.episode_length,
        total_timesteps=args.total_timesteps,
        experiment_run_name=f"{experiment_name}_mat_tmasac",
        mamujoco_agent_conf=config.mamujoco_agent_conf,
        mamujoco_agent_obsk=args.agent_obsk,
        mamujoco_num_workers=args.num_workers,
    )


def _parse_args(config: ScenarioExperimentConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "variant",
        nargs="?",
        choices=tuple(VARIANT_CONFIGS),
        default="tmasac",
    )
    parser.add_argument("--num-envs", type=int, default=config.num_envs)
    parser.add_argument("--rollout-steps-per-env", type=int, default=None)
    parser.add_argument("--episode-length", type=int, default=config.episode_length)
    parser.add_argument("--total-timesteps", type=int, default=config.total_timesteps)
    parser.add_argument("--cuda_idx", "--cuda-idx", type=int, default=None)
    if config.suite == "mamujoco":
        parser.add_argument(
            "--num-workers", type=int, default=config.mamujoco_num_workers
        )
        parser.add_argument(
            "--agent-obsk", type=int, default=config.mamujoco_agent_obsk
        )
    else:
        parser.set_defaults(num_workers=None, agent_obsk=config.mamujoco_agent_obsk)
    return parser.parse_args()
