from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from gymnasium.vector import AutoresetMode, SyncVectorEnv
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.throughput_benchmarks.throughput_benchmark_paths import (
    default_throughput_benchmark_json_out,
)
from swarmbots.learn.action_dists.predicted_std_action_dist import PredictedStdConfig
from swarmbots.learn.algos.mat.mat_encoder import MATEncoderConfig
from swarmbots.learn.algos.off_policy.replay_buffer import OffPolicyReplayBatch
from swarmbots.learn.algos.sac import (
    SAC,
    TMASACActorHeadConfig,
    TMASACCriticConfig,
    TMASACPolicy,
    TMASACPolicyConfig,
)
from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import (
    SwarmBotsLearnEnvWrapper,
)
from swarmbots.learn.testing_env import TestingSwarmBotsEnv

DEFAULT_JSON_OUT = default_throughput_benchmark_json_out(__file__)


@dataclass(frozen=True)
class BenchmarkConfig:
    device: str
    batch_size: int
    warmup_updates: int
    measured_updates: int
    n_agents: int
    local_obs_dim: int
    global_obs_dim: int
    hidden_local_vars_dim: int
    hidden_global_vars_dim: int
    actuators_dim: int
    connectors_dim: int
    max_agents: int
    d_model: int
    encoder_layers: int
    nhead: int
    compile_mode: str
    seed: int


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    compile_policy_modules: bool
    compile_sac_operations: bool
    compile_optimizer_steps: bool


@dataclass(frozen=True)
class CaseResult:
    name: str
    compile_policy_modules: bool
    compile_sac_operations: bool
    compile_optimizer_steps: bool
    initialization_seconds: float
    first_update_seconds: float
    steady_update_seconds: float
    steady_milliseconds_per_update: float
    steady_updates_per_second: float
    peak_cuda_memory_bytes: int | None


ALL_CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        name="eager",
        compile_policy_modules=False,
        compile_sac_operations=False,
        compile_optimizer_steps=False,
    ),
    BenchmarkCase(
        name="compiled_fragmented",
        compile_policy_modules=True,
        compile_sac_operations=True,
        compile_optimizer_steps=True,
    ),
)


def configure_float32_matmul_precision() -> None:
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_compile_caches() -> None:
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None and hasattr(dynamo, "reset"):
        dynamo.reset()


def seed_everything(seed: int, *, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def make_env(config: BenchmarkConfig) -> SwarmBotsLearnEnvWrapper:
    def make_single_env() -> TestingSwarmBotsEnv:
        return TestingSwarmBotsEnv(
            n_agents=config.n_agents,
            n_local_obs=config.local_obs_dim,
            n_global_obs=config.global_obs_dim,
            actuators_dim=config.actuators_dim,
            connectors_dim=config.connectors_dim,
            n_hidden_local_vars=config.hidden_local_vars_dim,
            n_hidden_global_vars=config.hidden_global_vars_dim,
            continuous_connector_actions=True,
        )

    vector_env = SyncVectorEnv(
        [make_single_env],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    return SwarmBotsLearnEnvWrapper(vector_env)


def build_policy_config(
        config: BenchmarkConfig,
        *,
        compile_modules: bool,
) -> TMASACPolicyConfig:
    encoder_config = MATEncoderConfig(
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.encoder_layers,
        dim_feedforward=config.d_model * 2,
    )
    return TMASACPolicyConfig(
        actor_encoder_config=encoder_config,
        critic_encoder_config=encoder_config,
        actor_head_config=TMASACActorHeadConfig(hidden_dims=[config.d_model]),
        critic_config=TMASACCriticConfig(
            n_local_projection_hidden_layers=1,
            n_value_regressor_hidden_layers=1,
        ),
        continuous_config=PredictedStdConfig(base_std=0.7),
        max_agents=config.max_agents,
        compile_modules=compile_modules,
        compile_mode=config.compile_mode,
    )


def build_policy(
        *,
        env: SwarmBotsLearnEnvWrapper,
        config: BenchmarkConfig,
        compile_modules: bool,
) -> TMASACPolicy:
    return TMASACPolicy(
        env=env,
        config=build_policy_config(config, compile_modules=compile_modules),
    )


def build_batch(config: BenchmarkConfig, *, device: torch.device) -> OffPolicyReplayBatch:
    batch_size = config.batch_size
    obs_shape = (batch_size, config.n_agents, config.local_obs_dim)
    global_obs_shape = (batch_size, config.global_obs_dim)
    hidden_local_shape = (
        batch_size,
        config.n_agents,
        config.hidden_local_vars_dim,
    )
    hidden_global_shape = (batch_size, config.hidden_global_vars_dim)
    action_shape = (
        batch_size,
        config.n_agents,
        config.actuators_dim + config.connectors_dim,
    )

    active_counts = torch.randint(
        1,
        config.n_agents + 1,
        (batch_size,),
        device=device,
    )
    agent_indices = torch.arange(config.n_agents, device=device).unsqueeze(0)
    agent_mask = agent_indices < active_counts.unsqueeze(1)
    next_agent_mask = agent_mask.clone()
    terminations = torch.rand(batch_size, device=device) < 0.05
    next_agent_mask[terminations] = True

    def observations(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, device=device)

    actions = torch.rand(action_shape, device=device) * 2.0 - 1.0
    actions = actions.masked_fill(~agent_mask.unsqueeze(-1), 0.0)
    previous_actions = torch.rand(action_shape, device=device) * 2.0 - 1.0
    previous_actions = previous_actions.masked_fill(~agent_mask.unsqueeze(-1), 0.0)

    return OffPolicyReplayBatch(
        local_obs=observations(obs_shape),
        global_obs=observations(global_obs_shape),
        hidden_local_vars=observations(hidden_local_shape),
        hidden_global_vars=observations(hidden_global_shape),
        agent_mask=agent_mask,
        actions=actions,
        rewards=observations((batch_size,)),
        terminations=terminations,
        truncations=torch.zeros(batch_size, dtype=torch.bool, device=device),
        previous_actions=previous_actions,
        next_local_obs=observations(obs_shape),
        next_global_obs=observations(global_obs_shape),
        next_hidden_local_vars=observations(hidden_local_shape),
        next_hidden_global_vars=observations(hidden_global_shape),
        next_agent_mask=next_agent_mask,
    )


def build_algorithm(
        *,
        env: SwarmBotsLearnEnvWrapper,
        config: BenchmarkConfig,
        case: BenchmarkCase,
) -> SAC:
    policy = build_policy(
        env=env,
        config=config,
        compile_modules=case.compile_policy_modules,
    )
    algorithm = SAC(
        policy=policy,
        env=env,
        learning_rate=3e-4,
        learning_rate_warmup_updates=0,
        buffer_capacity_per_env=config.batch_size,
        learning_starts=1,
        batch_size=config.batch_size,
        ent_coef="auto",
        train_device=config.device,
        rollout_device="cpu",
        replay_storage_device="cpu",
        replay_compile_tensor_operations=False,
        sac_compile_tensor_operations=case.compile_sac_operations,
        sac_compile_optimizer_steps=case.compile_optimizer_steps,
        sac_compile_mode=config.compile_mode,
    )
    return algorithm


def execute_update(
        algorithm: SAC,
        batch: OffPolicyReplayBatch,
        *,
        update_idx: int,
) -> None:
    algorithm._train_step(
        batch,
        global_update_idx=update_idx,
        materialize_metrics=False,
    )


def run_case(
        *,
        env: SwarmBotsLearnEnvWrapper,
        batch: OffPolicyReplayBatch,
        config: BenchmarkConfig,
        case: BenchmarkCase,
) -> CaseResult:
    device = torch.device(config.device)
    reset_compile_caches()
    seed_everything(config.seed, device=device)

    synchronize(device)
    initialization_start = time.perf_counter()
    algorithm = build_algorithm(env=env, config=config, case=case)
    synchronize(device)
    initialization_seconds = time.perf_counter() - initialization_start

    synchronize(device)
    first_update_start = time.perf_counter()
    execute_update(algorithm, batch, update_idx=0)
    synchronize(device)
    first_update_seconds = time.perf_counter() - first_update_start

    for update_idx in range(1, config.warmup_updates + 1):
        execute_update(algorithm, batch, update_idx=update_idx)
    synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    measured_start = time.perf_counter()
    first_measured_update = config.warmup_updates + 1
    for offset in range(config.measured_updates):
        execute_update(
            algorithm,
            batch,
            update_idx=first_measured_update + offset,
        )
    synchronize(device)
    steady_update_seconds = time.perf_counter() - measured_start
    peak_cuda_memory_bytes = (
        torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None
    )

    result = CaseResult(
        name=case.name,
        compile_policy_modules=case.compile_policy_modules,
        compile_sac_operations=case.compile_sac_operations,
        compile_optimizer_steps=case.compile_optimizer_steps,
        initialization_seconds=initialization_seconds,
        first_update_seconds=first_update_seconds,
        steady_update_seconds=steady_update_seconds,
        steady_milliseconds_per_update=(
            steady_update_seconds * 1_000.0 / config.measured_updates
        ),
        steady_updates_per_second=config.measured_updates / steady_update_seconds,
        peak_cuda_memory_bytes=peak_cuda_memory_bytes,
    )

    del algorithm
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def log_speedups(results: list[CaseResult]) -> None:
    results_by_name = {result.name: result for result in results}
    eager = results_by_name.get("eager")
    compiled = results_by_name.get("compiled_fragmented")
    if eager is not None and compiled is not None:
        logger.info(
            "Compiled fragmented vs eager steady update speedup: {:.2f}x",
            compiled.steady_updates_per_second / eager.steady_updates_per_second,
        )


def parse_args() -> tuple[BenchmarkConfig, list[str], Path | None]:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark eager and fragmented torch.compile SAC update steps. "
            "Sampling and metric materialization are excluded."
        )
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default=default_device)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--warmup-updates", type=int, default=5)
    parser.add_argument("--measured-updates", type=int, default=100)
    parser.add_argument("--n-agents", type=int, default=6)
    parser.add_argument("--local-obs-dim", type=int, default=154)
    parser.add_argument("--global-obs-dim", type=int, default=0)
    parser.add_argument("--hidden-local-vars-dim", type=int, default=4)
    parser.add_argument("--hidden-global-vars-dim", type=int, default=3)
    parser.add_argument("--actuators-dim", type=int, default=8)
    parser.add_argument("--connectors-dim", type=int, default=4)
    parser.add_argument("--max-agents", type=int, default=20)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--compile-mode", type=str, default="default")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=[case.name for case in ALL_CASES],
        default=[case.name for case in ALL_CASES],
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")
    if args.warmup_updates < 0:
        raise ValueError("--warmup-updates must be >= 0.")
    if args.measured_updates <= 0:
        raise ValueError("--measured-updates must be > 0.")
    if args.d_model % args.nhead != 0:
        raise ValueError("--d-model must be divisible by --nhead.")
    if args.max_agents < args.n_agents:
        raise ValueError("--max-agents must be >= --n-agents.")

    config = BenchmarkConfig(
        device=str(args.device),
        batch_size=int(args.batch_size),
        warmup_updates=int(args.warmup_updates),
        measured_updates=int(args.measured_updates),
        n_agents=int(args.n_agents),
        local_obs_dim=int(args.local_obs_dim),
        global_obs_dim=int(args.global_obs_dim),
        hidden_local_vars_dim=int(args.hidden_local_vars_dim),
        hidden_global_vars_dim=int(args.hidden_global_vars_dim),
        actuators_dim=int(args.actuators_dim),
        connectors_dim=int(args.connectors_dim),
        max_agents=int(args.max_agents),
        d_model=int(args.d_model),
        encoder_layers=int(args.encoder_layers),
        nhead=int(args.nhead),
        compile_mode=str(args.compile_mode),
        seed=int(args.seed),
    )
    return config, list(args.cases), args.json_out


def main() -> None:
    from swarmbots.learn.torch_logging import enable_torch_compile_logging

    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <5}</level> | <level>{message}</level>"
        ),
    )
    enable_torch_compile_logging()
    configure_float32_matmul_precision()

    config, selected_case_names, json_out = parse_args()
    device = torch.device(config.device)
    logger.info("Running SAC update compile benchmark with config: {}", config)

    env = make_env(config)
    seed_everything(config.seed, device=device)
    batch = build_batch(config, device=device)
    cases = [case for case in ALL_CASES if case.name in selected_case_names]
    results = [
        run_case(
            env=env,
            batch=batch,
            config=config,
            case=case,
        )
        for case in cases
    ]
    env.close()

    for result in results:
        logger.info("{}", result)
    log_speedups(results)
    payload = {
        "config": asdict(config),
        "torch_version": torch.__version__,
        "cuda_device": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "scope": (
            "SAC._train_step with a fixed synthetic batch; replay sampling and final "
            "metric materialization are excluded, and all cases use current deferred metrics."
        ),
        "first_update_timing_note": (
            "Includes lazy compilation but may reuse TorchInductor's persistent disk cache; "
            "use separate clean-cache runs to compare cold compile latency."
        ),
        "results": [asdict(result) for result in results],
    }
    serialized_payload = json.dumps(payload, indent=2)
    print(serialized_payload)

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(serialized_payload, encoding="utf-8")
        logger.info("Wrote benchmark JSON to {}", json_out)


if __name__ == "__main__":
    main()
