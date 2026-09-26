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
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.throughput_benchmarks.throughput_benchmark_paths import default_throughput_benchmark_json_out

DEFAULT_JSON_OUT = default_throughput_benchmark_json_out(__file__)

from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliConfig
from swarmbots.learn.action_dists.entropy_utils import AgentActionsReduction, EntropyLossConfig
from swarmbots.learn.action_dists.sign_magnitude_beta_action_dist import SignMagnitudeBetaConfig
from swarmbots.learn.algos.mat_qcs.mat_qcs_decoder import MATQCSDecoderConfig, MATQCSDecoderSelfAttentionMode
from swarmbots.learn.algos.mat import MATEncoderConfig, MLPConfig
from swarmbots.learn.algos.mat_qcs.mat_qcs_policy import MATQCSCriticConfig, MATQCSPolicy, MATQCSPolicyConfig
from swarmbots.learn.algos.ppo.ppo_policy import PopArtConfig
from swarmbots.learn.algos.world_modeling.next_obs_pred_mixin import NextObsPredConfig
from swarmbots.learn.algos.world_modeling.next_obs_pred_ppo_wrapper import NOPWorldModelConfig, NextObsPredWrapper
from swarmbots.learn.algos.world_modeling.ppo_wm_sampler import PPOWMSamples
from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from swarmbots.learn.testing_env import TestingSwarmBotsEnv


def configure_float32_matmul_precision() -> None:
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")


@dataclass(frozen=True)
class BenchmarkConfig:
    device: str
    env_num_envs: int
    batch_size: int
    num_next_steps: int
    warmup_iters: int
    measured_iters: int
    n_agents: int
    local_obs_dim: int
    global_obs_dim: int
    hidden_local_vars_dim: int
    hidden_global_vars_dim: int
    actuators_dim: int
    connectors_dim: int
    max_agents: int
    encoder_d_model: int
    decoder_d_model: int
    transition_d_model: int
    encoder_layers: int
    decoder_layers: int
    transition_layers: int
    encoder_nhead: int
    decoder_nhead: int
    transition_nhead: int
    scalar_target_count: int
    angle_target_count: int
    rot6d_target_count: int
    binary_target_count: int
    compile_mode: str
    seed: int


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    policy_compile_modules: bool
    world_model_compile_modules: bool


@dataclass(frozen=True)
class NOPCoreInputs:
    local_latents: torch.Tensor
    next_local_obs: torch.Tensor
    actions: torch.Tensor
    local_obs: torch.Tensor
    agent_mask: torch.Tensor
    loss_agent_mask: torch.Tensor
    time_mask: torch.Tensor


@dataclass(frozen=True)
class CaseResult:
    name: str
    policy_compile_modules: bool
    world_model_compile_modules: bool
    compiled_policy_eval_core: bool
    compiled_world_model_core: bool
    init_seconds: float
    first_nop_core_seconds: float
    steady_nop_core_seconds: float
    nop_core_calls_per_second: float
    first_evaluate_seconds: float
    steady_evaluate_seconds: float
    evaluate_calls_per_second: float


ALL_CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        name="eager_policy_eager_nop",
        policy_compile_modules=False,
        world_model_compile_modules=False,
    ),
    BenchmarkCase(
        name="compiled_policy_eager_nop",
        policy_compile_modules=True,
        world_model_compile_modules=False,
    ),
    BenchmarkCase(
        name="eager_policy_compiled_nop",
        policy_compile_modules=False,
        world_model_compile_modules=True,
    ),
    BenchmarkCase(
        name="compiled_policy_compiled_nop",
        policy_compile_modules=True,
        world_model_compile_modules=True,
    ),
)


def make_env(config: BenchmarkConfig) -> SwarmBotsLearnEnvWrapper:
    def _make_single_env() -> TestingSwarmBotsEnv:
        return TestingSwarmBotsEnv(
            n_agents=config.n_agents,
            n_local_obs=config.local_obs_dim,
            n_global_obs=config.global_obs_dim,
            actuators_dim=config.actuators_dim,
            connectors_dim=config.connectors_dim,
            n_hidden_local_vars=config.hidden_local_vars_dim,
            n_hidden_global_vars=config.hidden_global_vars_dim,
        )

    vector_env = SyncVectorEnv(
        [_make_single_env for _ in range(config.env_num_envs)],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    return SwarmBotsLearnEnvWrapper(vector_env, device=torch.device(config.device))


def build_policy_config(config: BenchmarkConfig, *, compile_modules: bool) -> MATQCSPolicyConfig:
    return MATQCSPolicyConfig(
        encoder_config=MATEncoderConfig(
            d_model=config.encoder_d_model,
            nhead=config.encoder_nhead,
            num_layers=config.encoder_layers,
            dim_feedforward=config.encoder_d_model * 2,
            local_obs_encoder_config=MLPConfig(
                hidden_dims=[config.encoder_d_model, config.encoder_d_model]
            ),
            global_obs_encoder_config=MLPConfig(hidden_dims=[config.encoder_d_model]),
        ),
        decoder_config=MATQCSDecoderConfig(
            d_model=config.decoder_d_model,
            nhead=config.decoder_nhead,
            num_layers=config.decoder_layers,
            dim_feedforward=config.decoder_d_model * 2,
            query_encoder_hidden_dims=[config.decoder_d_model * 2],
            context_encoder_hidden_dims=[config.decoder_d_model * 2],
            memory_dims=None,
            self_attention_mode=MATQCSDecoderSelfAttentionMode.FULL_CAUSAL,
        ),
        critic_config=MATQCSCriticConfig(
            n_local_projection_hidden_layers=2,
            n_value_regressor_hidden_layers=1,
            use_popart=True,
            popart_config=PopArtConfig(
                beta=5e-4,
                init_sigma=0.65,
            ),
        ),
        dropout=0.0,
        act_fn_cls=nn.GELU,
        continuous_config=SignMagnitudeBetaConfig(
            ent_loss_coef=1e-3,
            beta_ent_scale=0.75,
            categorical_ent_loss_config=EntropyLossConfig(
                agent_actions_reduction=AgentActionsReduction.SUM,
                metrics_reduction=AgentActionsReduction.MEAN,
            ),
            beta_ent_loss_config=EntropyLossConfig(
                agent_actions_reduction=AgentActionsReduction.SUM,
                metrics_reduction=AgentActionsReduction.MEAN,
            ),
        ),
        bernoulli_config=BernoulliConfig(
            initial_prob=0.7,
            ent_loss_coef=1e-3,
        ),
        max_agents=config.max_agents,
        compile_modules=compile_modules,
        compile_mode=config.compile_mode,
    )


def build_next_obs_pred_config(config: BenchmarkConfig) -> NextObsPredConfig:
    target_width = (
        config.scalar_target_count
        + config.angle_target_count * 2
        + config.rot6d_target_count * 6
        + config.binary_target_count
    )
    if target_width > config.local_obs_dim:
        raise ValueError(
            "Requested NOP targets do not fit into local_obs_dim: "
            f"{target_width} > {config.local_obs_dim}"
        )

    cursor = 0
    scalar_indices = list(range(cursor, cursor + config.scalar_target_count))
    cursor += config.scalar_target_count

    angle_indices = list(range(cursor, cursor + config.angle_target_count * 2, 2))
    cursor += config.angle_target_count * 2

    rot6d_indices = list(range(cursor, cursor + config.rot6d_target_count * 6, 6))
    cursor += config.rot6d_target_count * 6

    binary_indices = list(range(cursor, cursor + config.binary_target_count))

    return NextObsPredConfig(
        local_scalar_target_indices=scalar_indices or None,
        local_angle_target_indices=angle_indices or None,
        local_rot6d_target_indices=rot6d_indices or None,
        local_binary_target_indices=binary_indices or None,
        scalar_loss_weight=1.0,
        angle_loss_weight=1.0,
        rot6d_loss_weight=1.0,
        binary_loss_weight=1.0,
    )


def build_world_model_config(
        config: BenchmarkConfig,
        *,
        action_dim: int,
        compile_modules: bool,
) -> NOPWorldModelConfig:
    return NOPWorldModelConfig(
        n_agents=config.n_agents,
        local_latent_dim=config.encoder_d_model,
        action_dim=action_dim,
        world_model_loss_coef=0.1,
        compile_modules=compile_modules,
        compile_mode=config.compile_mode,
        act_fn_cls=nn.GELU,
        transition_model_dropout=0.0,
        d_model_transition_model=config.transition_d_model,
        nhead_transition_model=config.transition_nhead,
        num_layers_transition_model=config.transition_layers,
        dim_feedforward_transition_model=config.transition_d_model * 2,
        add_agent_embeddings_transition_model=False,
        transition_model_coembed_hidden_dims=[config.transition_d_model],
        transition_model_head_hidden_dims=None,
        transition_model_predict_delta=True,
        wm_pre_transition_dims=[config.encoder_d_model],
        wm_pre_predictors_dims=[config.transition_d_model, config.transition_d_model],
        wm_scalar_predictor_hidden_dims=[],
        wm_angle_predictor_hidden_dims=[],
        wm_rot6d_predictor_hidden_dims=[],
        wm_binary_predictor_hidden_dims=[],
        scalar_loss_fn="smooth_l1",
        next_obs_pred_config=build_next_obs_pred_config(config),
    )


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


def build_agent_mask(batch_size: int, *, n_agents: int, device: torch.device) -> torch.Tensor:
    active_counts = torch.randint(1, n_agents + 1, (batch_size,), device=device)
    agent_indices = torch.arange(n_agents, device=device).unsqueeze(0)
    return agent_indices < active_counts.unsqueeze(1)


def build_hybrid_actions(
        batch_size: int,
        *,
        n_agents: int,
        actuators_dim: int,
        connectors_dim: int,
        device: torch.device,
) -> torch.Tensor:
    actuators = torch.rand((batch_size, n_agents, actuators_dim), device=device) * 2.0 - 1.0
    connectors = (
        torch.rand((batch_size, n_agents, connectors_dim), device=device) < 0.5
    ).to(dtype=actuators.dtype)
    return torch.cat((actuators, connectors), dim=-1)


def build_wm_batch(config: BenchmarkConfig, *, device: torch.device) -> PPOWMSamples:
    batch_size = config.batch_size
    agent_mask = build_agent_mask(batch_size, n_agents=config.n_agents, device=device)
    wm_agent_mask = agent_mask.unsqueeze(1).expand(-1, config.num_next_steps, -1).clone()
    time_mask = torch.ones((batch_size, config.num_next_steps), dtype=torch.bool, device=device)

    return PPOWMSamples(
        local_obs=torch.randn((batch_size, config.n_agents, config.local_obs_dim), device=device),
        global_obs=torch.randn((batch_size, config.global_obs_dim), device=device),
        hidden_local_vars=torch.randn((batch_size, config.n_agents, config.hidden_local_vars_dim), device=device),
        hidden_global_vars=torch.randn((batch_size, config.hidden_global_vars_dim), device=device),
        agent_mask=agent_mask,
        previous_actions=build_hybrid_actions(
            batch_size,
            n_agents=config.n_agents,
            actuators_dim=config.actuators_dim,
            connectors_dim=config.connectors_dim,
            device=device,
        ),
        actions=build_hybrid_actions(
            batch_size,
            n_agents=config.n_agents,
            actuators_dim=config.actuators_dim,
            connectors_dim=config.connectors_dim,
            device=device,
        ),
        wm_actions=build_hybrid_actions(
            batch_size * config.num_next_steps,
            n_agents=config.n_agents,
            actuators_dim=config.actuators_dim,
            connectors_dim=config.connectors_dim,
            device=device,
        ).reshape(batch_size, config.num_next_steps, config.n_agents, -1),
        log_probs=torch.zeros(batch_size, device=device),
        values=torch.zeros(batch_size, device=device),
        returns=torch.zeros(batch_size, device=device),
        advantages=torch.zeros(batch_size, device=device),
        next_local_obs=torch.randn(
            (batch_size, config.num_next_steps, config.n_agents, config.local_obs_dim),
            device=device,
        ),
        wm_target_time_mask=time_mask,
        next_global_obs=torch.randn((batch_size, config.num_next_steps, config.global_obs_dim), device=device),
        wm_agent_mask=wm_agent_mask,
        wm_loss_agent_mask=wm_agent_mask.clone(),
    )


def build_nop_core_inputs(
        config: BenchmarkConfig,
        *,
        batch: PPOWMSamples,
        device: torch.device,
) -> NOPCoreInputs:
    return NOPCoreInputs(
        local_latents=torch.randn((config.batch_size, config.n_agents, config.encoder_d_model), device=device),
        next_local_obs=batch.next_local_obs,
        actions=batch.wm_actions,
        local_obs=batch.local_obs,
        agent_mask=batch.wm_agent_mask,
        loss_agent_mask=batch.wm_loss_agent_mask,
        time_mask=batch.wm_target_time_mask,
    )


def time_nop_core(
        *,
        policy: NextObsPredWrapper,
        inputs: NOPCoreInputs,
        warmup_iters: int,
        measured_iters: int,
        device: torch.device,
) -> tuple[float, float, float]:
    policy.train()
    synchronize(device)
    start = time.perf_counter()
    policy._compute_next_obs_pred_loss_fn(
        local_latents=inputs.local_latents,
        next_local_obs=inputs.next_local_obs,
        actions=inputs.actions,
        local_obs=inputs.local_obs,
        agent_mask=inputs.agent_mask,
        loss_agent_mask=inputs.loss_agent_mask,
        time_mask=inputs.time_mask,
    )
    synchronize(device)
    first_call_seconds = time.perf_counter() - start

    for _ in range(warmup_iters):
        policy._compute_next_obs_pred_loss_fn(
            local_latents=inputs.local_latents,
            next_local_obs=inputs.next_local_obs,
            actions=inputs.actions,
            local_obs=inputs.local_obs,
            agent_mask=inputs.agent_mask,
            loss_agent_mask=inputs.loss_agent_mask,
            time_mask=inputs.time_mask,
        )

    synchronize(device)
    start = time.perf_counter()
    for _ in range(measured_iters):
        policy._compute_next_obs_pred_loss_fn(
            local_latents=inputs.local_latents,
            next_local_obs=inputs.next_local_obs,
            actions=inputs.actions,
            local_obs=inputs.local_obs,
            agent_mask=inputs.agent_mask,
            loss_agent_mask=inputs.loss_agent_mask,
            time_mask=inputs.time_mask,
        )
    synchronize(device)
    steady_seconds = time.perf_counter() - start

    calls_per_second = measured_iters / steady_seconds
    return first_call_seconds, steady_seconds, calls_per_second


def time_evaluate_actions(
        *,
        policy: NextObsPredWrapper,
        batch: PPOWMSamples,
        warmup_iters: int,
        measured_iters: int,
        device: torch.device,
) -> tuple[float, float, float]:
    policy.train()
    synchronize(device)
    start = time.perf_counter()
    policy.evaluate_actions(batch=batch)
    synchronize(device)
    first_call_seconds = time.perf_counter() - start

    for _ in range(warmup_iters):
        policy.evaluate_actions(batch=batch)

    synchronize(device)
    start = time.perf_counter()
    for _ in range(measured_iters):
        policy.evaluate_actions(batch=batch)
    synchronize(device)
    steady_seconds = time.perf_counter() - start

    calls_per_second = measured_iters / steady_seconds
    return first_call_seconds, steady_seconds, calls_per_second


def run_case(
        *,
        env: SwarmBotsLearnEnvWrapper,
        batch: PPOWMSamples,
        nop_inputs: NOPCoreInputs,
        config: BenchmarkConfig,
        case: BenchmarkCase,
) -> CaseResult:
    device = torch.device(config.device)
    reset_compile_caches()
    seed_everything(config.seed, device=device)

    start = time.perf_counter()
    mat_qcs_policy = MATQCSPolicy(
        env=env,
        config=build_policy_config(config, compile_modules=case.policy_compile_modules),
    )
    policy = NextObsPredWrapper(
        policy=mat_qcs_policy,
        world_model_config=build_world_model_config(
            config,
            action_dim=env.action_space.total_agent_action_dim,
            compile_modules=case.world_model_compile_modules,
        ),
    )
    policy.to(device)
    synchronize(device)
    init_seconds = time.perf_counter() - start

    first_nop_core_seconds, steady_nop_core_seconds, nop_core_calls_per_second = time_nop_core(
        policy=policy,
        inputs=nop_inputs,
        warmup_iters=config.warmup_iters,
        measured_iters=config.measured_iters,
        device=device,
    )
    first_evaluate_seconds, steady_evaluate_seconds, evaluate_calls_per_second = time_evaluate_actions(
        policy=policy,
        batch=batch,
        warmup_iters=config.warmup_iters,
        measured_iters=config.measured_iters,
        device=device,
    )

    result = CaseResult(
        name=case.name,
        policy_compile_modules=case.policy_compile_modules,
        world_model_compile_modules=case.world_model_compile_modules,
        compiled_policy_eval_core=case.policy_compile_modules and policy.action_dist.compile_friendly,
        compiled_world_model_core=case.world_model_compile_modules,
        init_seconds=init_seconds,
        first_nop_core_seconds=first_nop_core_seconds,
        steady_nop_core_seconds=steady_nop_core_seconds,
        nop_core_calls_per_second=nop_core_calls_per_second,
        first_evaluate_seconds=first_evaluate_seconds,
        steady_evaluate_seconds=steady_evaluate_seconds,
        evaluate_calls_per_second=evaluate_calls_per_second,
    )

    del policy
    del mat_qcs_policy
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def log_speedups(results: list[CaseResult]) -> None:
    results_by_name = {result.name: result for result in results}
    pairs = (
        ("eager_policy_eager_nop", "eager_policy_compiled_nop", "policy_compile=False"),
        ("compiled_policy_eager_nop", "compiled_policy_compiled_nop", "policy_compile=True"),
    )
    for eager_name, compiled_name, label in pairs:
        eager_result = results_by_name.get(eager_name)
        compiled_result = results_by_name.get(compiled_name)
        if eager_result is None or compiled_result is None:
            continue
        logger.info(
            "{} NOP core speedup: {:.2f}x | full evaluate_actions speedup: {:.2f}x",
            label,
            compiled_result.nop_core_calls_per_second / eager_result.nop_core_calls_per_second,
            compiled_result.evaluate_calls_per_second / eager_result.evaluate_calls_per_second,
        )


def parse_args() -> tuple[BenchmarkConfig, list[str]]:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark NextObsPredWrapper torch.compile speedups for the world-model path, "
            "including the NOP loss core and full evaluate_actions()."
        )
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default=default_device)
    parser.add_argument(
        "--env-num-envs",
        type=int,
        default=8,
        help="Vector-env size used only to build the env wrapper.",
    )
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for evaluate_actions().")
    parser.add_argument("--num-next-steps", type=int, default=3)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--measured-iters", type=int, default=1000)
    parser.add_argument("--n-agents", type=int, default=6)
    parser.add_argument("--local-obs-dim", type=int, default=154)
    parser.add_argument("--global-obs-dim", type=int, default=0)
    parser.add_argument("--hidden-local-vars-dim", type=int, default=4)
    parser.add_argument("--hidden-global-vars-dim", type=int, default=3)
    parser.add_argument("--actuators-dim", type=int, default=8)
    parser.add_argument("--connectors-dim", type=int, default=4)
    parser.add_argument("--max-agents", type=int, default=20)
    parser.add_argument("--encoder-d-model", type=int, default=256)
    parser.add_argument("--decoder-d-model", type=int, default=96)
    parser.add_argument("--transition-d-model", type=int, default=192)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--transition-layers", type=int, default=2)
    parser.add_argument("--encoder-nhead", type=int, default=4)
    parser.add_argument("--decoder-nhead", type=int, default=2)
    parser.add_argument("--transition-nhead", type=int, default=4)
    parser.add_argument("--scalar-target-count", type=int, default=64)
    parser.add_argument("--angle-target-count", type=int, default=8)
    parser.add_argument("--rot6d-target-count", type=int, default=8)
    parser.add_argument("--binary-target-count", type=int, default=8)
    parser.add_argument("--compile-mode", type=str, default="default")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=[case.name for case in ALL_CASES],
        default=[case.name for case in ALL_CASES],
        help="Benchmark case subset to run.",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if args.measured_iters <= 0:
        raise ValueError("--measured-iters must be > 0.")
    if args.num_next_steps <= 0:
        raise ValueError("--num-next-steps must be > 0.")

    return BenchmarkConfig(
        device=str(args.device),
        env_num_envs=int(args.env_num_envs),
        batch_size=int(args.batch_size),
        num_next_steps=int(args.num_next_steps),
        warmup_iters=int(args.warmup_iters),
        measured_iters=int(args.measured_iters),
        n_agents=int(args.n_agents),
        local_obs_dim=int(args.local_obs_dim),
        global_obs_dim=int(args.global_obs_dim),
        hidden_local_vars_dim=int(args.hidden_local_vars_dim),
        hidden_global_vars_dim=int(args.hidden_global_vars_dim),
        actuators_dim=int(args.actuators_dim),
        connectors_dim=int(args.connectors_dim),
        max_agents=int(args.max_agents),
        encoder_d_model=int(args.encoder_d_model),
        decoder_d_model=int(args.decoder_d_model),
        transition_d_model=int(args.transition_d_model),
        encoder_layers=int(args.encoder_layers),
        decoder_layers=int(args.decoder_layers),
        transition_layers=int(args.transition_layers),
        encoder_nhead=int(args.encoder_nhead),
        decoder_nhead=int(args.decoder_nhead),
        transition_nhead=int(args.transition_nhead),
        scalar_target_count=int(args.scalar_target_count),
        angle_target_count=int(args.angle_target_count),
        rot6d_target_count=int(args.rot6d_target_count),
        binary_target_count=int(args.binary_target_count),
        compile_mode=str(args.compile_mode),
        seed=int(args.seed),
    ), list(args.cases), args.json_out


def main() -> None:
    from swarmbots.learn.torch_logging import enable_torch_compile_logging

    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <5}</level> | <level>{message}</level>",
    )
    enable_torch_compile_logging()
    configure_float32_matmul_precision()

    config, selected_case_names, json_out = parse_args()
    device = torch.device(config.device)
    logger.info("Running NOP compile benchmark with config: {}", config)

    env = make_env(config)
    seed_everything(config.seed, device=device)
    batch = build_wm_batch(config, device=device)
    nop_inputs = build_nop_core_inputs(config, batch=batch, device=device)
    cases = [case for case in ALL_CASES if case.name in selected_case_names]

    results = [
        run_case(
            env=env,
            batch=batch,
            nop_inputs=nop_inputs,
            config=config,
            case=case,
        )
        for case in cases
    ]
    env.close()

    for result in results:
        logger.info("{}", result)
    log_speedups(results)
    payload = [asdict(result) for result in results]
    print(json.dumps(payload, indent=2))

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Wrote benchmark JSON to {}", json_out)


if __name__ == "__main__":
    main()

