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
from swarmbots.learn.action_dists.hybrid_action_dist import HybridActionDistribution
from swarmbots.learn.action_dists.sign_magnitude_beta_action_dist import SignMagnitudeBetaConfig
from swarmbots.learn.algos.mat_qcs.mat_qcs_decoder import MATQCSDecoderConfig, MATQCSDecoderSelfAttentionMode
from swarmbots.learn.algos.mat import MATEncoderConfig, MLPConfig
from swarmbots.learn.algos.mat_qcs.mat_qcs_policy import MATQCSCriticConfig, MATQCSPolicy, MATQCSPolicyConfig
from swarmbots.learn.algos.ppo.ppo_policy import PopArtConfig
from swarmbots.learn.algos.ppo.ppo_sampler import PPOSamples
from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from swarmbots.learn.testing_env import TestingSwarmBotsEnv


def configure_float32_matmul_precision() -> None:
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")


@dataclass(frozen=True)
class BenchmarkConfig:
    device: str
    env_num_envs: int
    rollout_batch_size: int
    train_batch_size: int
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
    encoder_layers: int
    decoder_layers: int
    encoder_nhead: int
    decoder_nhead: int
    compile_mode: str
    seed: int


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    compile_modules: bool
    expose_compile_friendly: bool


@dataclass(frozen=True)
class CaseResult:
    name: str
    compile_modules: bool
    expose_compile_friendly: bool
    actual_compile_friendly: bool
    compiled_rollout_helper: bool
    compiled_train_eval_core: bool
    init_seconds: float
    first_forward_seconds: float
    steady_forward_seconds: float
    forward_calls_per_second: float
    first_evaluate_seconds: float
    steady_evaluate_seconds: float
    evaluate_calls_per_second: float


@dataclass(frozen=True)
class RolloutInputs:
    local_obs: torch.Tensor
    global_obs: torch.Tensor
    hidden_local_vars: torch.Tensor
    hidden_global_vars: torch.Tensor
    agent_mask: torch.Tensor
    previous_actions: torch.Tensor


class NonCompileFriendlyHybridActionDistribution(HybridActionDistribution):
    @property
    def compile_friendly(self) -> bool:
        return False


class BenchmarkMATQCSPolicy(MATQCSPolicy):
    def __init__(
            self,
            *,
            env: SwarmBotsLearnEnvWrapper,
            config: MATQCSPolicyConfig,
            expose_compile_friendly: bool,
    ) -> None:
        self._expose_compile_friendly = bool(expose_compile_friendly)
        super().__init__(env=env, config=config)

    def _build_action_dist(
            self,
            *,
            env: SwarmBotsLearnEnvWrapper,
            latent_pi_dim: int,
    ) -> HybridActionDistribution:
        action_dist_cls = (
            HybridActionDistribution
            if self._expose_compile_friendly
            else NonCompileFriendlyHybridActionDistribution
        )
        return action_dist_cls(
            latent_dim=latent_pi_dim,
            action_space=env.action_space,
            continuous_config=self.config.continuous_config,
            bernoulli_config=self.config.bernoulli_config,
        )


ALL_CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        name="eager_compile_friendly",
        compile_modules=False,
        expose_compile_friendly=True,
    ),
    BenchmarkCase(
        name="compiled_compile_friendly",
        compile_modules=True,
        expose_compile_friendly=True,
    ),
    BenchmarkCase(
        name="eager_forced_non_compile_friendly",
        compile_modules=False,
        expose_compile_friendly=False,
    ),
    BenchmarkCase(
        name="compiled_forced_non_compile_friendly",
        compile_modules=True,
        expose_compile_friendly=False,
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


def build_rollout_inputs(config: BenchmarkConfig, *, device: torch.device) -> RolloutInputs:
    return RolloutInputs(
        local_obs=torch.randn(
            (config.rollout_batch_size, config.n_agents, config.local_obs_dim),
            device=device,
        ),
        global_obs=torch.randn(
            (config.rollout_batch_size, config.global_obs_dim),
            device=device,
        ),
        hidden_local_vars=torch.randn(
            (config.rollout_batch_size, config.n_agents, config.hidden_local_vars_dim),
            device=device,
        ),
        hidden_global_vars=torch.randn(
            (config.rollout_batch_size, config.hidden_global_vars_dim),
            device=device,
        ),
        agent_mask=build_agent_mask(config.rollout_batch_size, n_agents=config.n_agents, device=device),
        previous_actions=build_hybrid_actions(
            config.rollout_batch_size,
            n_agents=config.n_agents,
            actuators_dim=config.actuators_dim,
            connectors_dim=config.connectors_dim,
            device=device,
        ),
    )


def build_train_samples(config: BenchmarkConfig, *, device: torch.device) -> PPOSamples:
    batch_size = config.train_batch_size
    return PPOSamples(
        local_obs=torch.randn((batch_size, config.n_agents, config.local_obs_dim), device=device),
        global_obs=torch.randn((batch_size, config.global_obs_dim), device=device),
        hidden_local_vars=torch.randn((batch_size, config.n_agents, config.hidden_local_vars_dim), device=device),
        hidden_global_vars=torch.randn((batch_size, config.hidden_global_vars_dim), device=device),
        agent_mask=build_agent_mask(batch_size, n_agents=config.n_agents, device=device),
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
        log_probs=torch.zeros(batch_size, device=device),
        values=torch.zeros(batch_size, device=device),
        returns=torch.zeros(batch_size, device=device),
        advantages=torch.zeros(batch_size, device=device),
    )


def time_rollout_forward(
        *,
        policy: BenchmarkMATQCSPolicy,
        inputs: RolloutInputs,
        warmup_iters: int,
        measured_iters: int,
        device: torch.device,
) -> tuple[float, float, float]:
    policy.eval()
    with torch.no_grad():
        synchronize(device)
        start = time.perf_counter()
        policy.forward(
            inputs.local_obs,
            inputs.global_obs,
            hidden_local_vars=inputs.hidden_local_vars,
            hidden_global_vars=inputs.hidden_global_vars,
            agent_mask=inputs.agent_mask,
            previous_actions=inputs.previous_actions,
            deterministic=False,
        )
        synchronize(device)
        first_call_seconds = time.perf_counter() - start

        for _ in range(warmup_iters):
            policy.forward(
                inputs.local_obs,
                inputs.global_obs,
                hidden_local_vars=inputs.hidden_local_vars,
                hidden_global_vars=inputs.hidden_global_vars,
                agent_mask=inputs.agent_mask,
                previous_actions=inputs.previous_actions,
                deterministic=False,
            )

        synchronize(device)
        start = time.perf_counter()
        for _ in range(measured_iters):
            policy.forward(
                inputs.local_obs,
                inputs.global_obs,
                hidden_local_vars=inputs.hidden_local_vars,
                hidden_global_vars=inputs.hidden_global_vars,
                agent_mask=inputs.agent_mask,
                previous_actions=inputs.previous_actions,
                deterministic=False,
            )
        synchronize(device)
        steady_seconds = time.perf_counter() - start

    calls_per_second = measured_iters / steady_seconds
    return first_call_seconds, steady_seconds, calls_per_second


def time_evaluate_actions(
        *,
        policy: BenchmarkMATQCSPolicy,
        batch: PPOSamples,
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
        rollout_inputs: RolloutInputs,
        train_batch: PPOSamples,
        config: BenchmarkConfig,
        case: BenchmarkCase,
) -> CaseResult:
    device = torch.device(config.device)
    reset_compile_caches()
    seed_everything(config.seed, device=device)

    start = time.perf_counter()
    policy = BenchmarkMATQCSPolicy(
        env=env,
        config=build_policy_config(config, compile_modules=case.compile_modules),
        expose_compile_friendly=case.expose_compile_friendly,
    )
    policy.to(device)
    synchronize(device)
    init_seconds = time.perf_counter() - start

    first_forward_seconds, steady_forward_seconds, forward_calls_per_second = time_rollout_forward(
        policy=policy,
        inputs=rollout_inputs,
        warmup_iters=config.warmup_iters,
        measured_iters=config.measured_iters,
        device=device,
    )
    first_evaluate_seconds, steady_evaluate_seconds, evaluate_calls_per_second = time_evaluate_actions(
        policy=policy,
        batch=train_batch,
        warmup_iters=config.warmup_iters,
        measured_iters=config.measured_iters,
        device=device,
    )

    result = CaseResult(
        name=case.name,
        compile_modules=case.compile_modules,
        expose_compile_friendly=case.expose_compile_friendly,
        actual_compile_friendly=policy.action_dist.compile_friendly,
        compiled_rollout_helper=case.compile_modules and policy.action_dist.compile_friendly,
        compiled_train_eval_core=case.compile_modules and policy.action_dist.compile_friendly,
        init_seconds=init_seconds,
        first_forward_seconds=first_forward_seconds,
        steady_forward_seconds=steady_forward_seconds,
        forward_calls_per_second=forward_calls_per_second,
        first_evaluate_seconds=first_evaluate_seconds,
        steady_evaluate_seconds=steady_evaluate_seconds,
        evaluate_calls_per_second=evaluate_calls_per_second,
    )

    del policy
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def log_speedups(results: list[CaseResult]) -> None:
    results_by_name = {result.name: result for result in results}
    required_names = {
        "eager_compile_friendly",
        "compiled_compile_friendly",
        "eager_forced_non_compile_friendly",
        "compiled_forced_non_compile_friendly",
    }
    if not required_names.issubset(results_by_name):
        logger.info("Skipping speedup summary because not all four benchmark cases were run.")
        return

    friendly_eager = results_by_name["eager_compile_friendly"]
    friendly_compiled = results_by_name["compiled_compile_friendly"]
    logger.info(
        "compile_friendly=True steady rollout speedup: {:.2f}x | steady train-eval speedup: {:.2f}x",
        friendly_compiled.forward_calls_per_second / friendly_eager.forward_calls_per_second,
        friendly_compiled.evaluate_calls_per_second / friendly_eager.evaluate_calls_per_second,
    )

    forced_false_eager = results_by_name["eager_forced_non_compile_friendly"]
    forced_false_compiled = results_by_name["compiled_forced_non_compile_friendly"]
    logger.info(
        "compile_friendly=False steady rollout speedup: {:.2f}x | steady train-eval speedup: {:.2f}x",
        forced_false_compiled.forward_calls_per_second / forced_false_eager.forward_calls_per_second,
        forced_false_compiled.evaluate_calls_per_second / forced_false_eager.evaluate_calls_per_second,
    )


def parse_args() -> tuple[BenchmarkConfig, list[str]]:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark MAT-QCS policy torch.compile speedups with the same action distribution exposed as "
            "compile-friendly or forced non-compile-friendly."
        )
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default=default_device)
    parser.add_argument(
        "--env-num-envs",
        type=int,
        default=8,
        help="Vector-env size used only to build the env wrapper.",
    )
    parser.add_argument("--rollout-batch-size", type=int, default=512, help="Batch size for rollout forward().")
    parser.add_argument("--train-batch-size", type=int, default=4096, help="Batch size for evaluate_actions().")
    parser.add_argument("--warmup-iters", type=int, default=10, help="Warmup iterations after the first timed call.")
    parser.add_argument("--measured-iters", type=int, default=1000, help="Measured iterations per benchmark phase.")
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
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--encoder-nhead", type=int, default=4)
    parser.add_argument("--decoder-nhead", type=int, default=2)
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

    return BenchmarkConfig(
        device=str(args.device),
        env_num_envs=int(args.env_num_envs),
        rollout_batch_size=int(args.rollout_batch_size),
        train_batch_size=int(args.train_batch_size),
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
        encoder_layers=int(args.encoder_layers),
        decoder_layers=int(args.decoder_layers),
        encoder_nhead=int(args.encoder_nhead),
        decoder_nhead=int(args.decoder_nhead),
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
    logger.info("Running MAT-QCS compile benchmark with config: {}", config)

    env = make_env(config)
    seed_everything(config.seed, device=device)
    rollout_inputs = build_rollout_inputs(config, device=device)
    train_batch = build_train_samples(config, device=device)

    cases = [case for case in ALL_CASES if case.name in selected_case_names]

    results = [
        run_case(
            env=env,
            rollout_inputs=rollout_inputs,
            train_batch=train_batch,
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

