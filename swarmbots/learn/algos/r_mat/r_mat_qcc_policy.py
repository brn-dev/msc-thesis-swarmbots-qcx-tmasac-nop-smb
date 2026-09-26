from dataclasses import dataclass, field, replace

from torch import nn

from swarmbots.learn.algos.mat_qcc.mat_qcc_decoder import MATQCCDecoder, MATQCCDecoderConfig
from swarmbots.learn.algos.r_mat.r_mat_qcs_policy import RMATQCSPolicy, RMATQCSPolicyConfig
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper


@dataclass(frozen=True)
class RMATQCCPolicyConfig(RMATQCSPolicyConfig):
    decoder_config: MATQCCDecoderConfig = field(default_factory=MATQCCDecoderConfig)


class RMATQCCPolicy(RMATQCSPolicy):

    def __init__(
            self,
            env: BaseLearnEnvWrapper,
            config: RMATQCCPolicyConfig = RMATQCCPolicyConfig(),
    ) -> None:
        super().__init__(env=env, config=config)

    def get_hyper_parameters(self) -> dict[str, object]:
        return {
            "mat_qcc_policy_config": super().get_hyper_parameters()["mat_qcs_policy_config"],
        }

    def _build_decoder_config(self) -> MATQCCDecoderConfig:
        return replace(
            self.config.decoder_config,
            d_model=self.d_model_decoder,
            act_fn_cls=self.config.act_fn_cls,
            dropout=self.config.dropout,
        )

    def _build_decoder(
            self,
            decoder_config: MATQCCDecoderConfig,
    ) -> nn.Module:
        return MATQCCDecoder(
            config=decoder_config,
            max_agents=self.max_agents,
            memory_d_model=self.memory_d_model,
        )
