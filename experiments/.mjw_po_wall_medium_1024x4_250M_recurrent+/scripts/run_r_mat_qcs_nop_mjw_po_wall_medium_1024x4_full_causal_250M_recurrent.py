from pathlib import Path

from common import run_experiment
from swarmbots.learn.algos.mat_qcs.mat_qcs_decoder import MATQCSDecoderSelfAttentionMode


def main() -> None:
    run_experiment(
        variant_name="r_mat_qcs_full_causal",
        entrypoint_path=Path(__file__).resolve(),
        policy_variant="r_mat_qcs",
        mat_decoder_self_attention_mode=MATQCSDecoderSelfAttentionMode.FULL_CAUSAL,
    )


if __name__ == "__main__":
    main()
