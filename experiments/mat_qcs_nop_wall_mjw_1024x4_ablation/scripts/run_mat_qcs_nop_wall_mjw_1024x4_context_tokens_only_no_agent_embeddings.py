from pathlib import Path

from common import MATQCSDecoderSelfAttentionMode, run_ablation


def main() -> None:
    run_ablation(
        variant_name="sign_magnitude_beta_nop_context_tokens_only_no_agent_embeddings",
        entrypoint_path=Path(__file__).resolve(),
        mat_decoder_self_attention_mode=MATQCSDecoderSelfAttentionMode.CONTEXT_TOKENS_ONLY,
        mat_add_agent_embeddings=False,
    )


if __name__ == "__main__":
    main()
