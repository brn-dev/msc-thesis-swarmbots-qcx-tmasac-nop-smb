from pathlib import Path

from common import TMASACActorHeadKind, run_experiment


def main() -> None:
    run_experiment(
        variant_name="tmasac_decentralized_slstm",
        temporal_model_variant="slstm",
        actor_head_kind=TMASACActorHeadKind.DECENTRALIZED,
        entrypoint_path=Path(__file__).resolve(),
    )


if __name__ == "__main__":
    main()
