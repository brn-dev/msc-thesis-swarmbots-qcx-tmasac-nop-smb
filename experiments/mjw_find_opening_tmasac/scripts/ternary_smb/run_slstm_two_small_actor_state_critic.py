from pathlib import Path

from common import ACTOR_D_MODEL, ActorStateCriticInputConfig, run_experiment


def main() -> None:
    run_experiment(
        variant_name="slstm_two_small_actor_state_critic",
        temporal_model_variant="slstm",
        actor_state_critic_input_config=ActorStateCriticInputConfig(
            projection_dim=ACTOR_D_MODEL,
            projection_hidden_dims=(ACTOR_D_MODEL,),
        ),
        entrypoint_path=Path(__file__).resolve(),
    )


if __name__ == "__main__":
    main()
