from __future__ import annotations

from pathlib import Path

from sign_magnitude_sac_probe import run_probe

from swarmbots.learn.action_dists.reparameterized_sign_magnitude_kumaraswamy_action_dist import (
    ReparameterizedSignMagnitudeKumaraswamyActionDist,
)


def make_action_dist(
        latent_dim: int,
        action_dim: int,
        gumbel_temperature: float | None,
) -> ReparameterizedSignMagnitudeKumaraswamyActionDist:
    _ = gumbel_temperature
    return ReparameterizedSignMagnitudeKumaraswamyActionDist(
        latent_dim=latent_dim,
        action_dim=action_dim,
        action_net_initialization=None,
        initial_positive_prob=0.5,
        negative_a=2.0,
        negative_b=2.5,
        positive_a=2.0,
        positive_b=2.5,
    )


def main() -> None:
    run_probe(
        description="Train RSMK with SAC actor updates against a mock quadratic critic.",
        default_action_hist_dir=Path("action_hists_rsmk"),
        action_dist_factory=make_action_dist,
    )


if __name__ == "__main__":
    main()
