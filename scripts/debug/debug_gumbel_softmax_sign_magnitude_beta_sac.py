from __future__ import annotations

from pathlib import Path

from sign_magnitude_sac_probe import run_probe

from swarmbots.learn.action_dists.gumbel_softmax_sign_magnitude_action_dist import (
    GumbelSoftmaxSignMagnitudeBetaActionDist,
)


def make_action_dist(
        latent_dim: int,
        action_dim: int,
        gumbel_temperature: float | None,
) -> GumbelSoftmaxSignMagnitudeBetaActionDist:
    return GumbelSoftmaxSignMagnitudeBetaActionDist(
        latent_dim=latent_dim,
        action_dim=action_dim,
        action_net_initialization=None,
        initial_positive_prob=0.5,
        negative_alpha=2.0,
        negative_beta=2.5,
        positive_alpha=2.0,
        positive_beta=2.5,
        gumbel_temperature=1.0 if gumbel_temperature is None else gumbel_temperature,
    )


def main() -> None:
    run_probe(
        description="Train Gumbel sign-magnitude beta with SAC actor updates against a mock quadratic critic.",
        default_action_hist_dir=Path("action_hists_gumbel_smb"),
        action_dist_factory=make_action_dist,
        supports_gumbel_temperature=True,
    )


if __name__ == "__main__":
    main()
