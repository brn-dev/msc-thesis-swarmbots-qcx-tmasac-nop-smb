from dataclasses import dataclass

from swarmbots.learn.action_dists.gumbel_softmax_sign_magnitude_action_dist import (
    _GumbelSoftmaxSignMagnitudeMixin,
)
from swarmbots.learn.action_dists.sign_magnitude_beta_action_dist import (
    SignMagnitudeBetaActionDist,
    SignMagnitudeBetaConfig,
)


@dataclass(frozen=True)
class TernarySignMagnitudeBetaConfig(SignMagnitudeBetaConfig):
    initial_zero_prob: float | None = None
    gumbel_temperature: float = 1.0


class TernarySignMagnitudeBetaActionDist(
        _GumbelSoftmaxSignMagnitudeMixin,
        SignMagnitudeBetaActionDist,
):
    """Samples exactly from [negative Beta, zero, positive Beta].

    ``rsample`` uses a hard straight-through Gumbel selector so SAC actor gradients
    reach both the component logits and Beta parameters.
    """

    _N_MIXTURE_COMPONENTS = 3
    _OUTPUTS_PER_ACTION = 7

    _NEGATIVE_INDEX = 0
    _ZERO_INDEX = 1
    _POSITIVE_INDEX = 2
