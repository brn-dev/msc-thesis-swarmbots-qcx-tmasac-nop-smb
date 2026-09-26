from __future__ import annotations

from collections.abc import Mapping

import numpy as np

PAYLOAD_GOAL_OBS_RECORD_DIM = 12
PAYLOAD_GOAL_IDENTITY_ROT6D: tuple[float, float, float, float, float, float] = (
    1.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
)
PAYLOAD_GOAL_COLORS: tuple[tuple[float, float, float, float], ...] = (
    (0.92, 0.24, 0.18, 1.0),
    (0.10, 0.62, 0.95, 1.0),
    (0.20, 0.78, 0.36, 1.0),
    (0.95, 0.72, 0.15, 1.0),
    (0.72, 0.32, 0.90, 1.0),
    (0.95, 0.42, 0.70, 1.0),
)


def normalize_count_probs(
    *,
    probs: Mapping[int, float] | None,
    max_payloads: int,
) -> dict[int, float]:
    if probs is None:
        return {max_payloads: 1.0}
    counts = np.asarray(list(probs.keys()), dtype=int)
    weights = np.asarray(list(probs.values()), dtype=float)
    if counts.size == 0:
        raise ValueError("active_payload_count_probs must not be empty")
    if (counts < 1).any() or (counts > max_payloads).any():
        raise ValueError(f"active payload counts must be in [1, {max_payloads}]")
    if (weights < 0.0).any():
        raise ValueError("active_payload_count_probs must not contain negative probabilities")
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        raise ValueError("active_payload_count_probs must sum to a positive value")
    weights /= weight_sum
    return dict(zip(counts.tolist(), weights.tolist(), strict=True))
