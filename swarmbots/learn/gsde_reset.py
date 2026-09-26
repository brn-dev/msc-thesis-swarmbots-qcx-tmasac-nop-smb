import abc
from dataclasses import dataclass


@dataclass(frozen=True)
class GSDEResetMode(abc.ABC):
    pass


@dataclass(frozen=True)
class GSDEIntervalResetMode(GSDEResetMode):
    interval: int


@dataclass(frozen=True)
class GSDEProbabilityResetMode(GSDEResetMode):
    probability: float


def resolve_gsde_reset_mode(
        *,
        gsde_enabled: bool,
        reset_mode: GSDEResetMode | None,
) -> GSDEResetMode | None:
    if not gsde_enabled:
        return None
    if reset_mode is None:
        raise ValueError("gsde_reset_mode is required when gSDE is enabled.")
    if isinstance(reset_mode, GSDEIntervalResetMode):
        if reset_mode.interval <= 0:
            raise ValueError(f"GSDEIntervalResetMode.interval must be > 0, got {reset_mode.interval}")
        return reset_mode
    if isinstance(reset_mode, GSDEProbabilityResetMode):
        if not (0.0 < reset_mode.probability < 1.0):
            raise ValueError(
                f"GSDEProbabilityResetMode.probability must be in (0, 1), got {reset_mode.probability}"
            )
        return reset_mode
    raise TypeError(f"Unknown gsde_reset_mode type: {type(reset_mode)}")


