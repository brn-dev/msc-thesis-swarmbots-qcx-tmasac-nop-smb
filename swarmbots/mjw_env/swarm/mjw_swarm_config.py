from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from swarmbots.mj_env.swarm.unit_config import UnitConfig


def get_connector_suffix(connector: int) -> str:
    return f"-{connector}-connector"


@dataclass
class MJWSwarmConfig:
    num_units: int
    unit_config: UnitConfig
    connection_torquescale: float
    quantize_connection_twist: int

    def __post_init__(self) -> None:
        if self.quantize_connection_twist <= 0:
            raise ValueError(
                f"Expected quantize_connection_twist > 0 for MJWarp, got {self.quantize_connection_twist}"
            )
        self.limbs_per_unit = len(self.unit_config)
        self.unit_prefixes = [f"Unit{i}-" for i in range(self.num_units)]
        self.connection_twist_values = np.linspace(
            0.0,
            2.0 * np.pi,
            int(self.quantize_connection_twist),
            endpoint=False,
            dtype=float,
        )

    def get_settings(self) -> dict[str, object]:
        return {
            "num_units": self.num_units,
            "unit_config": [limb_config.toJSON() for limb_config in self.unit_config],
            "connection_torquescale": self.connection_torquescale,
            "quantize_connection_twist": self.quantize_connection_twist,
        }

    def get_connector_name(self, unit: int, connector: int) -> str:
        return f"{self.unit_prefixes[unit]}{get_connector_suffix(connector)}"

    def get_eq_name(self, unit1: int, conn1: int, unit2: int, conn2: int) -> str:
        return f"eq_{unit1}-{conn1}_{unit2}-{conn2}"

    def get_eq_variant_name(self, unit1: int, conn1: int, unit2: int, conn2: int, twist_idx: int) -> str:
        return f"{self.get_eq_name(unit1, conn1, unit2, conn2)}_twist{twist_idx}"

    def get_nearest_connection_twist_index(self, twist: float) -> int:
        angle_delta = np.arctan2(
            np.sin(float(twist) - self.connection_twist_values),
            np.cos(float(twist) - self.connection_twist_values),
        )
        return int(np.argmin(np.abs(angle_delta)))

    def sample_connection_twist_index(self, rng: np.random.Generator) -> int:
        return int(rng.integers(len(self.connection_twist_values)))

    def sample_connection_twist(self, rng: np.random.Generator) -> float:
        return float(self.connection_twist_values[self.sample_connection_twist_index(rng)])
