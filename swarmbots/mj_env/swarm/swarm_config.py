from dataclasses import dataclass
from typing import Optional

import numpy as np

from swarmbots.mj_env.swarm.unit_config import UnitConfig


def get_connector_suffix(connector: int):
    return f'-{connector}-connector'

@dataclass
class SwarmConfig:

    def __init__(
            self,
            num_units: int,
            unit_config: UnitConfig,
            connection_torquescale: float,
            quantize_connection_twist: Optional[int] = None,
    ):
        if quantize_connection_twist is not None:
            quantize_connection_twist = int(quantize_connection_twist)
            if quantize_connection_twist <= 0:
                raise ValueError(
                    f"Expected quantize_connection_twist to be None or > 0, got {quantize_connection_twist}"
                )

        self.num_units = num_units
        self.unit_config = unit_config

        self.connection_torquescale = connection_torquescale
        self.quantize_connection_twist = quantize_connection_twist
        if quantize_connection_twist is None:
            self.connection_twist_values = np.zeros(1, dtype=float)
        else:
            self.connection_twist_values = np.linspace(
                0.0,
                2.0 * np.pi,
                quantize_connection_twist,
                endpoint=False,
                dtype=float,
            )

        self.limbs_per_unit = len(unit_config)

        self.unit_prefixes = [f'Unit{i}-' for i in range(0, self.num_units)]

    def get_settings(self):
        return {
            'num_units': self.num_units,
            'unit_config': [limb_config.toJSON() for limb_config in self.unit_config],
            'connection_torquescale': self.connection_torquescale,
            'quantize_connection_twist': self.quantize_connection_twist,
        }

    def get_connector_name(self, unit: int, connector: int):
        return f'{self.unit_prefixes[unit]}{get_connector_suffix(connector)}'

    def get_eq_name(self, unit1: int, conn1: int, unit2: int, conn2: int):
        return f'eq_{unit1}-{conn1}_{unit2}-{conn2}'

    @property
    def num_connection_eq_variants(self) -> int:
        return len(self.connection_twist_values)

    @property
    def uses_quantized_connection_twist(self) -> bool:
        return self.quantize_connection_twist is not None

    def get_eq_variant_name(
            self,
            unit1: int,
            conn1: int,
            unit2: int,
            conn2: int,
            twist_idx: int,
    ) -> str:
        eq_name = self.get_eq_name(unit1, conn1, unit2, conn2)
        if not self.uses_quantized_connection_twist:
            return eq_name
        return f"{eq_name}_twist{twist_idx}"

    def get_nearest_connection_twist_index(self, twist: float) -> int:
        if not self.uses_quantized_connection_twist:
            return 0
        angle_delta = np.arctan2(
            np.sin(float(twist) - self.connection_twist_values),
            np.cos(float(twist) - self.connection_twist_values),
        )
        return int(np.argmin(np.abs(angle_delta)))

    def sample_connection_twist(self, rng: np.random.Generator) -> float:
        if not self.uses_quantized_connection_twist:
            return float(rng.random() * 2.0 * np.pi)
        twist_idx = int(rng.integers(len(self.connection_twist_values)))
        return float(self.connection_twist_values[twist_idx])



