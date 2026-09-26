import abc
from dataclasses import dataclass
from typing import Optional

import mujoco
import numpy as np

from swarmbots.mj_env.swarm.swarm_config import SwarmConfig
from swarmbots.mj_env.swarm.swarm_connections import SwarmConnections


class BaseSwarm(abc.ABC):

    def __init__(self, config: SwarmConfig, max_unit_extent: float):
        self.config = config
        self.max_unit_extent = max_unit_extent

    def get_settings(self):
        return {
            'config': self.config.get_settings()
        }

    @property
    @abc.abstractmethod
    def can_have_inactive_units(self) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def _create_swarm_spec(self, rng: np.random.Generator | None = None) -> mujoco.MjSpec:
        raise NotImplementedError()

    @abc.abstractmethod
    def reset_swarm(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            rng: np.random.Generator,
            swarm_start_location: np.ndarray,
            inactive_unit_positions: np.ndarray
    ) -> tuple[SwarmConnections, Optional[np.ndarray]]:
        """

        :param model:
        :param data:
        :param rng:
        :param swarm_start_location: shape (3,)
        :param inactive_unit_positions: shape (N, 3)

        initialize state and potentially randomize swarm

        :return: connections, units_active_mask
        """
        raise NotImplementedError()

    def create_swarm_spec(self, rng: np.random.Generator | None = None) -> mujoco.MjSpec:
        if rng is None:
            rng = np.random.default_rng(42)
        spec = self._create_swarm_spec(rng)
        self.add_eq_constraints(spec)
        return spec

    def add_eq_constraints(self, spec: mujoco.MjSpec):
        for unit1 in range(self.config.num_units - 1):
            for unit2 in range(unit1 + 1, self.config.num_units):
                for conn1 in range(self.config.limbs_per_unit):
                    for conn2 in range(self.config.limbs_per_unit):
                        for twist_idx, twist in enumerate(self.config.connection_twist_values):
                            eq = spec.add_equality(
                                name=self.config.get_eq_variant_name(unit1, conn1, unit2, conn2, twist_idx),
                                type=mujoco.mjtEq.mjEQ_WELD,
                                objtype=mujoco.mjtObj.mjOBJ_BODY,
                                name1=self.config.get_connector_name(unit1, conn1),
                                name2=self.config.get_connector_name(unit2, conn2),
                                active=False,
                            )

                            eq.data[:3] = [0, 0, 0]  # anchor
                            eq.data[3:6] = [0, 0, 0]  # relpose pos
                            eq.data[6:10] = [0, 1, 0, 0]  # relpose quat
                            eq.data[10] = self.config.connection_torquescale  # torquescale
                            if self.config.uses_quantized_connection_twist:
                                eq.data[7] = np.cos(twist / 2)
                                eq.data[8] = np.sin(twist / 2)

