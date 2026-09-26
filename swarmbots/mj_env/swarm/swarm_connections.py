import numpy as np

from swarmbots.mj_env.swarm.swarm_config import SwarmConfig


class SwarmConnections:

    def __init__(self, config: SwarmConfig):
        self.connections = np.full((config.num_units, config.limbs_per_unit, 2), -1, dtype=int)
        self.twist_angles = np.zeros((config.num_units, config.limbs_per_unit), dtype=np.float32)
        self.disconnect_potentials = np.zeros((config.num_units, config.limbs_per_unit), dtype=np.float32)

    def reset(self):
        self.connections[:] = -1

    def is_connected(
            self,
            unit: int,
            unit_connector: int
    ):
        return self.connections[unit, unit_connector, 0] != -1

    def connect(
            self,
            unit1: int,
            unit1_connector: int,
            unit2: int,
            unit2_connector: int,
            twist_angle: float
    ):
        if unit1 == unit2:
            raise ValueError(f'Cannot connect {unit1} to itself')
        if self.is_connected(unit1, unit1_connector):
            raise ValueError(f'Unit {unit1} connector {unit1_connector} already connected')
        if self.is_connected(unit2, unit2_connector):
            raise ValueError(f'Unit {unit2} connector {unit2_connector} already connected')

        self.connections[unit1, unit1_connector] = [unit2, unit2_connector]
        self.twist_angles[unit1, unit1_connector] = twist_angle
        self.disconnect_potentials[unit1, unit1_connector] = 0

        self.connections[unit2, unit2_connector] = [unit1, unit1_connector]
        self.twist_angles[unit2, unit2_connector] = twist_angle
        self.disconnect_potentials[unit2, unit2_connector] = 0

    def update_disconnect_potentials(
            self,
            currently_active_mask: np.ndarray,
            newly_deactivated_mask: np.ndarray,
            disconnect_potential_threshold: float,
    ):
        disconnect_potentials_update = np.zeros_like(self.disconnect_potentials)

        disconnect_potentials_update[newly_deactivated_mask] = 1

        partners = self.connections[newly_deactivated_mask]
        disconnect_potentials_update[partners[:, 0], partners[:, 1]] += 1

        stayed_active_mask = np.logical_and(currently_active_mask, disconnect_potentials_update == 0)
        disconnect_potentials_update[stayed_active_mask] = -2

        self.disconnect_potentials += disconnect_potentials_update
        self.disconnect_potentials[stayed_active_mask] = np.maximum(0, self.disconnect_potentials[stayed_active_mask])

        return self.disconnect_potentials >= disconnect_potential_threshold

    def update_disconnect_potentials_continuous(
            self,
            currently_active_mask: np.ndarray,
            connector_actions: np.ndarray,
            disconnect_potential_threshold: float,
    ) -> np.ndarray:
        active_units, active_connectors = np.nonzero(currently_active_mask)
        if len(active_units) == 0:
            return self.disconnect_potentials >= disconnect_potential_threshold

        partners = self.connections[active_units, active_connectors]
        partner_units = partners[:, 0]
        partner_connectors = partners[:, 1]

        own_actions = connector_actions[active_units, active_connectors]
        partner_actions = connector_actions[partner_units, partner_connectors]
        own_disconnect_intent = np.maximum(-own_actions, 0.0)
        partner_disconnect_intent = np.maximum(-partner_actions, 0.0)
        disconnect_update = own_disconnect_intent + partner_disconnect_intent

        own_hold_intent = np.maximum(own_actions, 0.0)
        partner_hold_intent = np.maximum(partner_actions, 0.0)
        hold_update = -2.0 * np.minimum(own_hold_intent, partner_hold_intent)

        potential_update = np.where(disconnect_update > 0.0, disconnect_update, hold_update)
        self.disconnect_potentials[active_units, active_connectors] += potential_update
        self.disconnect_potentials[currently_active_mask] = np.maximum(
            0.0,
            self.disconnect_potentials[currently_active_mask],
        )

        return self.disconnect_potentials >= disconnect_potential_threshold

    def disconnect(
            self,
            unit: int,
            unit_connector: int,
    ) -> tuple[int, int]:
        unit2, unit2_connector = self.connections[unit, unit_connector]
        if unit2 == -1:
            raise ValueError(f'Unit {unit} connector {unit_connector} is not connected')

        # twist angles don't need to be reset necessarily
        self.connections[unit, unit_connector] = [-1, -1]
        self.connections[unit2, unit2_connector] = [-1, -1]
        self.disconnect_potentials[unit, unit_connector] = 0
        self.disconnect_potentials[unit2, unit2_connector] = 0

        return unit2, unit2_connector

    def get_is_active_mask(self):
        active_indices = np.where(self.connections[:, :, 0] != -1)

        is_active = np.zeros(self.connections.shape[:2], dtype=bool)
        is_active[active_indices] = True

        return is_active

    def get_active_connections(self):
        active_indices = np.where(self.connections[:, :, 0] != -1)

        rows = np.stack(active_indices).T  # (N, 2): (u1, c1)
        conns = self.connections[active_indices]  # (N, 2): (u2, c2)
        edges = np.concatenate((rows, conns), axis=-1)  # (N, 4): [u1, c1, u2, c2]
        angles = self.twist_angles[active_indices]  # (N,)

        u1, c1, u2, c2 = edges[:, 0], edges[:, 1], edges[:, 2], edges[:, 3]
        mask = (u1 < u2) | ((u1 == u2) & (c1 < c2))

        return edges[mask], angles[mask]
