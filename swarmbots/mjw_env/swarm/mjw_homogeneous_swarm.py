from __future__ import annotations

from dataclasses import dataclass
from typing import Collection

import mujoco
import numpy as np
import torch
from mujoco import MjsBody

from swarmbots.mj_env.random_utils import random_quat_shoemake
from swarmbots.mj_env.swarm.unit import init_unit
from swarmbots.mj_env.swarm.unit_config import UNIT_CONFIG_TETRAHEDRON_XY, UnitConfig
from swarmbots.mjw_env.swarm.mjw_swarm_config import MJWSwarmConfig


@dataclass
class MJWPreConnectedUnitLocationsConfig:
    num_units: int
    num_unit_probs: dict[int, float] | None = None
    unconnected_prob: float = 0.0
    max_radius: float = 1e8
    z_pos: float = 0.0
    center: bool = True
    pool_seeds: Collection[int] | None = None
    active_pool_size: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.unconnected_prob <= 1.0:
            raise ValueError("unconnected_prob must be in [0, 1]")
        if self.pool_seeds is not None:
            pool_seeds = tuple(sorted({int(seed) for seed in self.pool_seeds}))
            if not pool_seeds:
                raise ValueError("pool_seeds must not be empty")
            self.pool_seeds = pool_seeds
        if self.pool_seeds is None:
            raise ValueError("MJW preconnected swarms require explicit pool_seeds.")
        pool_size = self.get_pool_size()
        if self.active_pool_size is None:
            self.active_pool_size = pool_size
        if self.active_pool_size < 1 or self.active_pool_size > pool_size:
            raise ValueError(f"active_pool_size must be in [1, {pool_size}], got {self.active_pool_size}")
        if self.num_unit_probs is not None:
            counts = np.asarray(list(self.num_unit_probs.keys()), dtype=int)
            probs = np.asarray(list(self.num_unit_probs.values()), dtype=float)
            if counts.max() != self.num_units:
                raise ValueError("num_unit_probs must contain an entry for count == num_units")
            if np.any(counts < 1) or np.any(counts > self.num_units):
                raise ValueError("num_unit_probs contains invalid counts")
            if np.any(probs < 0):
                raise ValueError("num_unit_probs must not contain negative probabilities")
            probs_sum = probs.sum()
            if probs_sum <= 0:
                raise ValueError("num_unit_probs must sum to a positive value")
            probs = probs / probs_sum
            self.num_unit_probs = dict(zip(counts.tolist(), probs.tolist(), strict=True))

    def get_pool_size(self) -> int:
        if self.pool_seeds is not None:
            return len(self.pool_seeds)
        raise ValueError("Expected explicit pool_seeds for MJW preconnected swarms.")


@dataclass
class MJWSwarmPool:
    positions: torch.Tensor
    quats: torch.Tensor
    active_mask: torch.Tensor
    partner_unit: torch.Tensor
    partner_connector: torch.Tensor
    twist_idx: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.positions.shape[0])


class MJWHomogeneousSwarm:
    def __init__(
        self,
        unit_start_locations: MJWPreConnectedUnitLocationsConfig,
        *,
        unit_config: UnitConfig = UNIT_CONFIG_TETRAHEDRON_XY,
        body_radius: float = 0.1,
        leg_length: float = 0.2,
        leg_radius: float = 0.025,
        segment_1_ratio: float = 0.1,
        hinge_range: tuple[float | None, ...] = (np.pi / 3, np.pi / 3),
        hinge_armature: float | tuple[float, ...] = (0.0, 0.0),
        hinge_damping: float | tuple[float, ...] = (0.0, 0.0),
        hinge_frictionloss: float | tuple[float, ...] = (0.0, 0.0),
        connection_torquescale: float = 50.0,
        quantize_connection_twist: int = 16,
    ) -> None:
        self.unit_start_locations = unit_start_locations
        self.num_units = unit_start_locations.num_units
        self.config = MJWSwarmConfig(
            num_units=self.num_units,
            unit_config=unit_config,
            connection_torquescale=connection_torquescale,
            quantize_connection_twist=quantize_connection_twist,
        )
        self.body_radius = float(body_radius)
        self.leg_length = float(leg_length)
        self.leg_radius = float(leg_radius)
        self.segment_1_ratio = float(segment_1_ratio)
        self.hinge_range = hinge_range
        self.hinge_armature = hinge_armature
        self.hinge_damping = hinge_damping
        self.hinge_frictionloss = hinge_frictionloss
        self.max_unit_extent = self.body_radius + self.leg_length

    @property
    def can_have_inactive_units(self) -> bool:
        return self.unit_start_locations.num_unit_probs is not None

    def get_settings(self) -> dict[str, object]:
        return {
            "config": self.config.get_settings(),
            "unit_start_locations": self.unit_start_locations,
            "body_radius": self.body_radius,
            "leg_length": self.leg_length,
            "leg_radius": self.leg_radius,
            "segment_1_ratio": self.segment_1_ratio,
            "minimal_contacts": True,
            "use_cylinders": False,
            "hinge_range": self.hinge_range,
            "hinge_armature": self.hinge_armature,
            "hinge_damping": self.hinge_damping,
            "hinge_frictionloss": self.hinge_frictionloss,
        }

    def get_pool_size(self) -> int:
        return self.unit_start_locations.get_pool_size()

    def get_active_pool_size(self) -> int:
        return self.unit_start_locations.active_pool_size

    def set_active_pool_size(self, active_pool_size: int) -> int:
        pool_size = self.get_pool_size()
        if active_pool_size < 1 or active_pool_size > pool_size:
            raise ValueError(f"active_pool_size must be in [1, {pool_size}], got {active_pool_size}")
        self.unit_start_locations.active_pool_size = int(active_pool_size)
        return int(self.unit_start_locations.active_pool_size)

    def get_active_pool_seeds(self) -> tuple[int, ...]:
        return self._resolve_pool_seeds()[:self.get_active_pool_size()]

    def create_swarm_spec(self, *, seed: int | None = None) -> mujoco.MjSpec:
        rng = np.random.default_rng(42 if seed is None else seed)
        positions, quats, _partner_unit, _partner_connector, _twist_idx = self._generate_preconnected_sample(
            rng=rng,
            force_full=True,
        )

        spec = mujoco.MjSpec()
        spec.compiler.degree = 0
        worldbody: MjsBody = spec.worldbody

        for unit_idx in range(self.num_units):
            shade = 1.0 - (unit_idx / max(self.num_units - 1, 1))
            body_rgba = (shade, shade, shade, 0.35)
            unit = init_unit(
                body_radius=self.body_radius,
                leg_length=self.leg_length,
                leg_radius=self.leg_radius,
                segment_1_ratio=self.segment_1_ratio,
                minimal_contacts=True,
                use_cylinders=False,
                hinge_range=self.hinge_range,
                hinge_armature=self.hinge_armature,
                hinge_damping=self.hinge_damping,
                hinge_frictionloss=self.hinge_frictionloss,
                unit_config=self.config.unit_config,
                body_rgba=body_rgba,
            )
            unit.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)
            worldbody.add_frame(
                pos=positions[unit_idx].tolist(),
                quat=quats[unit_idx].tolist(),
            ).attach_body(unit, self.config.unit_prefixes[unit_idx], "")

        self._add_eq_constraints(spec)
        return spec

    def build_pool(self, *, device: torch.device) -> MJWSwarmPool:
        pool_seeds = self._resolve_pool_seeds()

        positions_list: list[np.ndarray] = []
        quats_list: list[np.ndarray] = []
        partner_units_list: list[np.ndarray] = []
        partner_connectors_list: list[np.ndarray] = []
        twist_idx_list: list[np.ndarray] = []

        for seed in pool_seeds:
            rng = np.random.default_rng(int(seed))
            positions, quats, partner_unit, partner_connector, twist_idx = self._generate_preconnected_sample(rng=rng)
            positions_list.append(positions.astype(np.float32, copy=False))
            quats_list.append(quats.astype(np.float32, copy=False))
            partner_units_list.append(partner_unit.astype(np.int64, copy=False))
            partner_connectors_list.append(partner_connector.astype(np.int64, copy=False))
            twist_idx_list.append(twist_idx.astype(np.int64, copy=False))

        positions = torch.as_tensor(np.stack(positions_list), device=device, dtype=torch.float32)
        quats = torch.as_tensor(np.stack(quats_list), device=device, dtype=torch.float32)
        partner_unit = torch.as_tensor(np.stack(partner_units_list), device=device, dtype=torch.int64)
        partner_connector = torch.as_tensor(np.stack(partner_connectors_list), device=device, dtype=torch.int64)
        twist_idx = torch.as_tensor(np.stack(twist_idx_list), device=device, dtype=torch.int64)
        active_mask = (partner_unit != -2).any(dim=-1)
        return MJWSwarmPool(
            positions=positions,
            quats=quats,
            active_mask=active_mask,
            partner_unit=partner_unit,
            partner_connector=partner_connector,
            twist_idx=twist_idx,
        )

    def _resolve_pool_seeds(self) -> tuple[int, ...]:
        if self.unit_start_locations.pool_seeds is not None:
            return tuple(int(seed) for seed in self.unit_start_locations.pool_seeds)
        raise ValueError("Expected explicit pool_seeds for MJW preconnected swarms.")

    def _add_eq_constraints(self, spec: mujoco.MjSpec) -> None:
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
                            eq.data[:3] = [0, 0, 0]
                            eq.data[3:6] = [0, 0, 0]
                            eq.data[6:10] = [0, 1, 0, 0]
                            eq.data[7] = np.cos(float(twist) / 2.0)
                            eq.data[8] = np.sin(float(twist) / 2.0)
                            eq.data[10] = self.config.connection_torquescale

    def _generate_preconnected_sample(
        self,
        *,
        rng: np.random.Generator,
        force_full: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        random_config = self.unit_start_locations
        unit_config = self.config.unit_config
        num_units = random_config.num_units
        max_radius = random_config.max_radius

        if force_full or random_config.num_unit_probs is None:
            num_active_units = num_units
        else:
            counts = list(random_config.num_unit_probs.keys())
            probs = list(random_config.num_unit_probs.values())
            num_active_units = int(rng.choice(counts, p=probs))

        def quat_to_mat(quat: np.ndarray) -> np.ndarray:
            mat = np.empty(9, dtype=float)
            mujoco.mju_quat2Mat(mat, quat)
            return mat.reshape(3, 3)

        def mat_to_quat(mat: np.ndarray) -> np.ndarray:
            quat = np.empty(4, dtype=float)
            mujoco.mju_mat2Quat(quat, mat.reshape(9))
            return quat

        limb_rot_mats = np.zeros((len(unit_config), 3, 3), dtype=float)
        for i, limb in enumerate(unit_config):
            quat = np.empty(4, dtype=float)
            mujoco.mju_quatZ2Vec(quat, limb.vec)
            mat = np.empty(9, dtype=float)
            mujoco.mju_quat2Mat(mat, quat)
            limb_rot_mats[i] = mat.reshape(3, 3)

        positions = np.zeros((num_units, 3), dtype=float)
        quats = np.zeros((num_units, 4), dtype=float)
        partner_unit = np.full((num_units, self.config.limbs_per_unit), -2, dtype=np.int64)
        partner_connector = np.full((num_units, self.config.limbs_per_unit), -2, dtype=np.int64)
        twist_idx = np.full((num_units, self.config.limbs_per_unit), -1, dtype=np.int64)

        rot_mats = np.zeros((num_active_units, 3, 3), dtype=float)
        connector_distance = self.body_radius + self.leg_length
        min_center_distance = 2.0 * self.max_unit_extent
        available_connectors = [list(range(len(unit_config))) for _ in range(num_active_units)]

        locations_found = 1
        quats[0] = random_quat_shoemake(rng)
        rot_mats[0] = quat_to_mat(quats[0])

        rejected_count = 0
        while locations_found < num_active_units:
            existing_choices = [
                (unit_idx, conn_idx)
                for unit_idx in range(locations_found)
                for conn_idx in available_connectors[unit_idx]
            ]
            if not existing_choices:
                raise RuntimeError("No available connectors left to build a preconnected swarm")

            unit1, conn1 = existing_choices[int(rng.integers(len(existing_choices)))]
            conn2 = available_connectors[locations_found][int(rng.integers(len(available_connectors[locations_found])))]

            rot1 = rot_mats[unit1]
            conn1_rot = rot1 @ limb_rot_mats[conn1]
            z1 = conn1_rot[:, 2]
            pos1 = positions[unit1] + z1 * connector_distance
            z2 = -z1
            pos2 = pos1 + z1 * connector_distance

            if float(np.linalg.norm(pos2)) > max_radius:
                rejected_count += 1
                if rejected_count > 1000:
                    raise RuntimeError("Failed to generate preconnected swarm inside max_radius")
                continue

            if locations_found > 0:
                distances = np.linalg.norm(positions[:locations_found] - pos2, axis=1)
                if np.any(distances + 1e-6 < min_center_distance):
                    rejected_count += 1
                    if rejected_count > 1000:
                        raise RuntimeError("Failed to generate collision-free preconnected swarm")
                    continue

            sampled_twist_idx = self.config.sample_connection_twist_index(rng)
            twist = float(self.config.connection_twist_values[sampled_twist_idx])
            x1 = conn1_rot[:, 0]
            y1 = conn1_rot[:, 1]
            x2 = np.cos(twist) * x1 + np.sin(twist) * y1
            x2 /= np.linalg.norm(x2)
            y2 = np.cross(z2, x2)
            y2 /= np.linalg.norm(y2)
            conn2_rot = np.stack([x2, y2, z2], axis=1)
            unit2_rot = conn2_rot @ limb_rot_mats[conn2].T

            positions[locations_found] = pos2
            rot_mats[locations_found] = unit2_rot
            quats[locations_found] = mat_to_quat(unit2_rot)

            if random_config.unconnected_prob <= 0.0 or rng.random() >= random_config.unconnected_prob:
                partner_unit[unit1, conn1] = locations_found
                partner_connector[unit1, conn1] = conn2
                twist_idx[unit1, conn1] = sampled_twist_idx
                partner_unit[locations_found, conn2] = unit1
                partner_connector[locations_found, conn2] = conn1
                twist_idx[locations_found, conn2] = sampled_twist_idx

            available_connectors[unit1].remove(conn1)
            available_connectors[locations_found].remove(conn2)
            locations_found += 1
            rejected_count = 0

        if random_config.center:
            mean = positions[:num_active_units].mean(axis=0, keepdims=True)
            positions[:num_active_units] -= mean
        positions[:num_active_units, 2] += random_config.z_pos
        quats[num_active_units:] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        positions[num_active_units:] = 0.0

        inactive_mask = np.arange(num_units) >= num_active_units
        partner_unit[inactive_mask] = -2
        partner_connector[inactive_mask] = -2
        twist_idx[inactive_mask] = -1
        active_unit_mask = (~inactive_mask)[:, np.newaxis]
        partner_unit[active_unit_mask & (partner_unit == -2)] = -1
        partner_connector[active_unit_mask & (partner_connector == -2)] = -1

        return positions, quats, partner_unit, partner_connector, twist_idx
