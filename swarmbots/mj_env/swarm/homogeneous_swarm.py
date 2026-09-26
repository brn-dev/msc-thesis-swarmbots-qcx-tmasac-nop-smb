from dataclasses import dataclass
from typing import Collection, Optional, TypeVar, TypeAlias

import mujoco
import numpy as np
from loguru import logger
from mujoco import MjsBody

from swarmbots.mj_env.float_or_dist_params import FloatOrDistParams, eval_fodp, DistParams, eval_fodp_3d
from swarmbots.mj_env.random_utils import random_quat_shoemake
from swarmbots.mj_env.swarm.base_swarm import BaseSwarm
from swarmbots.mj_env.swarm.swarm_config import SwarmConfig
from swarmbots.mj_env.swarm.swarm_connections import SwarmConnections
from swarmbots.mj_env.swarm.unit import init_unit
from swarmbots.mj_env.swarm.unit_config import UnitConfig, UNIT_CONFIG_TETRAHEDRON_XY, UNIT_CONFIG_TETRAHEDRON_ZX

UNIT_START_LOCATION_PRESETS = {
    '4:diamond': [
        (0.0, 0.0, 0.0),
        (0.4, -0.4, 0.0),
        (-0.4, -0.4, 0.0),
        (0.0, -0.8, 0.0),
    ],
    '5:X': [
        (0.0, 0.0, 0.0),
        (0.4, 0.4, 0.0),
        (0.4, -0.4, 0.0),
        (-0.4, 0.4, 0.0),
        (-0.4, -0.4, 0.0),
    ],
    '5:T': [
        (0.0, 0.0, 0.0),
        (0.0, -0.4, 0.0),
        (0.0, 0.4, 0.0),
        (0.0, 0.4, -0.4),
        (0.0, 0.4, 0.4),
    ],
    '5:T-reverse': [
        (0.0, 0.0, 0.0),
        (0.0, 0.4, 0.0),
        (0.0, -0.4, 0.0),
        (0.0, -0.4, -0.4),
        (0.0, -0.4, 0.4),
    ],
    '5:3+2': [
        (0.0, 0.0, 0.0),
        (-0.6, 0.0, 0.0),
        (-0.3, -0.4, 0.0),
        (0.8, 0.0, 0.0),
        (1.3, 0.0, 0.0),
    ],
    '5:W': [
        (0.0, 0.0, 0.0),
        (0.8, 0.0, 0.0),
        (-0.8, 0.0, 0.0),
        (0.4, -0.4, 0.0),
        (-0.4, -0.4, 0.0),
    ],
    '5:M': [
        (0.0, -0.4, 0.0),
        (0.8, -0.4, 0.0),
        (-0.8, -0.4, 0.0),
        (0.4, 0.0, 0.0),
        (-0.4, 0.0, 0.0),
    ],
    '8:hourglass': [
        (0.0, 0.0, 0.0),
        (0.8, 0.0, 0.0),
        (-0.8, 0.0, 0.0),
        (0.4, -0.4, 0.0),
        (-0.4, -0.4, 0.0),
        (0.0, -0.8, 0.0),
        (0.8, -0.8, 0.0),
        (-0.8, -0.8, 0.0),
    ],
}

T = TypeVar('T')
Tuple2: TypeAlias = tuple[T, T]
Tuple3: TypeAlias = tuple[T, T, T]
Tuple2or3: TypeAlias = Tuple2 | Tuple3
HingeJointParam: TypeAlias = float | tuple[float, ...]

def _normalize_tuple2or3(tup: Tuple2or3[T], default_val: T) -> Tuple3[T]:
    l = len(tup)
    if l == 2:
        # noinspection PyTypeChecker
        return tup + (default_val,)
    if l == 3:
        return tup
    raise ValueError(tup)

def _normalize_tuples2or3(tuples: list[Tuple2or3[T]], default_val: T) -> list[Tuple3[T]]:
    return [_normalize_tuple2or3(tup, default_val) for tup in tuples]

@dataclass
class PoissonDiscUnitLocationsConfig:
    num_units: int

    num_unit_probs: Optional[dict[int, float]] = None
    max_radius: float = 1e8
    z_pos: float = 0.0
    center: bool = True

@dataclass
class PreConnectedUnitLocationsConfig:
    num_units: int

    num_unit_probs: Optional[dict[int, float]] = None
    unconnected_prob: float = 0.0

    max_radius: float = 1e8
    z_pos: float = 0.0
    center: bool = True

    pool_seeds: Optional[Collection[int]] = None
    active_pool_size: int | None = None

@dataclass
class RandomWiggleUnitLocationsConfig:
    num_units: int
    unit_locations: list[Tuple3[float]]
    wiggle_params: list[Tuple3[FloatOrDistParams]]

    def __init__(
            self,
            unit_locations: list[Tuple2or3[float]] | str,
            wiggle_params: DistParams | Tuple2or3[DistParams] | list[Tuple2or3[FloatOrDistParams]],
    ):
        if isinstance(unit_locations, str):
            self.unit_locations = UNIT_START_LOCATION_PRESETS[unit_locations]
        else:
            self.unit_locations = _normalize_tuples2or3(unit_locations, 0.0)

        self.num_units = len(self.unit_locations)
        if isinstance(wiggle_params, DistParams):
            self.wiggle_params = [(wiggle_params,) * 3] * self.num_units
        elif isinstance(wiggle_params, tuple):
            self.wiggle_params = [_normalize_tuple2or3(wiggle_params, 0.0)] * self.num_units
        else:
            assert len(wiggle_params) == self.num_units
            self.wiggle_params = _normalize_tuples2or3(wiggle_params, 0.0)

UnitStartLocations = (
        list[Tuple2or3[FloatOrDistParams]]
        | PoissonDiscUnitLocationsConfig
        | PreConnectedUnitLocationsConfig
        | RandomWiggleUnitLocationsConfig
)

class HomogeneousSwarm(BaseSwarm):

    unit_start_locations: UnitStartLocations

    def __init__(
            self,
            unit_start_locations: UnitStartLocations | str,
            unit_start_quats: list[tuple[float, float, float, float]] | None = None,
            unit_config: UnitConfig = UNIT_CONFIG_TETRAHEDRON_XY,
            body_radius: float = 0.1,
            leg_length: float = 0.2,
            leg_radius: float = 0.025,
            segment_1_ratio: float = 0.1,
            minimal_contacts: bool = True,
            use_cylinders: bool = False,
            hinge_range: tuple[float | None, ...] = (np.pi / 3, np.pi / 3),
            hinge_armature: HingeJointParam = (0.0, 0.0),
            hinge_damping: HingeJointParam = (0.0, 0.0),
            hinge_frictionloss: HingeJointParam = (0.0, 0.0),
            connection_torquescale: float = 50.0,
            quantize_connection_twist: Optional[int] = None,
            randomize_unit_orientations: bool = False
    ) -> None:
        assert unit_start_quats is None or not randomize_unit_orientations

        self._can_have_inactive_units = False

        if isinstance(unit_start_locations, str):
            self.unit_start_locations = UNIT_START_LOCATION_PRESETS[unit_start_locations]
            self.num_units = len(self.unit_start_locations)
        elif isinstance(unit_start_locations, PoissonDiscUnitLocationsConfig):
            self.num_units = unit_start_locations.num_units
            self.unit_start_locations = unit_start_locations
            if unit_start_locations.num_unit_probs is not None:
                counts = np.array(list(unit_start_locations.num_unit_probs.keys()), dtype=int)
                probs = np.array(list(unit_start_locations.num_unit_probs.values()), dtype=float)
                if counts.max() != self.num_units:
                    raise ValueError("num_unit_probs must contain an entry for count == num_units")
                if (counts < 1).any():
                    raise ValueError("num_unit_probs must only contain counts >= 1")
                if (counts > unit_start_locations.num_units).any():
                    raise ValueError("num_unit_probs must not exceed num_units")
                if (probs < 0).any():
                    raise ValueError("num_unit_probs must not contain negative probabilities")
                probs_sum = probs.sum()
                if probs_sum <= 0:
                    raise ValueError("num_unit_probs must sum to a positive value")
                probs = probs / probs_sum
                unit_start_locations.num_unit_probs = dict(zip(counts.tolist(), probs.tolist()))
                self._can_have_inactive_units = True
        elif isinstance(unit_start_locations, PreConnectedUnitLocationsConfig):
            self.num_units = unit_start_locations.num_units
            self.unit_start_locations = unit_start_locations
            if not 0.0 <= unit_start_locations.unconnected_prob <= 1.0:
                raise ValueError("unconnected_prob must be in [0.0, 1.0]")
            if unit_start_locations.num_unit_probs is not None:
                counts = np.array(list(unit_start_locations.num_unit_probs.keys()), dtype=int)
                probs = np.array(list(unit_start_locations.num_unit_probs.values()), dtype=float)
                if counts.max() != self.num_units:
                    raise ValueError("num_unit_probs must contain an entry for count == num_units")
                if (counts < 1).any():
                    raise ValueError("num_unit_probs must only contain counts >= 1")
                if (counts > unit_start_locations.num_units).any():
                    raise ValueError("num_unit_probs must not exceed num_units")
                if (probs < 0).any():
                    raise ValueError("num_unit_probs must not contain negative probabilities")
                probs_sum = probs.sum()
                if probs_sum <= 0:
                    raise ValueError("num_unit_probs must sum to a positive value")
                probs = probs / probs_sum
                unit_start_locations.num_unit_probs = dict(zip(counts.tolist(), probs.tolist()))
                self._can_have_inactive_units = True
            if unit_start_locations.pool_seeds is not None:
                pool_seeds = tuple(sorted(set(unit_start_locations.pool_seeds)))
                if not pool_seeds:
                    raise ValueError("pool_seeds must contain at least one seed when provided")
                if any(isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) for seed in pool_seeds):
                    raise ValueError("pool_seeds must only contain integer seeds")
                unit_start_locations.pool_seeds = tuple(int(seed) for seed in pool_seeds)
            if unit_start_locations.pool_seeds is not None:
                resolved_pool_size = self._get_pool_size()
                if unit_start_locations.active_pool_size is None:
                    unit_start_locations.active_pool_size = resolved_pool_size
                self._validate_active_pool_size(unit_start_locations.active_pool_size)
            elif unit_start_locations.active_pool_size is not None:
                raise ValueError("active_pool_size requires pool_seeds for a fixed precomputed pool.")
        elif isinstance(unit_start_locations, RandomWiggleUnitLocationsConfig):
            self.num_units = unit_start_locations.num_units
            self.unit_start_locations = unit_start_locations
        elif isinstance(unit_start_locations, list):
            self.num_units = len(unit_start_locations)
            self.unit_start_locations = _normalize_tuples2or3(unit_start_locations, 0.0)
        else:
            raise ValueError(unit_start_locations)

        self.unit_start_quats = unit_start_quats
        self.randomize_unit_orientations = randomize_unit_orientations

        if isinstance(unit_start_locations, PreConnectedUnitLocationsConfig):
            if unit_start_quats is not None:
                raise ValueError("PreConnectedUnitLocationsConfig determines unit orientations")

        assert unit_start_quats is None or len(unit_start_quats) == self.num_units

        super().__init__(
            config=SwarmConfig(
                num_units=self.num_units,
                unit_config=unit_config,
                connection_torquescale=connection_torquescale,
                quantize_connection_twist=quantize_connection_twist,
            ),
            max_unit_extent=body_radius + leg_length
        )

        self.body_radius = body_radius
        self.leg_length = leg_length
        self.leg_radius = leg_radius
        self.segment_1_ratio = segment_1_ratio
        self.minimal_contacts = minimal_contacts
        self.use_cylinders = use_cylinders
        self.hinge_range = hinge_range
        self.hinge_armature = hinge_armature
        self.hinge_damping = hinge_damping
        self.hinge_frictionloss = hinge_frictionloss

    @property
    def can_have_inactive_units(self) -> bool:
        return self._can_have_inactive_units

    def get_settings(self):
        settings = super().get_settings()
        settings.update({
            'unit_start_locations': self.unit_start_locations,
            'unit_start_quats': self.unit_start_quats,
            'body_radius': self.body_radius,
            'leg_length': self.leg_length,
            'leg_radius': self.leg_radius,
            'segment_1_ratio': self.segment_1_ratio,
            'minimal_contacts': self.minimal_contacts,
            'use_cylinders': self.use_cylinders,
            'hinge_range': self.hinge_range,
            'hinge_armature': self.hinge_armature,
            'hinge_damping': self.hinge_damping,
            'hinge_frictionloss': self.hinge_frictionloss,
            'randomize_unit_orientations': self.randomize_unit_orientations,
        })
        return settings

    def has_fixed_pool(self) -> bool:
        return isinstance(self.unit_start_locations, PreConnectedUnitLocationsConfig) and (
            self.unit_start_locations.pool_seeds is not None
        )

    def get_pool_size(self) -> int | None:
        if not self.has_fixed_pool():
            return None
        return self._get_pool_size()

    def get_active_pool_size(self) -> int | None:
        if not self.has_fixed_pool():
            return None
        return int(self.unit_start_locations.active_pool_size)

    def set_active_pool_size(self, active_pool_size: int) -> int:
        if not self.has_fixed_pool():
            raise ValueError("This swarm does not use a fixed precomputed pool.")
        self._validate_active_pool_size(active_pool_size)
        self.unit_start_locations.active_pool_size = int(active_pool_size)
        return int(self.unit_start_locations.active_pool_size)

    def get_active_pool_seeds(self) -> tuple[int, ...] | None:
        if not self.has_fixed_pool():
            return None
        active_pool_size = self.unit_start_locations.active_pool_size
        return self._resolve_pool_seeds()[:active_pool_size]

    def _get_pool_size(self) -> int:
        pool_seeds = self.unit_start_locations.pool_seeds
        if pool_seeds is not None:
            return len(pool_seeds)
        raise ValueError("Expected pool_seeds to be configured for a fixed precomputed pool.")

    def _resolve_pool_seeds(self) -> tuple[int, ...]:
        pool_seeds = self.unit_start_locations.pool_seeds
        if pool_seeds is not None:
            return tuple(int(seed) for seed in pool_seeds)
        raise ValueError("Expected pool_seeds to be configured for a fixed precomputed pool.")

    def _validate_active_pool_size(self, active_pool_size: int) -> None:
        pool_size = self._get_pool_size()
        if active_pool_size < 1 or active_pool_size > pool_size:
            raise ValueError(f"active_pool_size must be in [1, {pool_size}], got {active_pool_size}")

    def _create_swarm_spec(self, rng: np.random.Generator | None = None) -> mujoco.MjSpec:
        spec = mujoco.MjSpec()
        spec.compiler.degree = 0
        worldbody: MjsBody = spec.worldbody
        if rng is None:
            rng = np.random.default_rng()

        unit_start_locations: list[Tuple3[FloatOrDistParams]]
        unit_start_quats: list[tuple[float, float, float, float]] | None = None
        if isinstance(self.unit_start_locations, PoissonDiscUnitLocationsConfig):
            unit_start_locations = self._generate_poisson_disc_start_locations(
                rng=rng,
                force_full=True
            )
        elif isinstance(self.unit_start_locations, PreConnectedUnitLocationsConfig):
            unit_start_locations, unit_start_quats, _ = self._generate_preconnected_swarm(
                rng=rng,
                force_full=True
            )
        elif isinstance(self.unit_start_locations, RandomWiggleUnitLocationsConfig):
            unit_start_locations = self.unit_start_locations.unit_locations
        else:
            unit_start_locations = self.unit_start_locations

        for i, unit_start_location in enumerate(unit_start_locations):
            unit_start_location = np.array([eval_fodp(coord, rng) for coord in unit_start_location])
            unit_start_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            if unit_start_quats is not None:
                unit_start_quat = np.array(unit_start_quats[i])
            elif self.unit_start_quats is not None:
                unit_start_quat = np.array(self.unit_start_quats[i])

            if self.num_units > 1:
                shade = 1.0 - (i / (self.num_units - 1))
            else:
                shade = 1.0
            body_rgba = (shade, shade, shade, 0.35)

            unit = init_unit(
                body_radius=self.body_radius,
                leg_length=self.leg_length,
                leg_radius=self.leg_radius,
                segment_1_ratio=self.segment_1_ratio,
                minimal_contacts=self.minimal_contacts,
                use_cylinders=self.use_cylinders,
                hinge_range=self.hinge_range,
                hinge_armature=self.hinge_armature,
                hinge_damping=self.hinge_damping,
                hinge_frictionloss=self.hinge_frictionloss,
                unit_config=self.config.unit_config,
                body_rgba=body_rgba,
            )
            unit.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)
            (worldbody
             .add_frame(pos=unit_start_location, quat=unit_start_quat)
             .attach_body(unit, self.config.unit_prefixes[i], ''))

        return spec

    def reset_swarm(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            rng: np.random.Generator,
            swarm_start_location: np.ndarray,
            inactive_unit_positions: np.ndarray
    ) -> tuple[SwarmConnections, Optional[np.ndarray]]:
        connections = SwarmConnections(self.config)

        start_locations: list[Tuple3[float]]
        start_quats: list[tuple[float, float, float, float]] | None = None
        if isinstance(self.unit_start_locations, PoissonDiscUnitLocationsConfig):
            start_locations = self._generate_poisson_disc_start_locations(rng)
        elif isinstance(self.unit_start_locations, PreConnectedUnitLocationsConfig):
            start_locations, start_quats, connections = self._generate_preconnected_swarm(rng)
        elif isinstance(self.unit_start_locations, RandomWiggleUnitLocationsConfig):
            start_locations = []
            for unit_loc, unit_wiggle in zip(
                    self.unit_start_locations.unit_locations,
                    self.unit_start_locations.wiggle_params
            ):
                # noinspection PyTypeChecker
                start_locations.append(tuple(
                    coord + eval_fodp(wiggle, rng)
                    for coord, wiggle in zip(unit_loc, unit_wiggle)
                ))
        else:
            start_locations = [
                eval_fodp_3d(unit_loc, rng)
                for unit_loc in self.unit_start_locations
            ]

        for i, unit_start_location in enumerate(start_locations):
            qpos_adr, dof_adr = self._get_unit_main_body_addresses(model, i)

            data.qpos[qpos_adr:qpos_adr + 3] = swarm_start_location + np.array(unit_start_location)

            if start_quats is not None:
                data.qpos[qpos_adr + 3:qpos_adr + 7] = start_quats[i]
            elif self.randomize_unit_orientations:
                data.qpos[qpos_adr + 3:qpos_adr + 7] = random_quat_shoemake(rng)
            elif self.unit_start_quats is not None:
                data.qpos[qpos_adr + 3:qpos_adr + 7] = self.unit_start_quats[i]
            else:
                data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]

            data.qvel[dof_adr:dof_adr+6] = 0

        for i in range(len(start_locations), self.num_units):
            qpos_adr, dof_adr = self._get_unit_main_body_addresses(model, i)

            data.qpos[qpos_adr:qpos_adr + 3] = inactive_unit_positions[i]
            data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]
            data.qvel[dof_adr:dof_adr + 6] = 0

        data.ctrl[:] = 0

        units_active_mask: Optional[np.ndarray] = None
        if self.can_have_inactive_units:
            units_active_mask = np.zeros(self.num_units, dtype=bool)
            units_active_mask[:len(start_locations)] = True

        return connections, units_active_mask

    def _get_unit_main_body_addresses(self, model: mujoco.MjModel, unit_idx: int):
        body_name = f"{self.config.unit_prefixes[unit_idx]}-main_body"
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        jnt_adr = model.body_jntadr[body_id]
        qpos_adr = model.jnt_qposadr[jnt_adr]
        dof_adr = model.jnt_dofadr[jnt_adr]
        return qpos_adr, dof_adr

    def _generate_poisson_disc_start_locations(
            self,
            rng: np.random.Generator,
            eps: float = 1e-6,
            force_full: bool = False
    ) -> list[tuple[float, float, float]]:
        random_config: PoissonDiscUnitLocationsConfig = self.unit_start_locations
        num_units = random_config.num_units
        pairwise_distance = 2.1 * self.max_unit_extent
        max_distance = random_config.max_radius

        if force_full or random_config.num_unit_probs is None:
            num_active_units = num_units
        else:
            unit_probs = random_config.num_unit_probs
            num_active_units = rng.choice(list(unit_probs.keys()), p=list(unit_probs.values()))
        if num_active_units < 1:
            raise ValueError("num_active_units must be >= 1")

        points = np.zeros((num_active_units, 2))
        points_found = 1
        rejected_count = 0

        while points_found < num_active_units:
            source_point = points[rng.choice(points_found)]

            angle = rng.random() * 2 * np.pi
            proposal_x = np.cos(angle) * pairwise_distance
            proposal_y = np.sin(angle) * pairwise_distance

            target_point = source_point + np.array([proposal_x, proposal_y])

            if float(np.linalg.norm(target_point)) > max_distance:
                rejected_count += 1
                continue

            distances = np.linalg.norm(points[:points_found] - target_point, axis=-1)

            if np.any(distances + eps < pairwise_distance):
                rejected_count += 1
            else:
                points[points_found] = target_point
                points_found += 1

            if rejected_count > 1000:
                raise RuntimeError(f"Failed to generate random start locations after 1000 attempts: {random_config=}")

        if random_config.center:
            mean = points.mean(axis=0, keepdims=True)
            points -= mean

            counter = 0
            while np.any(np.linalg.norm(points, axis=-1) > max_distance) and counter < 4:
                points += mean / 4
                counter += 1


        return [(float(p[0]), float(p[1]), random_config.z_pos) for p in points]


    def _generate_preconnected_swarm(
            self,
            rng: np.random.Generator,
            force_full: bool = False
    ) -> tuple[
        list[tuple[float, float, float]],
        list[tuple[float, float, float, float]],
        SwarmConnections
    ]:
        random_config: PreConnectedUnitLocationsConfig = self.unit_start_locations
        unit_config = self.config.unit_config
        num_units = random_config.num_units
        max_radius = random_config.max_radius

        active_pool_seeds = self.get_active_pool_seeds()
        if active_pool_seeds is not None:
            seed = int(rng.choice(active_pool_seeds))
            rng = np.random.default_rng(seed)

        if force_full or random_config.num_unit_probs is None:
            num_active_units = num_units
        else:
            unit_probs = random_config.num_unit_probs
            num_active_units = rng.choice(list(unit_probs.keys()), p=list(unit_probs.values()))
        if num_active_units < 1:
            raise ValueError("num_active_units must be >= 1")

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

        positions = np.zeros((num_active_units, 3), dtype=float)
        quats = np.zeros((num_active_units, 4), dtype=float)
        rot_mats = np.zeros((num_active_units, 3, 3), dtype=float)

        connections = SwarmConnections(self.config)

        connector_distance = self.body_radius + self.leg_length
        min_center_distance = 2 * self.max_unit_extent

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

            unit1, conn1 = existing_choices[rng.integers(len(existing_choices))]
            new_available = available_connectors[locations_found]
            conn2 = new_available[rng.integers(len(new_available))]

            rot1 = rot_mats[unit1]
            conn1_rot = rot1 @ limb_rot_mats[conn1]
            z1 = conn1_rot[:, 2]
            pos1 = positions[unit1] + z1 * connector_distance

            z2 = -z1
            pos2 = pos1 + z1 * connector_distance


            if rejected_count > 100:
                logger.warning(f'{rejected_count = }')

            if float(np.linalg.norm(pos2)) > max_radius:
                rejected_count += 1
                if rejected_count > 1000:
                    raise RuntimeError(
                        f"Failed to generate preconnected swarm after 1000 attempts: {random_config=}"
                    )
                continue

            if locations_found > 0:
                distances = np.linalg.norm(positions[:locations_found] - pos2, axis=1)
                if np.any(distances + 1e-6 < min_center_distance):
                    rejected_count += 1
                    if rejected_count > 1000:
                        raise RuntimeError(
                            f"Failed to generate preconnected swarm after 1000 attempts: {random_config=}"
                        )
                    continue

            twist = self.config.sample_connection_twist(rng)
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
            if random_config.unconnected_prob <= 0 or rng.random() >= random_config.unconnected_prob:
                connections.connect(unit1, conn1, locations_found, conn2, twist)

            available_connectors[unit1].remove(conn1)
            available_connectors[locations_found].remove(conn2)
            locations_found += 1

            rejected_count = 0


        if random_config.center:
            mean = positions.mean(axis=0, keepdims=True)
            positions -= mean

            counter = 0
            while np.any(np.linalg.norm(positions, axis=-1) > max_radius) and counter < 4:
                positions += mean / 4
                counter += 1

        positions[:, 2] += random_config.z_pos

        pos_list = [tuple(map(float, pos)) for pos in positions]
        quat_list = [tuple(map(float, quat)) for quat in quats]

        return pos_list, quat_list, connections


