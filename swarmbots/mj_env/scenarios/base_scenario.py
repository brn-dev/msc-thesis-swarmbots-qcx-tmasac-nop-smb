import abc
import math
from copy import deepcopy
from typing import Any, Iterable, TypedDict, Literal, NotRequired, Optional

import mujoco
import numpy as np
from gymnasium import spaces
from mujoco import MjsBody

from swarmbots.utils.connector_actions import connector_action_space
import swarmbots.mj_env.mujoco_utils as mj_utils
from swarmbots.learn.performance_timer import PerformanceTimer
from swarmbots.mj_env.float_or_dist_params import FloatOrDistParams, eval_fodp_2d
from swarmbots.mj_env.quat_rot6d import quat_to_rot6d
from swarmbots.mj_env.swarm.base_swarm import BaseSwarm
from swarmbots.mj_env.swarm.swarm_connections import SwarmConnections

class SwarmObsDict(TypedDict):
    local_obs: np.ndarray  # shape (n_unit, n_obs_per_unit)
    global_obs: np.ndarray  # shape (n_global_features,)
    hidden_local_vars: np.ndarray  # shape (n_unit, n_hidden_local_vars)
    hidden_global_vars: np.ndarray  # shape (n_hidden_global_vars,)
    agent_mask: NotRequired[Optional[np.ndarray]]  # shape (n_unit,), type bool

class SwarmActDict(TypedDict):
    actuators: np.ndarray  # shape (n_unit, n_actuators_per_unit), type float
    connectors: np.ndarray  # shape (n_unit, n_connectors_per_unit), type bool or float

class RewardWeights(TypedDict, total=False):
    progress_reward_weight: float
    guidance_reward_weight: float

    units_without_connections_reward_weight: float

class RewardWeightsUpdateResult(TypedDict):
    ok: bool
    error: str | None
    unknown_keys: list[str]
    updated_keys: list[str]

DEFAULT_GEOM_FRICTION: tuple[float, float, float] = (1.0, 0.005, 0.0001)
HIGH_FRICTION_SLIDING_THRESHOLD: float = 2.0
DEFAULT_INACTIVE_AREA_LOCATION: tuple[float, float, float] = (50.0, 0.0, 0.1)

_REWARD_WEIGHT_KEYS: frozenset[str] = frozenset(RewardWeights.__annotations__.keys())


def _validate_geom_friction(
        friction: float | Iterable[float] | None,
) -> tuple[float, float, float] | None:
    if friction is None:
        return None
    if isinstance(friction, (int, float)):
        return (float(friction), DEFAULT_GEOM_FRICTION[1], DEFAULT_GEOM_FRICTION[2])
    friction_tuple = tuple(float(x) for x in friction)
    if len(friction_tuple) != 3:
        raise ValueError(f"Expected friction to be a float or 3-tuple, got {friction_tuple!r}")
    return friction_tuple


class BaseScenario(abc.ABC):

    def __init__(
        self,
            swarm: BaseSwarm,
            timestep: float,
            action_repeat: int,
            actuator_strength: float,
            progress_reward_weight: float,
            guidance_reward_weight: float,
            units_without_connections_reward_weight: float,
            potential_reward_discount_factor: float,
            include_connectors_xpos_in_obs: bool,
            include_connectors_xquat_in_obs: bool,
            quat_rot6d_representation: bool,
            connection_dist_threshold: float,
            connection_angle_threshold: float,
            disconnect_potential_threshold: float,
            friction: float | Iterable[float] | None,
            force_elliptic_cone: bool,
            reset_settle_time: float,
            reset_settle_timestep_scale: float,
            swarm_start_x: FloatOrDistParams,
            swarm_start_y: FloatOrDistParams,
            randomize_initial_swarm_z_rotation: bool,
            inactive_area_location: Iterable[float] | None,
            continuous_connector_actions: bool,
            seed: int | None,
            _reset_in_init: bool = True,
    ) -> None:
        self.seed = seed
        self.rng: np.random.Generator = np.random.default_rng(seed)

        self.swarm = swarm
        self.num_units = swarm.config.num_units
        self.limbs_per_unit = swarm.config.limbs_per_unit
        self.timestep = float(timestep)
        if self.timestep <= 0.0:
            raise ValueError(f"Expected timestep > 0, got {self.timestep}")
        self.action_repeat = int(action_repeat)
        if self.action_repeat <= 0:
            raise ValueError(f"Expected action_repeat > 0, got {self.action_repeat}")
        self.actuator_strength = actuator_strength
        self.connection_dist_threshold = connection_dist_threshold
        self.connection_angle_threshold = connection_angle_threshold
        self.disconnect_potential_threshold = disconnect_potential_threshold
        self.continuous_connector_actions = bool(continuous_connector_actions)
        self.swarm_start_x = swarm_start_x
        self.swarm_start_y = swarm_start_y
        self.randomize_initial_swarm_z_rotation = bool(randomize_initial_swarm_z_rotation)
        if reset_settle_time < 0:
            raise ValueError(f"Expected reset_settle_time >= 0, got {reset_settle_time}")
        self.reset_settle_time = reset_settle_time
        if reset_settle_timestep_scale <= 0:
            raise ValueError(
                f"Expected reset_settle_timestep_scale > 0, got {reset_settle_timestep_scale}"
            )
        self.reset_settle_timestep_scale = reset_settle_timestep_scale

        self.reward_weights: RewardWeights = {
            "progress_reward_weight": progress_reward_weight,
            "guidance_reward_weight": guidance_reward_weight,
            "units_without_connections_reward_weight": units_without_connections_reward_weight,
        }
        self.potential_reward_discount_factor = float(potential_reward_discount_factor)
        self.include_connectors_xpos_in_obs = include_connectors_xpos_in_obs
        self.include_connectors_xquat_in_obs = include_connectors_xquat_in_obs
        self.quat_rot6d_representation = quat_rot6d_representation
        self.friction = _validate_geom_friction(friction)
        self.force_elliptic_cone = force_elliptic_cone

        if inactive_area_location is None:
            inactive_area_location = DEFAULT_INACTIVE_AREA_LOCATION
        self.inactive_area_location = np.asarray(inactive_area_location, dtype=float)
        self.inactive_unit_positions = self._generate_inactive_unit_positions(swarm, self.inactive_area_location)

        self.spec = self.create_scenario_spec()
        self.dummy_model, self.dummy_data = self.build()

        unit_prefixes = self.swarm.config.unit_prefixes
        self._qpos_indices = np.array(
            [mj_utils.qpos_indices_for_prefix(self.dummy_model, prefix) for prefix in unit_prefixes],
            dtype=int,
        )
        self._qvel_indices = np.array(
            [mj_utils.dof_indices_for_prefix(self.dummy_model, prefix) for prefix in unit_prefixes],
            dtype=int,
        )
        self._ctrl_indices = np.array(
            [mj_utils.ctrl_indices_for_prefix(self.dummy_model, prefix) for prefix in unit_prefixes],
            dtype=int,
        )

        num_units = self.num_units
        num_connectors = self.limbs_per_unit
        self._connector_body_indices = np.zeros((num_units, num_connectors), dtype=int)
        for u in range(num_units):
            for c in range(num_connectors):
                conn_body_name = self.swarm.config.get_connector_name(u, c)
                self._connector_body_indices[u, c] = mujoco.mj_name2id(
                    self.dummy_model, mujoco.mjtObj.mjOBJ_BODY, conn_body_name
                )

        num_eq_variants = self.swarm.config.num_connection_eq_variants
        self._eq_indices = np.full(
            (num_units, num_connectors, num_units, num_connectors, num_eq_variants),
            -1,
            dtype=int,
        )

        for u1 in range(num_units - 1):
            for u2 in range(u1 + 1, num_units):
                for c1 in range(num_connectors):
                    for c2 in range(num_connectors):
                        for twist_idx in range(num_eq_variants):
                            eq_name = self.swarm.config.get_eq_variant_name(u1, c1, u2, c2, twist_idx)
                            eq_id = mujoco.mj_name2id(self.dummy_model, mujoco.mjtObj.mjOBJ_EQUALITY, eq_name)
                            if eq_id == -1:
                                raise ValueError(f"Equality constraint {eq_name} not found")
                            self._eq_indices[u1, c1, u2, c2, twist_idx] = eq_id
                            self._eq_indices[u2, c2, u1, c1, twist_idx] = eq_id

        self._unit_body_ids: list[np.ndarray] = []
        self._unit_geom_ids: list[np.ndarray] = []
        self._unit_geom_contype: list[np.ndarray] = []
        self._unit_geom_conaffinity: list[np.ndarray] = []
        self._unit_body_gravcomp: list[np.ndarray] = []
        for prefix in unit_prefixes:
            body_ids = np.asarray(mj_utils.body_ids_for_prefix(self.dummy_model, prefix), dtype=int)
            if body_ids.size == 0:
                geom_ids = np.zeros(0, dtype=int)
            else:
                geom_ids = np.nonzero(np.isin(self.dummy_model.geom_bodyid, body_ids))[0].astype(int)
            self._unit_body_ids.append(body_ids)
            self._unit_geom_ids.append(geom_ids)
            self._unit_geom_contype.append(self.dummy_model.geom_contype[geom_ids].copy())
            self._unit_geom_conaffinity.append(self.dummy_model.geom_conaffinity[geom_ids].copy())
            self._unit_body_gravcomp.append(self.dummy_model.body_gravcomp[body_ids].copy())

        if _reset_in_init:
            self._dummy_state, self._dummy_connections = self.reset_scenario(self.dummy_model, self.dummy_data)
        else:
            self._dummy_state, self._dummy_connections = None, None

    def get_settings(self) -> dict[str, Any]:
        return {
            'swarm': self.swarm.get_settings(),
            'timestep': self.timestep,
            'action_repeat': self.action_repeat,
            'actuator_strength': self.actuator_strength,
            'reward_weights': dict(self.reward_weights),
            'include_connectors_xpos_in_obs': self.include_connectors_xpos_in_obs,
            'include_connectors_xquat_in_obs': self.include_connectors_xquat_in_obs,
            'quat_rot6d_representation': self.quat_rot6d_representation,
            'connection_dist_threshold': self.connection_dist_threshold,
            'connection_angle_threshold': self.connection_angle_threshold,
            'disconnect_potential_threshold': self.disconnect_potential_threshold,
            'continuous_connector_actions': self.continuous_connector_actions,
            'swarm_start_x': self.swarm_start_x,
            'swarm_start_y': self.swarm_start_y,
            'randomize_initial_swarm_z_rotation': self.randomize_initial_swarm_z_rotation,
            'friction': self.friction,
            'force_elliptic_cone': self.force_elliptic_cone,
            'seed': self.seed,
            'reset_settle_time': self.reset_settle_time,
            'reset_settle_timestep_scale': self.reset_settle_timestep_scale,
            'inactive_area_location': self.inactive_area_location,
            'potential_reward_discount_factor': self.potential_reward_discount_factor,
        }

    def potential_reward_delta(self, current_potential: Any, previous_potential: Any) -> Any:
        discount_factor = float(getattr(self, "potential_reward_discount_factor", 1.0))
        return (discount_factor * current_potential) - previous_potential

    def clone_for_worker_pool(self) -> "BaseScenario":
        """Clone a fully-built scenario without recompiling its MuJoCo spec."""
        clone = object.__new__(type(self))
        for attr_name, attr_value in self.__dict__.items():
            if attr_name in {"rng", "spec", "dummy_model", "dummy_data"}:
                continue
            clone.__dict__[attr_name] = deepcopy(attr_value)

        clone_seed = int(
            self.rng.integers(
                low=0,
                high=np.iinfo(np.uint64).max,
                dtype=np.uint64,
            )
        )
        clone.rng = np.random.default_rng(clone_seed)
        clone.spec = None
        clone.dummy_model = deepcopy(self.dummy_model)
        clone.dummy_data = deepcopy(self.dummy_data)
        return clone

    def get_reward_weights(self) -> RewardWeights:
        return self.reward_weights

    @abc.abstractmethod
    def _create_scenario_spec(self) -> mujoco.MjSpec:
        raise NotImplementedError()

    @abc.abstractmethod
    def compute_progress_reward(
            self,
            data: mujoco.MjData,
            state: dict,
    ) -> float:
        raise NotImplementedError()

    def evaluate_step(
            self,
            action: SwarmActDict,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            state: dict,
            connections: SwarmConnections
    ) -> tuple[float, bool]:
        """
        :return: (reward, done)
        """
        units_active_mask = state.get("units_active_mask")
        if units_active_mask is not None:
            self._enforce_inactive_units_state(model, data, units_active_mask)

        progress_reward = self.compute_progress_reward(data, state)

        guidance_reward = self.compute_guidance_reward(data, action, state, connections)
        state['guidance_reward'] = guidance_reward

        weighted_progress_reward = progress_reward * self.reward_weights['progress_reward_weight']
        weighted_guidance_reward = guidance_reward * self.reward_weights['guidance_reward_weight']
        state['weighted_progress_reward'] = weighted_progress_reward
        state['weighted_guidance_reward'] = weighted_guidance_reward
        state['reward_terms'] = {
            'progress': weighted_progress_reward,
            'guidance': weighted_guidance_reward,
        }

        return weighted_progress_reward + weighted_guidance_reward, False

    def create_scenario_spec(self) -> mujoco.MjSpec:
        spec = mujoco.MjSpec()
        spec.compiler.degree = 0
        worldbody: MjsBody = spec.worldbody

        seed = self.seed if self.seed is not None else 42
        spec_rng = np.random.default_rng(seed)
        swarm_spec = self.swarm.create_swarm_spec(rng=spec_rng)

        swarm_site = worldbody.add_site(pos=self.get_swarm_start_location(), name='swarm_site')
        spec.attach(swarm_spec, '', site=swarm_site)

        scenario_site = worldbody.add_site(pos=[0, 0, 0], name='scenario_site')
        spec.attach(self._create_scenario_spec(), '', site=scenario_site)

        self.add_cameras(spec)

        return spec

    def add_cameras(self, spec: mujoco.MjSpec):
        desired_pos_world = np.array([5.0, 0.0, 3.0], dtype=float)
        desired_quat_world = np.empty(4, dtype=float)
        desired_euler = np.array([0.0, np.pi / 3, np.pi / 2], dtype=float)
        mujoco.mju_euler2Quat(desired_quat_world, desired_euler, "xyz")

        for unit_prefix in self.swarm.config.unit_prefixes:
            unit1_body = spec.body(unit_prefix + '-main_body')
            body_quat = np.array(unit1_body.quat, dtype=float)
            if unit1_body.frame is not None:
                frame_quat = np.array(unit1_body.frame.quat, dtype=float)
                combined = np.empty(4, dtype=float)
                mujoco.mju_mulQuat(combined, frame_quat, body_quat)
                body_quat = combined
            body_quat_inv = body_quat.copy()
            body_quat_inv[1:] *= -1.0

            pos_body = np.empty(3, dtype=float)
            mujoco.mju_rotVecQuat(pos_body, desired_pos_world, body_quat_inv)
            cam_quat = np.empty(4, dtype=float)
            mujoco.mju_mulQuat(cam_quat, body_quat_inv, desired_quat_world)

            unit1_body.add_camera(
                pos=pos_body.tolist(), quat=cam_quat.tolist(),
                mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACK)
            
    def build(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        model = self.spec.compile()
        model.opt.timestep = self.timestep
        if self.friction is not None:
            model.geom_friction[:] = np.asarray(self.friction, dtype=float)

            # Elliptic cone is more stable for high friction forces but slower to compute.
            # If a high friction is set while using pyramidal cone is used, geoms can fall through the plane.
            sliding = float(self.friction[0])
            if self.force_elliptic_cone or (
                sliding >= HIGH_FRICTION_SLIDING_THRESHOLD
                and int(model.opt.cone) == int(mujoco.mjtCone.mjCONE_PYRAMIDAL)
            ):
                model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        data = mujoco.MjData(model)
        return model, data

    def add_render_geoms(self, scene: mujoco.MjvScene) -> None:
        return

    def get_swarm_start_location(self) -> np.ndarray:
        start_xy = eval_fodp_2d((self.swarm_start_x, self.swarm_start_y), self.rng)
        return np.array(
            [start_xy[0], start_xy[1], self.swarm.max_unit_extent * 1.1],
            dtype=float,
        )

    def reset_scenario(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            settle: bool = True,
    ) -> tuple[dict, SwarmConnections]:
        mujoco.mj_resetData(model, data)

        state = dict()
        swarm_start_location = self.get_swarm_start_location()
        state["swarm_start_location"] = swarm_start_location.copy()
        connections, units_active_mask = self.swarm.reset_swarm(
            model,
            data,
            self.rng,
            swarm_start_location,
            self.inactive_unit_positions
        )
        initial_swarm_z_rotation = self._sample_initial_swarm_z_rotation()
        state["initial_swarm_z_rotation"] = initial_swarm_z_rotation
        self._apply_initial_swarm_z_rotation(
            data=data,
            units_active_mask=units_active_mask,
            swarm_start_location=swarm_start_location,
            angle=initial_swarm_z_rotation,
        )
        if self.swarm.can_have_inactive_units:
            if units_active_mask is None:
                units_active_mask = np.ones(self.num_units, dtype=bool)
            self._apply_units_active_mask(model, data, units_active_mask)
        else:
            units_active_mask = None

        for (u1, c1, u2, c2), angle in zip(*connections.get_active_connections()):
            self._activate_equality_constraint(model, data, u1, c1, u2, c2, angle)

        state['units_active_mask'] = units_active_mask

        if settle:
            self.settle_reset(
                model=model,
                data=data,
                state=state,
            )

        return state, connections

    def _sample_initial_swarm_z_rotation(self) -> float:
        if not self.randomize_initial_swarm_z_rotation:
            return 0.0
        return float(self.rng.uniform(0.0, 2.0 * math.pi))

    def _apply_initial_swarm_z_rotation(
            self,
            *,
            data: mujoco.MjData,
            units_active_mask: np.ndarray | None,
            swarm_start_location: np.ndarray,
            angle: float,
    ) -> None:
        if angle == 0.0:
            return

        active_mask = (
            np.ones(self.num_units, dtype=bool)
            if units_active_mask is None
            else units_active_mask
        )
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        yaw_quat = np.array([math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)], dtype=float)

        for unit_idx, is_active in enumerate(active_mask):
            if not is_active:
                continue
            qpos_indices = self._qpos_indices[unit_idx]
            pos_indices = qpos_indices[:3]
            pos_offset = data.qpos[pos_indices] - swarm_start_location
            data.qpos[pos_indices[0]] = swarm_start_location[0] + cos_angle * pos_offset[0] - sin_angle * pos_offset[1]
            data.qpos[pos_indices[1]] = swarm_start_location[1] + sin_angle * pos_offset[0] + cos_angle * pos_offset[1]

            quat_indices = qpos_indices[3:7]
            rotated_quat = np.empty(4, dtype=float)
            mujoco.mju_mulQuat(rotated_quat, yaw_quat, data.qpos[quat_indices].copy())
            data.qpos[quat_indices] = rotated_quat

    def settle_reset(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            state: dict,
    ) -> None:
        if self.reset_settle_time <= 0:
            return

        remaining_time = self.reset_settle_time - data.time
        if remaining_time <= 0:
            return
        original_timestep = model.opt.timestep
        effective_timestep = original_timestep * self.reset_settle_timestep_scale
        nstep = math.ceil(remaining_time / effective_timestep)
        if self.reset_settle_timestep_scale != 1.0:
            model.opt.timestep = effective_timestep
        try:
            mujoco.mj_step(model, data, nstep=int(nstep))
        finally:
            if model.opt.timestep != original_timestep:
                model.opt.timestep = original_timestep

    def get_obs(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            state: dict,
            connections: SwarmConnections
    ) -> SwarmObsDict:
        """
        Local observations contain for each unit:

        * qpos - hinge angles are encoded via sin/cos pairs
        * qvel
        * connector obs - containing is-connected status, twist angle sin/cos and disconnect potentials
        * [Optionally] xpos of connectors
        * [Optionally] xquat of connectors
        """
        qpos = data.qpos[self._qpos_indices]
        qvel = data.qvel[self._qvel_indices]

        # encoding hinge angles to sin and cos
        free_joint_qpos = qpos[:, :7]
        free_joint_cartesian = free_joint_qpos[:, :3]
        free_joint_quats = free_joint_qpos[:, 3:]
        if self.quat_rot6d_representation:
            free_joint_quats = quat_to_rot6d(free_joint_quats, axis=-1)

        hinge_qpos = qpos[:, 7:]
        hinge_sin = np.sin(hinge_qpos)
        hinge_cos = np.cos(hinge_qpos)
        hinge_obs = np.stack([hinge_sin, hinge_cos], axis=-1).reshape(self.num_units, -1)
        qpos_obs = np.concatenate([free_joint_cartesian, free_joint_quats, hinge_obs], axis=1)

        # connector obs
        is_active = connections.get_is_active_mask()
        active_indices = np.where(is_active)
        non_active_indices = np.where(np.logical_not(is_active))
        twist_angles = connections.twist_angles
        disconnect_potentials = connections.disconnect_potentials

        connector_obs = np.zeros((self.num_units, self.limbs_per_unit, 5), dtype=float)
        connector_obs[non_active_indices[0], non_active_indices[1], 0] = 1
        connector_obs[active_indices[0], active_indices[1], 1] = 1
        connector_obs[active_indices[0], active_indices[1], 2] = np.sin(twist_angles[is_active])
        connector_obs[active_indices[0], active_indices[1], 3] = np.cos(twist_angles[is_active])
        connector_obs[active_indices[0], active_indices[1], 4] = disconnect_potentials[is_active]
        connector_obs = connector_obs.reshape((self.num_units, -1))

        obs_list = [qpos_obs, qvel, connector_obs]

        if self.include_connectors_xpos_in_obs:
            conn_xpos = data.xpos[self._connector_body_indices]
            conn_xpos = conn_xpos.reshape((self.num_units, -1))
            obs_list.append(conn_xpos)

        if self.include_connectors_xquat_in_obs:
            conn_xquat = data.xquat[self._connector_body_indices]
            if self.quat_rot6d_representation:
                conn_xquat = quat_to_rot6d(conn_xquat, axis=-1)
            conn_xquat = conn_xquat.reshape((self.num_units, -1))
            obs_list.append(conn_xquat)

        units_active_mask = state.get('units_active_mask')
        active_units_count = self.num_units
        if units_active_mask is not None:
            active_units_count = int(np.asarray(units_active_mask, dtype=bool).sum())

        obs: SwarmObsDict = {
            'local_obs': np.concatenate(obs_list, axis=1),
            'global_obs': np.empty(0, dtype=float),
            'hidden_local_vars': np.zeros((self.num_units, 0), dtype=float),
            'hidden_global_vars': np.array([active_units_count], dtype=float),
        }
        if self.swarm.can_have_inactive_units:
            if units_active_mask is None:
                raise ValueError("units_active_mask must be set when can_have_inactive_units is True")
            obs['agent_mask'] = units_active_mask.copy()
        return obs

    def apply_action(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            action: SwarmActDict,
            state: dict,
            connections: SwarmConnections
    ) -> None:
        agent_mask = state['units_active_mask']
        actuators_action = np.asarray(action['actuators'], dtype=float)
        if agent_mask is not None:
            actuators_action = actuators_action.copy()
            actuators_action[~agent_mask] = 0.0
        data.ctrl[self._ctrl_indices] = actuators_action * self.actuator_strength

        if self.continuous_connector_actions:
            self._apply_continuous_connector_action(
                model=model,
                data=data,
                action=action,
                state=state,
                connections=connections,
            )
            return

        connectors_action = np.asarray(action['connectors'], dtype=bool)
        if agent_mask is not None:
            connectors_action = connectors_action.copy()
            connectors_action[~agent_mask] = False

        currently_active_mask = connections.get_is_active_mask()
        newly_activated_mask = np.logical_and(connectors_action, np.logical_not(currently_active_mask))
        newly_deactivated_mask = np.logical_and(np.logical_not(connectors_action), currently_active_mask)

        self.try_connect(
            model,
            data,
            connections,
            newly_activated_mask
        )

        deactivation_mask = connections.update_disconnect_potentials(
            currently_active_mask,
            newly_deactivated_mask,
            self.disconnect_potential_threshold
        )
        self.disconnect(data, connections, deactivation_mask)

    def _apply_continuous_connector_action(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            action: SwarmActDict,
            state: dict,
            connections: SwarmConnections,
    ) -> None:
        agent_mask = state['units_active_mask']
        connectors_action = np.asarray(action['connectors'], dtype=np.float32)
        if agent_mask is not None:
            connectors_action = connectors_action.copy()
            connectors_action[~agent_mask] = -1.0

        currently_active_mask = connections.get_is_active_mask()
        newly_activated_mask = np.logical_and(connectors_action > 0.0, np.logical_not(currently_active_mask))

        self.try_connect(
            model,
            data,
            connections,
            newly_activated_mask
        )

        deactivation_mask = connections.update_disconnect_potentials_continuous(
            currently_active_mask,
            connectors_action,
            self.disconnect_potential_threshold,
        )
        self.disconnect(data, connections, deactivation_mask)


    def get_obs_space(self):
        obs = self.get_obs(self.dummy_model, self.dummy_data, self._dummy_state, self._dummy_connections)
        obs_space = spaces.Dict({
            'local_obs': spaces.Box(
                low=-np.inf, high=np.inf, shape=obs['local_obs'].shape, dtype=np.float32
            ),
            'global_obs': spaces.Box(
                low=-np.inf, high=np.inf, shape=obs['global_obs'].shape, dtype=np.float32
            ),
            'hidden_local_vars': spaces.Box(
                low=-np.inf, high=np.inf, shape=obs['hidden_local_vars'].shape, dtype=np.float32
            ),
            'hidden_global_vars': spaces.Box(
                low=-np.inf, high=np.inf, shape=obs['hidden_global_vars'].shape, dtype=np.float32
            ),
        })
        if self.swarm.can_have_inactive_units:
            obs_space['agent_mask'] = spaces.MultiBinary((self.num_units,))
        return obs_space

    def get_actuator_action_shape(self):
        return self._ctrl_indices.shape

    def get_connector_action_shape(self):
        return self.swarm.config.num_units, self.swarm.config.limbs_per_unit

    def get_action_space(self):
        return spaces.Dict({
            'actuators': spaces.Box(
                low=-1, high=1, shape=self.get_actuator_action_shape(), dtype=np.float32
            ),
            'connectors': connector_action_space(
                self.get_connector_action_shape(),
                continuous=self.continuous_connector_actions,
            )
        })

    def try_connect(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            connections: SwarmConnections,
            newly_activated_mask: np.ndarray
    ) -> None:
        """
        Considers newly activated connectors (those that are not connected but are activated by the policy) and connects
        the closest pairs if the following criteria are fulfilled:
        * They don't belong to the same unit
        * They are below a certain threshold in distance
        * Their z-axes are anti-aligned up to a threshold - meaning they face each other
        * They are in front of each other (positive relative z-distance)
        """
        activated_units, activated_connectors = np.nonzero(newly_activated_mask)
        num_activated = len(activated_units)
        if num_activated < 2:
            return

        body_ids = self._connector_body_indices[activated_units, activated_connectors]
        positions = data.xpos[body_ids]
        connector_mats = data.xmat[body_ids].reshape(num_activated, 3, 3)
        x_axes = connector_mats[:, :, 0]
        y_axes = connector_mats[:, :, 1]
        z_axes = connector_mats[:, :, 2]

        unused = np.ones(num_activated, dtype=bool)
        distance_threshold_sq = self.connection_dist_threshold ** 2

        for i in range(num_activated - 1):
            if not unused[i]:
                continue

            unit1 = int(activated_units[i])
            conn1 = int(activated_connectors[i])

            candidate_mask = unused[i + 1:] & (activated_units[i + 1:] != unit1)
            if not candidate_mask.any():
                continue

            candidate_indices = np.flatnonzero(candidate_mask) + i + 1
            rel_pos = positions[candidate_indices] - positions[i]
            dist_sq = np.einsum("ij,ij->i", rel_pos, rel_pos)
            close_enough_mask = dist_sq < distance_threshold_sq
            if not close_enough_mask.any():
                continue

            candidate_indices = candidate_indices[close_enough_mask]
            rel_pos = rel_pos[close_enough_mask]
            dist_sq = dist_sq[close_enough_mask]

            closest_to_farthest_order = np.argsort(dist_sq)
            candidate_indices = candidate_indices[closest_to_farthest_order]
            rel_pos = rel_pos[closest_to_farthest_order]

            z1 = z_axes[i]
            is_anti_aligned = z_axes[candidate_indices] @ z1 <= self.connection_angle_threshold
            is_in_front = rel_pos @ z1 >= 0.0
            valid_candidates = np.flatnonzero(is_anti_aligned & is_in_front)
            if valid_candidates.size == 0:
                continue

            j = int(candidate_indices[valid_candidates[0]])
            unit2 = int(activated_units[j])
            conn2 = int(activated_connectors[j])

            x1 = x_axes[i]
            y1 = y_axes[i]
            x2 = x_axes[j]
            twist = float(np.arctan2(np.dot(x2, y1), np.dot(x2, x1)))

            connections.connect(unit1, conn1, unit2, conn2, twist)
            self._activate_equality_constraint(model, data, unit1, conn1, unit2, conn2, twist)

            unused[j] = False

    def disconnect(
            self,
            data: mujoco.MjData,
            connections: SwarmConnections,
            deactivation_mask: np.ndarray,
    ) -> None:
        already_disconnected = np.zeros((self.num_units, self.limbs_per_unit), dtype=bool)

        disconnect_indices = np.stack(np.where(deactivation_mask)).T
        for unit, connector in disconnect_indices:
            if already_disconnected[unit, connector]:
                continue

            other_unit, other_connector = connections.disconnect(unit, connector)

            already_disconnected[unit, connector] = True
            already_disconnected[other_unit, other_connector] = True

            self._deactivate_equality_constraint(data, unit, connector, other_unit, other_connector)

    def _activate_equality_constraint(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            unit1: int,
            conn1: int,
            unit2: int,
            conn2: int,
            twist: float
    ):
        if self.swarm.config.uses_quantized_connection_twist:
            twist_idx = self.swarm.config.get_nearest_connection_twist_index(twist)
            eq_idx = self._eq_indices[unit1, conn1, unit2, conn2, twist_idx]
            data.eq_active[eq_idx] = 1
            return

        eq_idx = self._eq_indices[unit1, conn1, unit2, conn2, 0]
        data.eq_active[eq_idx] = 1
        mj_utils.apply_twist(model, eq_idx, twist)

    def _deactivate_equality_constraint(
            self,
            data: mujoco.MjData,
            unit1: int,
            conn1: int,
            unit2: int,
            conn2: int
    ):
        if self.swarm.config.uses_quantized_connection_twist:
            eq_indices = self._eq_indices[unit1, conn1, unit2, conn2]
            data.eq_active[eq_indices] = 0
            return

        eq_idx = self._eq_indices[unit1, conn1, unit2, conn2, 0]
        data.eq_active[eq_idx] = 0

    def compute_guidance_reward(
            self,
            data: mujoco.MjData,
            action: SwarmActDict,
            state: dict,
            connections: SwarmConnections
    ) -> float:
        reward = 0.0

        rw = self.reward_weights
        units_active_mask = state.get("units_active_mask", None)
        active_units_mask = None if units_active_mask is None else np.asarray(units_active_mask, dtype=bool)
        active_units_count = self.num_units if active_units_mask is None else int(active_units_mask.sum())

        connection_mask = connections.get_is_active_mask()
        if active_units_mask is not None:
            connection_mask = connection_mask[active_units_mask]
        num_units_without_connections = 0
        if active_units_count > 0:
            num_units_without_connections = np.logical_not(connection_mask).all(axis=1).sum()
        state['num_units_without_connections'] = num_units_without_connections
        units_without_connections_ratio = (
            num_units_without_connections / active_units_count if active_units_count > 0 else 0.0
        )
        reward += units_without_connections_ratio * rw['units_without_connections_reward_weight']

        return reward


    def update_reward_weights(self, reward_weights: RewardWeights) -> RewardWeightsUpdateResult:
        unknown = set(reward_weights.keys()) - _REWARD_WEIGHT_KEYS
        if unknown:
            return {
                "ok": False,
                "error": f"Unknown reward weight keys: {sorted(unknown)}.",
                "unknown_keys": sorted(unknown),
                "updated_keys": [],
            }

        updated_keys: list[str] = []
        new_reward_weights = dict(self.reward_weights)
        try:
            for key, value in reward_weights.items():
                new_reward_weights[key] = float(value)
                updated_keys.append(key)
        except Exception as e:
            return {
                "ok": False,
                "error": f"Failed to apply reward weights update: {type(e).__name__}: {e}",
                "unknown_keys": [],
                "updated_keys": [],
            }

        self.reward_weights = new_reward_weights
        return {"ok": True, "error": None, "unknown_keys": [], "updated_keys": updated_keys}

    def _apply_units_active_mask(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            units_active_mask: np.ndarray,
    ) -> None:
        active_mask = np.asarray(units_active_mask, dtype=bool)
        if active_mask.shape[0] != self.num_units:
            raise ValueError(f"Expected units_active_mask length {self.num_units}, got {active_mask.shape[0]}")

        for unit_idx in range(self.num_units):
            self._apply_unit_active_state(model, data, unit_idx, bool(active_mask[unit_idx]))

        mujoco.mj_forward(model, data)

    def _apply_unit_active_state(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            unit_idx: int,
            is_active: bool,
    ) -> None:
        geom_ids = self._unit_geom_ids[unit_idx]
        body_ids = self._unit_body_ids[unit_idx]
        dof_ids = self._qvel_indices[unit_idx]

        if is_active:
            if geom_ids.size > 0:
                model.geom_contype[geom_ids] = self._unit_geom_contype[unit_idx]
                model.geom_conaffinity[geom_ids] = self._unit_geom_conaffinity[unit_idx]
            if body_ids.size > 0:
                model.body_gravcomp[body_ids] = self._unit_body_gravcomp[unit_idx]
        else:
            if geom_ids.size > 0:
                model.geom_contype[geom_ids] = 0
                model.geom_conaffinity[geom_ids] = 0
            if body_ids.size > 0:
                model.body_gravcomp[body_ids] = 1.0
            if dof_ids.size > 0:
                data.qvel[dof_ids] = 0.0
                data.qacc[dof_ids] = 0.0

    def _enforce_inactive_units_state(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            units_active_mask: np.ndarray,
    ) -> None:
        inactive_mask = np.logical_not(np.asarray(units_active_mask, dtype=bool))
        if not inactive_mask.any():
            return
        qpos_indices = self._qpos_indices[inactive_mask]
        qvel_indices = self._qvel_indices[inactive_mask].ravel()
        if qpos_indices.size > 0:
            data.qpos[qpos_indices[:, :3]] = self.inactive_unit_positions[inactive_mask]
        if qvel_indices.size > 0:
            data.qvel[qvel_indices] = 0.0
        if qpos_indices.size > 0 or qvel_indices.size > 0:
            mujoco.mj_forward(model, data)

    @staticmethod
    def _generate_inactive_unit_positions(swarm: BaseSwarm, inactive_area_location: np.ndarray):
        num_units = swarm.config.num_units
        max_unit_extent = swarm.max_unit_extent

        num_units_sqrt = int(math.ceil(math.sqrt(num_units)))
        unit_spacing = max_unit_extent * 3

        inactive_unit_positions = np.zeros((num_units, 3), dtype=float)
        inactive_unit_positions[:, 2] = max_unit_extent

        for i in range(num_units):
            inactive_unit_positions[i, 0] = (i // num_units_sqrt) * unit_spacing
            inactive_unit_positions[i, 1] = (i % num_units_sqrt) * unit_spacing

        inactive_unit_positions[:, :2] -= inactive_unit_positions[:, :2].mean(axis=0)

        return inactive_area_location + inactive_unit_positions
