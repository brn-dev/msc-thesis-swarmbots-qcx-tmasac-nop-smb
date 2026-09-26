import mujoco
from typing import TypeAlias

from swarmbots.mj_env.swarm.swarm_config import get_connector_suffix
from swarmbots.mj_env.swarm.unit_config import UnitConfig, LimbConfig, LimbType

HingeJointParam: TypeAlias = float | tuple[float, ...]


def _limb_joint_specs(limb_type: LimbType) -> tuple[tuple[str, tuple[float, float, float]], ...]:
    if limb_type == LimbType.xy:
        return (
            ("hinge1x", (1.0, 0.0, 0.0)),
            ("hinge2y", (0.0, 1.0, 0.0)),
        )
    if limb_type == LimbType.zx:
        return (
            ("hinge1z", (0.0, 0.0, 1.0)),
            ("hinge2x", (1.0, 0.0, 0.0)),
        )
    if limb_type == LimbType.xyz:
        return (
            ("hinge1x", (1.0, 0.0, 0.0)),
            ("hinge2y", (0.0, 1.0, 0.0)),
            ("hinge3z", (0.0, 0.0, 1.0)),
        )
    raise NotImplementedError(limb_type)


def get_limb_segment_names(limb_idx: int, limb_config: LimbConfig) -> tuple[str, ...]:
    joint_specs = _limb_joint_specs(limb_config.type)
    return tuple(
        limb_config.name if joint_idx == 0 else f"-{limb_idx}-seg{joint_idx + 1}"
        for joint_idx in range(len(joint_specs))
    )


def _resolve_joint_params(value: HingeJointParam, n_joints: int) -> tuple[float, ...]:
    if isinstance(value, tuple):
        if len(value) != n_joints:
            raise ValueError(len(value), n_joints)
        return value
    scalar = float(value)
    return (scalar,) * n_joints


def init_unit(
    body_radius: float,
    leg_length: float,
    leg_radius: float,
    hinge_range: tuple[float | None, ...],
    unit_config: UnitConfig,
    body_rgba=(0.75, 0, 0, 0.1),
    segment_1_ratio: float = 0.1,
    hinge_armature: HingeJointParam = 0.0,
    hinge_damping: HingeJointParam = 0.0,
    hinge_frictionloss: HingeJointParam = 0.0,
    minimal_contacts: bool = True,  # makes short limb segments and tips non-collidable and excludes long limb segments from the same unit from collision
    use_cylinders: bool = False,  # if false, uses capsules instead of cylinders
) -> mujoco.MjsBody:
    if not 0.0 < segment_1_ratio < 1.0:
        raise ValueError(f"Expected segment_1_ratio in (0, 1), got {segment_1_ratio}")

    spec = mujoco.MjSpec()
    spec.compiler.degree = False

    worldbody = spec.worldbody

    # Main Body
    body = worldbody.add_body(name='-main_body', pos=[0, 0, 0])

    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[body_radius, 0, 0], # Sphere only uses first size param
        rgba=body_rgba
    )

    for i, limb_config in enumerate(unit_config):
        hip_pos = body_radius * limb_config.vec
        joint_specs = _limb_joint_specs(limb_config.type)
        joint_ranges = tuple(r if r is not None else 0.0 for r in hinge_range)
        joint_armatures = _resolve_joint_params(hinge_armature, len(joint_specs))
        joint_dampings = _resolve_joint_params(hinge_damping, len(joint_specs))
        joint_frictionlosses = _resolve_joint_params(hinge_frictionloss, len(joint_specs))

        limb_root = body.add_body(
            name=f'-{i}-limb_root',
            pos=hip_pos,
            zaxis=limb_config.vec
        )

        _build_limb(
            spec=spec,
            parent_body=limb_root,
            limb_idx=i,
            limb_config=limb_config,
            length=leg_length,
            radius=leg_radius,
            hinge_ranges=joint_ranges,
            short_segment_ratio=segment_1_ratio,
            joint_specs=joint_specs,
            joint_armatures=joint_armatures,
            joint_dampings=joint_dampings,
            joint_frictionlosses=joint_frictionlosses,
            minimal_contacts=minimal_contacts,
            use_cylinders=use_cylinders,
        )

    if minimal_contacts:
        _add_same_unit_long_segment_excludes(spec, unit_config)

    return body


def _add_same_unit_long_segment_excludes(spec: mujoco.MjSpec, unit_config: UnitConfig) -> None:
    main_body_name = "-main_body"
    long_segment_body_names: list[str] = []
    for limb_idx, limb_config in enumerate(unit_config):
        long_segment_body_names.extend(get_limb_segment_names(limb_idx, limb_config)[1:])

    for long_segment_body_name in long_segment_body_names:
        exclude = spec.add_exclude()
        exclude.bodyname1 = main_body_name
        exclude.bodyname2 = long_segment_body_name

    for body_idx1, body_name1 in enumerate(long_segment_body_names[:-1]):
        for body_name2 in long_segment_body_names[body_idx1 + 1:]:
            exclude = spec.add_exclude()
            exclude.bodyname1 = body_name1
            exclude.bodyname2 = body_name2


def _build_limb(
        spec: mujoco.MjSpec,
        parent_body: mujoco.MjsBody,
        limb_idx: int,
        limb_config: LimbConfig,
        length: float,
        radius: float,
        hinge_ranges: tuple[float, ...],
        short_segment_ratio: float,
        joint_specs: tuple[tuple[str, tuple[float, float, float]], ...],
        joint_armatures: tuple[float, ...],
        joint_dampings: tuple[float, ...],
        joint_frictionlosses: tuple[float, ...],
        minimal_contacts: bool,
        use_cylinders: bool,
):
    rgba = tuple(limb_config.rgba)
    if len(joint_specs) < 2:
        raise ValueError(f"Expected at least 2 joints per limb, got {len(joint_specs)}")

    segment_lengths = [length * short_segment_ratio]
    remaining_length = length * (1 - short_segment_ratio)
    segment_lengths.extend([remaining_length / (len(joint_specs) - 1)] * (len(joint_specs) - 1))
    segment_names = get_limb_segment_names(limb_idx, limb_config)
    geom_type = mujoco.mjtGeom.mjGEOM_CYLINDER if use_cylinders else mujoco.mjtGeom.mjGEOM_CAPSULE

    hinge_names: list[str] = []
    parent_segment = parent_body
    tip_offset = 0.0

    for joint_idx, ((joint_suffix, axis), seg_length, segment_name) in enumerate(
            zip(joint_specs, segment_lengths, segment_names, strict=True)
    ):
        segment_body = parent_segment.add_body(name=segment_name, pos=[0, 0, tip_offset])
        hinge_name = f'-{limb_idx}-{joint_suffix}'
        segment_body.add_joint(
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=list(axis),
            range=[-hinge_ranges[joint_idx], hinge_ranges[joint_idx]],
            name=hinge_name,
            armature=joint_armatures[joint_idx],
            damping=joint_dampings[joint_idx],
            frictionloss=joint_frictionlosses[joint_idx],
        )
        geom_kwargs: dict[str, int] = {}
        if minimal_contacts and joint_idx == 0:
            geom_kwargs = {"contype": 0, "conaffinity": 0}
        segment_body.add_geom(
            type=geom_type,
            fromto=[0, 0, 0, 0, 0, seg_length],
            size=[radius, 0, 0],
            rgba=rgba,
            **geom_kwargs,
        )
        hinge_names.append(hinge_name)
        parent_segment = segment_body
        tip_offset = seg_length

    connector: mujoco.MjsBody = parent_segment.add_body(
        name=get_connector_suffix(limb_idx),
        pos=[0, 0, tip_offset]
    )
    connector_geom_kwargs: dict[str, int] = {}
    if minimal_contacts:
        connector_geom_kwargs = {"contype": 0, "conaffinity": 0}
    connector.add_geom(
        type=geom_type,
        fromto=[0, 0, 0, 0, 0, -length * 0.02],
        size=[radius * 1.05, 0, 0],
        rgba=rgba,
        **connector_geom_kwargs,
    )

    for actuator_idx, hinge_name in enumerate(hinge_names):
        spec.add_actuator(
            target=hinge_name,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            name=f'-{limb_idx}-actuator{actuator_idx}'
        )
