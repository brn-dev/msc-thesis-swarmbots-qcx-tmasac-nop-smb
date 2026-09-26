import mujoco
import numpy as np

from swarmbots.mj_env.float_or_dist_params import BoundedDistParams, FloatOrDistParams


def add_line_geom(
    scene: mujoco.MjvScene,
    *,
    from_pos: tuple[float, float, float],
    to_pos: tuple[float, float, float],
    rgba: tuple[float, float, float, float],
    width: float,
) -> None:
    if scene.ngeom >= scene.maxgeom:
        return

    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_LINE,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_LINE,
        width,
        np.asarray(from_pos, dtype=np.float64),
        np.asarray(to_pos, dtype=np.float64),
    )
    scene.ngeom += 1


def add_payload_centering_boundary_geoms(
    scene: mujoco.MjvScene,
    *,
    tolerance: float,
    half_length: float,
    z: float = 0.025,
) -> None:
    tolerance = float(tolerance)
    x_positions = (0.0,) if tolerance == 0.0 else (-tolerance, tolerance)
    for x_pos in x_positions:
        add_line_geom(
            scene,
            from_pos=(float(x_pos), -float(half_length), z),
            to_pos=(float(x_pos), float(half_length), z),
            rgba=(1.0, 0.16, 0.05, 1.0),
            width=6.0,
        )


def add_y_reference_line_geom(
    scene: mujoco.MjvScene,
    *,
    y: float,
    half_width: float = 0.5,
    z: float = 0.025,
    rgba: tuple[float, float, float, float] = (1.0, 0.75, 0.05, 1.0),
) -> None:
    add_line_geom(
        scene,
        from_pos=(-float(half_width), float(y), float(z)),
        to_pos=(float(half_width), float(y), float(z)),
        rgba=rgba,
        width=5.0,
    )


def add_wall_y_reference_line_geoms(
    scene: mujoco.MjvScene,
    *,
    wall_distance: FloatOrDistParams,
) -> None:
    bounds_rgba = (0.1, 0.8, 1.0, 1.0)
    if isinstance(wall_distance, (float, int)):
        add_y_reference_line_geom(scene, y=float(wall_distance), rgba=bounds_rgba)
        return
    if isinstance(wall_distance, BoundedDistParams):
        add_y_reference_line_geom(scene, y=wall_distance.low, rgba=bounds_rgba)
        add_y_reference_line_geom(scene, y=wall_distance.high, rgba=bounds_rgba)
