import enum
from dataclasses import dataclass

import numpy as np

def _unit_vec(vec: list[int | float]) -> np.ndarray:
    arr = np.array(vec, dtype=float)
    arr /= np.linalg.norm(arr)
    return arr


class LimbType(int, enum.Enum):
    xy = 0
    zx = 1
    xyz = 2

@dataclass
class LimbConfig:
    name: str
    vec: np.ndarray
    type: LimbType
    rgba: tuple[float, float, float, float]

    def toJSON(self):
        return {
            'name': self.name,
            'vec': self.vec.tolist(),
            'type': str(self.type),
            'rgba': self.rgba,
        }

UnitConfig = tuple[LimbConfig, ...]

UNIT_CONFIG_CUBE_ZX: UnitConfig = (
    LimbConfig('xp', _unit_vec([ 1,  0,  0]), LimbType.zx, (  1,   0,   0,   1)),
    LimbConfig('xn', _unit_vec([-1,  0,  0]), LimbType.zx, (0.7, 0.3,   0,   1)),
    LimbConfig('yp', _unit_vec([ 0,  1,  0]), LimbType.zx, (  0,   1,   0,   1)),
    LimbConfig('yn', _unit_vec([ 0, -1,  0]), LimbType.zx, (  0, 0.7, 0.3,   1)),
    LimbConfig('zp', _unit_vec([ 0,  0,  1]), LimbType.zx, (  0,   0,   1,   1)),
    LimbConfig('zn', _unit_vec([ 0,  0, -1]), LimbType.zx, (0.3,   0, 0.7,   1))
)

UNIT_CONFIG_TETRAHEDRON_ZX: UnitConfig = (
    LimbConfig('ppp', _unit_vec([ 1,  1,  1]), LimbType.zx, (1, 0, 0, 1)),
    LimbConfig('pmm', _unit_vec([ 1, -1, -1]), LimbType.zx, (0, 1, 0, 1)),
    LimbConfig('mpm', _unit_vec([-1,  1, -1]), LimbType.zx, (0, 0, 1, 1)),
    LimbConfig('mmp', _unit_vec([-1, -1,  1]), LimbType.zx, (1, 1, 0, 1)),
)

UNIT_CONFIG_TETRAHEDRON_XY: UnitConfig = (
    LimbConfig('ppp', _unit_vec([ 1,  1,  1]), LimbType.xy, (1, 0, 0, 1)),
    LimbConfig('pmm', _unit_vec([ 1, -1, -1]), LimbType.xy, (0, 1, 0, 1)),
    LimbConfig('mpm', _unit_vec([-1,  1, -1]), LimbType.xy, (0, 0, 1, 1)),
    LimbConfig('mmp', _unit_vec([-1, -1,  1]), LimbType.xy, (1, 1, 0, 1)),
)

UNIT_CONFIG_TETRAHEDRON_XYZ: UnitConfig = (
    LimbConfig('ppp', _unit_vec([ 1,  1,  1]), LimbType.xyz, (1, 0, 0, 1)),
    LimbConfig('pmm', _unit_vec([ 1, -1, -1]), LimbType.xyz, (0, 1, 0, 1)),
    LimbConfig('mpm', _unit_vec([-1,  1, -1]), LimbType.xyz, (0, 0, 1, 1)),
    LimbConfig('mmp', _unit_vec([-1, -1,  1]), LimbType.xyz, (1, 1, 0, 1)),
)
