import numpy as np

def random_quat_shoemake(gen: np.random.Generator | None = None) -> np.ndarray:
    if gen is None:
        gen = np.random.default_rng()
    u1, u2, u3 = gen.random(3)

    s1 = np.sqrt(1.0 - u1)
    s2 = np.sqrt(u1)
    theta1 = 2.0 * np.pi * u2
    theta2 = 2.0 * np.pi * u3

    w = s2 * np.cos(theta2)
    x = s1 * np.sin(theta1)
    y = s1 * np.cos(theta1)
    z = s2 * np.sin(theta2)

    return np.array([w, x, y, z], dtype=float)
