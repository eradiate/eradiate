import numpy as np


def square_to_uniform_disk_concentric(sample):
    """Low-distortion concentric square to disk mapping (unvectorized)."""

    x = 2.0 * sample[0] - 1.0
    y = 2.0 * sample[1] - 1.0

    if x == 0 and y == 0:
        r = phi = 0
    elif x * x > y * y:
        r = x
        phi = (np.pi / 4.0) * (y / x)
    else:
        r = y
        phi = (np.pi / 2.0) - (x / y) * (np.pi / 4.0)

    s = np.sin(phi)
    c = np.cos(phi)
    return np.array((r * c, r * s))


def square_to_uniform_hemisphere(sample):
    """Uniformly sample a vector on the unit hemisphere with respect to solid
    angles."""

    p = square_to_uniform_disk_concentric(sample)
    z = 1.0 - np.dot(p, p)
    p *= np.sqrt(z + 1.0)
    return np.array((p[0], p[1], z))
