import numpy as np


def square_to_uniform_disk_concentric(sample):
    """Low-distortion concentric square to disk mapping (unvectorized)."""

    x = 2. * sample[0] - 1.
    y = 2. * sample[1] - 1.

    if x == 0 and y == 0:
        r = phi = 0
    elif x * x > y * y:
        r = x
        phi = (np.pi / 4.) * (y / x)
    else:
        r = y
        phi = (np.pi / 2.) - (x / y) * (np.pi / 4.)

    s = np.sin(phi)
    c = np.cos(phi)
    return np.array((r * c, r * s))


def square_to_uniform_hemisphere(sample):
    """Uniformly sample a vector on the unit hemisphere with respect to solid
    angles."""

    p = square_to_uniform_disk_concentric(sample)
    z = 1. - np.dot(p, p)
    p *= np.sqrt(z + 1.)
    return np.array((p[0], p[1], z))