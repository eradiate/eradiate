import math
from typing import List, Tuple, Union

import numpy as np


def find_closest(
    x: np.ndarray, target: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Find closest value to target in array.

    Parameter ``x`` (:class:`numpy.ndarray`):
        Array to search (sorted).

    Parameter ``target`` (float or :class:`numpy.ndarray`):
        Values to reach.

    Returns -> float or :class:`numpy.ndarray`:
        Closest values.
    """
    idx = x.searchsorted(target)
    idx = np.clip(idx, 1, len(x) - 1)
    left = x[idx - 1]
    right = x[idx]
    idx -= target - left < right - target
    return x[idx]


def conciliate(
    z_mol: np.ndarray, z_par: List[np.ndarray], atol: float = 0.1
) -> np.ndarray:
    """
    Conciliate molecular atmopshere altitude mesh with particle layers altitude
    meshes.

    The method subdivides the initial molecular atmosphere's altitude mesh
    until it reaches a resolution sufficient for the accurate representation
    of the particles layers, and returns that refined altitude mesh.
    The resolution is sufficient when one can find new particle layer bounds
    in the refined altitude mesh so that the difference between the initial
    layer altitude bounds and the new layer altitude bounds is less than
    the value specified by ``atol``.

    .. note::
        Each particle layer is garanteed to have a greater or equal number of
        sub-layers than in their initial level altitude mesh.

    .. note::
        In the process of merging the altitude meshes, the particle layers
        altitude mesh may be displaced a little bit from their initial values.
        Set the maximum value of this displacement using the ``atol`` parameter.

    Parameter ``z_mol`` (:class`numpy.ndarray`):
        Molecular atmosphere level altitude mesh.
        Sorted linearly spaced 1D array.

    Parameter ``z_par`` (list of :class`numpy.ndarray`):
        Particle level altitude meshes.
        Linearly spaced sorted 1D arrays.

    Parameter ``atol`` (float):
        Absolute tolerance on particle layers bottom and top altitudes.

    Returns -> :class:``numpy.ndarray``
        Conciliatory altitude mesh.
    """
    # Initial guess
    dz_mol = z_mol[1] - z_mol[0]  # molecules layers thickness
    dz_par = [x[1] - x[0] for x in z_par]  # particle layers thicknesses
    n_mol = len(z_mol)
    ratio = dz_mol / min(dz_par)  # maximum molecules/particles layers thickness ratio
    # if-else block below deals with floating point arithmetic issues:
    # n_1 is the number sub-layers inside molecular atmosphere's layers
    if np.isclose(ratio, math.floor(ratio), rtol=1e-9):
        n_1 = math.floor(ratio)
    else:
        n_1 = math.ceil(ratio)
    # total number of layers in the molecular atmosphere (initial guess):
    n = (n_mol - 1) * n_1 + 1
    z = np.linspace(z_mol[0], z_mol[-1], n)

    # Compute altitude difference between initial and new particle layers edges
    bottoms = [x[0] for x in z_par]
    tops = [x[-1] for x in z_par]
    edges = np.concatenate([bottoms, tops])
    new_bottoms = find_closest(z, bottoms)
    new_tops = find_closest(z, tops)
    new_edges = np.concatenate([new_bottoms, new_tops])
    diff = max(np.abs(new_edges - edges))

    # Refine altitude mesh until tolerance criterion on particle layers edges
    # altitudes is met
    while diff > atol:
        n_1 *= 2
        n = (n_mol - 1) * n_1 + 1
        z = np.linspace(z_mol[0], z_mol[-1], n)
        new_bottoms = find_closest(z, bottoms)
        new_tops = find_closest(z, tops)
        new_edges = np.concatenate([new_bottoms, new_tops])
        diff = max(np.abs(new_edges - edges))

    return z


def extract_layer_mesh(z: np.ndarray, bottom: float, top: float) -> np.ndarray:
    """
    Extract a particle layer altitude mesh from the input molecular atmosphere
    altitude mesh corresponding to the particle layer's altitude region.

    Parameter ``z`` (:class:`~numpy.ndarray`):
        Molecular atmosphere altitude mesh.

    Parameter ``bottom`` (float):
        Particle layer bottom altitude.

    Parameter ``top`` (float):
        Particle layer top altitude.

    Returns → :class:`~numpy.ndarray`:
        Particle layer altitude mesh.
    """
    if not bottom < top:
        raise ValueError(
            f"bottom must be less than top (got bottom={bottom} which is not less than top={top})"
        )
    if bottom < z.min() or bottom > z.max():
        raise ValueError(
            f"bottom must be within z (got {bottom} which is not within {z})"
        )
    if top < z.min() or top > z.max():
        raise ValueError(f"top must be within z (got {top} which is not within {z})")

    return z[(z >= find_closest(z, bottom)) & (z <= find_closest(z, top))]
