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


def merge(
    z_mol: np.ndarray, z_par: List[np.ndarray], atol: float = 0.1
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Merge molecular atmopshere altitude mesh with particle layers altitude meshes.

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

    Returns -> Tuple[:class:``numpy.ndarray``, List[:class:``numpy.ndarray``]]:
        Merged altitude mesh, new particle layers individual altitude meshs.
    """
    # initial guess
    dz_mol = z_mol[1] - z_mol[0]
    dz_par = [x[1] - x[0] for x in z_par]
    n_mol = len(z_mol)
    factor = dz_mol / min(dz_par)
    if np.isclose(factor, math.floor(factor), rtol=1e-9):
        n_1 = math.floor(factor)
    else:
        n_1 = math.ceil(factor)
    n = (n_mol - 1) * n_1 + 1
    z = np.linspace(z_mol[0], z_mol[-1], n)

    # compute altitude difference between initial and new particle layers edges
    bottoms = [x[0] for x in z_par]
    tops = [x[-1] for x in z_par]
    edges = np.concatenate([bottoms, tops])
    new_bottoms = find_closest(z, bottoms)
    new_tops = find_closest(z, tops)
    new_edges = np.concatenate([new_bottoms, new_tops])
    diff = max(np.abs(new_edges - edges))

    # refine altitude mesh until tolerance criterion on particle layers edges
    # altitudes is met
    while diff > atol:
        n *= 2
        z = np.linspace(z_mol[0], z_mol[-1], n)
        new_bottoms = find_closest(z, bottoms)
        new_tops = find_closest(z, tops)
        new_edges = np.concatenate([new_bottoms, new_tops])
        diff = max(np.abs(new_edges - edges))

    # compute new particle layers altitude meshes
    new_z_par = []
    for bottom, top in zip(new_bottoms, new_tops):
        new_z_par.append(z[(z >= bottom) & (z <= top)])

    return z, new_z_par
