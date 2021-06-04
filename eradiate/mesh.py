"""Mesh utilities."""
from typing import List, Tuple

import math
import numpy as np


def merge(meshes: List[np.ndarray], atol: float = 1e-6) -> np.ndarray:
    """
    Merge meshes and remove the potential duplicate values.

    Parameter ``meshes`` (list of :class:`~numpy.ndarray`):
        Meshes to merge.

    Parameter ``atol`` (float):
        Absolute tolerance parameter. If the difference between two elements in
        the arrays is less than ``atol``, the elements will be considered as
        duplicates.

    Returns → :class:`~numpy.ndarray`:
        The merged mesh.

    .. admonition:: Example

       .. code:: python

          import numpy as np

          mesh_1 = np.linspace(0., 20., 5)
          mesh_2 = np.linspace(0., .7, 4)
          mesh_3 = np.linspace(9., 10., 3)

          merge([mesh_1, mesh_2, mesh_3])

       returns:

       .. code:: shell

          array([0., 0.233333, 0.466667, 0.7, 5., 9., 9.5, 10., 15., 20.])
    """
    x = np.sort(np.concatenate(meshes))
    return unique(x=x, atol=atol)


def unique(x: np.ndarray, atol: float) -> np.ndarray:
    """
    Returns the unique elements of an array with tolerance.

    Parameter ``x`` (:class:`~numpy.ndarray`):
        Array with potential duplicate elements.

    Parameter ``atol`` (float):
        Absolute tolerance parameter. If the difference between two elements in
        ``x`` is smaller than ``atol``, they are considered duplicate elements.

    Returns → :class:`~numpy.ndarray`:
        Array with unique elements.

    .. admonition:: Example

       .. code:: python

          import numpy as np

          mesh = np.array([0., 1., 1.0001, 3., 6.])
          unique(mesh, atol=1e-3)

       returns:

       .. code:: shell

          array([0., 1., 3., 6.])
    """
    n = int(math.ceil(-math.log10(atol)))
    unique_values = [x[0]]
    for i in range(1, len(x)):
        if abs(x[i] - x[i - 1]) >= atol:
            unique_values.append(round(x[i], n))
    return np.array(unique_values)


def to_regular(x: np.ndarray, atol: float) -> np.ndarray:
    """
    Converts an irregular mesh into an approximately-equivalent regular
    mesh.

    .. note::
        The bound values in the irregular mesh remain the same in
        the output regular mesh. Only the intermediate node values are modified.

    .. warning::
        The algorithm is not optimised to find the approximating regular mesh
        with the smallest number of cells. Depending on the value of ``atol``,
        the resulting mesh size can be very large.

    Parameter ``x`` (:class:`~numpy.ndarray`):
        Irregular mesh.

    Parameter ``atol`` (float):
        Absolute tolerance parameter.

    Returns → :class:`~numpy.ndarray`:
        Regular mesh.

    .. admonition:: Example

       .. code:: python

          import numpy as np

          irregular = np.array([0., 1., 3.5, 6.])
          to_regular(irregular, atol=1.)

       returns:

       .. code:: shell

         array([0., 1., 2., 3., 4., 5., 6.])

       To improve the accuracy of the conversion, you can specify a lower
       value of ``atol``:

       .. code:: python

          to_regular(irregular, atol=.1)

       which returns:

       .. code:: shell

          array([0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6.])
    """
    x = np.sort(x)
    num, _ = find_regular_params_gcd(x=x, unit_number=atol)
    return np.linspace(start=x[0], stop=x[-1], num=num)


def find_regular_params_gcd(
    x: np.ndarray, unit_number: float = 1.0
) -> Tuple[int, float]:
    """
    Find the parameters (cells number and cell width) of the regular mesh that
    approximates the irregular input mesh as best as possible.

    The algorithm finds the greatest common divisor (GCD) of all cells widths in
    the integer representation specified by the parameter ``unit_number``.
    This GCD is used to define the constant cells width of the approximating
    regular mesh.

    .. warning::
        There are no safeguards regarding how large the number of cells in the
        regular mesh can be. Use the parameter ``unit_number`` with caution.

    Parameter ``x`` (:class:`~numpy.ndarray`):
        1-D array with floating point values.
        Values must be sorted by increasing order.

    Parameter ``unit_number`` (float):
        Defines the unit used to convert the floating point numbers to integer.
        numbers.

        Default: 1.

    Returns -> Tuple[int, float]:
        Number of points in the regular mesh and the value of the constant cells
        width.

    .. admonition:: Example

       .. code:: python

          import numpy as np
          x = np.array([0., 1., 3.5, 6.])
          num, width = find_regular_params_gcd(x)

       where ``num`` is 7 and ``width`` is 1.0, corresponding to the regular
       mesh:

       .. code:: shell

          array([0., 1., 2., 3., 4., 5., 6.])

       To increase the accuracy of the approximation, set ``unit_number`` to
       a smaller value:

       .. code:: python

          num, width = find_regular_params_gcd(x, unit_number=0.1)

       where ``num`` is now 13 and ``width`` is 0.5, corresponding to the
       regular mesh:

       .. code:: shell

          array([0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6.])
    """

    # Convert float cell widths to integer cell widths
    eps = np.finfo(float).eps
    if unit_number >= eps:
        w = np.divide(x, unit_number).astype(int)
    else:
        raise ValueError(
            f"Parameter unit_number ({unit_number}) must be "
            f"larger than machine epsilon ({eps})."
        )
    widths = w[1:] - w[:-1]

    # Find the greatest common divisor (GCD) of all integer cell widths
    # The constant cell width in the regular mesh is given by that GCD.
    from math import gcd

    w = gcd(widths[0], widths[1])
    for wi in widths[2:]:
        w = gcd(wi, w)

    # Compute the number of points in the regular mesh
    total_width = (x[-1] - x[0]) / unit_number
    n = total_width // w + 1

    return int(n), float(w) * unit_number


# def find_regular_params_tol(mesh, rtol=1e-3, n_cells_max=10000):
#     r"""Finds the number of cells and constant cell width of the regular 1-D
#     mesh that approximates a 1-D irregular mesh the best.
#
#     Parameter ``mesh`` (:class:`~numpy.ndarray`):
#         Irregular 1-D mesh. Values must be sorted in increasing order.
#
#     Parameter ``rtol`` (float):
#         Relative tolerance on the cells widths. This parameter controls the
#         accuracy of the approximation.
#         The parameters of the approximating regular mesh are computed so that
#         for each layer of the irregular mesh, the corresponding layer or layers
#         in the regular mesh has a width or have a total width that is not larger
#         than ``rtol`` times the width of the cell in the irregular mesh.
#
#         Default: 1e-3
#
#     Parameter ``n_cells_max`` (float):
#         Maximum number of cells in the regular mesh. This parameter controls the
#         size of the resulting regular mesh.
#
#         Default: 10000
#
#     Returns -> Tuple[int, float]:
#         Number of cells and constant cells width in the approximating regular
#         mesh.
#     """
#
#     raise NotImplemented
# TODO: implement this function
