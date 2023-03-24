"""
Geometric transforms.
"""
from __future__ import annotations

import mitsuba as mi


def map_unit_cube(
    xmin: float, xmax: float, ymin: float, ymax: float, zmin: float, zmax: float
) -> mi.ScalarTransform4f:
    r"""
    Map the unit cube :math:`[0, 1]^3` to
    :math:`[x_\mathrm{min}, x_\mathrm{max}] \times [y_\mathrm{min}, y_\mathrm{max}] \times [z_\mathrm{min}, z_\mathrm{max}]`.

    Parameters
    ----------
    xmin : float
        Minimum X value.

    xmax : float
        Maximum X value.

    ymin : float
        Minimum Y value.

    ymax : float
        Maximum Y value.

    zmin : float
        Minimum Z value.

    zmax : float
        Maximum Z value.

    Returns
    -------
    :class:`mitsuba.core.ScalarTransform4f`
        Computed transform matrix.

    Warnings
    --------
    You must select a Mitsuba variant before calling this function.
    """
    return mi.ScalarTransform4f.translate(
        [xmin, ymin, zmin]
    ) @ mi.ScalarTransform4f.scale([xmax - xmin, ymax - ymin, zmax - zmin])


def map_cube(
    xmin: float, xmax: float, ymin: float, ymax: float, zmin: float, zmax: float
) -> mi.ScalarTransform4f:
    r"""
    Map the cube :math:`[-1, 1]^3` to
    :math:`[x_\mathrm{min}, x_\mathrm{max}] \times [y_\mathrm{min}, y_\mathrm{max}] \times [z_\mathrm{min}, z_\mathrm{max}]`.

    Parameters
    ----------
    xmin : float
        Minimum X value.

    xmax : float
        Maximum X value.

    ymin : float
        Minimum Y value.

    ymax : float
        Maximum Y value.

    zmin : float
        Minimum Z value.

    zmax : float
        Maximum Z value.

    Returns
    -------
    :class:`mitsuba.core.ScalarTransform4f`
        Computed transform matrix.

    Warnings
    --------
    You must select a Mitsuba variant before calling this function.
    """
    half_edge_x = 0.5 * (xmax - xmin)
    half_edge_y = 0.5 * (ymax - ymin)
    half_edge_z = 0.5 * (zmax - zmin)

    return mi.ScalarTransform4f.translate(
        [half_edge_x + xmin, half_edge_y + ymin, half_edge_z + zmin]
    ) @ mi.ScalarTransform4f.scale([half_edge_x, half_edge_y, half_edge_z])
