from typing import Any


def map_unit_cube(
    xmin: float, xmax: float, ymin: float, ymax: float, zmin: float, zmax: float
) -> Any:
    """
    Map the unit cube to [xmin, xmax] x [ymin, ymax] x [zmin, zmax].

    .. note::
        You must select a Mitsuba variant before calling this function.

    Parameter ``xmin`` (float):
        Minimum X value.

    Parameter ``xmax`` (float):
        Maximum X value.

    Parameter ``ymin`` (float):
        Minimum Y value.

    Parameter ``ymax`` (float):
        Maximum Y value.

    Parameter ``zmin`` (float):
        Minimum Z value.

    Parameter ``zmax`` (float):
        Maximum Z value.

    Returns → :class:`~mitsuba.core.ScalarTransform4f`
        Transform matrix.
    """
    from mitsuba.core import ScalarTransform4f

    scale_trafo = ScalarTransform4f.scale([xmax - xmin, ymax - ymin, zmax - zmin])
    translate_trafo = ScalarTransform4f.translate([xmin, ymin, zmin])
    return translate_trafo * scale_trafo


def map_cube(
    xmin: float, xmax: float, ymin: float, ymax: float, zmin: float, zmax: float
) -> Any:
    """
    Map the cube [-1, 1]^3 to [xmin, xmax] x [ymin, ymax] x [zmin, zmax].

    .. note::
        You must select a Mitsuba variant before calling this function.

    Parameter ``xmin`` (float):
        Minimum X value.

    Parameter ``xmax`` (float):
        Maximum X value.

    Parameter ``ymin`` (float):
        Minimum Y value.

    Parameter ``ymax`` (float):
        Maximum Y value.

    Parameter ``zmin`` (float):
        Minimum Z value.

    Parameter ``zmax`` (float):
        Maximum Z value.

    Returns → :class:`~mitsuba.core.ScalarTransform4f`
        Transform matrix.
    """
    from mitsuba.core import ScalarTransform4f

    half_dx = (xmax - xmin) * 0.5
    half_dy = (ymax - ymin) * 0.5
    half_dz = (zmax - zmin) * 0.5
    scale_trafo = ScalarTransform4f.scale([half_dx, half_dy, half_dz])
    translate_trafo = ScalarTransform4f.translate(
        [xmin + half_dx, ymin + half_dy, half_dz + zmin]
    )
    return translate_trafo * scale_trafo
