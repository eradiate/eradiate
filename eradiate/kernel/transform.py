def map_unit_cube(
    xmin: float, xmax: float, ymin: float, ymax: float, zmin: float, zmax: float
) -> "mitsuba.core.ScalarTransform4f":
    """
    Map the unit cube to [xmin, xmax] x [ymin, ymax] x [zmin, zmax].

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
    :class:`~mitsuba.core.ScalarTransform4f`
        Transform matrix.

    Warnings
    --------
    You must select a Mitsuba variant before calling this function.
    """
    from mitsuba.core import ScalarTransform4f

    scale_trafo = ScalarTransform4f.scale([xmax - xmin, ymax - ymin, zmax - zmin])
    translate_trafo = ScalarTransform4f.translate([xmin, ymin, zmin])
    return translate_trafo * scale_trafo


def map_cube(
    xmin: float, xmax: float, ymin: float, ymax: float, zmin: float, zmax: float
) -> "mitsuba.core.ScalarTransform4f":
    """
    Map the cube [-1, 1]^3 to [xmin, xmax] x [ymin, ymax] x [zmin, zmax].

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
    :class:`~mitsuba.core.ScalarTransform4f`
        Transform matrix.

    Warnings
    --------
    You must select a Mitsuba variant before calling this function.
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
