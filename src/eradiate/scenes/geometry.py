from __future__ import annotations

import abc
import typing as t

import attrs
import pint
import pinttr

from ..attrs import documented, parse_docs
from ..constants import EARTH_RADIUS
from ..units import unit_context_config as ucc
from ..units import unit_registry as ureg


@parse_docs
@attrs.define
class SceneGeometry(abc.ABC):
    """
    Abstract base class defining a scene geometry.
    """

    @classmethod
    def convert(cls, value: t.Any) -> t.Any:
        """
        Attempt conversion of a value to a :class:`.SceneGeometry` subtype.

        Parameters
        ----------
        value
            Value to attempt conversion of. If a dictionary is passed, its
            ``"type"`` key is used to route its other entries as keyword
            arguments to the appropriate subtype's constructor. If a string is
            passed, this method calls itself with the parameter
            ``{"type": value}``.

        Returns
        -------
        result
            If `value` is a dictionary, the constructed :class:`.SceneGeometry`
            instance is returned. Otherwise, `value` is returned.

        Raises
        ------
        ValueError
            A dictionary was passed but the requested type is unknown.
        """
        if isinstance(value, str):
            return cls.convert({"type": value})

        if isinstance(value, dict):
            value = value.copy()
            geometry_type = value.pop("type")

            # Note: if this conditional becomes large, use a dictionary
            if geometry_type == "plane_parallel":
                geometry_cls = PlaneParallelGeometry
            elif geometry_type == "spherical_shell":
                geometry_cls = SphericalShellGeometry
            else:
                raise ValueError(f"unknown geometry type '{geometry_type}'")

            return geometry_cls(**pinttr.interpret_units(value, ureg=ureg))

        return value


@parse_docs
@attrs.define
class PlaneParallelGeometry(SceneGeometry):
    """
    Plane parallel geometry.

    A plane parallel atmosphere is translation-invariant in the X and Y
    directions. However, Eradiate represents it with a finite 3D geometry
    consisting of a cuboid. By default, the cuboid's size is computed
    automatically; however, it can also be forced by assigning a value to
    the `width` field.
    """

    width: pint.Quantity = documented(
        pinttr.field(default=1e6 * ureg.km, units=ucc.deferred("length")),
        doc="Cuboid shape width.",
        type="quantity",
        init_type="quantity or float",
        default="1.000.000 km",
    )


@parse_docs
@attrs.define
class SphericalShellGeometry(SceneGeometry):
    """
    Spherical shell geometry.

    A spherical shell atmosphere has a spherical symmetry. Eradiate represents
    it with a finite 3D geometry consisting of a sphere. By default, the
    sphere's radius is set equal to Earth's radius.
    """

    planet_radius: pint.Quantity = documented(
        pinttr.field(default=EARTH_RADIUS, units=ucc.deferred("length")),
        doc="Planet radius. Defaults to Earth's radius.",
        type="quantity",
        init_type="quantity or float",
        default=":data:`.EARTH_RADIUS`",
    )
