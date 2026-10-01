from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod
from enum import Enum

import attrs
import mitsuba as mi
import numpy as np
import pint
import pinttrs

from .shapes import CuboidShape, RectangleShape, Shape, SphereShape
from .. import _repr_diagrams
from ..attrs import define, documented
from ..constants import EARTH_RADIUS
from ..grid import GridCoords, SphericalShellGridCoords
from ..kernel import map_cube, map_unit_cube
from ..units import unit_context_config as ucc
from ..units import unit_context_kernel as uck
from ..units import unit_registry as ureg


class WrapMode(Enum):
    CLAMP = 0
    REPEAT = 1
    MIRROR = 2

    def __str__(self):
        return self.name.lower()


class FilterType(Enum):
    TRILINEAR = 0
    NEAREST = 1

    def __str__(self) -> str:
        return self.name.lower()


@define
class SceneGeometry(ABC):
    """
    Abstract base class defining a scene geometry.

    Warnings
    --------
    If a ``grid`` value is passed to the constructor together with explicit
    ``ground_altitude`` and/or ``toa_altitude`` values, its extent must be
    [``ground_altitude``, ``toa_altitude``]. The constructor will raise a
    :class:`ValueError` otherwise. If ``ground_altitude`` and/or
    ``toa_altitude`` are left unset, they are instead derived from the
    ``grid`` extent.
    """

    toa_altitude: pint.Quantity | None = documented(
        pinttrs.field(default=None, units=ucc.deferred("length")),
        doc="Top-of-atmosphere level altitude. If unset, defaults to the "
        "``grid`` top level when ``grid`` is set, and to 120 km otherwise. "
        'Unit-enabled field (default: ``ucc["length"]``).',
        default="120 km",
        type="pint.Quantity",
        init_type="float or quantity, optional",
    )

    ground_altitude: pint.Quantity | None = documented(
        pinttrs.field(default=None, units=ucc.deferred("length")),
        doc="Baseline ground altitude. If unset, defaults to the ``grid`` "
        "bottom level when ``grid`` is set, and to 0 km otherwise. "
        'Unit-enabled field (default: ``ucc["length"]``).',
        default="0 km",
        type="pint.Quantity",
        init_type="float or quantity, optional",
    )

    grid: GridCoords = documented(
        attrs.field(
            default=None,
            converter=attrs.converters.optional(
                lambda x: GridCoords.convert(x) if not isinstance(x, GridCoords) else x
            ),
            validator=attrs.validators.optional(
                attrs.validators.instance_of(GridCoords)
            ),
        ),
        doc="The altitude mesh on which the radiative properties of "
        "heterogeneous atmosphere components are evaluated. "
        "If unset, a default grid with one layer per 100 m (or 10 layers if "
        "the atmosphere object height is less than 100 m) is used.",
        type=".GridCoords",
        init_type=".GridCoords, quantity or ndarray, optional",
    )

    filter_type: FilterType = documented(
        attrs.field(
            default=FilterType.NEAREST,
            converter=FilterType,
            validator=attrs.validators.instance_of(FilterType),
            kw_only=True,
        ),
        doc="Volume filter type.",
        default="FilterType.NEAREST",
        type=".FilterType",
        init_type=".FilterType",
    )

    wrap_mode: WrapMode = documented(
        attrs.field(
            default=WrapMode.CLAMP,
            converter=WrapMode,
            validator=attrs.validators.instance_of(WrapMode),
            kw_only=True,
        ),
        doc="Volume wrap mode.",
        default="WrapMode.CLAMP",
        type=".WrapMode",
        init_type=".WrapMode",
    )

    def __attrs_post_init__(self) -> None:
        # Resolve ground_altitude and toa_altitude
        if self.grid is not None:
            grid_bottom = self.grid.levels[0]
            if self.ground_altitude is None:
                self.ground_altitude = grid_bottom
            elif not np.isclose(grid_bottom, self.ground_altitude):
                raise ValueError(
                    "grid bottom must match ground_altitude; "
                    f"expected {self.ground_altitude}, got {grid_bottom}"
                )

            grid_top = self.grid.levels[-1]
            if self.toa_altitude is None:
                self.toa_altitude = grid_top
            elif not np.isclose(grid_top, self.toa_altitude):
                raise ValueError(
                    "grid top must match toa_altitude; "
                    f"expected {self.toa_altitude}, got {grid_top}"
                )
        else:
            if self.ground_altitude is None:
                self.ground_altitude = 0.0 * ureg.km
            if self.toa_altitude is None:
                self.toa_altitude = 120.0 * ureg.km

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

            return geometry_cls(**pinttrs.interpret_units(value, ureg=ureg))

        return value

    @property
    @abstractmethod
    def atmosphere_shape(self) -> Shape:
        """
        :class:`.Shape`: Stencil of the participating medium representing the
        atmosphere.
        """

    @property
    @abstractmethod
    def atmosphere_volume_to_world(self) -> mi.ScalarTransform4f:
        """
        :class:`mi.ScalarTransform4f` : Mitsuba transform mapping volume texture
            coordinates to world coordinates for heterogeneous atmosphere
            components.
        """

    @property
    @abstractmethod
    def surface_shape(self) -> Shape:
        """
        :class:`.Shape` : Shape representing the surface.
        """

    @property
    def bbox(self):
        """
        :class:`eradiate.scenes.core.BoundingBox` : Bounding box of the geometry
        """
        return self.atmosphere_shape.bbox

    def _repr_html_(self) -> str | None:
        return _repr_diagrams.geometry_repr_html(self)


@define
class PlaneParallelGeometry(SceneGeometry):
    """
    Plane parallel geometry [``plane_parallel``].

    A plane parallel atmosphere is translation-invariant in the X and Y
    directions. However, Eradiate represents it with a finite 3D geometry
    consisting of a cuboid. By default, the cuboid's size is computed
    automatically; however, it can also be forced by assigning a value to
    the `width` field.
    """

    width: pint.Quantity = documented(
        pinttrs.field(default=1e6 * ureg.km, units=ucc.deferred("length")),
        doc="Cuboid shape width.",
        type="quantity",
        init_type="quantity or float",
        default="1,000,000 km",
    )

    @property
    def atmosphere_shape(self) -> CuboidShape:
        return CuboidShape.atmosphere(
            top=self.toa_altitude, bottom=0.0 * ureg.km, width=self.width
        )

    @property
    def atmosphere_volume_to_world(self) -> mi.ScalarTransform4f:
        length_units = uck.get("length")
        shape = self.atmosphere_shape

        # The bounding box corresponds to the vertices of the cuboid
        center = shape.center.m_as(length_units)
        edges = shape.edges.m_as(length_units)
        xmin, ymin = center[0:2] - edges[0:2] * 0.5
        xmax, ymax = center[0:2] + edges[0:2] * 0.5
        zmin = self.ground_altitude.m_as(length_units)
        zmax = self.toa_altitude.m_as(length_units)

        return map_unit_cube(xmin, xmax, ymin, ymax, zmin, zmax)

    @property
    def surface_shape(self) -> RectangleShape:
        return RectangleShape.surface(altitude=self.ground_altitude, width=self.width)

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()

        if self.grid is None:
            bottom = self.ground_altitude
            top = self.toa_altitude
            step = min(100.0 * ureg.m, (top - bottom) / 10.0)

            self.grid = GridCoords.make_onedim_arange(top, bottom, step, self.width)


@define
class SphericalShellGeometry(SceneGeometry):
    """
    Spherical shell geometry [``spherical_shell``].

    A spherical shell atmosphere has a spherical symmetry. Eradiate represents
    it with a finite 3D geometry consisting of a sphere. By default, the
    sphere's radius is set equal to Earth's radius.
    """

    planet_radius: pint.Quantity = documented(
        pinttrs.field(default=EARTH_RADIUS, units=ucc.deferred("length")),
        doc="Planet radius. Defaults to Earth's radius.",
        type="quantity",
        init_type="quantity or float",
        default=":data:`.EARTH_RADIUS`",
    )

    @property
    def atmosphere_shape(self) -> SphereShape:
        return SphereShape.atmosphere(
            top=self.toa_altitude, planet_radius=self.planet_radius
        )

    @property
    def atmosphere_volume_to_world(self) -> mi.ScalarTransform4f:
        length_units = ucc.get("length")

        # The bounding box corresponds to the vertices of the bounding box
        bbox = self.atmosphere_shape.bbox
        xmin, ymin, zmin = bbox.min.m_as(length_units)
        xmax, ymax, zmax = bbox.max.m_as(length_units)

        return map_cube(xmin, xmax, ymin, ymax, zmin, zmax)

    @property
    def atmosphere_volume_rmin(self) -> float:
        """
        Value for the ``rmin`` parameter of the ``sphericalcoordsvolume`` plugin
        describing the volume data in the spherical shell geometry.
        """
        return (self.planet_radius / (self.planet_radius + self.toa_altitude)).m_as(
            ureg.dimensionless
        )

    @property
    def surface_shape(self) -> Shape:
        return SphereShape.surface(
            altitude=self.ground_altitude, planet_radius=self.planet_radius
        )

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()

        if self.grid is None:
            bottom = self.ground_altitude.m_as(ureg.m)
            top = self.toa_altitude.m_as(ureg.m)
            step = min(100.0, (top - bottom) / 10.0)
            self.grid = SphericalShellGridCoords(
                levels=ureg.convert(
                    np.arange(bottom, top + step * 0.1, step),
                    ureg.m,
                    ucc.get("length"),
                ),
                azimuths=np.asarray([0.0, 360.0]) * ureg.degree,
                colatitudes=np.asarray([0.0, 180.0]) * ureg.degree,
            )
