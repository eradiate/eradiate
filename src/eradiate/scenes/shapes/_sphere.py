from __future__ import annotations

import mitsuba as mi
import numpy as np
import pint
import pinttr
from pinttr.util import ensure_units

from ._core import ShapeNode
from ..bsdfs import BSDF
from ..core import BoundingBox
from ...attrs import define, documented
from ...constants import EARTH_RADIUS
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg


def _normalize(v: np.typing.ArrayLike) -> np.ndarray:
    return np.array(v) / np.linalg.norm(v)


@define(eq=False, slots=False)
class SphereShape(ShapeNode):
    """
    Sphere shape [``sphere``].

    This shape represents a sphere parametrized by its centre and radius.

    Notes
    -----
    * If the `to_world` parameter is set, it will be appended to the position
      and scaling defined by the `center` and `radius` parameters.
    """

    center: pint.Quantity = documented(
        pinttr.field(factory=lambda: [0, 0, 0], units=ucc.deferred("length")),
        doc="Location of the centre of the sphere. Unit-enabled field "
        "(default: ``ucc['length']``).",
        type="quantity",
        init_type="quantity or array-like, optional",
        default="[0, 0, 0]",
    )

    radius: pint.Quantity = documented(
        pinttr.field(
            factory=lambda: 1.0 * ucc.get("length"), units=ucc.deferred("length")
        ),
        doc="Sphere radius. Unit-enabled field (default: ``ucc['length']``).",
        type="quantity",
        init_type="quantity or float, optional",
        default="1.0",
    )

    @property
    def bbox(self) -> BoundingBox:
        length_units = ucc.get("length")
        if self.to_world is not None:
            trafo = (
                self.to_world
                @ mi.Transform4f.translate(self.center.m_as(length_units))
                @ mi.Transform4f.scale(self.radius.m_as(length_units))
            )
        else:
            trafo = mi.Transform4f.translate(
                self.center.m_as(length_units)
            ) @ mi.Transform4f.scale(self.radius.m_as(length_units))

        c = trafo @ (0, 0, 0)
        r = np.linalg.norm(trafo @ (1, 0, 0) - c)

        p1 = pint.Quantity(np.array(c + np.array((-1, -1, -1)) * r), length_units)
        p2 = pint.Quantity(np.array(c + np.array((1, 1, 1)) * r), length_units)

        return BoundingBox(p1, p2)

    @property
    def template(self) -> dict:
        result = {
            "type": "sphere",
            "center": self.center.m_as(uck.get("length")),
            "radius": self.radius.m_as(uck.get("length")),
        }
        if self.to_world is not None:
            result["to_world"] = self.to_world
        return result

    def contains(self, p: np.typing.ArrayLike, strict: bool = False) -> bool:
        """
        Test whether a point lies within the sphere.

        Parameters
        ----------
        p : quantity or array-like
            An array of shape (3,) (resp. (N, 3)) representing one (resp. N)
            points. If a unitless value is passed, it is interpreted as
            ``ucc['length']``.

        strict : bool
            If ``True``, comparison is done using strict inequalities (<, >).

        Returns
        -------
        result : array of bool or bool
            ``True`` iff ``p`` in within the sphere.
        """
        length_units = ucc.get("length")
        if self.to_world is not None:
            trafo = (
                self.to_world
                @ mi.Transform4f.translate(self.center.m_as(length_units))
                @ mi.Transform4f.scale(self.radius.m_as(length_units))
            )
        else:
            trafo = mi.Transform4f.translate(
                self.center.m_as(length_units)
            ) @ mi.Transform4f.scale(self.radius.m_as(length_units))
        p = np.atleast_2d(ensure_units(p, ucc.get("length")).m_as(length_units))
        c = trafo @ (0, 0, 0)
        d = np.linalg.norm(p - c, axis=1)
        r = np.linalg.norm(trafo @ (1, 0, 0) - c)
        return d < r if strict else d <= r

    @classmethod
    def surface(
        cls,
        altitude=0.0 * ureg.km,
        planet_radius: pint.Quantity = EARTH_RADIUS,
        bsdf: BSDF | None = None,
    ) -> SphereShape:
        """
        This class method constructor provides a simplified parametrization of
        the sphere shape better suited for the definition of the surface when
        configuring the one-dimensional model.

        The resulting sphere shape is centred at [0, 0, -`planet_radius`] and
        has a radius equal to `planet_radius` + `altitude`.

        Parameters
        ----------
        altitude : quantity or array-like, optional, default: 0 km
            Surface altitude. If a unitless value is passed, it is interpreted
            as ``ucc['length']``.

        planet_radius : quantity or float, optional
            Planet radius. If a unitless value is passed, it is interpreted
            as ``ucc['length']``. The default is Earth's radius.

        bsdf : BSDF or dict, optional, default: None
            A BSDF specification, forwarded to the main constructor.

        Returns
        -------
        SphereShape
            A sphere shape which can be used as the surface in a spherical shell
            geometry.
        """
        altitude = pinttr.util.ensure_units(altitude, default_units=ucc.get("length"))

        planet_radius = pinttr.util.ensure_units(
            planet_radius, default_units=ucc.get("length")
        )

        return cls(
            center=[0.0, 0.0, 0.0] * planet_radius.units,
            radius=planet_radius + altitude,
            bsdf=bsdf,
        )

    @classmethod
    def atmosphere(
        cls,
        top: pint.Quantity = 100.0 * ureg.km,
        planet_radius: pint.Quantity = EARTH_RADIUS,
        bsdf: BSDF | None = None,
    ) -> SphereShape:
        """
        This class method constructor provides a simplified parametrization of
        the sphere shape better suited for the definition of the surface when
        configuring the one-dimensional model.

        The resulting sphere shape is centred at [0, 0, 0] and
        has a radius equal to `planet_radius` + `top`.

        Parameters
        ----------
        top : quantity or array-like, optional, default: 100 km
            Top-of-atmosphere altitude. If a unitless value is passed, it is
            interpreted as ``ucc['length']``.

        planet_radius : quantity or float, optional
            Planet radius. If a unitless value is passed, it is interpreted
            as ``ucc['length']``. The default is Earth's radius.

        bsdf : BSDF or dict, optional, default: None
            A BSDF specification, forwarded to the main constructor.

        Returns
        -------
        SphereShape
            A sphere shape which can be used as the stencil of a participating
            medium in a spherical shell geometry.
        """
        top = pinttr.util.ensure_units(top, default_units=ucc.get("length"))

        planet_radius = pinttr.util.ensure_units(
            planet_radius, default_units=ucc.get("length")
        )

        return cls(
            center=[0.0, 0.0, 0.0] * planet_radius.units,
            radius=planet_radius + top,
            bsdf=bsdf,
        )
