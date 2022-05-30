from __future__ import annotations

import typing as t

import attr
import mitsuba as mi
import numpy as np
import pint
import pinttr

from ._core import Shape, shape_factory
from ..bsdfs import BSDF
from ..core import KernelDict
from ... import validators
from ...attrs import documented, parse_docs
from ...contexts import KernelDictContext
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg
from ...util.misc import onedict_value


def _normalize(v: np.typing.ArrayLike) -> np.ndarray:
    # Return L2 norm of a vector
    return np.array(v) / np.linalg.norm(v)


def _edges_converter(value):
    # Basic unit conversion and array reshaping
    length_units = ucc.get("length")
    value = np.reshape(
        pinttr.util.ensure_units(value, default_units=length_units).m_as(length_units),
        (-1,),
    )

    # Broadcast if relevant
    if len(value) == 1:
        value = np.full((2,), value[0])

    return value * length_units


@shape_factory.register(type_id="rectangle")
@parse_docs
@attr.s
class RectangleShape(Shape):
    """
    Rectangle shape [``rectangle``].

    This shape represents a rectangle parametrised by the length of its edges,
    the coordinates of its central point, a normal vector and an orientation
    vector.

    Notes
    -----
    This is a wrapper around the ``rectangle`` kernel plugin.
    """

    edges: pint.Quantity = documented(
        pinttr.ib(
            factory=lambda: [1, 1],
            converter=_edges_converter,
            units=ucc.deferred("length"),
        ),
        doc="Length of the rectangle's edges. "
        'Unit-enabled field (default: ``ucc["length"]``).',
        type="quantity",
        init_type="quantity or array-like, optional",
        default="[1, 1]",
    )

    center: pint.Quantity = documented(
        pinttr.ib(factory=lambda: [0, 0, 0], units=ucc.deferred("length")),
        doc="Cartesian coordinates of the rectangle's central point. "
        'Unit-enabled field (default: ``ucc["length"]``).',
        type="quantity",
        init_type="quantity or array-like, optional",
        default="[0, 0, 0]",
    )

    normal: np.ndarray = documented(
        attr.ib(
            factory=lambda: [0, 0, 1],
            converter=_normalize,
            validator=validators.is_vector3,
        ),
        doc="Normal vector of the plane containing the rectangle. Defaults to "
        "the +Z direction.",
        type="array",
        init_type="array-like, optional",
        default="[0, 0, 1]",
    )

    up: np.ndarray = documented(
        attr.ib(
            factory=lambda: [0, 1, 0],
            converter=_normalize,
            validator=validators.is_vector3,
        ),
        doc="Orientation vector defining the rotation of the rectangle around "
        "the normal vector. Defaults to the +Y direction.",
        type="array",
        init_type="array-like, optional",
        default="[0, 1, 0]",
    )

    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        # Inherit docstring

        length_units = uck.get("length")
        scale = self.edges.m_as(length_units) * 0.5
        center = self.center.m_as(length_units)
        normal = self.normal
        up = self.up

        trafo = mi.ScalarTransform4f.look_at(
            origin=center, target=center + normal, up=up
        ) @ mi.ScalarTransform4f.scale([scale[0], scale[1], 1.0])

        result = KernelDict({self.id: {"type": "rectangle", "to_world": trafo}})

        if self.bsdf is not None:
            result[self.id]["bsdf"] = onedict_value(self.bsdf.kernel_dict(ctx))

        return result

    @classmethod
    def surface(
        cls,
        altitude: pint.Quantity = 0.0 * ureg.km,
        width: pint.Quantity = 1.0 * ureg.km,
        bsdf: t.Optional[BSDF] = None,
    ) -> RectangleShape:
        """
        This class method constructor provides a simplified parametrisation of
        the rectangle shape better suited for the definition of the surface when
        configuring the one-dimensional model.

        The resulting rectangle shape is a square with edge length equal to
        `width`, centred at [0, 0, `altitude`], with normal vector +Z.

        Parameters
        ----------
        altitude : quantity or array-like, optional, default: 0 km
            Surface altitude. If a unitless value is passed, it is interpreted
            as ``ucc["length"]``.

        width : quantity or float, optional, default: 1 km
            Edge length. If a unitless value is passed, it is interpreted
            as ``ucc["length"]``.

        bsdf : BSDF or dict, optional, default: None
            A BSDF specification, forwarded to the main constructor.

        Returns
        -------
        RectangleShape
            A rectangle shape which can be used as the surface in a
            plane parallel geometry.
        """
        altitude = pinttr.util.ensure_units(altitude, default_units=ucc.get("length"))
        return cls(
            edges=width,
            center=[0.0, 0.0, altitude.m] * altitude.units,
            normal=np.array([0, 0, 1]),
            up=np.array([0, 1, 0]),
            bsdf=bsdf,
        )
