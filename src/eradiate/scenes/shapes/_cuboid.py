from __future__ import annotations

import typing as t

import attr
import mitsuba as mi
import numpy as np
import pint
import pinttr
from pinttr.util import ensure_units

from ._core import Shape, shape_factory
from ..bsdfs import BSDF
from ..core import KernelDict
from ..._util import onedict_value
from ...attrs import documented, parse_docs
from ...contexts import KernelDictContext
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg


def _edges_converter(x):
    # Basic unit conversion and array reshaping
    length_units = ucc.get("length")
    x = np.reshape(
        pinttr.util.ensure_units(x, default_units=length_units).m_as(length_units),
        (-1,),
    )

    # Broadcast if relevant
    if len(x) == 1:
        x = np.full((3,), x[0])

    return x * length_units


@shape_factory.register(type_id="cuboid")
@parse_docs
@attr.s
class CuboidShape(Shape):
    """
    Cuboid shape [``cuboid``].

    This shape represents an axis-aligned cuboid parametrised by the length of
    its edges and the coordinates of its central point.

    Notes
    -----
    This is a wrapper around the ``cube`` kernel plugin.
    """

    center: pint.Quantity = documented(
        pinttr.ib(factory=lambda: [0, 0, 0], units=ucc.deferred("length")),
        doc="Coordinates of the centre of the cube. "
        'Unit-enabled field (default: ``ucc["length"]``).',
        type="quantity",
        init_type="quantity or array-like, optional",
        default="[0, 0, 0]",
    )

    edges: pint.Quantity = documented(
        pinttr.ib(
            factory=lambda: [1, 1, 1],
            converter=_edges_converter,
            units=ucc.deferred("length"),
        ),
        doc="Lengths of the edges of the cuboid. "
        'Unit-enabled field (default: ``ucc["length"]``).',
        type="quantity",
        init_type="quantity or array-like",
        default="[1, 1, 1]",
    )

    def contains(self, p: np.typing.ArrayLike, strict: bool = False) -> bool:
        """
        Test whether a point lies within the cuboid.

        Parameters
        ----------
        p : quantity or array-like
            An array of shape (3,) (resp. (N, 3)) representing one (resp. N)
            points. If a unitless value is passed, it is interpreted as
            ``ucc["length"]``.

        strict : bool
            If ``True``, comparison is done using strict inequalities (<, >).

        Returns
        -------
        result : array of bool or bool
            ``True`` iff ``p`` in within the cuboid.
        """

        node_min = self.center - 0.5 * self.edges
        node_max = self.center + 0.5 * self.edges

        p = np.atleast_2d(ensure_units(p, ucc.get("length")))
        cmp = (
            np.logical_and(p > node_min, p < node_max)
            if strict
            else np.logical_and(p >= node_min, p <= node_max)
        )
        return np.all(cmp, axis=1)

    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        # Inherit docstring

        length_units = uck.get("length")
        to_world = mi.ScalarTransform4f.translate(
            self.center.m_as(length_units)
        ) * mi.ScalarTransform4f.scale(0.5 * self.edges.m_as(length_units))

        result = KernelDict({self.id: {"type": "cube", "to_world": to_world}})

        if self.bsdf is not None:
            result[self.id]["bsdf"] = onedict_value(self.bsdf.kernel_dict(ctx))

        return result

    @classmethod
    def atmosphere(
        cls,
        top: pint.Quantity = 100.0 * ureg.km,
        bottom: pint.Quantity = 0.0 * ureg.km,
        bottom_offset: pint.Quantity = None,
        width: pint.Quantity = 100.0 * ureg.km,
        bsdf: t.Optional[BSDF] = None,
    ) -> CuboidShape:
        """
        This class method constructor provides a simplified parametrisation of
        the cuboid shape better suited for the definition of the atmosphere when
        configuring the one-dimensional model with a plane parallel geometry.

        Parameters
        ----------
        top : quantity, optional, default: 100 km
            Top of atmosphere altitude. If a unitless value is passed, it is
            interpreted as ucc["length"].

        bottom : quantity, optional, default: 0 km
            Ground altitude. If a unitless value is passed, it is interpreted as
            ``ucc["length"]``.

        bottom_offset : quantity, optional
            Additional offset by which the cuboid with be extended to avoid an
            exact match of its bottom face and the shape representing the
            surface. If left unset, defaults to a negative offset of  0.1 % of
            the atmosphere's height, *i.e.*
            :math:`- 0.001 \\times (\\mathtt{top} - \\mathtt{bottom})`.
            If a unitless value is passed, it is interpreted as
            ``ucc["length"]``.

        width : quantity, optional, default: 100 km
            Length of the horizontal edges of the cuboid. If a unitless value is
            passed, it is interpreted as ``ucc["length"]``.

        bsdf : BSDF or dict, optional, default: None
            A BSDF specification, forwarded to the main constructor.

        Returns
        -------
        CuboidShape
            A cuboid shape which can be used as the atmosphere in a plane
            parallel geometry.
        """
        top = pinttr.util.ensure_units(top, default_units=ucc.get("length"))
        bottom = pinttr.util.ensure_units(bottom, default_units=ucc.get("length"))
        width = pinttr.util.ensure_units(width, default_units=ucc.get("length"))

        if bottom_offset is None:
            bottom_offset = -0.001 * (top - bottom)
        else:
            bottom_offset = pinttr.util.ensure_units(
                bottom_offset, default_units=ucc.get("length")
            )

        length_units = top.units
        center_z = (bottom + bottom_offset).m_as(length_units)
        edge_x = np.squeeze(width.m_as(length_units))
        edge_y = np.squeeze(width.m_as(length_units))
        edge_z = np.squeeze(top.m_as(length_units) - center_z)

        return cls(
            center=[0.0, 0.0, center_z] * length_units,
            edges=[edge_x, edge_y, edge_z] * length_units,
            bsdf=bsdf,
        )
