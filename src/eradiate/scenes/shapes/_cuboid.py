from __future__ import annotations

import typing as t

import attrs
import mitsuba as mi
import numpy as np
import pint
import pinttr
from pinttr.util import ensure_units

from ._core import ShapeNode
from ..bsdfs import BSDF
from ..core import BoundingBox
from ...attrs import documented, parse_docs
from ...contexts import KernelContext
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


@parse_docs
@attrs.define(eq=False, slots=False)
class CuboidShape(ShapeNode):
    """
    Cuboid shape [``cuboid``].

    This shape represents an axis-aligned cuboid parametrized by the length of
    its edges and the coordinates of its central point.

    Notes
    -----
    * If the `to_world` parameter is set, the `center` and `edges` parameters will
      be ignored. The transform will be applied to a cube reaching from
      [-1, -1, -1] to [1, 1, 1].
    """

    center: pint.Quantity = documented(
        pinttr.field(factory=lambda: [0, 0, 0], units=ucc.deferred("length")),
        doc="Coordinates of the centre of the cube. "
        "Unit-enabled field (default: ``ucc['length']``).",
        type="quantity",
        init_type="quantity or array-like, optional",
        default="[0, 0, 0]",
    )

    edges: pint.Quantity = documented(
        pinttr.field(
            factory=lambda: [1, 1, 1],
            converter=_edges_converter,
            units=ucc.deferred("length"),
        ),
        doc="Lengths of the edges of the cuboid. "
        "Unit-enabled field (default: ``ucc['length]``).",
        type="quantity",
        init_type="quantity or array-like",
        default="[1, 1, 1]",
    )

    @property
    def bbox(self) -> BoundingBox:
        # Inherit docstring

        if self.to_world is None:
            return BoundingBox(
                self.center - 0.5 * self.edges, self.center + 0.5 * self.edges
            )
        else:
            p1 = np.array((-1, -1, -1))
            p2 = np.array((1, -1, -1))
            p3 = np.array((-1, 1, -1))
            p4 = np.array((-1, -1, 1))
            p5 = np.array((-1, 1, 1))
            p6 = np.array((1, -1, 1))
            p7 = np.array((1, 1, -1))
            p8 = np.array((1, 1, 1))

            max = np.array([-np.inf, -np.inf, -np.inf])
            min = np.array([np.inf, np.inf, np.inf])

            for p in [p1, p2, p3, p4, p5, p6, p7, p8]:
                p_t = np.array(self.to_world @ p)

                max = np.where(p_t > max, p_t, max)
                min = np.where(p_t < min, p_t, min)

            return BoundingBox(min, max)

    def contains(
        self, p: np.typing.ArrayLike, strict: bool = False
    ) -> t.Sequence[bool]:
        """
        Test whether a point lies within the cuboid.

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
            ``True`` iff ``p`` in within the cuboid.
        """

        if self.to_world is None:
            node_min = self.center - 0.5 * self.edges
            node_max = self.center + 0.5 * self.edges

            p = np.atleast_2d(ensure_units(p, ucc.get("length")))
            cmp = (
                np.logical_and(p > node_min, p < node_max)
                if strict
                else np.logical_and(p >= node_min, p <= node_max)
            )
            return np.all(cmp, axis=1)
        else:
            p0 = np.array(self.to_world @ np.array((0, 0, 0)))
            p1 = np.array(self.to_world @ np.array((-1, -1, -1)))
            p2 = np.array(self.to_world @ np.array((1, -1, -1)))
            p3 = np.array(self.to_world @ np.array((-1, 1, -1)))
            p4 = np.array(self.to_world @ np.array((-1, -1, 1)))

            v1 = p2 - p1
            l1 = np.linalg.norm(v1)
            n1 = v1 / l1

            v2 = p3 - p1
            l2 = np.linalg.norm(v2)
            n2 = v2 / l2

            v3 = p4 - p1
            l3 = np.linalg.norm(v3)
            n3 = v3 / l3

            p = np.atleast_2d(
                ensure_units(p, ucc.get("length")).m_as(ucc.get("length"))
            )
            vp = p - p0

            if strict:
                cmp = np.prod(
                    [
                        np.dot(vp, n1) < l1 / 2.0,
                        np.dot(vp, n2) < l2 / 2.0,
                        np.dot(vp, n3) < l3 / 2.0,
                    ],
                    axis=0,
                )
            else:
                cmp = np.prod(
                    [
                        np.dot(vp, n1) <= l1 / 2.0,
                        np.dot(vp, n2) <= l2 / 2.0,
                        np.dot(vp, n3) <= l3 / 2.0,
                    ],
                    axis=0,
                )
            return cmp

    def eval_to_world(self, ctx: KernelContext | None = None) -> mi.ScalarTransform4f:
        kwargs = ctx.kwargs.get(self.id, {}) if ctx is not None else {}

        if "to_world" in kwargs:
            return kwargs["to_world"]

        else:
            length_units = uck.get("length")
            converters = {field.name: field.converter for field in self.__attrs_attrs__}

            edges = (
                converters["edges"](kwargs["edges"])
                if "edges" in kwargs
                else self.edges
            )

            center = (
                converters["center"](kwargs["center"])
                if "center" in kwargs
                else self.center
            )

            return mi.ScalarTransform4f.translate(
                center.m_as(length_units)
            ) @ mi.ScalarTransform4f.scale(0.5 * edges.m_as(length_units))

    @property
    def template(self) -> dict:
        length_units = uck.get("length")

        if self.to_world is not None:
            trafo = self.to_world
        else:
            trafo = mi.ScalarTransform4f.translate(
                self.center.m_as(length_units)
            ) @ mi.ScalarTransform4f.scale(0.5 * self.edges.m_as(length_units))

        return {
            "type": "cube",
            "to_world": trafo,
        }

    @classmethod
    def atmosphere(
        cls,
        top: pint.Quantity = 100.0 * ureg.km,
        bottom: pint.Quantity = 0.0 * ureg.km,
        bottom_offset: pint.Quantity = None,
        width: pint.Quantity = 100.0 * ureg.km,
        bsdf: BSDF | None = None,
    ) -> CuboidShape:
        """
        This class method constructor provides a simplified parametrization of
        the cuboid shape better suited for the definition of the atmosphere when
        configuring the one-dimensional model with a plane parallel geometry.

        Parameters
        ----------
        top : quantity, optional, default: 100 km
            Top of atmosphere altitude. If a unitless value is passed, it is
            interpreted as ucc['length'].

        bottom : quantity, optional, default: 0 km
            Ground altitude. If a unitless value is passed, it is interpreted as
            ``ucc['length']``.

        bottom_offset : quantity, optional
            Additional offset by which the cuboid with be extended to avoid an
            exact match of its bottom face and the shape representing the
            surface. If left unset, defaults to a negative offset of  1 % of
            the atmosphere's height, *i.e.*
            :math:`- 0.01 \\times (\\mathtt{top} - \\mathtt{bottom})`.
            If a unitless value is passed, it is interpreted as ``ucc['length']``.

        width : quantity, optional, default: 100 km
            Length of the horizontal edges of the cuboid.
            If a unitless value is passed, it is interpreted as ``ucc['length']``.

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
            bottom_offset = -0.01 * (top - bottom)
        else:
            bottom_offset = pinttr.util.ensure_units(
                bottom_offset, default_units=ucc.get("length")
            )

        length_units = top.units
        center_z = 0.5 * (top + bottom + bottom_offset).m_as(length_units)
        edge_x = np.squeeze(width.m_as(length_units))
        edge_y = np.squeeze(width.m_as(length_units))
        edge_z = np.squeeze((top - bottom - bottom_offset).m_as(length_units))

        return cls(
            center=[0.0, 0.0, center_z] * length_units,
            edges=[edge_x, edge_y, edge_z] * length_units,
            bsdf=bsdf,
        )
