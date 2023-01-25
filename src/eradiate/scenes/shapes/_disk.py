from __future__ import annotations

import attrs
import numpy as np
import pint
import pinttr

import mitsuba as mi

from ._core import Shape
from ..core import KernelDict
from ... import validators
from ...attrs import documented, parse_docs
from ...contexts import KernelDictContext
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
from ...util.misc import onedict_value


def _normalize(v: np.typing.ArrayLike) -> np.ndarray:
    return np.array(v) / np.linalg.norm(v)


@parse_docs
@attrs.define
class DiskShape(Shape):
    """
    Disk shape [``disk``].

    This shape represents a disk parametrised by its center, normal and radius.
    """

    center: pint.Quantity = documented(
        pinttr.field(factory=lambda: [0, 0, 0], units=ucc.deferred("length")),
        doc="Location of the centre of the disk. Unit-enabled field "
        '(default: ``ucc["length"]``).',
        type="quantity",
        init_type="quantity or array-like, optional",
        default="[0, 0, 0]",
    )

    normal: np.ndarray = documented(
        attrs.field(
            factory=lambda: [0, 0, 1],
            converter=_normalize,
            validator=validators.is_vector3,
        ),
        doc="Normal vector of the plane containing the disk. Defaults to "
        "the +Z direction.",
        type="array",
        init_type="array-like, optional",
        default="[0, 0, 1]",
    )

    radius: pint.Quantity = documented(
        pinttr.field(
            factory=lambda: 1.0 * ucc.get("length"), units=ucc.deferred("length")
        ),
        doc='Disk radius. Unit-enabled field (default: ``ucc["length"]``).',
        type="quantity",
        init_type="quantity or float, optional",
        default="1.0",
    )

    up: np.ndarray = documented(
        attrs.field(
            factory=lambda: [0, 1, 0],
            converter=_normalize,
            validator=validators.is_vector3,
        ),
        doc="Orientation vector defining the rotation of the disk around "
        "the normal vector. Defaults to the +Y direction.",
        type="array",
        init_type="array-like, optional",
        default="[0, 1, 0]",
    )

    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        # Inherit docstring

        length_units = uck.get("length")
        scale = self.radius.m_as(length_units)
        center = self.center.m_as(length_units)
        up = self.up

        trafo = mi.ScalarTransform4f.look_at(
            origin=center, target=center + self.normal, up=up
        ) @ mi.ScalarTransform4f.scale(scale)

        result = KernelDict({self.id: {"type": "disk", "to_world": trafo}})

        if self.bsdf is not None:
            result[self.id]["bsdf"] = onedict_value(self.bsdf.kernel_dict(ctx))

        return result
