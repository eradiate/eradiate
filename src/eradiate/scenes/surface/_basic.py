from __future__ import annotations

import typing as t
import warnings

import attr

from ._core import Surface, surface_factory
from ..bsdfs import BSDF, LambertianBSDF, bsdf_factory
from ..core import KernelDict
from ..shapes import RectangleShape, Shape, SphereShape, shape_factory
from ...attrs import documented, parse_docs
from ...contexts import KernelDictContext
from ...exceptions import OverriddenValueWarning


@surface_factory.register(type_id="basic")
@parse_docs
@attr.s
class BasicSurface(Surface):
    """
    Basic surface [``basic``].

    A basic surface description consisting of a single shape and BSDF.
    """

    shape: t.Optional[Shape] = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(shape_factory.convert),
            validator=attr.validators.optional(
                attr.validators.instance_of((RectangleShape, SphereShape))
            ),
        ),
        doc="Shape describing the surface. This parameter may be left unset "
        "for situations in which setting its value is delegated to another "
        "component (*e.g.* an :class:`.Experiment` instance owning the "
        "surface object); however, if it is still unset upon kernel dictionary "
        "generation, the call to :meth:`.kernel_dict` will raise.",
        type=".RectangleShape or .SphereShape or None",
        init_type=".RectangleShape or .SphereShape or dict, optional",
        default="None",
    )

    @shape.validator
    def _shape_validator(self, attribute, value):
        if value is not None:  # Means it's a Shape
            if value.bsdf is not None:
                warnings.warn(
                    f"while validating '{attribute.name}': "
                    f"'{attribute.name}.bsdf' should be set to None; it will "
                    "be overridden upon kernel dictionary generation",
                    OverriddenValueWarning,
                )

    bsdf: BSDF = documented(
        attr.ib(
            factory=LambertianBSDF,
            converter=bsdf_factory.convert,
            validator=attr.validators.instance_of(BSDF),
        ),
        doc="The reflection model attached to the surface.",
        type=".BSDF",
        init_type=".BSDF or dict, optional",
        default=":class:`LambertianBSDF() <.LambertianBSDF>`",
    )

    def kernel_shapes(self, ctx: KernelDictContext) -> KernelDict:
        # Inherit docstring

        if self.shape is None:
            # This is allowed for clarity: many Surface instances will actually
            # have their surface overridden by a higher-level component such as
            # and Experiment. However, the 'shape' field must be a Shape for
            # kernel dict generation to happen.
            raise ValueError(
                "The 'shape' field must be a Shape for kernel dictionary "
                "generation to work (got None)."
            )
        else:
            # Note: No coupling between the shape and BSDF is not done at this
            # level: this part is delegate to the kernel_dict() method.
            return self.shape.kernel_dict(ctx)

    def kernel_bsdfs(self, ctx: KernelDictContext) -> KernelDict:
        # Inherit docstring
        return self.bsdf.kernel_dict(ctx)
