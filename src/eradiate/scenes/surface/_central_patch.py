import typing as t
import warnings

import attr
import mitsuba as mi
import numpy as np
import pint
import pinttr

import eradiate

from ._core import Surface
from ..bsdfs import BSDF, BlackBSDF, LambertianBSDF, bsdf_factory
from ..core import KernelDict
from ..shapes import RectangleShape, shape_factory
from ...attrs import documented, parse_docs
from ...contexts import KernelDictContext
from ...exceptions import OverriddenValueWarning
from ...units import unit_context_config as ucc
from ...util.misc import onedict_value


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


@parse_docs
@attr.s
class CentralPatchSurface(Surface):
    """
    Central patch surface [``central_patch``].

    This surface consists of a rectangular patch, described by its `field`
    parameter, with a composite reflection model composed of a background
    uniform component, and a central patch.


    This class creates a square surface to which two BSDFs will be attached.

    The two constituent surfaces ``central_patch`` and ``background_surface`` define the
    properties of the two sections of this surface.

    The size of the central surface is controlled by setting the ``width`` parameter of the
    ``central_patch`` surface, while the ``width`` of the ``background_surface`` must be set to
    ``AUTO`` and the total width of the surface is set by the ``width`` of the main surface object.
    Note that the ``width`` of a surface defaults to ``AUTO``, which means, omitting the parameter
    in the ``background_surface`` will yield the correct behaviour.

    If the ``central_patch`` width is set to ``AUTO`` as well it defaults to one third of the
    overall surface size, unless a contextual constraint (*e.g.* to match the size of an
    atmosphere or canopy) is applied.
    """

    shape: t.Optional[RectangleShape] = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(shape_factory.convert),
            validator=attr.validators.optional(
                attr.validators.instance_of(RectangleShape)
            ),
        ),
        doc="Shape describing the surface. This parameter may be left unset "
        "for situations in which setting its value is delegated to another "
        "component (*e.g.* an :class:`.Experiment` instance owning the "
        "surface object); however, if it is still unset upon kernel dictionary "
        "generation, the call to :meth:`.kernel_dict` will raise.",
        type=".RectangleShape or None",
        init_type=".RectangleShape or dict, optional",
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

    patch_edges: t.Optional[pint.Quantity] = documented(
        pinttr.ib(
            default=None,
            converter=attr.converters.optional(_edges_converter),
            units=ucc.deferred("length"),
        ),
        doc="Length of the central patch's edges. If unset, the central patch "
        "edges will be 1/3 of the surface's edges. "
        'Unit-enabled field (default: ``ucc["length"]``).',
        type="quantity or None",
        init_type="quantity or array-like, optional",
    )

    patch_bsdf: BSDF = documented(
        attr.ib(
            factory=BlackBSDF,
            converter=bsdf_factory.convert,
            validator=attr.validators.instance_of(BSDF),
        ),
        doc="The reflection model attached to the central patch.",
        type=".BSDF",
        init_type=".BSDF or dict, optional",
        default=":class:`BlackBSDF() <.BlackBSDF>`",
    )

    def _texture_scale(self):
        """
        Compute patch texture scaling factor based on configuration.
        """
        # Note: The texture file has a central patch covering 1/3 of its
        # surface, hence the 1/3 factor.
        return (
            [1.0, 1.0]
            if self.patch_edges is None
            else (self.shape.edges / (3.0 * self.patch_edges)).m_as("dimensionless")
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

        scale = self._texture_scale()
        to_uv = mi.ScalarTransform4f.scale(
            [scale[0], scale[1], 1]
        ) @ mi.ScalarTransform4f.translate(
            [-0.5 + (0.5 / scale[0]), -0.5 + (0.5 / scale[1]), 0.0]
        )

        return KernelDict(
            {
                self.bsdf_id: {
                    "type": "blendbsdf",
                    "bsdf_0": onedict_value(self.patch_bsdf.kernel_dict(ctx=ctx)),
                    "bsdf_1": onedict_value(self.bsdf.kernel_dict(ctx=ctx)),
                    "weight": {
                        "type": "bitmap",
                        "filename": str(
                            eradiate.data.data_store.fetch(
                                "textures/central_patch_surface_mask.bmp"
                            )
                        ),
                        "filter_type": "nearest",
                        "to_uv": to_uv,
                        "wrap_mode": "clamp",
                    },
                }
            }
        )
