import typing as t

import attr
import pint
import pinttr

import eradiate
from ._core import Surface, surface_factory
from ._lambertian import LambertianSurface
from ..core import KernelDict
from ... import converters, validators
from ..._util import onedict_value
from ...attrs import documented, parse_docs, AUTO, AutoType
from ...contexts import KernelDictContext
from ...units import unit_context_config as ucc


@surface_factory.register(type_id="central_patch")
@parse_docs
@attr.s
class CentralPatchSurface(Surface):
    """
    Central patch surface element [``central_patch``]

    This class creates a square surface to which two BSDFs will be attached.
    The two constituent  surfaces ``central_patch`` and ``background_surface`` define the
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

    central_patch: Surface = documented(
        attr.ib(
            factory=LambertianSurface,
            converter=surface_factory.convert,
            validator=attr.validators.instance_of(Surface),
        ),
        doc="Central patch specification. "
        "This parameter can be specified as a dictionary which will be "
        "interpreted by "
        ":meth:`SurfaceFactory.convert() <.SurfaceFactory.convert>`.",
        type=":class:`.Surface`",
        init_type=":class:`.Surface` or dict",
        default=":class:`LambertianSurface() <.LambertianSurface>`",
    )

    background_surface: Surface = documented(
        attr.ib(
            factory=LambertianSurface,
            converter=surface_factory.convert,
            validator=[
                attr.validators.instance_of(Surface),
            ],
        ),
        doc="Outer surface specification. "
        "This parameter can be specified as a dictionary which will be "
        "interpreted by "
        ":meth:`SurfaceFactory.convert() <.SurfaceFactory.convert>`.",
        type=":class:`.Surface`",
        init_type=":class:`.Surface` or dict",
        default=":class:`LambertianSurface() <.LambertianSurface>`",
    )

    @background_surface.validator
    def _bg_surface_width_is_auto_validator(self, attribute, value):
        if not value.width == AUTO:
            raise ValueError(
                f"background_surface.width must be set to 'AUTO'\n"
                f"got: {value.width}"
            )

    def _compute_scale_parameter(self, ctx: KernelDictContext) -> float:
        """
        Compute the scaling parameter for the bitmap texture in the blendbsdf.
        """
        if self.central_patch.width is AUTO and not ctx.override_canopy_width:
            return 1
        else:
            width = ctx.override_scene_width if ctx.override_scene_width else self.width
            patch_width = (
                ctx.override_canopy_width
                if ctx.override_canopy_width
                else self.central_patch.width
            )
            # the size of the central patch is one third of the overall size of the texture
            return width / (3 * patch_width)

    def bsdfs(self, ctx: KernelDictContext) -> KernelDict:
        from mitsuba.core import ScalarTransform4f

        returndict = KernelDict(
            {
                f"bsdf_{self.id}": {
                    "type": "blendbsdf",
                    "inner_bsdf": onedict_value(self.central_patch.bsdfs(ctx=ctx)),
                    "outer_bsdf": onedict_value(self.background_surface.bsdfs(ctx=ctx)),
                    "weight": {
                        "type": "bitmap",
                        "filename": str(
                            eradiate.path_resolver.resolve(
                                "textures/rami4atm_experiment_surface_mask.bmp"
                            )
                        ),
                        "filter_type": "nearest",
                        "wrap_mode": "clamp",
                    },
                }
            }
        )

        scale = self._compute_scale_parameter(ctx=ctx)
        trafo = ScalarTransform4f.scale(scale) * ScalarTransform4f.translate(
            (
                -0.5 + (0.5 / scale),
                -0.5 + (0.5 / scale),
                0,
            )
        )

        returndict[f"bsdf_{self.id}"]["weight"]["to_uv"] = trafo

        return returndict
