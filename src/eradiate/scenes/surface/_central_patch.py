from __future__ import annotations

import warnings

import attrs
import mitsuba as mi
import numpy as np
import pint
import pinttr

from ._core import Surface
from ..bsdfs import BSDF, BlackBSDF, LambertianBSDF, bsdf_factory
from ..core import Ref, SceneTraversal, traverse
from ..shapes import RectangleShape, shape_factory
from ...attrs import define, documented
from ...data import fresolver
from ...exceptions import OverriddenValueWarning, TraversalError
from ...units import unit_context_config as ucc


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


@define(eq=False, slots=False)
class CentralPatchSurface(Surface):
    """
    Central patch surface [``central_patch``].

    This surface consists of a rectangular patch, described by its `field`
    parameter, with a composite reflection model composed of a background
    uniform component, and a central patch.


    This class creates a square surface to which two BSDFs will be attached.

    The two constituent surfaces ``central_patch`` and ``background_surface``
    define the properties of the two sections of this surface.

    The size of the central surface is controlled by setting the ``width``
    parameter of the ``central_patch`` surface, while the ``width`` of the
    ``background_surface`` must be set to ``AUTO`` and the total width of the
    surface is set by the ``width`` of the main surface object.
    Note that the ``width`` of a surface defaults to ``AUTO``, which means,
    omitting the parameter in the ``background_surface`` will yield the correct
    behaviour.

    If the ``central_patch`` width is set to ``AUTO`` as well it defaults to one
    third of the overall surface size, unless a contextual constraint (*e.g.* to
    match the size of an atmosphere or canopy) is applied.
    """

    shape: RectangleShape | None = documented(
        attrs.field(
            default=None,
            converter=attrs.converters.optional(shape_factory.convert),
            validator=attrs.validators.optional(
                attrs.validators.instance_of(RectangleShape)
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
        attrs.field(
            factory=LambertianBSDF,
            converter=bsdf_factory.convert,
            validator=attrs.validators.instance_of(BSDF),
        ),
        doc="The reflection model attached to the surface.",
        type=".BSDF",
        init_type=".BSDF or dict, optional",
        default=":class:`LambertianBSDF() <.LambertianBSDF>`",
    )

    patch_edges: pint.Quantity | None = documented(
        pinttr.field(
            default=None,
            converter=attrs.converters.optional(_edges_converter),
            units=ucc.deferred("length"),
        ),
        doc="Length of the central patch's edges. If unset, the central patch "
        "edges will be 1/3 of the surface's edges. "
        "Unit-enabled field (default: ``ucc['length']``).",
        type="quantity or None",
        init_type="quantity or array-like, optional",
    )

    patch_bsdf: BSDF = documented(
        attrs.field(
            factory=BlackBSDF,
            converter=bsdf_factory.convert,
            validator=attrs.validators.instance_of(BSDF),
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

    def update(self) -> None:
        # Inherit docstring

        # Fix BSDF IDs
        self.bsdf.id = self._background_bsdf_id
        self.patch_bsdf.id = self._patch_bsdf_id

        # Force BSDF referencing if the shape is defined
        if self.shape is not None:
            if isinstance(self.shape.bsdf, BSDF):
                warnings.warn("Set BSDF will be overridden by surface BSDF settings.")
            self.shape.bsdf = Ref(id=self._bsdf_id)

    @property
    def _shape_id(self):
        """
        Mitsuba shape object identifier.
        """
        return f"{self.id}_shape"

    @property
    def _bsdf_id(self):
        """
        Mitsuba BSDF object identifier (blend).
        """
        return f"{self.id}_bsdf"

    @property
    def _background_bsdf_id(self):
        """
        Mitsuba BSDF object identifier (background).
        """
        return f"{self.id}_background_bsdf"

    @property
    def _patch_bsdf_id(self):
        """
        Mitsuba BSDF object identifier (patch).
        """
        return f"{self.id}_patch_bsdf"

    @property
    def _template_bsdfs(self) -> dict:
        objects = {
            "bsdf_0": traverse(self.bsdf)[0].data,
            "bsdf_1": traverse(self.patch_bsdf)[0].data,
        }

        scale = self._texture_scale()
        to_uv = mi.ScalarTransform4f.scale(
            [scale[0], scale[1], 1]
        ) @ mi.ScalarTransform4f.translate(
            [-0.5 + (0.5 / scale[0]), -0.5 + (0.5 / scale[1]), 0.0]
        )

        result = {f"{self._bsdf_id}.type": "blendbsdf"}

        for obj_key, obj_params in objects.items():
            for key, param in obj_params.items():
                result[f"{self._bsdf_id}.{obj_key}.{key}"] = param

        weight_dict = {
            "type": "bitmap",
            "filename": str(
                fresolver.resolve("texture/central_patch_surface_mask.bmp")
            ),
            "filter_type": "nearest",
            "to_uv": to_uv,
            "wrap_mode": "clamp",
        }

        for key, param in weight_dict.items():
            result[f"{self._bsdf_id}.weight.{key}"] = param

        return result

    @property
    def _template_shapes(self) -> dict:
        kdict_template = traverse(self.shape)[0].data

        result = {}

        for key, param in kdict_template.items():
            result[f"{self._shape_id}.{key}"] = param

        return result

    @property
    def _params_bsdfs(self) -> dict:
        objects = {
            "bsdf_0": traverse(self.bsdf)[1].data,
            "bsdf_1": traverse(self.patch_bsdf)[1].data,
        }

        result = {}

        for obj_key, obj_params in objects.items():
            for key, param in obj_params.items():
                result[f"{self._bsdf_id}.{obj_key}.{key}"] = param

        return result

    @property
    def _params_shapes(self) -> dict:
        return {}

    def traverse(self, callback: SceneTraversal) -> None:
        # Inherit docstring

        if self.shape is None:
            raise TraversalError(
                "A 'CentralPatchSurface' cannot be traversed if its 'shape' field "
                "is unset."
            )

        super().traverse(callback)
