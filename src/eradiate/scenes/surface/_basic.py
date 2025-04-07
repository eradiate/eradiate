from __future__ import annotations

import warnings

import attrs
import mitsuba as mi

from ._core import Surface
from ..bsdfs import BSDF, LambertianBSDF, bsdf_factory
from ..core import Ref, SceneTraversal, traverse
from ..shapes import RectangleShape, SphereShape, shape_factory
from ...attrs import define, documented
from ...exceptions import OverriddenValueWarning, TraversalError
from ...kernel import SceneParameter, SearchSceneParameter


@define(eq=False, slots=False)
class BasicSurface(Surface):
    """
    Basic surface [``basic``].

    A basic surface description consisting of a single shape and BSDF.
    """

    shape: None | RectangleShape | SphereShape = documented(
        attrs.field(
            default=None,
            converter=attrs.converters.optional(shape_factory.convert),
            validator=attrs.validators.optional(
                attrs.validators.instance_of((RectangleShape, SphereShape))
            ),
        ),
        doc="Shape describing the surface. This parameter may be left unset "
        "for situations in which the task of setting its value is delegated to "
        "another component (*e.g.* an :class:`.Experiment` instance owning the "
        "surface object); however, if it is still unset upon kernel dictionary "
        "generation, the call to :meth:`.traverse` will raise a "
        ":class:`.TraversalError`.",
        type=".RectangleShape or .SphereShape or None",
        init_type=".RectangleShape or .SphereShape or dict, optional",
        default=":class:`.RectangleShape <RectangleShape()>`",
    )

    @shape.validator
    def _shape_validator(self, attribute, value):
        if value is not None and value.bsdf is not None:
            warnings.warn(
                f"while validating '{attribute.name}': "
                f"'{attribute.name}.bsdf' should be set to None; it will "
                "be overridden during kernel dictionary generation",
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

    def update(self) -> None:
        # Inherit docstring

        # Fix BSDF ID
        self.bsdf.id = self._bsdf_id

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
        Mitsuba BSDF object identifier.
        """
        return f"{self.id}_bsdf"

    @property
    def _template_bsdfs(self) -> dict:
        kdict_template = traverse(self.bsdf)[0].data

        result = {}

        for key, param in kdict_template.items():
            result[f"{self._bsdf_id}.{key}"] = param

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
        umap_template = traverse(self.bsdf)[1].data

        result = {}

        for key, param in umap_template.items():
            # If no lookup strategy is set, we must add one
            if isinstance(param, SceneParameter) and param.search is None:
                param = attrs.evolve(
                    param,
                    search=SearchSceneParameter(
                        mi.BSDF,
                        self._bsdf_id,
                        parameter_relpath=key,
                    ),
                )

            result[f"{self._bsdf_id}.{key}"] = param

        return result

    @property
    def _params_shapes(self) -> dict:
        return {}

    def traverse(self, callback: SceneTraversal) -> None:
        # Inherit docstring
        if self.shape is None:
            raise TraversalError(
                "A 'BasicSurface' cannot be traversed if its 'shape' field is unset."
            )
        super().traverse(callback)
