from __future__ import annotations

import typing as t
import warnings

import attrs

from ._core import SurfaceComposite
from ..bsdfs import BSDF, LambertianBSDF, bsdf_factory
from ..core import NodeSceneElement, Ref, SceneTraversal
from ..shapes import RectangleShape, SphereShape, shape_factory
from ...attrs import documented, parse_docs
from ...exceptions import OverriddenValueWarning, TraversalError


@parse_docs
@attrs.define(eq=False, slots=False)
class BasicSurface(SurfaceComposite):
    """
    Basic surface [``basic``].

    A basic surface description consisting of a single shape and BSDF.
    """

    shape: t.Union[None, RectangleShape, SphereShape] = documented(
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
        "generation, the call to :meth:`.kernel_dict` will raise.",
        type=".RectangleShape or .SphereShape or None",
        init_type=".RectangleShape or .SphereShape or dict, optional",
        default=":class:`.RectangleShape <RectangleShape()>",
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
    def objects(self) -> t.Dict[str, NodeSceneElement]:
        # Mind the order: the BSDF has to be BEFORE the shape
        return {self._bsdf_id: self.bsdf, self._shape_id: self.shape}

    def traverse(self, callback: SceneTraversal) -> None:
        if self.shape is None:
            raise TraversalError(
                "A 'BasicSurface' cannot be traversed if its 'shape' field is unset."
            )
        super().traverse(callback)
