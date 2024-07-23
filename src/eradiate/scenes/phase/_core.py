from __future__ import annotations

from abc import ABC

import attrs

from ..core import NodeSceneElement, SceneElement
from ..._factory import Factory
from ...attrs import define, documented, get_doc

phase_function_factory = Factory()
phase_function_factory.register_lazy_batch(
    [
        (
            "_blend.BlendPhaseFunction",
            "blend_phase",
            {},
        ),
        (
            "_hg.HenyeyGreensteinPhaseFunction",
            "hg",
            {},
        ),
        (
            "_isotropic.IsotropicPhaseFunction",
            "isotropic",
            {},
        ),
        (
            "_rayleigh.RayleighPhaseFunction",
            "rayleigh",
            {},
        ),
        (
            "_tabulated.TabulatedPhaseFunction",
            "tab_phase",
            {},
        ),
    ],
    cls_prefix="eradiate.scenes.phase",
)


@define(eq=False, slots=False)
class PhaseFunction(NodeSceneElement, ABC):
    """
    An abstract base class defining common facilities for all phase functions.
    """

    id: str | None = documented(
        attrs.field(
            default="phase",
            validator=attrs.validators.optional(attrs.validators.instance_of(str)),
        ),
        doc=get_doc(SceneElement, "id", "doc"),
        type=get_doc(SceneElement, "id", "type"),
        init_type=get_doc(SceneElement, "id", "init_type"),
        default='"phase"',
    )
