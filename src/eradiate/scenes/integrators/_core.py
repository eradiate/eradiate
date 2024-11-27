from __future__ import annotations

from abc import ABC

import attrs

from ..core import NodeSceneElement
from ..._factory import Factory
from ...attrs import define, documented, get_doc

integrator_factory = Factory()
integrator_factory.register_lazy_batch(
    [
        ("_path_tracers.PathIntegrator", "path", {}),
        ("_path_tracers.VolPathIntegrator", "volpath", {}),
        ("_path_tracers.VolPathMISIntegrator", "volpathmis", {}),
        ("_path_tracers.PiecewiseVolPathIntegrator", "piecewise_volpath", {}),
    ],
    cls_prefix="eradiate.scenes.integrators",
)


@define(eq=False, slots=False)
class Integrator(NodeSceneElement, ABC):
    """
    Abstract base class for all integrator elements.

    Notes
    -----
    * This class is to be used as a mixin.
    """

    id: str | None = documented(
        attrs.field(
            default="integrator",
            validator=attrs.validators.optional(attrs.validators.instance_of(str)),
        ),
        doc=get_doc(NodeSceneElement, "id", "doc"),
        type=get_doc(NodeSceneElement, "id", "type"),
        init_type=get_doc(NodeSceneElement, "id", "init_type"),
        default='"integrator"',
    )

    timeout: float | None = documented(
        attrs.field(
            default=None,
            converter=attrs.converters.optional(float),
        ),
        doc="Maximum amount of time to spend during a kernel render pass in "
        "second (excluding scene parsing).",
        type="float or None",
        init_type="float, optional",
        default="None",
    )

    moment: bool = documented(
        attrs.field(
            default=False,
            converter=attrs.converters.optional(bool),
        ),
        doc="If ``True``, calculates the variance of the output film.",
        type="bool",
        init_type="bool, optional",
        default="False",
    )

    stokes: bool = documented(
        attrs.field(
            default=False,
            converter=attrs.converters.optional(bool),
        ),
        doc="If ``True``, calculate and output the polarized components of "
        "the Stokes vector. Note that this will only work with a polarized "
        "mode and a sensor that measures radiance.",
        type="bool",
        init_type="bool, optional",
        default="False",
    )

    meridian_align: bool = documented(
        attrs.field(
            default=True,
            converter=attrs.converters.optional(bool),
        ),
        doc="If ``True``, align the Stokes vector to the meridian plane. "
        "Otherwise, default to aligning to the sensor's x-axis. This parameter "
        "is only relevant in polarized mode, when measuring radiance with "
        "``stokes=True``.",
        type="bool",
        init_type="bool, optional",
        default="True",
    )
