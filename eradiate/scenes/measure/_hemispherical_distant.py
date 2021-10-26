from __future__ import annotations

import typing as t

import attr
import numpy as np
import pint
import pinttr

from eradiate import unit_context_config as ucc
from eradiate import unit_registry as ureg

from ._core import Measure, MeasureFlags, measure_factory
from ._target import Target, TargetPoint, TargetRectangle
from ..core import KernelDict
from ... import validators
from ...attrs import documented, get_doc, parse_docs
from ...contexts import KernelDictContext


@measure_factory.register(type_id="hemispherical_distant")
@parse_docs
@attr.s
class HemisphericalDistantMeasure(Measure):

    # --------------------------------------------------------------------------
    #                           Fields and properties
    # --------------------------------------------------------------------------

    _film_resolution: t.Tuple[int, int] = documented(
        attr.ib(
            default=(32, 32),
            validator=attr.validators.deep_iterable(
                member_validator=attr.validators.instance_of(int),
                iterable_validator=validators.has_len(2),
            ),
        ),
        doc="Film resolution as a (width, height) 2-tuple. "
        "If the height is set to 1, direction sampling will be restricted to a "
        "plane.",
        type="array-like",
        default="(32, 32)",
    )

    orientation: pint.Quantity = documented(
        pinttr.ib(
            default=ureg.Quantity(0.0, ureg.deg),
            validator=validators.is_positive,
            units=ucc.deferred("angle"),
        ),
        doc="Azimuth angle defining the orientation of the sensor in the "
        "horizontal plane.\n"
        "\n"
        "Unit-enabled field (default: ucc['angle']).",
        type="float",
        default="0.0 deg",
    )

    direction = documented(
        attr.ib(
            default=[0, 0, 1],
            converter=np.array,
            validator=validators.is_vector3,
        ),
        doc="A 3-vector orienting the hemisphere mapped by the measure.",
        type="array-like",
        default="[0, 0, 1]",
    )

    flip_directions = documented(
        attr.ib(default=None, converter=attr.converters.optional(bool)),
        doc=" If ``True``, sampled directions will be flipped.",
        type="bool",
        default="False",
    )

    target: t.Optional[Target] = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(Target.convert),
            validator=attr.validators.optional(
                attr.validators.instance_of(
                    (
                        TargetPoint,
                        TargetRectangle,
                    )
                )
            ),
            on_setattr=attr.setters.pipe(attr.setters.convert, attr.setters.validate),
        ),
        doc="Target specification. The target can be specified using an "
        "array-like with 3 elements (which will be converted to a "
        ":class:`.TargetOriginPoint`) or a dictionary interpreted by "
        ":meth:`TargetOrigin.convert() <.TargetOrigin.convert>`. If set to "
        "``None`` (not recommended), the default target point selection "
        "method is used: rays will not target a particular region of the "
        "scene.",
        type=":class:`.TargetOrigin` or None",
        init_type=":class:`.TargetOrigin` or dict or array-like, optional",
    )

    @property
    def film_resolution(self):
        return self._film_resolution

    flags: MeasureFlags = documented(
        attr.ib(default=MeasureFlags.DISTANT, converter=MeasureFlags, init=False),
        doc=get_doc(Measure, "flags", "doc"),
        type=get_doc(Measure, "flags", "type"),
    )

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    def _kernel_dict(self, sensor_id, spp):
        result = {
            "type": "distant",
            "id": sensor_id,
            "direction": self.direction,
            "orientation": [
                np.cos(self.orientation.m_as(ureg.rad)),
                np.sin(self.orientation.m_as(ureg.rad)),
                0.0,
            ],
            "sampler": {
                "type": "independent",
                "sample_count": spp,
            },
            "film": {
                "type": "hdrfilm",
                "width": self.film_resolution[0],
                "height": self.film_resolution[1],
                "pixel_format": "luminance",
                "component_format": "float32",
                "rfilter": {"type": "box"},
            },
        }

        if self.target is not None:
            result["ray_target"] = self.target.kernel_item()

        if self.flip_directions is not None:
            result["flip_directions"] = self.flip_directions

        return result

    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        sensor_ids = self._sensor_ids()
        sensor_spps = self._sensor_spps()
        result = KernelDict()

        for spp, sensor_id in zip(sensor_spps, sensor_ids):
            result.data[sensor_id] = self._kernel_dict(sensor_id, spp)

        return result

    # --------------------------------------------------------------------------
    #                        Post-processing information
    # --------------------------------------------------------------------------

    @property
    def var(self) -> str:
        return "radiance"
