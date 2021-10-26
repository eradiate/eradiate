from __future__ import annotations

import typing as t

import attr
import numpy as np

from ._core import Measure, measure_factory
from ._target import Target, TargetPoint, TargetRectangle
from ..core import KernelDict
from ... import validators
from ...attrs import documented, parse_docs
from ...contexts import KernelDictContext


@measure_factory.register(type_id="distant_flux")
@parse_docs
@attr.s
class DistantFluxMeasure(Measure):
    """
    Record the exitant flux density (in W/m²(/nm)) at infinite distance in a
    hemisphere defined by its ``direction`` parameter.

    When used with a backward tracing algorithm, rays traced by the sensor
    target a shape which can be controlled through the ``target`` parameter.
    This feature is useful if one wants to compute the average flux leaving
    a particular subset of the scene.

    Notes
    -----
    * Setting the ``target`` parameter is required to get meaningful results.
      Solver applications should take care of setting it appropriately.
    * The film resolution can be adjusted to manually stratify film sampling
      and reduce variance in results. The default 32x32 is generally a good
      choice, but scenes with sharp reflection lobes may benefit from higher
      values.
    * This scene element is a thin wrapper around the ``distantflux``
      sensor kernel plugin.
    """

    # --------------------------------------------------------------------------
    #                           Fields and properties
    # --------------------------------------------------------------------------

    direction: np.ndarray = documented(
        attr.ib(
            default=[0, 0, 1],
            converter=np.array,
            validator=validators.is_vector3,
        ),
        doc="A 3-vector defining the normal to the reference surface for which "
        "the exitant flux density is computed.",
        type="array-like",
        default="[0, 0, 1]",
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

    _film_resolution: t.Tuple[int, int] = documented(
        attr.ib(
            default=(32, 32),
            validator=attr.validators.deep_iterable(
                member_validator=attr.validators.instance_of(int),
                iterable_validator=validators.has_len(2),
            ),
        ),
        doc="Film resolution as a (width, height) 2-tuple.",
        type="array-like",
        default="(32, 32)",
    )

    @property
    def film_resolution(self) -> t.Tuple[int, int]:
        return self._film_resolution

    # --------------------------------------------------------------------------
    #                        Kernel dictionary generation
    # --------------------------------------------------------------------------

    def _kernel_dict(self, sensor_id, spp):
        from mitsuba.core import ScalarTransform4f, ScalarVector3f, coordinate_system

        _, up = coordinate_system(self.direction)

        result = {
            "type": "distantflux",
            "id": sensor_id,
            "to_world": ScalarTransform4f.look_at(
                origin=[0, 0, 0],
                target=ScalarVector3f(self.direction),
                up=up,
            ),
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
            result["target"] = self.target.kernel_item()

        return result

    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        sensor_ids = self._sensor_ids()
        sensor_spps = self._sensor_spps()
        result = KernelDict()

        for spp, sensor_id in zip(sensor_spps, sensor_ids):
            result.data[sensor_id] = self._kernel_dict(sensor_id, spp)

        return result
