import attr
import numpy as np
import pinttr

from ._core import Measure, MeasureFactory
from ... import validators
from ..._attrs import documented, parse_docs
from ..._units import unit_context_config as ucc
from ..._units import unit_context_kernel as uck
from ..._units import unit_registry as ureg


@MeasureFactory.register("perspective")
@parse_docs
@attr.s
class PerspectiveCameraMeasure(Measure):
    """Perspective camera scene element [:factorykey:`perspective`].

    This scene element is a thin wrapper around the ``perspective`` sensor
    kernel plugin. It positions a perspective camera based on a set of vectors,
    specifying the origin, viewing direction and 'up' direction of the camera.
    """

    origin = documented(
        pinttr.ib(
            default=ureg.Quantity([1, 1, 1], ureg.m),
            validator=validators.has_len(3),
            units=ucc.deferred("length"),
        ),
        doc="A 3-element vector specifying the position of the camera.\n"
        "\n"
        "Unit-enabled field (default: cdu[length]).",
        type="array-like[float, float, float]",
        default="[1, 1, 1] m",
    )

    target = documented(
        pinttr.ib(
            default=ureg.Quantity([0, 0, 0], ureg.m),
            validator=validators.has_len(3),
            units=ucc.deferred("length"),
        ),
        doc="A 3-element vector specifying the location targeted by the camera.\n"
        "\n"
        "Unit-enabled field (default: cdu[length]).",
        type="array-like[float, float, float]",
        default="[0, 0, 0] m",
    )

    up = documented(
        attr.ib(default=[0, 0, 1], validator=validators.has_len(3)),
        doc="A 3-element vector specifying the up direction of the camera.\n"
        "This vector must be different from the camera's viewing direction,\n"
        "which is given by ``target - origin``.",
        type="array-like[float, float, float]",
        default="[0, 0, 1]",
    )

    @target.validator
    @origin.validator
    def _target_origin_validator(self, attribute, value):
        if np.allclose(self.target, self.origin):
            raise ValueError(
                f"While initializing {attribute}: "
                f"Origin and target must not be equal, "
                f"got target = {self.target}, origin = {self.origin}"
            )

    @up.validator
    def _up_validator(self, attribute, value):
        direction = self.target - self.origin
        if np.allclose(np.cross(direction, value), 0):
            raise ValueError(
                f"While initializing {attribute}:"
                f"Up direction must differ from viewing direction, "
                f"got up = {self.up}, direction = {direction}"
            )

    def _base_dicts(self):
        from eradiate.kernel.core import ScalarTransform4f

        target = self.target.m_as(uck.get("length"))
        origin = self.origin.m_as(uck.get("length"))
        result = []

        for sensor_info in self.sensor_infos():
            result.append(
                {
                    "type": "perspective",
                    "id": sensor_info.id,
                    "far_clip": 1e7,
                    "to_world": ScalarTransform4f.look_at(
                        origin=origin, target=target, up=self.up
                    ),
                }
            )

        return result
