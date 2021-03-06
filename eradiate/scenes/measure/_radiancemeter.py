import attr
import pinttr
import numpy as np

from ._core import Measure, MeasureFactory
from ... import validators
from ..._attrs import documented, parse_docs
from ..._units import unit_context_config as ucc
from ..._units import unit_context_kernel as uck
from ..._units import unit_registry as ureg


@MeasureFactory.register("radiancemeter")
@parse_docs
@attr.s
class RadiancemeterMeasure(Measure):
    """Radiance meter measure scene element [:factorykey:`radiancemeter`].

    This measure scene element is a thin wrapper around the ``radiancemeter``
    sensor kernel plugin. It records the incident power per unit area per unit
    solid angle along a certain ray."""

    film_resolution = documented(
        attr.ib(
            default=(1, 1),
            validator=attr.validators.deep_iterable(
                member_validator=attr.validators.instance_of(int),
                iterable_validator=validators.has_len(2),
            ),
        ),
        doc="Film resolution, set to (1, 1).",
        type="array-like[int, int]",
        default="(1, 1)",
    )

    origin = documented(
        pinttr.ib(
            default=ureg.Quantity([0.0, 0.0, 0.0], ureg.m),
            validator=validators.has_len(3),
            units=ucc.deferred("length"),
        ),
        doc="A 3-element vector specifying the position of the radiance meter.\n"
        "\n"
        "Unit-enabled field (default: ucc[length]).",
        type="array-like[float, float, float]",
        default="[0, 0, 0] m",
    )

    target = documented(
        pinttr.ib(
            default=ureg.Quantity([0.0, 0.0, 1.0], ureg.m),
            validator=validators.has_len(3),
            units=ucc.deferred("length"),
        ),
        doc="A 3-element vector specifying the location targeted by the sensor.\n"
        "\n"
        "Unit-enabled field (default: ucc[length]).",
        type="array-like[float, float, float]",
        default="[0, 0, 1] m",
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

    def _base_dicts(self):
        target = self.target.m_as(uck.get("length"))
        origin = self.origin.m_as(uck.get("length"))
        direction = target - origin
        result = []

        for sensor_info in self.sensor_infos():
            result.append(
                {
                    "type": "radiancemeter",
                    "id": sensor_info.id,
                    "origin": origin,
                    "direction": direction,
                }
            )

        return result
