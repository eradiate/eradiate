from __future__ import annotations

from typing import Dict, List, Tuple

import attr
import numpy as np
import pint
import pinttr

from ._core import Measure, measure_factory
from ... import validators
from ...attrs import documented, parse_docs
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg


@measure_factory.register(type_id="radiancemeter")
@parse_docs
@attr.s
class RadiancemeterMeasure(Measure):
    """
    Radiance meter measure scene element [``radiancemeter``].

    This measure scene element is a thin wrapper around the ``radiancemeter``
    sensor kernel plugin. It records the incident power per unit area per unit
    solid angle along a certain ray.
    """

    origin: pint.Quantity = documented(
        pinttr.ib(
            default=ureg.Quantity([0.0, 0.0, 0.0], ureg.m),
            validator=validators.has_len(3),
            units=ucc.deferred("length"),
        ),
        doc="A 3-element vector specifying the position of the radiance meter.\n"
        "\n"
        "Unit-enabled field (default: ucc['length']).",
        type="array-like[float, float, float]",
        default="[0, 0, 0] m",
    )

    target: pint.Quantity = documented(
        pinttr.ib(
            default=ureg.Quantity([0.0, 0.0, 1.0], ureg.m),
            validator=validators.has_len(3),
            units=ucc.deferred("length"),
        ),
        doc="A 3-element vector specifying the location targeted by the sensor.\n"
        "\n"
        "Unit-enabled field (default: ucc['length']).",
        type="array[float, float, float]",
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

    @property
    def film_resolution(self) -> Tuple[int, int]:
        return (1, 1)

    def _base_dicts(self) -> List[Dict]:
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
