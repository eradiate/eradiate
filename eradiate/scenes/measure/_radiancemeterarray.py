from __future__ import annotations

from typing import Dict, List, Tuple

import attr
import numpy as np
import pint
import pinttr

from ._core import Measure, measure_factory
from ...attrs import documented, parse_docs
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg


@measure_factory.register(type_id="radiancemeterarray")
@parse_docs
@attr.s
class RadiancemeterArrayMeasure(Measure):
    """
    Radiance meter array measure scene element [``radiancemeterarray``].

    This measure scene element is a thin wrapper around the ``radiancemeterarray``
    sensor kernel plugin. It records the incident power per unit area per unit
    solid angle along a number of rays defined by its ``origins`` and
    ``directions`` parameters.
    """

    origins: pint.Quantity = documented(
        pinttr.ib(
            default=ureg.Quantity([[0.0, 0.0, 0.0]], ureg.m),
            units=ucc.deferred("length"),
        ),
        doc="A sequence of 3D points specifying radiance meter array positions.\n"
        "\n"
        "Unit-enabled field (default: ucc['length']).",
        type="array-like",
        default="[[0, 0, 0]] m",
    )

    directions: np.ndarray = documented(
        attr.ib(
            default=np.array([[0.0, 0.0, 1.0]]),
            converter=np.array,
        ),
        doc="A sequence of 3-vectors specifying radiance meter array directions.",
        type="array-like",
        default="[[0, 0, 1]]",
    )

    @directions.validator
    @origins.validator
    def _target_origin_validator(self, attribute, value):
        if value.shape[1] != 3:
            raise ValueError(
                f"While initializing {attribute}: "
                f"Expected shape (N, 3), got {value.shape}"
            )

        if not self.origins.shape == self.directions.shape:
            raise ValueError(
                f"While initializing {attribute}: "
                f"Origin and direction arrays must have the same shape, "
                f"got origins.shape = {self.origins.shape}, "
                f"directions.shape = {self.directions.shape}"
            )

    @property
    def film_resolution(self) -> Tuple[int, int]:
        return (self.origins.shape[0], 1)

    def _base_dicts(self) -> List[Dict]:
        origins = self.origins.m_as(uck.get("length"))
        directions = self.directions
        result = []

        for sensor_info in self.sensor_infos():
            result.append(
                {
                    "type": "radiancemeterarray",
                    "id": sensor_info.id,
                    "origins": ", ".join([str(x) for x in origins.ravel(order="C")]),
                    "directions": ", ".join(
                        [str(x) for x in directions.ravel(order="C")]
                    ),
                }
            )

        return result
