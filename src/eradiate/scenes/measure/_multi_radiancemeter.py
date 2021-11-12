from __future__ import annotations

import typing as t

import attr
import numpy as np
import pint
import pinttr

from ._core import Measure, measure_factory
from ..core import KernelDict
from ...attrs import documented, parse_docs
from ...contexts import KernelDictContext
from ...units import symbol
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg


@measure_factory.register(type_id="mradiancemeter", allow_aliases=True)
@measure_factory.register(type_id="multi_radiancemeter")
@parse_docs
@attr.s
class MultiRadiancemeterMeasure(Measure):
    """
    Radiance meter array measure scene element [``mradiancemeter``,
    ``multi_radiancemeter``].

    This measure scene element is a thin wrapper around the ``mradiancemeter``
    sensor kernel plugin. It records the incident power per unit area per unit
    solid angle along a number of rays defined by its ``origins`` and
    ``directions`` parameters.
    """

    # --------------------------------------------------------------------------
    #                           Fields and properties
    # --------------------------------------------------------------------------

    origins: pint.Quantity = documented(
        pinttr.ib(
            default=ureg.Quantity([[0.0, 0.0, 0.0]], ureg.m),
            units=ucc.deferred("length"),
        ),
        doc="A sequence of points specifying radiance meter array positions.\n"
        "\n"
        "Unit-enabled field (default: ucc['length']).",
        type="quantity",
        init_type="array-like",
        default="[[0, 0, 0]] m",
    )

    directions: np.ndarray = documented(
        attr.ib(
            default=np.array([[0.0, 0.0, 1.0]]),
            converter=np.array,
        ),
        doc="A sequence of 3-vectors specifying radiance meter array directions.",
        type="quantity",
        init_type="array-like",
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
    def film_resolution(self) -> t.Tuple[int, int]:
        return (self.origins.shape[0], 1)

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    def _kernel_dict(self, sensor_id, spp):
        origins = self.origins.m_as(uck.get("length"))
        directions = self.directions

        result = {
            "type": "mradiancemeter",
            "id": sensor_id,
            "origins": ",".join(map(str, origins.ravel(order="C"))),
            "directions": ",".join(map(str, directions.ravel(order="C"))),
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
    def var(self) -> t.Tuple[str, t.Dict]:
        return "radiance", {
            "standard_name": "radiance",
            "long_description": "radiance",
            "units": symbol(uck.get("radiance")),
        }
