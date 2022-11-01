from __future__ import annotations

import typing as t

import attrs
import numpy as np
import pint
import pinttr

from ._core import Measure
from ... import validators
from ...attrs import documented, parse_docs
from ...units import symbol
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg


@parse_docs
@attrs.define
class RadiancemeterMeasure(Measure):
    """
    Radiance meter measure scene element [``radiancemeter``].

    This measure scene element is a thin wrapper around the ``radiancemeter``
    sensor kernel plugin. It records the incident power per unit area per unit
    solid angle along a certain ray.
    """

    # --------------------------------------------------------------------------
    #                           Fields and properties
    # --------------------------------------------------------------------------

    origin: pint.Quantity = documented(
        pinttr.field(
            default=ureg.Quantity([0.0, 0.0, 0.0], ureg.m),
            validator=[validators.has_len(3), pinttr.validators.has_compatible_units],
            units=ucc.deferred("length"),
        ),
        doc="A 3-element vector specifying the position of the radiance meter.\n"
        "\n"
        "Unit-enabled field (default: ucc['length']).",
        type="quantity",
        init_type="array-like",
        default="[0, 0, 0] m",
    )

    target: pint.Quantity = documented(
        pinttr.field(
            default=ureg.Quantity([0.0, 0.0, 1.0], ureg.m),
            validator=[validators.has_len(3), pinttr.validators.has_compatible_units],
            units=ucc.deferred("length"),
        ),
        doc="A 3-element vector specifying the location targeted by the sensor.\n"
        "\n"
        "Unit-enabled field (default: ucc['length']).",
        type="quantity",
        init_type="array-like",
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
    def film_resolution(self) -> t.Tuple[int, int]:
        return (1, 1)

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    def _kernel_dict_impl(self, sensor_id, spp):
        target = self.target.m_as(uck.get("length"))
        origin = self.origin.m_as(uck.get("length"))
        direction = target - origin

        result = {
            "type": "radiancemeter",
            "id": sensor_id,
            "origin": origin,
            "direction": direction,
            "sampler": {
                "type": self.sampler,
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

    # --------------------------------------------------------------------------
    #                        Post-processing information
    # --------------------------------------------------------------------------

    @property
    def var(self) -> t.Tuple[str, t.Dict]:
        return "radiance", {
            "standard_name": "radiance",
            "long_name": "radiance",
            "units": symbol(uck.get("radiance")),
        }
