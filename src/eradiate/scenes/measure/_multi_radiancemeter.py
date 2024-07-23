from __future__ import annotations

import attrs
import numpy as np
import pint
import pinttr

from ._core import Measure
from ...attrs import define, documented
from ...units import symbol
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg


@define(eq=False, slots=False)
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
        pinttr.field(
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
        attrs.field(
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
    def film_resolution(self) -> tuple[int, int]:
        return (self.origins.shape[0], 1)

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    @property
    def kernel_type(self) -> str:
        # Inherit docstring
        return "mradiancemeter"

    @property
    def template(self) -> dict:
        # Inherit docstring
        result = super().template

        origins = self.origins.m_as(uck.get("length"))
        result["origins"] = ",".join(map(str, origins.ravel(order="C")))
        directions = self.directions
        result["directions"] = ",".join(map(str, directions.ravel(order="C")))

        return result

    # --------------------------------------------------------------------------
    #                        Post-processing information
    # --------------------------------------------------------------------------

    @property
    def var(self) -> tuple[str, dict]:
        return "radiance", {
            "standard_name": "radiance",
            "long_name": "radiance",
            "units": symbol(uck.get("radiance")),
        }
