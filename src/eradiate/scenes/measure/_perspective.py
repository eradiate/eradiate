from __future__ import annotations

import attrs
import mitsuba as mi
import numpy as np
import pint
import pinttr

from ._core import Measure
from ... import validators
from ...attrs import define, documented
from ...units import symbol
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg


@define(eq=False, slots=False)
class PerspectiveCameraMeasure(Measure):
    """
    Perspective camera scene element [``perspective``].

    This scene element is a thin wrapper around the ``perspective`` sensor
    kernel plugin. It positions a perspective camera based on a set of vectors,
    specifying the origin, viewing direction and 'up' direction of the camera.
    """

    # --------------------------------------------------------------------------
    #                           Fields and properties
    # --------------------------------------------------------------------------

    spp: int = documented(
        attrs.field(default=32, converter=int, validator=validators.is_positive),
        doc="Number of samples per pixel.",
        type="int",
        default="32",
    )

    _film_resolution: tuple[int, int] = documented(
        attrs.field(
            default=(32, 32),
            converter=tuple,
            validator=attrs.validators.deep_iterable(
                member_validator=attrs.validators.instance_of(int),
                iterable_validator=validators.has_len(2),
            ),
        ),
        doc="Film resolution as a (width, height) 2-tuple.",
        type="tuple of int",
        init_type="array-like",
        default="(32, 32)",
    )

    @property
    def film_resolution(self) -> tuple[int, int]:
        return self._film_resolution

    origin: pint.Quantity = documented(
        pinttr.field(
            factory=lambda: [1, 1, 1] * ureg.m,
            validator=[validators.has_len(3), pinttr.validators.has_compatible_units],
            units=ucc.deferred("length"),
        ),
        doc="A 3-vector specifying the position of the camera.\n"
        "\n"
        "Unit-enabled field (default: ucc['length']).",
        type="quantity",
        init_type="array-like",
        default="[1, 1, 1] m",
    )

    target: pint.Quantity = documented(
        pinttr.field(
            factory=lambda: [0, 0, 0] * ureg.m,
            validator=[validators.has_len(3), pinttr.validators.has_compatible_units],
            units=ucc.deferred("length"),
        ),
        doc="Point location targeted by the camera.\n"
        "\n"
        "Unit-enabled field (default: ucc['length']).",
        type="quantity",
        init_type="array-like",
        default="[0, 0, 0] m",
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

    up: np.ndarray = documented(
        attrs.field(
            factory=lambda: [0, 0, 1],
            converter=np.array,
            validator=validators.has_len(3),
        ),
        doc="A 3-vector specifying the up direction of the camera.\n"
        "This vector must be different from the camera's viewing direction,\n"
        "which is given by ``target - origin``.",
        type="array",
        default="[0, 0, 1]",
    )

    @up.validator
    def _up_validator(self, attribute, value):
        direction = self.target - self.origin
        if np.allclose(np.cross(direction, value), 0):
            raise ValueError(
                f"While initializing '{attribute.name}': "
                f"up direction must not be colinear with viewing direction, "
                f"got up = {self.up}, direction = {direction}"
            )

    far_clip: pint.Quantity = documented(
        pinttr.field(
            default=1e4 * ureg.km,
            units=ucc.deferred("length"),
        ),
        doc="Distance to the far clip plane.\n"
        "\n"
        "Unit-enabled field (default: ucc[length]).",
        type="quantity",
        init_type="quantity of float",
        default="10 000 km",
    )

    fov: pint.Quantity = documented(
        pinttr.field(default=50.0 * ureg.deg, units=ureg.deg),
        doc="Camera field of view.\n\nUnit-enabled field (default: degree).",
        type="quantity",
        init_type="quantity or float",
        default="50°",
    )

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    @property
    def kernel_type(self) -> str:
        # Inherit docstring
        return "perspective"

    @property
    def template(self) -> dict:
        # Inherit docstring
        result = super().template

        result["far_clip"] = self.far_clip.m_as(uck.get("length"))
        result["fov"] = self.fov.m_as(ureg.deg)

        target = self.target.m_as(uck.get("length"))
        origin = self.origin.m_as(uck.get("length"))
        result["to_world"] = mi.ScalarTransform4f.look_at(
            origin=origin, target=target, up=self.up
        )

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
