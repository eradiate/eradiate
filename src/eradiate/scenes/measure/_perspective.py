from __future__ import annotations

import typing as t

import attr
import numpy as np
import pint
import pinttr

from ._core import Measure, measure_factory
from ..core import KernelDict
from ... import validators
from ...attrs import documented, parse_docs
from ...contexts import KernelDictContext
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg


@measure_factory.register(type_id="perspective")
@parse_docs
@attr.s
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

    _film_resolution: t.Tuple[int, int] = documented(
        attr.ib(
            default=(32, 32),
            converter=tuple,
            validator=attr.validators.deep_iterable(
                member_validator=attr.validators.instance_of(int),
                iterable_validator=validators.has_len(2),
            ),
        ),
        doc="Film resolution as a (width, height) 2-tuple.",
        type="tuple of int",
        init_type="array-like",
        default="(32, 32)",
    )

    @property
    def film_resolution(self) -> t.Tuple[int, int]:
        return self._film_resolution

    origin: pint.Quantity = documented(
        pinttr.ib(
            factory=lambda: [1, 1, 1] * ureg.m,
            validator=validators.has_len(3),
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
        pinttr.ib(
            factory=lambda: [0, 0, 0] * ureg.m,
            validator=validators.has_len(3),
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
        attr.ib(
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
        pinttr.ib(
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
        pinttr.ib(default=50.0 * ureg.deg, units=ureg.deg),
        doc="Camera field of view.\n\nUnit-enabled field (default: degree).",
        type="quantity",
        init_type="quantity or float",
        default="50°",
    )

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    def _kernel_dict(self, sensor_id, spp):
        from mitsuba.core import ScalarTransform4f

        target = self.target.m_as(uck.get("length"))
        origin = self.origin.m_as(uck.get("length"))

        result = {
            "type": "perspective",
            "id": sensor_id,
            "far_clip": self.far_clip.m_as(uck.get("length")),
            "fov": self.fov.m_as(ureg.deg),
            "to_world": ScalarTransform4f.look_at(
                origin=origin, target=target, up=self.up
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
        return "img", dict()
