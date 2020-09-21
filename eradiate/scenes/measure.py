"""Measurement-related scene generation facilities.

.. admonition:: Factory-enabled scene generation helpers
    :class: hint

    .. factorytable::
        :modules: measure
"""

import attr
import numpy as np

from .core import Factory, SceneHelper
from .core import kernel_default_units as kdu
from ..util.config_object import config_default_units
from ..util.frame import angles_to_direction, spherical_to_cartesian


@attr.s
@Factory.register(name="distant")
class DistantMeasure(SceneHelper):
    """Distant measure scene generation helper [:factorykey:`distant`].

    The sensor is oriented based on the classical angular convention used
    in Earth observation.

    .. admonition:: Configuration example
        :class: hint

        Default:
            .. code:: python

               {
                   "zenith": 0.,
                   "azimuth": 0.,
                   "spp": 10000,
               }

    .. admonition:: Configuration format
        :class: hint

        ``zenith`` (float):
            Zenith angle [deg].

            Default value: 0.

        ``azimuth`` (float):
            Azimuth angle value [deg].

            Default value: 0.

        ``spp`` (int):
            Number of samples.

            Default: 10000.
    """

    @classmethod
    def config_schema(cls):
        return dict({
            "zenith": {
                "type": "number",
                "min": 0.,
                "max": 90.,
                "default": 0.0,
            },
            "zenith_unit": {
                "type": "string",
                "default": str(config_default_units.units.get("angle")())
            },
            "azimuth": {
                "type": "number",
                "min": 0.,
                "max": 360.,
                "default": 0.0,
            },
            "azimuth_unit": {
                "type": "string",
                "default": str(config_default_units.units.get("angle")())
            },
            "spp": {
                "type": "integer",
                "min": 0,
                "default": 10000
            }
        })

    id = attr.ib(default="measure")

    def kernel_dict(self, **kwargs):
        zenith = self.get_quantity("zenith").to(kdu.units.get("angle")()).magnitude
        azimuth = self.get_quantity("azimuth").to(kdu.units.get("angle")()).magnitude
        spp = self.get_quantity("spp")
        return {
            self.id: {
                "type": "distant",
                "direction": list(-angles_to_direction(
                    theta=np.deg2rad(zenith),
                    phi=np.deg2rad(azimuth)
                )),
                "target": [0, 0, 0],
                "sampler": {
                    "type": "independent",
                    "sample_count": spp
                },
                "film": {
                    "type": "hdrfilm",
                    "width": 1,
                    "height": 1,
                    "pixel_format": "luminance",
                    "component_format": "float32",
                    "rfilter": {"type": "box"}
                }
            }
        }


@attr.s
@Factory.register(name="perspective")
class PerspectiveCameraMeasure(SceneHelper):
    """Perspective camera scene generation helper [:factorykey:`perspective`].

    The sensor is oriented based on the classical angular convention used
    in Earth observation.

    The film is a square.

    .. admonition:: Configuration example
        :class: hint

        Default:
            .. code:: python

               {
                   "target": [0, 0, 0],
                   "zenith": 0.,
                   "azimuth": 0.,
                   "distance": 1.,
                   "res": 64,
                   "spp": 32,
               }

    .. admonition:: Configuration format
        :class: hint

        ``target`` (list[float]):
            A 3-element vector specifying the location targeted by the camera
            [u_length].

            Default: [0, 0, 0].

        ``zenith`` (float):
            Zenith angle [deg].

            Default value: 0.

        ``azimuth`` (float):
            Azimuth angle value [deg].

            Default value: 0.

        ``distance`` (float):
            Distance from the ``target`` point to the camera [u_length].

            Default: 1.

        ``res`` (int):
            Resolution of the film in pixels.

            Default: 64.

        ``spp`` (int):
            Number of samples per pixel.

            Default: 32.
    """

    @classmethod
    def config_schema(cls):
        return dict({
            "target": {
                "type": "list",
                "items": [{"type": "number"}] * 3,
                "default": [0, 0, 0]
            },
            "target_unit": {
                "type": "string",
                "default": str(config_default_units.units.get("length")())
            },
            "zenith": {
                "type": "number",
                "min": 0.,
                "max": 90.,
                "default": 45.
            },
            "zenith_unit": {
                "type": "string",
                "default": str(config_default_units.units.get("angle")())
            },
            "azimuth": {
                "type": "number",
                "min": 0.,
                "max": 360.,
                "default": 180.
            },
            "azimuth_unit": {
                "type": "string",
                "default": str(config_default_units.units.get("angle")())
            },
            "distance": {
                "type": "number",
                "min": 0.,
                "default": 1.,
            },
            "distance_unit": {
                "type": "string",
                "default": str(config_default_units.units.get("length")())
            },
            "res": {
                "type": "integer",
                "min": 0,
                "default": 64
            },
            "spp": {
                "type": "integer",
                "min": 0,
                "default": 32
            }
        })

    id = attr.ib(default="measure")

    def kernel_dict(self, **kwargs):
        target = self.get_quantity("target").to(kdu.units.get("length")()).magnitude
        zenith = self.get_quantity("zenith").to(kdu.units.get("angle")()).magnitude
        azimuth = self.get_quantity("azimuth").to(kdu.units.get("angle")()).magnitude
        distance = self.get_quantity("distance").to(kdu.units.get("length")()).magnitude
        res = self.get_quantity("res"),
        spp = self.get_quantity("spp")

        from eradiate.kernel.core import ScalarTransform4f

        origin = spherical_to_cartesian(distance,
                                        np.deg2rad(zenith),
                                        np.deg2rad(azimuth))

        if np.allclose(origin, target):
            raise ValueError("target is too close to the camera")

        return {
            self.id: {
                "type": "perspective",
                "far_clip": 1e7,
                "to_world": ScalarTransform4f
                    .look_at(origin=origin, target=target, up=[0, 0, 1]),
                "sampler": {
                    "type": "independent",
                    "sample_count": spp
                },
                "film": {
                    "type": "hdrfilm",
                    "width": res,
                    "height": res,
                    "pixel_format": "luminance",
                    "component_format": "float32",
                    "rfilter": {"type": "box"}
                }
            }
        }
