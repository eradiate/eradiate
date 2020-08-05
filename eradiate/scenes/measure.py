"""Measurement-related scene generation facilities.

.. admonition:: Factory-enabled scene generation helpers
    :class: hint

    .. factorytable::
        :modules: measure
"""

import attr
import numpy as np

from .core import Factory, SceneHelper
from ..util.collections import frozendict
from ..util.frame import angles_to_direction, spherical_to_cartesian
from ..util.units import ureg


@ureg.wraps(None, (ureg.deg, ureg.deg, None), strict=False)
def _distant(zenith=0., azimuth=0., spp=10000):
    """Create a dictionary which will instantiate a ``distant`` kernel plugin
    based on the provided angular geometry.

    Parameter ``zenith`` (float)
        Zenith angle [deg].

    Parameter ``azimuth`` (float)
        Azimuth angle [deg].

    Parameter ``spp`` (int)
        Number of samples used from this sensor.

    Returns → dict
        A dictionary which can be used to instantiate a ``distant`` kernel plugin
        facing the direction specified by the angular configuration and pointing
        towards the origin :math:`(0, 0, 0)` in world coordinates.
    """

    return {
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

    CONFIG_SCHEMA = frozendict({
        "zenith": {
            "type": "number",
            "min": 0.,
            "max": 90.,
            "default": 0.0,
        },
        "azimuth": {
            "type": "number",
            "min": 0.,
            "max": 360.,
            "default": 0.0,
        },
        "spp": {
            "type": "integer",
            "min": 0,
            "default": 10000
        }
    })

    id = attr.ib(default="measure")

    def kernel_dict(self, **kwargs):
        return {
            self.id: _distant(
                self.config["zenith"],
                self.config["azimuth"],
                self.config["spp"]
            )
        }


@ureg.wraps(None, (ureg.km, ureg.deg, ureg.deg, ureg.km, None, None), strict=False)
def _perspective(target=[0, 0, 0], zenith=45., azimuth=180., distance=1.,
                 res=64, spp=32):
    """Create a dictionary which will instantiate a ``perspective`` kernel
    plugin based on the provided angular geometry.

    Parameter ``target`` (list(float))
        Target point location [km].

    Parameter ``zenith`` (float)
        Zenith angle [deg].

    Parameter ``azimuth`` (float)
        Azimuth angle [deg].

    Parameter ``distance`` (float)
        Distance to ``target`` [km].

    Parameter ``res`` (int)
        Film resolution.

    Parameter ``spp`` (int)
        Number of samples used from this sensor.

    Returns → dict
        A dictionary which can be used to instantiate a ``perspective`` kernel
        plugin facing the direction specified by the angular configuration and
        pointing towards the origin ``target`` in world coordinates.
    """

    from eradiate.kernel.core import ScalarTransform4f

    origin = spherical_to_cartesian(distance,
                                    np.deg2rad(zenith),
                                    np.deg2rad(azimuth))

    if np.allclose(origin, target):
        raise ValueError("target is too close to the camera")

    return {
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

    CONFIG_SCHEMA = frozendict({
        "target": {
            "type": "list",
            "items": [{"type": "number"}] * 3,
            "default": [0, 0, 0]
        },
        "zenith": {
            "type": "number",
            "min": 0.,
            "max": 90.,
            "default": 45.
        },
        "azimuth": {
            "type": "number",
            "min": 0.,
            "max": 360.,
            "default": 180.
        },
        "distance": {
            "type": "number",
            "min": 0.,
            "default": 1.,
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
        return {
            self.id: _perspective(
                target=self.config["target"],
                zenith=self.config["zenith"],
                azimuth=self.config["azimuth"],
                distance=self.config["distance"],
                res=self.config["res"],
                spp=self.config["spp"]
            )
        }
