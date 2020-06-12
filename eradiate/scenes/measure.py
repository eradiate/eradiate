"""Scene generation facilities related with measurement."""

import attr
import numpy as np

from .base import SceneHelper
from .factory import Factory
from ..util.frame import angles_to_direction, spherical_to_cartesian
from ..util.units import ureg


@ureg.wraps(None, (ureg.deg, ureg.deg, None), strict=False)
def _distant(zenith=0., azimuth=0., spp=10000):
    """Create a dictionary which will instantiate a `distant` Mitsuba plugin
    based on the provided angular geometry.

    Parameter ``zenith`` (float)
        Zenith angle [deg].

    Parameter ``azimuth`` (float)
        Azimuth angle [deg].

    Parameter ``spp`` (int)
        Number of samples used from this sensor.

    Returns → dict
        A dictionary which can be used to instantiate a `distant` Mitsuba plugin
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
@Factory.register()
class Distant(SceneHelper):
    """TODO: add docs"""

    DEFAULT_CONFIG = {
        "zenith": 0.0,
        "azimuth": 0.0,
        "spp": 10000
    }

    id = attr.ib(default="measure")

    def kernel_dict(self, **kwargs):
        return {
            self.id: _distant(
                self.config["zenith"],
                self.config["azimuth"],
                self.config["spp"]
            )
        }


@ureg.wraps(None, (ureg.deg, ureg.deg, ureg.km, None, None), strict=False)
def _perspective(zenith=45., azimuth=180., distance=1., res=64, spp=10000):
    """Create a dictionary which will instantiate a `perspective` Mitsuba plugin
    based on the provided angular geometry.

    Parameter ``zenith`` (float)
        Zenith angle [deg].

    Parameter ``azimuth`` (float)
        Azimuth angle [deg].

    Parameter ``distance`` (float)
        Distance to scene center [km].

    Parameter ``res`` (int)
        Film resolution.

    Parameter ``spp`` (int)
        Number of samples used from this sensor.

    Returns → dict
        A dictionary which can be used to instantiate a `perspective` Mitsuba
        plugin facing the direction specified by the angular configuration and
        pointing towards the origin :math:`(0, 0, 0)` in world coordinates.
    """

    from eradiate.kernel.core import ScalarTransform4f

    x, y, z = spherical_to_cartesian(distance,
                                     np.deg2rad(zenith),
                                     np.deg2rad(azimuth))

    if [x, y, z] == [0, 0, 0]:
        raise ValueError("invalid look_at transform")

    return {
        "type": "perspective",
        "far_clip": 1e7,
        "to_world": ScalarTransform4f
            .look_at(origin=[x, y, z], target=[0, 0, 0], up=[0, 0, 1]),
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
@Factory.register()
class Perspective(SceneHelper):
    """TODO: add docs"""

    DEFAULT_CONFIG = {
        "zenith": 45.0,
        "azimuth": 180.0,
        "distance": 1.0,
        "res": 64,
        "spp": 10000
    }

    id = attr.ib(default="measure")

    def kernel_dict(self, **kwargs):
        return {
            self.id: _perspective(
                self.config["zenith"],
                self.config["azimuth"],
                self.config["distance"],
                self.config["res"],
                self.config["spp"]
            )
        }
