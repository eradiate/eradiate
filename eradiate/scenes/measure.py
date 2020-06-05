"""Scene generation facilities related with measurement."""

import attr
import numpy as np

from .base import SceneHelper
from .factory import Factory
from ..util.frame import angles_to_direction
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