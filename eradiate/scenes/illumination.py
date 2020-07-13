"""Illumination-related scene generation facilities.

.. admonition:: Factory-enabled scene generation helpers
    :class: hint

    .. factorytable::
        :modules: illumination
"""

import attr
import numpy as np

from . import SceneHelper
from .core import Factory
from ..util.collections import frozendict
from ..util.frame import angles_to_direction
from ..util.units import ureg


def _constant(radiance=1.):
    """Create a dictionary which will instantiate a ``constant`` kernel plugin.

    Parameter ``radiance`` (float or dict)
        Emitted radiance [(u_power)/(u_length)^2/sr/nm].

    Returns → dict
        A dictionary which can be used to instantiate a ``constant`` kernel
        plugin facing the direction specified by the angular configuration.
    """
    # Design note: this function provides experimental support for units.
    # It will be removed when general support for units will be added.

    return {
        "type": "constant",
        "radiance":
            {"type": "uniform", "value": radiance}
            if isinstance(radiance, float)
            else radiance
    }


@attr.s
@Factory.register(name="constant")
class ConstantIllumination(SceneHelper):
    """Constant illumination scene generation helper [:factorykey:`constant`].

    .. admonition:: Configuration format
        :class: hint

        ``radiance`` (float):
            Emitted radiance [(u_power)/(u_length)^2/nm].

            Default: 1.
    """

    CONFIG_SCHEMA = frozendict({
        "radiance": {
            "type": "number",
            "min": 0.,
            "default": 1.
        }
    })

    id = attr.ib(default="illumination")

    def kernel_dict(self, **kwargs):
        return {self.id: _constant(self.config["radiance"])}


@ureg.wraps(None, (ureg.deg, ureg.deg, None), strict=False)
def _directional(zenith=0., azimuth=0., irradiance=1.):
    """Create a dictionary which will instantiate a ``directional`` kernel
    plugin based on the provided angular geometry.

    Parameter ``zenith`` (float)
        Zenith angle [deg].

    Parameter ``azimuth`` (float)
        Azimuth angle [deg].

    Parameter ``irradiance`` (float or dict)
        Emitted irradiance in the plane orthogonal to the emitter's direction
        [(u_power)/(u_length)^2/nm].

    Returns → dict
        A dictionary which can be used to instantiate a ``directional`` kernel
        plugin facing the direction specified by the angular configuration.
    """
    # Design note: see _constant()

    return {
        "type": "directional",
        "direction": list(-angles_to_direction(
            theta=np.deg2rad(zenith),
            phi=np.deg2rad(azimuth)
        )),
        "irradiance":
            {"type": "uniform", "value": irradiance}
            if isinstance(irradiance, float)
            else irradiance
    }


@attr.s
@Factory.register(name="directional")
class DirectionalIllumination(SceneHelper):
    """Directional illumination scene generation helper [:factorykey:`directional`].

    The illumination is oriented based on the classical angular convention used
    in Earth observation.

    .. admonition:: Configuration format
        :class: hint

        ``zenith`` (float):
             Zenith angle [deg].

             Default: 0.

        ``azimuth`` (float):
            Azimuth angle value [deg].

            Default: 0.

        ``irradiance`` (float):
            Emitted radiant power flux in the plane orthogonal to the
            illumination direction [(u_power)/(u_length)^2/nm].

            Default: 1.
    """

    CONFIG_SCHEMA = frozendict({
        "zenith": {"type": "number", "default": 0.},
        "azimuth": {"type": "number", "default": 0.},
        "irradiance": {"type": "number", "default": 1.}
    })

    id = attr.ib(default="illumination")

    def kernel_dict(self, **kwargs):
        return {
            self.id: _directional(
                self.config["zenith"],
                self.config["azimuth"],
                self.config["irradiance"]
            )
        }
