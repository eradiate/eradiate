"""Illumination-related scene generation facilities.

.. admonition:: Factory-enabled scene generation helpers
    :class: hint

    .. factorytable::
        :modules: illumination
"""

import attr
import numpy as np

from .core import Factory, SceneHelper
from ..util.collections import frozendict
from ..util.frame import angles_to_direction
from ..util.units import ureg


@attr.s
@Factory.register(name="constant")
class ConstantIllumination(SceneHelper):
    """Constant illumination scene generation helper [:factorykey:`constant`].

    .. admonition:: Configuration examples
        :class: hint

        Default:
            .. code:: python

               {"radiance": {"type": "uniform"}}

        Fully specified:
            .. code:: python

               {"radiance": {"type": "uniform", "value": 10.}}

    .. admonition:: Configuration format
        :class: hint

        ``radiance`` (dict):
            Emitted radiance spectrum. This section must be a factory
            configuration dictionary which will be passed to
            :meth:`eradiate.scenes.core.Factory.create`.

            Allowed scene generation helpers:
            :factorykey:`uniform`,
            :factorykey:`solar_irradiance`

            Default:
            :factorykey:`uniform`.
    """

    CONFIG_SCHEMA = frozendict({
        "radiance": {
            "type": "dict",
            "default": {},
            "allow_unknown": True,
            "schema": {
                "type": {
                    "type": "string",
                    "allowed": ["uniform", "solar_irradiance"],
                    "default": "uniform"
                }
            }
        }
    })

    id = attr.ib(default="illumination")

    def kernel_dict(self, **kwargs):
        radiance = Factory().create(self.config["radiance"])
        return {
            self.id: {
                "type": "constant",
                "radiance": radiance.kernel_dict()["spectrum"]
            }
        }


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

    .. admonition:: Configuration examples
        :class: hint

        Default:
            .. code:: python

               {
                   "zenith": 0.,
                   "azimuth": 0.,
                   "irradiance": {"type": "solar_irradiance"}
               }

        Fully specified:
            .. code:: python

               {
                   "zenith": 0.,
                   "azimuth": 0.,
                   "irradiance": {
                       "type": "uniform",
                       "value": 10.
                   }
               }

    .. admonition:: Configuration format
        :class: hint

        ``zenith`` (float):
             Zenith angle [deg].

             Default: 0.

        ``azimuth`` (float):
            Azimuth angle value [deg].

            Default: 0.

        ``irradiance`` (dict):
            Emitted power flux in the plane orthogonal to the
            illumination direction. This section must be a factory
            configuration dictionary which will be passed to
            :meth:`eradiate.scenes.core.Factory.create`.

            Allowed scene generation helpers:
            :factorykey:`uniform`,
            :factorykey:`solar_irradiance`

            Default:
            :factorykey:`solar_irradiance`.
    """

    CONFIG_SCHEMA = frozendict({
        "zenith": {"type": "number", "default": 0.},
        "azimuth": {"type": "number", "default": 0.},
        "irradiance": {
            "type": "dict",
            "default": {},
            "allow_unknown": True,
            "schema": {
                "type": {
                    "type": "string",
                    "allowed": ["uniform", "solar_irradiance"],
                    "default": "solar_irradiance"
                }
            }
        }
    })

    id = attr.ib(default="illumination")

    def kernel_dict(self, **kwargs):
        irradiance = Factory().create(self.config["irradiance"])

        return {
            self.id: _directional(
                self.config["zenith"],
                self.config["azimuth"],
                irradiance.kernel_dict()["spectrum"]
            )
        }
