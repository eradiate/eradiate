"""Illumination-related scene generation facilities.

.. admonition:: Factory-enabled scene generation helpers
    :class: hint

    .. factorytable::
        :modules: illumination
"""

import attr
import numpy as np

from .core import Factory, SceneHelper
from ..util.frame import angles_to_direction
from ..util.units import config_default_units as cdu
from ..util.units import kernel_default_units as kdu


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

    @classmethod
    def config_schema(cls):
        d = super(ConstantIllumination, cls).config_schema()
        d["id"]["default"] = "illumination"
        d.update({
            "radiance": {
                "type": "dict",
                "default": {},
                "allow_unknown": True,
                "schema": {
                    "type": {
                        "type": "string",
                        "allowed": ["uniform", "solar_irradiance"],
                        "default": "uniform"
                    },
                    "quantity": {
                        "type": "string",
                        "allowed": ["radiance"],
                        "default": "radiance"
                    }
                }
            },
        })
        return d

    def kernel_dict(self, **kwargs):
        radiance = Factory().create(self.config["radiance"])
        return {
            self.id: {
                "type": "constant",
                "radiance": radiance.kernel_dict()["spectrum"]
            }
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

    @classmethod
    def config_schema(cls):
        d = super(DirectionalIllumination, cls).config_schema()
        d["id"]["default"] = "illumination"
        d.update({
            "zenith": {"type": "number", "default": 0.},
            "zenith_unit": {
                "type": "string",
                "default": cdu.get_str("angle")
            },
            "azimuth": {"type": "number", "default": 0.},
            "azimuth_unit": {
                "type": "string",
                "default": cdu.get_str("angle")
            },
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
            },
        })
        return d

    def kernel_dict(self, **kwargs):
        irradiance_ = Factory().create(self.config["irradiance"])

        zenith = self.config.get_quantity("zenith").to(kdu.get("angle")).magnitude,
        azimuth = self.config.get_quantity("azimuth").to(kdu.get("angle")).magnitude
        irradiance = irradiance_.kernel_dict()["spectrum"]

        return {
            self.id: {
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
        }
