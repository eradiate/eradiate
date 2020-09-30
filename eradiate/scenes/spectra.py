"""Spectrum-related scene generation facilities.

.. admonition:: Factory-enabled scene generation helpers
    :class: hint

    .. factorytable::
        :modules: spectra
"""

import attr
import numpy as np

from .. import data
from ..data import SOLAR_IRRADIANCE_SPECTRA
from ..util.exceptions import ModeError
from ..util.units import config_default_units as cdu
from ..util.units import kernel_default_units as kdu
from ..util.units import ureg
from .core import Factory, SceneHelper


@attr.s
@Factory.register(name="uniform")
class UniformSpectrum(SceneHelper):
    """Uniform spectrum scene generation helper [:factorykey:`uniform`].

    .. admonition:: Configuration examples
        :class: hint

        Default:
            .. code:: python

               {"value": 1.}

    .. admonition:: Configuration format
        :class: hint

        ``value`` (float):
            Spectrum constant value.

            Default: 1.
    """

    @classmethod
    def config_schema(cls):
        return {
            "value": {
                "type": "number",
                "min": 0.,
                "default": 1.0,
            },
            "quantity": {
                "type": "string",
                "allowed": ["radiance", "irradiance"],
                "default": "radiance",
                "nullable": True
            }
        }

    def kernel_dict(self, **kwargs):
        if self.config["quantity"] == "radiance":
            value = self.config["value"] * cdu.get("radiance")
            value = value.to(kdu.get("radiance")).magnitude
        elif self.config["quantity"] == "irradiance":
            value = self.config["value"] * cdu.get("irradiance")
            value = value.to(kdu.get("irradiance")).magnitude
        elif self.config["quantity"] is None:
            value = self.config["value"]
        else:
            raise ValueError(f"Unsupported quantity {self.config['quantity']}.")
        return {
            "spectrum": {
                "type": "uniform",
                "value": value,
            }
        }


@attr.s
@Factory.register(name="solar_irradiance")
class SolarIrradianceSpectrum(SceneHelper):
    """Solar irradiance spectrum scene generation helper
    [:factorykey:`solar_irradiance`].

    This scene generation helper produces the scene dictionary required to
    instantiate a kernel plugin using the Sun irradiance spectrum. The data set
    used by this helper is controlled by the ``dataset`` configuration parameter
    (see configuration format for the list of available data sets). The spectral
    range of the data sets shipped can vary and an attempt for use outside of
    the supported spectral range will raise a :class:`ValueError` upon calling
    :meth:`kernel_dict`.

    The generated kernel dictionary varies based on the selected mode of
    operation. By default, irradiance values are given in W/km^2/nm. The
    ``scale`` parameter can be used to adjust the value based on unit conversion
    or to account for variations of the Sun-planet distance.

    .. admonition:: Configuration examples
        :class: hint

        Default:
            .. code:: python

               {
                   "dataset": "thuillier_2003",
                   "scale": 1.
               }

    .. admonition:: Configuration format
        :class: hint

        ``dataset`` (str):
            Dataset key.

            Allowed values: see :attr:`eradiate.data.SOLAR_IRRADIANCE_SPECTRA`.

            Default: ``"thuillier_2003"``.

        ``scale`` (float):
            Scaling factor.

            Default: 1.
    """

    @classmethod
    def config_schema(cls):
        return {
            "dataset": {
                "type": "string",
                "allowed": list(SOLAR_IRRADIANCE_SPECTRA.keys()),
                "default": "thuillier_2003",
            },
            "scale": {
                "type": "number",
                "min": 0.,
                "default": 1.0,
            }
        }

    dataset = attr.ib(default=None)

    def init(self):
        dataset = self.config["dataset"]

        # Select dataset
        try:
            self.dataset = data.get(SOLAR_IRRADIANCE_SPECTRA[dataset])
        except KeyError:
            raise ValueError(f"unknown dataset {dataset}")

    def kernel_dict(self, **kwargs):
        from eradiate import mode

        if mode.type == "mono":
            wavelength = mode.config.get("wavelength", None)

            irradiance_magnitude = float(
                self.dataset["spectral_irradiance"].interp(
                    wavelength=wavelength,
                    method="linear",
                ).values
            )

            # Raise if out of bounds or ill-formed dataset
            if np.isnan(irradiance_magnitude):
                raise ValueError(f"dataset evaluation returned nan")

            # Apply units
            irradiance = ureg.Quantity(
                irradiance_magnitude,
                self.dataset["spectral_irradiance"].attrs["units"]
            )

            # Apply scaling, build kernel dict
            return {
                "spectrum": {
                    "type": "uniform",
                    "value": irradiance.to(kdu.get("irradiance")).magnitude *
                             self.config["scale"]
                }
            }

        else:
            raise ModeError(f"unsupported mode {mode.type}")
