"""Spectrum-related scene generation facilities.

.. admonition:: Factory-enabled scene generation helpers
    :class: hint

    .. factorytable::
        :modules: spectra
"""

import attr
import numpy as np

from .core import Factory, SceneHelper
from .. import data
from ..util.exceptions import ModeError
from ..util.units import ureg, kernel_default_units as kdu, config_default_units as cdu


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
        return dict({
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
        })

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
    (see configuration format). The spectral range of the data sets shipped can
    vary and an attempt for use outside of the supported spectral range will
    raise a :class:`ValueError` upon calling :meth:`kernel_dict`.

    The generated kernel dictionary varies based on the selected mode of
    operation. By default, irradiance values are given in W/km^2/nm. The
    ``scale`` parameter can be used to adjust the value based on unit conversion
    or to account for variations of the Sun-planet distance.

    .. admonition:: Configuration examples
        :class: hint

        Default:
            .. code:: python

               {
                   "dataset": "thuillier2003",
                   "scale": 1.
               }

    .. admonition:: Configuration format
        :class: hint

        ``dataset`` (str):
            Dataset key.

            Allowed values: see table below.

            Default: ``"thuillier2003"``.

        ``scale`` (float):
            Scaling factor.

            Default: 1.

    .. list-table:: Solar irradiance models
       :widths: 1 1 1
       :header-rows: 1

       * - Key
         - Reference
         - Spectral range [nm]
       * - ``thuillier2003``
         - :cite:`Thuillier2003SolarSpectralIrradiance`
         - [200, 2397]
    """

    @classmethod
    def config_schema(cls):
        return dict({
            "dataset": {
                "type": "string",
                "allowed": ["thuillier2003"],
                "default": "thuillier2003",
            },
            "scale": {
                "type": "number",
                "min": 0.,
                "default": 1.0,
            },
            "quantity": {
                "type": "string",
                "allowed": ["radiance", "irradiance"],
                "default": "irradiance"
            }
        })

    dataset = attr.ib(default=None)

    def init(self):
        dataset = self.config["dataset"]

        # Select dataset
        if dataset == "thuillier2003":
            self.dataset = data.get("spectra/thuillier_2003.nc")
        else:
            raise ValueError(f"unsupported dataset {dataset}")

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

            if self.config["quantity"] == "irradiance":
                irradiance = irradiance.to(kdu.get("irradiance")).magnitude
            else:
                raise NotImplementedError(f"Cannot convert to {self.config['quantity']}.")

            # Apply unit conversion and scaling, build kernel dict
            return {
                "spectrum": {
                    "type": "uniform",
                    "value": irradiance * self.config["scale"]
                }
            }

        else:
            raise ModeError(f"unsupported mode {mode.type}")
