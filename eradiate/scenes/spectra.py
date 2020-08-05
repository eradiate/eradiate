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
from ..util.collections import frozendict
from ..util.exceptions import ModeError
from ..util.units import Q_


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
    CONFIG_SCHEMA = frozendict({
        "value": {
            "type": "number",
            "min": 0.,
            "default": 1.0,
        },
    })

    def kernel_dict(self, **kwargs):
        return {
            "spectrum": {
                "type": "uniform",
                "value": self.config["value"],
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

    CONFIG_SCHEMA = frozendict({
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
    })

    dataset = attr.ib(default=None)

    def init(self):
        dataset = self.config["dataset"]

        # Select dataset
        if dataset == "thuillier2003":
            self.dataset = data.get("spectra/thuillier2003.nc")
        else:
            raise ValueError(f"unsupported dataset {dataset}")

    def kernel_dict(self, **kwargs):
        from eradiate import mode

        if mode.type == "mono":
            wavelength = mode.config.get("wavelength", None)

            irradiance_magnitude = float(
                self.dataset["irradiance"].interp(
                    wavelength=wavelength,
                    method="linear",
                ).values
            )

            # Raise if out of bounds or ill-formed dataset
            if np.isnan(irradiance_magnitude):
                raise ValueError(f"dataset evaluation returned nan")

            # Apply units
            irradiance = Q_(
                irradiance_magnitude,
                self.dataset["irradiance"].attrs["units"]
            )

            # Apply unit conversion and scaling, build kernel dict
            return {
                "spectrum": {
                    "type": "uniform",
                    "value": irradiance.to("W/km^2/nm").magnitude *
                             self.config["scale"]
                }
            }

        else:
            raise ModeError(f"unsupported mode {mode.type}")
