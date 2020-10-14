"""Spectrum-related scene generation facilities.

.. admonition:: Factory-enabled scene elements
    :class: hint

    .. factorytable::
        :modules: spectra
"""
from abc import ABC

import attr
import numpy as np
from pint import DimensionalityError

from .core import SceneElementFactory, SceneElement
from .. import data
from ..data import SOLAR_IRRADIANCE_SPECTRA
from ..util.attrs import attrib, attrib_float_positive, attrib_units, validator_is_positive, validator_is_string
from ..util.exceptions import ModeError
from ..util.units import compatible
from ..util.units import config_default_units as cdu
from ..util.units import kernel_default_units as kdu
from ..util.units import ureg


@attr.s
class Spectrum(SceneElement, ABC):
    """Spectrum abstract base class.

    See :class:`SceneElement` for undocumented members.
    """
    pass


@SceneElementFactory.register(name="uniform")
@attr.s
class UniformSpectrum(Spectrum):
    """Uniform spectrum scene element [:factorykey:`uniform`].

    See :class:`Spectrum` for undocumented members.

    Constructor arguments / instance attributes:
        ``quantity`` ("radiance" or "irradiance" or "reflectance"):
            Physical quantity represented by the current instance. This field is
            used to automatically determine compatible units for the ``value``
            field, as well as its default unit.

        ``value`` (float):
            Spectrum constant value. Default: 1.

            Unit-enabled field (default: cdu[quantity]).
    """

    _valid_quantities = ("radiance", "irradiance", "reflectance")

    quantity = attrib(
        default="radiance",
        validator=attr.validators.in_(_valid_quantities),
    )

    value = attrib_float_positive(
        default=1.0,
        has_units=True
    )

    value_units = attrib_units(
        default=None,  # Note: default is None here but handled in post-init step
        compatible_units=[
            cdu.get("radiance"), cdu.get("irradiance"), cdu.get("reflectance")
        ],
    )

    def __attrs_post_init__(self):
        # Check unit and quantity consistency
        quantity_units = cdu.get(self.quantity)

        if self.value_units is None:
            # If no unit was specified, get default
            self.value_units = quantity_units

        else:
            if not compatible(self.value_units, quantity_units):
                raise DimensionalityError(
                    self.value_units,
                    quantity_units,
                    extra_msg="inconsistent units between value and quantity "
                              "fields"
                )

        # Apply parent class post-init (unit scaling)
        super(UniformSpectrum, self).__attrs_post_init__()

    def kernel_dict(self, **kwargs):
        value = self.get_quantity("value")
        kernel_units = kdu.get(self.quantity)

        return {
            "spectrum": {
                "type": "uniform",
                "value": value.to(kernel_units).magnitude,
            }
        }


@SceneElementFactory.register(name="solar_irradiance")
@attr.s
class SolarIrradianceSpectrum(Spectrum):
    """Solar irradiance spectrum scene element
    [:factorykey:`solar_irradiance`].

    This scene element produces the scene dictionary required to
    instantiate a kernel plugin using the Sun irradiance spectrum. The data set
    used by this element is controlled by the ``dataset`` attribute (see
    :data:`eradiate.data.SOLAR_IRRADIANCE_SPECTRA` for available data sets).

    The spectral range of the data sets shipped can vary and an attempt for use
    outside of the supported spectral range will raise a :class:`ValueError`
    upon calling :meth:`kernel_dict`.

    The generated kernel dictionary varies based on the selected mode of
    operation. The ``scale`` parameter can be used to adjust the value based on
    unit conversion or to account for variations of the Sun-planet distance.

    The produced kernel dictionary automatically adjusts its irradiance units
    depending on the selected kernel default units.

    Constructor arguments / instance attributes:
        ``dataset`` (str):
            Dataset key. Allowed values: see
            :attr:`eradiate.data.SOLAR_IRRADIANCE_SPECTRA`.
            Default: ``"thuillier_2003"``.

        ``scale`` (float):
            Scaling factor. Default: 1.
    """

    #: Dataset identifier
    dataset = attr.ib(
        default="thuillier_2003",
        validator=validator_is_string,
    )

    scale = attr.ib(
        default=1.,
        converter=float,
        validator=validator_is_positive,
    )

    @dataset.validator
    def _dataset_validator(self, attribute, value):
        if value not in SOLAR_IRRADIANCE_SPECTRA:
            raise ValueError(f"while setting {attribute.name}: '{value}' not in "
                             f"list of supported solar irradiance spectra "
                             f"{str(list(SOLAR_IRRADIANCE_SPECTRA.keys()))}")

    @property
    def quantity(self):
        """Physical quantity associated to this spectrum.

        Returns → str:
            Always returns ``"irradiance"``."""
        return "irradiance"

    def __attrs_post_init__(self):
        super(SolarIrradianceSpectrum, self).__attrs_post_init__()

        # Load dataset
        try:
            self._data = data.get(SOLAR_IRRADIANCE_SPECTRA[self.dataset])
        except KeyError:
            raise ValueError(f"unknown dataset {self.dataset}")

    def kernel_dict(self, ref=True):
        from eradiate import mode

        if mode.type == "mono":
            wavelength = mode.config.get("wavelength", None)

            irradiance_magnitude = float(
                self._data["spectral_irradiance"].interp(
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
                self._data["spectral_irradiance"].attrs["units"]
            )

            # Apply scaling, build kernel dict
            return {
                "spectrum": {
                    "type": "uniform",
                    "value": irradiance.to(kdu.get("irradiance")).magnitude *
                             self.scale
                }
            }

        else:
            raise ModeError(f"unsupported mode '{mode.type}'")
