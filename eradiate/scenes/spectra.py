"""Spectrum-related scene generation facilities.

.. admonition:: Registered factory members
    :class: hint

    .. factorytable::
       :factory: SceneElementFactory
       :modules: eradiate.scenes.spectra
"""
from abc import ABC

import attr
import numpy as np
import pint

from .core import SceneElement, SceneElementFactory
from .. import data
from ..util import rstrip
from ..util.attrs import (
    MKey, validator_is_positive, validator_is_string, validator_units_compatible
)
from ..util.exceptions import ModeError
from ..util.units import config_default_units as cdu
from ..util.units import ensure_units
from ..util.units import kernel_default_units as kdu
from ..util.units import ureg


@attr.s
class Spectrum(SceneElement, ABC):
    """Spectrum abstract base class.

    See :class:`SceneElement` for undocumented members.
    """

    @staticmethod
    def converter(quantity):
        """This function generates a converter wrapping
        :meth:`SceneElementFactory.convert` to handle defaults for shortened
        spectrum defintions. The produced converter processes a parameter
        ``value`` as follows:

        * if ``value`` is a float or a :class:`pint.Quantity`, the converter
          calls itself using a dictionary ``{"type": "uniform", "value": value}``;
        * if ``value`` is a dictionary, it replaces a ``"type": "uniform"``
          entry with ``"type": f"uniform_{quantity}"`` and hands the resulting
          dictionary to :meth:`.SceneElementFactory.convert`;
        * otherwise, it calls :meth:`.SceneElementFactory.convert`.

        Parameter ``quantity`` (str):
            Quantity string. Supported values:

            * ``"radiance"``
            * ``"irradiance"``
            * ``"reflectance"``
            * ``"transmittance"``

        Returns → callable:
            Generated converter.
        """

        # TODO: remove when specialised factories will be implemented
        #  (merge into SpectrumFactory.convert())

        def f(value):
            if isinstance(value, (float, pint.Quantity)):
                return f({"type": "uniform", "value": value})

            if isinstance(value, dict):
                try:
                    if value["type"] == "uniform":
                        return SceneElementFactory.convert(
                            {**value, "type": f"uniform_{quantity}"}
                        )
                except KeyError:
                    pass

            return SceneElementFactory.convert(value)

        return f

    _quantity = None  # String containing the physical quantity held by the spectrum
    _units_compatible = None  # Associated basic units


@attr.s
class UniformSpectrum(Spectrum):
    """Base class for uniform spectra. This class must be subclassed to a
    child class also inheriting from a quantity mixin class; otherwise, its
    initialisation will be incomplete.

    See :class:`Spectrum` for undocumented members.
    """
    _quantity = None
    _units_compatible = None
    value = attr.ib(default=None, init=False, repr=True)

    def kernel_dict(self, ref=True):
        kernel_units = kdu.get(self._quantity)

        return {
            "spectrum": {
                "type": "uniform",
                "value": self.value.to(kernel_units).magnitude,
            }
        }

    @staticmethod
    def _value_validator(instance, attribute, val):
        """This static method defines a validator to attach to the definition
        of the child class.
        """
        v = validator_units_compatible(instance._units_compatible)
        return v(instance, attribute, val)

    @staticmethod
    def _attrib_value(mixin):
        """This static method creates an attribute specification for the
        ``value`` field. It should be used to override the ``value`` field
        in the definition of the child class.

        Parameter ``mixin`` (type):
            The quantity mixin class from which the child class inherits.

        Returns → :class:`attr._make._CountingAttr`:
            Generated attribute field.
        """
        return attr.ib(
            default=attr.Factory(lambda self: ureg.Quantity(1., mixin._units_compatible),
                                 takes_self=True),
            converter=mixin._value_converter,
            validator=UniformSpectrum._value_validator,
            on_setattr=attr.setters.pipe(attr.setters.convert, attr.setters.validate),
            metadata={MKey.compatible_units: mixin._units_compatible}
        )


@attr.s
class RadianceMixin:
    """Radiance quantity mixin."""
    _quantity = "radiance"
    _units_compatible = ureg.Unit("W/m^2/sr/nm")

    @staticmethod
    def _value_converter(val):
        return ensure_units(val, cdu.generator("radiance"))


@attr.s
class IrradianceMixin:
    """Irradiance quantity mixin."""
    _quantity = "irradiance"
    _units_compatible = ureg.Unit("W/m^2/nm")

    @staticmethod
    def _value_converter(val):
        return ensure_units(val, cdu.generator("irradiance"))


@attr.s
class ReflectanceMixin:
    """Reflectance quantity mixin."""
    _quantity = "reflectance"
    _units_compatible = ureg.dimensionless

    @staticmethod
    def _value_converter(val):
        return ensure_units(val, cdu.generator("reflectance"))


@attr.s
class TransmittanceMixin:
    """Transmittance quantity mixin."""
    _quantity = "transmittance"
    _units_compatible = ureg.dimensionless

    @staticmethod
    def _value_converter(val):
        return ensure_units(val, cdu.generator("transmittance"))


def create_specialized_spectrum(mixin, cls, register_as=None, return_value=False,
                                docstring=""):
    """This function generates a new spectrum class based on a generic spectrum
    type and a quantity mixin.

    Parameter ``mixin`` (type):
        Quantity mixin attached to the spectrum.

    Parameter ``cls`` (type):
        Generic spectrum class.

    Parameter ``register_as`` (str or None):
        If not ``None``, string used to register the created class to the
        :class:`.SceneElementFactory`. Otherwise, the created class is not
        registered to the factory.

    Parameter ``return_value`` (bool):
        If ``True``, return the created class. Otherwise, register it to the
        current module's namespace.

    Returns → type:
        If ``return_value`` is ``True``, created class.
    """
    # Create class name
    mixin_basename = rstrip(mixin.__name__, "Mixin")
    cls_basename = rstrip(cls.__name__, "Spectrum")
    new_cls_name = f"{cls_basename}{mixin_basename}Spectrum"

    # Create class
    new_cls = attr.make_class(
        name=new_cls_name,
        attrs={
            "value": UniformSpectrum._attrib_value(mixin)
        },
        bases=(mixin, cls)
    )
    new_cls.__doc__ = docstring

    if register_as is not None:
        # Apply factory decorator
        new_cls = SceneElementFactory.register(name=register_as)(new_cls)

    if return_value:
        return new_cls
    else:
        # Register class to current module
        globals()[new_cls_name] = new_cls


uniform_spectra_definitions = {
    ("radiance", RadianceMixin),
    ("irradiance", IrradianceMixin),
    ("reflectance", ReflectanceMixin),
    ("transmittance", TransmittanceMixin)
}

for quantity, mixin in uniform_spectra_definitions:
    create_specialized_spectrum(
        mixin, UniformSpectrum,
        register_as=f"uniform_{quantity}", return_value=False,
        docstring= \
            f"""Uniform spectrum scene element distribution ({quantity})
            [:factorykey:`uniform_{quantity}`].

            .. rubric:: Constructor arguments / instance attributes

            ``value`` (float):
                Uniform distribution value. Default: 1 cdu[{quantity}].

                Unit-enabled field (default: cdu[{quantity}]).
            """
    )


@SceneElementFactory.register(name="solar_irradiance")
@attr.s(frozen=True)
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

    .. rubric:: Constructor arguments / instance attributes

    ``dataset`` (str):
        Dataset key. Allowed values: see
        :attr:`solar irradiance dataset documentation <eradiate.data.solar_irradiance_spectra>`.
        Default: ``"thuillier_2003"``.

    ``scale`` (float):
        Scaling factor. Default: 1.
    """

    _quantity = "irradiance"
    _units_compatible = ureg.Unit("W/m^2/nm")

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
        if value not in data.registered("solar_irradiance_spectrum"):
            raise ValueError(f"while setting {attribute.name}: '{value}' not in "
                             f"list of supported solar irradiance spectra "
                             f"{data.registered('solar_irradiance_spectrum')}")

    data = attr.ib(
        init=False,
        repr=False
    )

    @data.default
    def _data_factory(self):
        # Load dataset
        try:
            return data.open("solar_irradiance_spectrum", self.dataset)
        except KeyError:
            raise ValueError(f"unknown dataset {self.dataset}")

    def kernel_dict(self, ref=True):
        from eradiate import mode

        if mode.id == "mono":
            wavelength = mode.wavelength.to(ureg.nm).magnitude

            irradiance_magnitude = float(
                self.data["spectral_irradiance"].interp(
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
                self.data["spectral_irradiance"].attrs["units"]
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
