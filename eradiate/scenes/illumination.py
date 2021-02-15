"""Illumination-related scene generation facilities.

.. admonition:: Registered factory members [:class:`IlluminationFactory`]
   :class: hint

   .. factorytable::
      :factory: IlluminationFactory
"""

from abc import ABC

import attr

from .core import SceneElement
from .spectra import SolarIrradianceSpectrum, Spectrum, SpectrumFactory
from ..util.attrs import (
    attrib_quantity,
    documented, parse_docs, validator_has_quantity,
    validator_is_positive
)
from ..util.factory import BaseFactory
from ..util.frame import angles_to_direction
from ..util.units import config_default_units as cdu
from ..util.units import ureg


@attr.s
class Illumination(SceneElement, ABC):
    """Abstract base class for all illumination scene elements.

    See :class:`~eradiate.scenes.core.SceneElement` for undocumented members.
    """

    id = attr.ib(
        default="illumination",
        validator=attr.validators.optional(attr.validators.instance_of(str)),
    )


class IlluminationFactory(BaseFactory):
    """This factory constructs objects whose classes are derived from
    :class:`Illumination`.

    .. admonition:: Registered factory members
       :class: hint

       .. factorytable::
          :factory: IlluminationFactory
    """
    _constructed_type = Illumination
    registry = {}


@IlluminationFactory.register("constant")
@parse_docs
@attr.s
class ConstantIllumination(Illumination):
    """Constant illumination scene element [:factorykey:`constant`].
    """

    radiance = documented(
        attr.ib(
            default=1.,
            converter=SpectrumFactory.converter("radiance"),
            validator=[attr.validators.instance_of(Spectrum),
                       validator_has_quantity("radiance")]
        ),
        doc="Emitted radiance spectrum. Must be a radiance spectrum "
            "(in W/m^2/sr/nm or compatible units).",
        type="float or :class:`~eradiate.scenes.spectra.Spectrum`",
        default="1.0 cdu[radiance]",
    )

    def kernel_dict(self, ref=True):
        return {
            self.id: {
                "type": "constant",
                "radiance": self.radiance.kernel_dict()["spectrum"]
            }
        }


@IlluminationFactory.register("directional")
@parse_docs
@attr.s
class DirectionalIllumination(Illumination):
    """Directional illumination scene element [:factorykey:`directional`].

    The illumination is oriented based on the classical angular convention used
    in Earth observation.
    """

    zenith = documented(
        attrib_quantity(
            default=ureg.Quantity(0., ureg.deg),
            validator=validator_is_positive,
            units_compatible=cdu.generator("angle"),
        ),
        doc="Zenith angle. \n"
            "\n"
            "Unit-enabled field (default units: cdu[angle]).",
        type="float",
        default="0.0 deg"
    )

    azimuth = documented(
        attrib_quantity(
            default=ureg.Quantity(0., ureg.deg),
            validator=validator_is_positive,
            units_compatible=cdu.generator("angle"),
        ),
        doc="Azimuth angle value.\n"
            "\n"
            "Unit-enabled field (default units: cdu[angle]).",
        type="float",
        default="0.0 deg",
    )

    irradiance = documented(
        attr.ib(
            factory=SolarIrradianceSpectrum,
            converter=SpectrumFactory.converter("irradiance"),
            validator=[attr.validators.instance_of(Spectrum),
                       validator_has_quantity("irradiance")]
        ),
        doc="Emitted power flux in the plane orthogonal to the illumination direction. "
            "Must be an irradiance spectrum (in W/m^2/nm or compatible unit). "
            "Can be initialised with a dictionary processed by "
            ":meth:`.SpectrumFactory.convert`.",
        type=":class:`~eradiate.scenes.spectra.Spectrum`",
        default=":class:`SolarIrradianceSpectrum() <.SolarIrradianceSpectrum>`",
    )

    def kernel_dict(self, ref=True):
        return {
            self.id: {
                "type": "directional",
                "direction": list(-angles_to_direction(
                    theta=self.zenith.to(ureg.rad).magnitude,
                    phi=self.azimuth.to(ureg.rad).magnitude
                )),
                "irradiance": self.irradiance.kernel_dict()["spectrum"]
            }
        }
