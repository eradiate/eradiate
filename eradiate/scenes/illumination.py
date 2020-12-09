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
    validator_has_quantity,
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
        validator=attr.validators.optional((attr.validators.instance_of(str))),
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


@IlluminationFactory.register(name="constant")
@attr.s
class ConstantIllumination(Illumination):
    """Constant illumination scene element [:factorykey:`constant`].

    See :class:`Illumination` for undocumented members.

    .. rubric:: Constructor arguments / instance attributes

    ``radiance`` (float or :class:`~eradiate.scenes.spectra.Spectrum`):
        Emitted radiance spectrum. Must be a radiance spectrum (in W/m^2/sr/nm
        or compatible units).
        Default: 1 cdu[radiance].
    """

    radiance = attr.ib(
        default=1.,
        converter=SpectrumFactory.converter("radiance"),
        validator=[attr.validators.instance_of(Spectrum),
                   validator_has_quantity("radiance")]
    )

    def kernel_dict(self, ref=True):
        return {
            self.id: {
                "type": "constant",
                "radiance": self.radiance.kernel_dict()["spectrum"]
            }
        }


@IlluminationFactory.register(name="directional")
@attr.s
class DirectionalIllumination(Illumination):
    """Directional illumination scene element [:factorykey:`directional`].

    The illumination is oriented based on the classical angular convention used
    in Earth observation.

    See :class:`Illumination` for undocumented members.

    .. rubric:: Constructor arguments / instance attributes

    ``zenith`` (float):
         Zenith angle. Default: 0 deg.

        Unit-enabled field (default units: cdu[angle]).

    ``azimuth`` (float):
        Azimuth angle value. Default: 0 deg.

        Unit-enabled field (default units: cdu[angle]).

    ``irradiance`` (:class:`~eradiate.scenes.spectra.Spectrum`):
        Emitted power flux in the plane orthogonal to the illumination direction.
        Must be an irradiance spectrum (in W/m^2/nm or compatible unit).
        Default: :class:`~eradiate.scenes.spectra.SolarIrradianceSpectrum`.

        Can be initialised with a dictionary processed by
        :meth:`.SpectrumFactory.convert`.
    """

    zenith = attrib_quantity(
        default=ureg.Quantity(0., ureg.deg),
        validator=validator_is_positive,
        units_compatible=cdu.generator("angle"),
    )

    azimuth = attrib_quantity(
        default=ureg.Quantity(0., ureg.deg),
        validator=validator_is_positive,
        units_compatible=cdu.generator("angle"),
    )

    irradiance = attr.ib(
        factory=SolarIrradianceSpectrum,
        converter=SpectrumFactory.converter("irradiance"),
        validator=[attr.validators.instance_of(Spectrum),
                   validator_has_quantity("irradiance")]
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
