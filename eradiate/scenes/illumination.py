"""Illumination-related scene generation facilities.

.. admonition:: Factory-enabled scene elements
    :class: hint

    .. factorytable::
        :modules: illumination
"""

from abc import ABC

import attr

from .core import SceneElement, SceneElementFactory
from .spectra import SolarIrradianceSpectrum, UniformSpectrum
from ..util.attrs import attrib, attrib_float_positive, attrib_units
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


@SceneElementFactory.register(name="constant")
@attr.s
class ConstantIllumination(Illumination):
    """Constant illumination scene element [:factorykey:`constant`].

    See :class:`Illumination` for undocumented members.

    Constructor arguments / instance attributes:
        ``radiance`` (:class:`~eradiate.scenes.spectra.UniformSpectrum`):
            Emitted radiance spectrum.

            Default: :class:`~eradiate.scenes.spectra.UniformSpectrum`.
    """
    # TODO: reject non-radiance spectra

    radiance = attrib(
        default=attr.Factory(UniformSpectrum),
        converter=SceneElementFactory.convert,
        validator=attr.validators.instance_of(UniformSpectrum),
    )

    def kernel_dict(self, **kwargs):
        return {
            self.id: {
                "type": "constant",
                "radiance": self.radiance.kernel_dict()["spectrum"]
            }
        }


@SceneElementFactory.register(name="directional")
@attr.s
class DirectionalIllumination(Illumination):
    """Directional illumination scene element [:factorykey:`directional`].

    The illumination is oriented based on the classical angular convention used
    in Earth observation.

    See :class:`Illumination` for undocumented members.

    Constructor arguments / instance attributes:
        ``zenith`` (float):
             Zenith angle. Default: 0.

            Unit-enabled field (default unit: cdu[angle]).

        ``azimuth`` (float):
            Azimuth angle value. Default: 0.

            Unit-enabled field (default unit: cdu[angle]).

        ``irradiance`` (:class:`~eradiate.scenes.spectra.UniformSpectrum` or :class:`~eradiate.scenes.spectra.SolarIrradianceSpectrum`):
            Emitted power flux in the plane orthogonal to the illumination direction.
            Default: :class:`~eradiate.scenes.spectra.SolarIrradianceSpectrum`.
    """
    zenith = attrib_float_positive(
        default=0.,
        has_units=True
    )
    zenith_units = attrib_units(
        default=attr.Factory(lambda: cdu.get("angle")),
        compatible_units=ureg.deg,
    )

    azimuth = attrib_float_positive(
        default=0.,
        has_units=True
    )
    azimuth_units = attrib_units(
        default=attr.Factory(lambda: cdu.get("angle")),
        compatible_units=ureg.deg,
    )

    irradiance = attrib(
        default=attr.Factory(SolarIrradianceSpectrum),
        converter=SceneElementFactory.convert,
        validator=[attr.validators.instance_of(
            (UniformSpectrum, SolarIrradianceSpectrum)
        )],
    )

    def kernel_dict(self, ref=True):
        zenith = self.get_quantity("zenith")
        azimuth = self.get_quantity("azimuth")
        irradiance = self.irradiance.kernel_dict()["spectrum"]

        return {
            self.id: {
                "type": "directional",
                "direction": list(-angles_to_direction(
                    theta=zenith.to(ureg.rad).magnitude,
                    phi=azimuth.to(ureg.rad).magnitude
                )),
                "irradiance": irradiance
            }
        }
