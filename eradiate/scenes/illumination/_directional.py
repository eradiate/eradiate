import attr
import pinttr

from ._core import Illumination, IlluminationFactory
from ..spectra import SolarIrradianceSpectrum, Spectrum, SpectrumFactory
from ..._attrs import documented, parse_docs
from ..._units import unit_context_config as ucc
from ..._units import unit_registry as ureg
from ...frame import angles_to_direction
from ...validators import has_quantity, is_positive


@IlluminationFactory.register("directional")
@parse_docs
@attr.s
class DirectionalIllumination(Illumination):
    """
    Directional illumination scene element [:factorykey:`directional`].

    The illumination is oriented based on the classical angular convention used
    in Earth observation.
    """

    zenith = documented(
        pinttr.ib(
            default=ureg.Quantity(0.0, ureg.deg),
            validator=is_positive,
            units=ucc.deferred("angle"),
        ),
        doc="Zenith angle. \n" "\n" "Unit-enabled field (default units: cdu[angle]).",
        type="float",
        default="0.0 deg",
    )

    azimuth = documented(
        pinttr.ib(
            default=ureg.Quantity(0.0, ureg.deg),
            validator=is_positive,
            units=ucc.deferred("angle"),
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
            validator=[
                attr.validators.instance_of(Spectrum),
                has_quantity("irradiance"),
            ],
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
                "direction": list(
                    -angles_to_direction(
                        theta=self.zenith.to(ureg.rad).magnitude,
                        phi=self.azimuth.to(ureg.rad).magnitude,
                    )
                ),
                "irradiance": self.irradiance.kernel_dict()["spectrum"],
            }
        }
