from typing import Dict

import attr
import pint
import pinttr

from ._core import Illumination, illumination_factory
from ..spectra import SolarIrradianceSpectrum, Spectrum, spectrum_factory
from ...attrs import documented, parse_docs
from ...contexts import KernelDictContext
from ...frame import angles_to_direction
from ...units import unit_context_config as ucc
from ...units import unit_registry as ureg
from ...validators import has_quantity, is_positive


@illumination_factory.register(type_id="directional")
@parse_docs
@attr.s
class DirectionalIllumination(Illumination):
    """
    Directional illumination scene element [``directional``].

    The illumination is oriented based on the classical angular convention used
    in Earth observation.
    """

    zenith: pint.Quantity = documented(
        pinttr.ib(
            default=ureg.Quantity(0.0, ureg.deg),
            validator=is_positive,
            units=ucc.deferred("angle"),
        ),
        doc="Zenith angle.\n\nUnit-enabled field (default units: ucc[angle]).",
        type="float",
        default="0.0 deg",
    )

    azimuth: pint.Quantity = documented(
        pinttr.ib(
            default=ureg.Quantity(0.0, ureg.deg),
            validator=is_positive,
            units=ucc.deferred("angle"),
        ),
        doc="Azimuth angle value.\n"
        "\n"
        "Unit-enabled field (default units: ucc[angle]).",
        type="float",
        default="0.0 deg",
    )

    irradiance: Spectrum = documented(
        attr.ib(
            factory=SolarIrradianceSpectrum,
            converter=spectrum_factory.converter("irradiance"),
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

    def kernel_dict(self, ctx: KernelDictContext) -> Dict:
        return {
            self.id: {
                "type": "directional",
                "direction": list(
                    -angles_to_direction(
                        [self.zenith.m_as(ureg.rad), self.azimuth.m_as(ureg.rad)]
                    ).squeeze(),
                ),
                "irradiance": self.irradiance.kernel_dict(ctx=ctx)["spectrum"],
            }
        }
