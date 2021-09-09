from typing import MutableMapping, Optional

import attr
import pinttr

from ._core import Illumination, IlluminationFactory
from ..spectra import SolarIrradianceSpectrum, Spectrum, SpectrumFactory
from ...attrs import documented, parse_docs
from ...contexts import KernelDictContext
from ...frame import angles_to_direction
from ...units import unit_context_config as ucc
from ...units import unit_registry as ureg
from ...validators import has_len, has_quantity, is_positive


@IlluminationFactory.register("spot")
@parse_docs
@attr.s
class SpotIllumination(Illumination):
    """
    Spot light like illumination, adapted to be used in earth observation contexts.
    This measure is parametrized through zenith and azimuth angles, a distance
    and a target point, instead of directly specifying its position and emission
    direction.
    """

    zenith = documented(
        pinttr.ib(
            default=ureg.Quantity(0.0, ureg.deg),
            validator=is_positive,
            units=ucc.deferred("angle"),
        ),
        doc="Zenith angle.\n\nUnit-enabled field (default units: ucc[angle]).",
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
        "Unit-enabled field (default units: ucc[angle]).",
        type="float",
        default="0.0 deg",
    )

    distance = documented(
        pinttr.ib(
            default=ureg.Quantity(1.0, ureg.m),
            validator=is_positive,
            units=ucc.deferred("length"),
        ),
        doc="Emitter distance from target point.\n"
        "\n"
        "Unit-enabled field (default units: ucc[length]).",
        type="float",
        default="1.0 m",
    )

    target = documented(
        pinttr.ib(
            default=ureg.Quantity([0.0, 0.0, 1.0], ureg.m),
            validator=has_len(3),
            units=ucc.deferred("length"),
        ),
        doc="A 3-element vector specifying the location targeted by the emitter.\n"
        "\n"
        "Unit-enabled field (default: ucc[length]).",
        type="array-like[float, float, float]",
        default="[0, 0, 1] m",
    )

    irradiance = documented(
        attr.ib(
            factory=IntensitySpectrum,
            converter=SpectrumFactory.converter("intensity"),
            validator=[
                attr.validators.instance_of(Spectrum),
                has_quantity("intensity"),
            ],
        ),
        doc="Emitted power into the cone. "
        "Must be an intensity spectrum (in W/sr/nm or compatible unit). "
        "Can be initialised with a dictionary processed by "
        ":meth:`.SpectrumFactory.convert`.",
        type=":class:`~eradiate.scenes.spectra.Spectrum`",
        default=":class:`SolarIrradianceSpectrum() <.IntensitySpectrum>`",
    )
