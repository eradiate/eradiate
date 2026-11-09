from __future__ import annotations

import attrs
import pint

from ._core import AtmosphereComponent, AtmosphereComponentFlag
from ..phase import PhaseFunction, RayleighPhaseFunction, phase_function_factory
from ..spectra import AirScatteringCoefficientSpectrum, Spectrum, spectrum_factory
from ...attrs import define, documented
from ...grid import SpatialGrid
from ...spectral.index import SpectralIndex
from ...util.deprecation import substitute
from ...validators import has_quantity


@define(eq=False, slots=False)
class HomogeneousComponent(AtmosphereComponent):
    """
    Homogeneous atmosphere component [``homogeneous``].

    This component has spatially uniform collision coefficients and a
    customizable phase function. Evaluation methods ignore the ``grid``
    argument and return scalar values.
    """

    flags = AtmosphereComponentFlag.HOMOGENEOUS

    sigma_s: Spectrum = documented(
        attrs.field(
            factory=AirScatteringCoefficientSpectrum,
            converter=spectrum_factory.converter("collision_coefficient"),
            validator=[
                attrs.validators.instance_of(Spectrum),
                has_quantity("collision_coefficient"),
            ],
        ),
        doc="Scattering coefficient value.\n"
        "\n"
        "Can be initialized with a dictionary processed by "
        ":data:`~eradiate.scenes.spectra.spectrum_factory`.",
        type=":class:`~eradiate.scenes.spectra.Spectrum` or float",
        default=":class:`AirScatteringCoefficientSpectrum() "
        "<.AirScatteringCoefficientSpectrum>`",
    )

    sigma_a: Spectrum = documented(
        attrs.field(
            default=0.0,
            converter=spectrum_factory.converter("collision_coefficient"),
            validator=[
                attrs.validators.instance_of(Spectrum),
                has_quantity("collision_coefficient"),
            ],
        ),
        doc="Absorption coefficient value. Defaults disable absorption.\n"
        "\n"
        "Can be initialized with a dictionary processed by "
        ":data:`~eradiate.scenes.spectra.spectrum_factory`.",
        type=":class:`~eradiate.scenes.spectra.Spectrum`",
        default="0.0 km**-1",
    )

    _phase: PhaseFunction = documented(
        attrs.field(
            factory=lambda: RayleighPhaseFunction(),
            converter=phase_function_factory.convert,
            validator=attrs.validators.instance_of(PhaseFunction),
        ),
        doc="Scattering phase function.\n"
        "\n"
        "Can be initialized with a dictionary processed by "
        ":data:`~eradiate.scenes.phase.phase_function_factory`.",
        type=":class:`~eradiate.scenes.phase.PhaseFunction`",
        default=":class:`RayleighPhaseFunction() <.RayleighPhaseFunction>`",
    )

    def update(self) -> None:
        # Inherit docstring
        self.phase.id = self.phase_id

    @property
    def phase(self) -> PhaseFunction:
        # Inherit docstring
        return self._phase

    def eval_albedo(
        self, si: SpectralIndex, grid: SpatialGrid | None = None
    ) -> pint.Quantity:
        # Inherit docstring
        return self.eval_sigma_s(si) / self.eval_sigma_t(si)

    def eval_sigma_a(
        self, si: SpectralIndex, grid: SpatialGrid | None = None
    ) -> pint.Quantity:
        # Inherit docstring
        return self.sigma_a.eval(si)

    def eval_sigma_s(
        self, si: SpectralIndex, grid: SpatialGrid | None = None
    ) -> pint.Quantity:
        # Inherit docstring
        return self.sigma_s.eval(si)

    def eval_sigma_t(
        self, si: SpectralIndex, grid: SpatialGrid | None = None
    ) -> pint.Quantity:
        # Inherit docstring
        return self.eval_sigma_a(si) + self.eval_sigma_s(si)


__getattr__ = substitute(
    {
        "HomogeneousAtmosphere": (
            HomogeneousComponent,
            {
                "deprecated_in": "1.4.0",
                "removed_in": "2.0.0",
                "details": "Use 'HomogeneousComponent' instead, inside an "
                "'Atmosphere' object. Experiments wrap a single component in "
                "an 'Atmosphere' automatically.",
            },
        )
    },
    stacklevel=3,
)
