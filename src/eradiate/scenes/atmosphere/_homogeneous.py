import typing as t

import attrs
import pint
import pinttr

from ._core import Atmosphere
from ..core import traverse
from ..phase import PhaseFunction, RayleighPhaseFunction, phase_function_factory
from ..spectra import AirScatteringCoefficientSpectrum, SpectrumNode, spectrum_factory
from ...attrs import documented, parse_docs
from ...contexts import KernelDictContext, SpectralContext
from ...kernel import InitParameter, UpdateParameter
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg
from ...validators import has_quantity


@parse_docs
@attrs.define(eq=False, slots=False)
class HomogeneousAtmosphere(Atmosphere):
    """
    Homogeneous atmosphere scene element [``homogeneous``].

    This class builds an atmosphere consisting of a homogeneous medium with
    customisable collision coefficients and phase function.
    """

    _bottom: pint.Quantity = documented(
        pinttr.field(
            default=ureg.Quantity(0.0, ureg.km),
            units=ucc.deferred("length"),
        ),
        doc="Atmosphere's bottom altitude.\n"
        "\n"
        'Unit-enabled field (default: ucc["length"]).',
        type="quantity",
        init_type="quantity or float",
        default="0 km",
    )

    _top: pint.Quantity = documented(
        pinttr.field(
            default=ureg.Quantity(10.0, ureg.km),
            units=ucc.deferred("length"),
        ),
        doc="Atmosphere's top altitude.\n"
        "\n"
        'Unit-enabled field (default: ucc["length"]).',
        type="quantity",
        init_type="quantity or float",
        default="10 km",
    )

    @_bottom.validator
    @_top.validator
    def _validate_bottom_and_top(self, attribute, value):
        if self.bottom >= self.top:
            raise ValueError("bottom altitude must be lower than top altitude")

    sigma_s: SpectrumNode = documented(
        attrs.field(
            factory=AirScatteringCoefficientSpectrum,
            converter=spectrum_factory.converter("collision_coefficient"),
            validator=[
                attrs.validators.instance_of(SpectrumNode),
                has_quantity("collision_coefficient"),
            ],
        ),
        doc="Atmosphere scattering coefficient value.\n"
        "\n"
        "Can be initialised with a dictionary processed by "
        ":data:`~eradiate.scenes.spectra.spectrum_factory`.",
        type=":class:`~eradiate.scenes.spectra.Spectrum` or float",
        default=":class:`AirScatteringCoefficientSpectrum() "
        "<.AirScatteringCoefficientSpectrum>`",
    )

    sigma_a: SpectrumNode = documented(
        attrs.field(
            default=0.0,
            converter=spectrum_factory.converter("collision_coefficient"),
            validator=[
                attrs.validators.instance_of(SpectrumNode),
                has_quantity("collision_coefficient"),
            ],
        ),
        doc="Atmosphere absorption coefficient value. Defaults disable "
        "absorption.\n"
        "\n"
        "Can be initialised with a dictionary processed by "
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
        "Can be initialised with a dictionary processed by "
        ":data:`~eradiate.scenes.phase.phase_function_factory`.",
        type=":class:`~eradiate.scenes.phase.PhaseFunction`",
        default=":class:`RayleighPhaseFunction() <.RayleighPhaseFunction>`",
    )

    def __attrs_post_init__(self) -> None:
        self.update()

    def update(self) -> None:
        # Inherit docstring
        self.phase.id = self.phase_id

    # --------------------------------------------------------------------------
    #                               Properties
    # --------------------------------------------------------------------------

    @property
    def bottom(self) -> pint.Quantity:
        # Inherit docstring
        return self._bottom

    @property
    def top(self) -> pint.Quantity:
        # Inherit docstring
        return self._top

    @property
    def phase(self) -> PhaseFunction:
        # Inherit docstring
        return self._phase

    # --------------------------------------------------------------------------
    #                           Evaluation methods
    # --------------------------------------------------------------------------

    def eval_mfp(self, ctx: KernelDictContext) -> pint.Quantity:
        # Inherit docstring
        return (
            1.0 / self.eval_sigma_s(ctx.spectral_ctx)
            if self.eval_sigma_s(ctx.spectral_ctx).m != 0.0
            else 1.0 / self.eval_sigma_a(ctx.spectral_ctx)
        )

    def eval_albedo(self, spectral_ctx: SpectralContext) -> pint.Quantity:
        """
        Return albedo.

        Parameters
        ----------
        spectral_ctx : .SpectralContext
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode).

        Returns
        -------
        quantity
            Albedo.
        """
        return self.eval_sigma_s(spectral_ctx) / (
            self.eval_sigma_s(spectral_ctx) + self.eval_sigma_a(spectral_ctx)
        )

    def eval_sigma_a(self, sctx: SpectralContext) -> pint.Quantity:
        """
        Return absorption coefficient.

        Parameters
        ----------
        sctx : .SpectralContext
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode).

        Returns
        -------
        quantity
            Absorption coefficient.
        """
        return self.sigma_a.eval(sctx)

    def eval_sigma_s(self, sctx: SpectralContext) -> pint.Quantity:
        """
        Return scattering coefficient.

        Parameters
        ----------
        sctx : .SpectralContext
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode).

        Returns
        -------
        quantity
            Scattering coefficient.
        """
        return self.sigma_s.eval(sctx)

    def eval_sigma_t(self, sctx: SpectralContext) -> pint.Quantity:
        """
        Return extinction coefficient.

        Parameters
        ----------
        sctx : .SpectralContext
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode).

        Returns
        -------
        quantity
            Extinction coefficient.
        """
        return self.eval_sigma_a(sctx) + self.eval_sigma_s(sctx)

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    @property
    def _template_phase(self) -> dict:
        # Inherit docstring
        result, _ = traverse(self.phase)
        return result.data

    @property
    def _template_medium(self) -> dict:
        # Inherit docstring
        return {
            "type": "homogeneous",
            "sigma_t": InitParameter(
                lambda ctx: self.eval_sigma_t(ctx.spectral_ctx).m_as(
                    uck.get("collision_coefficient")
                ),
            ),
            "albedo": InitParameter(
                lambda ctx: self.eval_albedo(ctx.spectral_ctx).m_as(uck.get("albedo"))
            ),
            # Note: "phase" is deliberately unset, this is left to the
            # Atmosphere.template property
        }

    @property
    def _params_medium(self) -> t.Dict[str, UpdateParameter]:
        # Inherit docstring
        return {
            # Note: "value" appears twice because the mi.Spectrum is
            # encapsulated in a mi.ConstVolume
            "sigma_t.value.value": UpdateParameter(
                lambda ctx: self.eval_sigma_t(ctx.spectral_ctx).m_as(
                    uck.get("collision_coefficient")
                ),
                UpdateParameter.Flags.SPECTRAL,
            ),
            "albedo.value.value": UpdateParameter(
                lambda ctx: self.eval_albedo(ctx.spectral_ctx).m_as(uck.get("albedo")),
                UpdateParameter.Flags.SPECTRAL,
            ),
        }

    @property
    def _params_phase(self) -> t.Dict[str, UpdateParameter]:
        # Inherit docstring
        _, params = traverse(self.phase)
        return params.data
