import attrs
import pint
import pinttr

from ._core import Atmosphere
from ..core import KernelDict
from ..phase import PhaseFunction, RayleighPhaseFunction, phase_function_factory
from ..spectra import AirScatteringCoefficientSpectrum, Spectrum, spectrum_factory
from ...attrs import documented, parse_docs
from ...contexts import KernelDictContext, SpectralContext
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg
from ...validators import has_quantity


@parse_docs
@attrs.define
class HomogeneousAtmosphere(Atmosphere):
    """
    Homogeneous atmosphere scene element [``homogeneous``].

    This class builds an atmosphere consisting of a homogeneous medium with
    customisable collision coefficients and phase function, attached to a
    cuboid shape.
    """

    _bottom: pint.Quantity = documented(
        pinttr.field(
            default=ureg.Quantity(0.0, ureg.km),
            units=ucc.deferred("length"),
        ),
        doc="Atmosphere's bottom altitude.\n\nUnit-enabled field (default: ucc[length])",
        type="quantity",
        init_type="quantity or float",
        default="0 km",
    )

    _top: pint.Quantity = documented(
        pinttr.field(
            default=ureg.Quantity(10.0, ureg.km),
            units=ucc.deferred("length"),
        ),
        doc="Atmosphere's top altitude.\n\nUnit-enabled field (default: ucc[length]).",
        type="quantity",
        init_type="quantity or float",
        default="10 km.",
    )

    @_bottom.validator
    @_top.validator
    def _validate_bottom_and_top(instance, attribute, value):
        if instance.bottom >= instance.top:
            raise ValueError("bottom altitude must be lower than top altitude")

    sigma_s: Spectrum = documented(
        attrs.field(
            factory=AirScatteringCoefficientSpectrum,
            converter=spectrum_factory.converter("collision_coefficient"),
            validator=[
                attrs.validators.instance_of(Spectrum),
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

    sigma_a: Spectrum = documented(
        attrs.field(
            default=0.0,
            converter=spectrum_factory.converter("collision_coefficient"),
            validator=[
                attrs.validators.instance_of(Spectrum),
                has_quantity("collision_coefficient"),
            ],
        ),
        doc="Atmosphere absorption coefficient value. Defaults disable "
        "absorption.\n"
        "\n"
        "Can be initialised with a dictionary processed by "
        ":data:`~eradiate.scenes.spectra.spectrum_factory`.",
        type=":class:`~eradiate.scenes.spectra.Spectrum`",
        default="0.0 ucc[collision_coefficient]",
    )

    phase: PhaseFunction = documented(
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
        self.phase.id = self.id_phase

    # --------------------------------------------------------------------------
    #                               Properties
    # --------------------------------------------------------------------------

    @property
    def bottom(self) -> pint.Quantity:
        return self._bottom

    @property
    def top(self) -> pint.Quantity:
        return self._top

    # --------------------------------------------------------------------------
    #                           Evaluation methods
    # --------------------------------------------------------------------------

    def eval_mfp(self, ctx: KernelDictContext) -> pint.Quantity:
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
        spectral_ctx : :class:`.SpectralContext`
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

    def eval_sigma_a(self, spectral_ctx: SpectralContext) -> pint.Quantity:
        """
        Return absorption coefficient.

        Parameters
        ----------
        spectral_ctx : :class:`.SpectralContext`
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode).

        Returns
        -------
        quantity
            Absorption coefficient.
        """
        return self.sigma_a.eval(spectral_ctx)

    def eval_sigma_s(self, spectral_ctx: SpectralContext) -> pint.Quantity:
        """
        Return scattering coefficient.

        Parameters
        ----------
        spectral_ctx : :class:`.SpectralContext`
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode).

        Returns
        -------
        quantity
            Scattering coefficient.
        """
        return self.sigma_s.eval(spectral_ctx)

    def eval_sigma_t(self, spectral_ctx: SpectralContext) -> pint.Quantity:
        """
        Return extinction coefficient.

        Parameters
        ----------
        spectral_ctx : :class:`.SpectralContext`
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode).

        Returns
        -------
        quantity
            Extinction coefficient.
        """
        return self.eval_sigma_a(spectral_ctx) + self.eval_sigma_s(spectral_ctx)

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    def kernel_phase(self, ctx: KernelDictContext) -> KernelDict:
        return self.phase.kernel_dict(ctx=ctx)

    def kernel_media(self, ctx: KernelDictContext) -> KernelDict:
        # Note: The "medium" param is set at a higher level: it is set as a
        # reference in the kernel_dict() method.
        return KernelDict(
            {
                self.id_medium: {
                    "type": "homogeneous",
                    "sigma_t": self.eval_sigma_t(ctx.spectral_ctx).m_as(
                        uck.get("collision_coefficient")
                    ),
                    "albedo": self.eval_albedo(ctx.spectral_ctx).m_as(
                        uck.get("albedo")
                    ),
                }
            }
        )
