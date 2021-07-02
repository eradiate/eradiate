import attr

from ._core import Atmosphere, AtmosphereFactory
from ..phase import PhaseFunction, PhaseFunctionFactory, RayleighPhaseFunction
from ..spectra import AirScatteringCoefficientSpectrum, Spectrum, SpectrumFactory
from ..._util import onedict_value
from ...attrs import documented, parse_docs
from ...units import unit_context_kernel as uck
from ...validators import has_quantity


@AtmosphereFactory.register("homogeneous")
@parse_docs
@attr.s()
class HomogeneousAtmosphere(Atmosphere):
    """
    Homogeneous atmosphere scene element [:factorykey:`homogeneous`].

    This class builds an atmosphere consisting of a homogeneous medium.
    Scattering uses the Rayleigh phase function.
    """

    sigma_s = documented(
        attr.ib(
            factory=AirScatteringCoefficientSpectrum,
            converter=SpectrumFactory.converter("collision_coefficient"),
            validator=[
                attr.validators.instance_of(Spectrum),
                has_quantity("collision_coefficient"),
            ],
        ),
        doc="Atmosphere scattering coefficient value.\n"
        "\n"
        "Can be initialised with a dictionary processed by "
        ":class:`.SpectrumFactory`.",
        type=":class:`~eradiate.scenes.spectra.Spectrum` or float",
        default=":class:`AirScatteringCoefficient() <.AirScatteringCoefficient>`",
    )

    sigma_a = documented(
        attr.ib(
            default=0.0,
            converter=SpectrumFactory.converter("collision_coefficient"),
            validator=[
                attr.validators.instance_of(Spectrum),
                has_quantity("collision_coefficient"),
            ],
        ),
        doc="Atmosphere absorption coefficient value. Defaults disable "
        "absorption.\n"
        "\n"
        "Can be initialised with a dictionary processed by "
        ":class:`.SpectrumFactory`.",
        type=":class:`~eradiate.scenes.spectra.Spectrum`",
        default="0.0 cdu[collision_coefficient]",
    )

    phase = documented(
        attr.ib(
            factory=lambda: RayleighPhaseFunction(),
            converter=PhaseFunctionFactory.convert,
            validator=attr.validators.instance_of(PhaseFunction),
        )
    )

    def __attrs_post_init__(self):
        self.update()

    def update(self):
        self.phase.id = f"phase_{self.id}"

    def kernel_width(self, ctx=None):
        """
        Width of the kernel object delimiting the atmosphere.
        """
        if self.width == "auto":
            spectral_ctx = ctx.spectral_ctx if ctx is not None else None
            return 10.0 / self.eval_sigma_s(spectral_ctx)
        else:
            return self.width

    def eval_albedo(self, spectral_ctx=None):
        """
        Return albedo.

        Parameter ``spectral_ctx`` (:class:`.SpectralContext` or None):
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode).

        Returns → :class:`pint.Quantity`:
            Albedo.
        """
        return self.eval_sigma_s(spectral_ctx) / (
            self.eval_sigma_s(spectral_ctx) + self.eval_sigma_a(spectral_ctx)
        )

    def eval_sigma_a(self, spectral_ctx=None):
        """
        Return absorption coefficient.

        Parameter ``spectral_ctx`` (:class:`.SpectralContext` or None):
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode).

        Returns → :class:`pint.Quantity`:
            Absorption coefficient.
        """
        return self.sigma_a.eval(spectral_ctx)

    def eval_sigma_s(self, spectral_ctx=None):
        """
        Return scattering coefficient.

        Parameter ``spectral_ctx`` (:class:`.SpectralContext` or None):
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode).

        Returns → :class:`pint.Quantity`:
            Scattering coefficient.
        """
        return self.sigma_s.eval(spectral_ctx)

    def eval_sigma_t(self, spectral_ctx=None):
        """
        Return extinction coefficient.

        Parameter ``spectral_ctx`` (:class:`.SpectralContext` or None):
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode).

        Returns → :class:`pint.Quantity`:
            Extinction coefficient.
        """
        return self.eval_sigma_a(spectral_ctx) + self.eval_sigma_s(spectral_ctx)

    def kernel_phase(self, ctx=None):
        return self.phase.kernel_dict(ctx=ctx)

    def kernel_media(self, ctx=None):
        if ctx.ref:
            phase = {"type": "ref", "id": self.phase.id}
        else:
            phase = onedict_value(self.kernel_phase(ctx=ctx))

        return {
            f"medium_{self.id}": {
                "type": "homogeneous",
                "phase": phase,
                "sigma_t": self.eval_sigma_t(ctx.spectral_ctx).m_as(
                    uck.get("collision_coefficient")
                ),
                "albedo": self.eval_albedo(ctx.spectral_ctx).m_as(uck.get("albedo")),
            }
        }

    def kernel_shapes(self, ctx=None):
        from mitsuba.core import ScalarTransform4f

        if ctx.ref:
            medium = {"type": "ref", "id": f"medium_{self.id}"}
        else:
            medium = self.kernel_media(ctx=ctx)[f"medium_{self.id}"]

        k_length = uck.get("length")
        k_width = self.kernel_width(ctx=ctx).m_as(k_length)
        k_height = self.kernel_height(ctx=ctx).m_as(k_length)
        k_offset = self.kernel_offset(ctx=ctx).m_as(k_length)

        return {
            f"shape_{self.id}": {
                "type": "cube",
                "to_world": ScalarTransform4f(
                    [
                        [0.5 * k_width, 0.0, 0.0, 0.0],
                        [0.0, 0.5 * k_width, 0.0, 0.0],
                        [0.0, 0.0, 0.5 * k_height, 0.5 * k_height - k_offset],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
                "bsdf": {"type": "null"},
                "interior": medium,
            }
        }
