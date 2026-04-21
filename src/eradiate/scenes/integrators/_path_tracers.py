from __future__ import annotations

import attrs

import eradiate

from ._core import Integrator
from ...attrs import define, documented


@define(eq=False, slots=False)
class MonteCarloIntegrator(Integrator):
    """
    Base class for integrator elements wrapping kernel classes
    deriving from
    :class:`mitsuba.MonteCarloIntegrator`.

    .. warning:: This class should not be instantiated.
    """

    max_depth: int | None = documented(
        attrs.field(default=None, converter=attrs.converters.optional(int)),
        doc="Longest path depth in the generated measure data (where -1 "
        "corresponds to ∞). A value of 1 will display only visible emitters. 2 "
        "computes only direct illumination (no multiple scattering), etc. If "
        "unset, the kernel default value (-1) is used.",
        type="int or None",
        init_type="int, optional",
    )

    rr_depth: int = documented(
        attrs.field(default=5, converter=int),
        doc="Minimum path depth after which the implementation starts applying "
        "the Russian roulette path termination criterion.",
        type="int",
    )

    hide_emitters: bool | None = documented(
        attrs.field(default=None, converter=attrs.converters.optional(bool)),
        doc="Hide directly visible emitters. If unset, the kernel default "
        "value (``false``) is used.",
        type="bool or None",
        init_type="bool, optional",
    )

    @property
    def kernel_type(self) -> str:
        raise NotImplementedError

    def _build_kernel_dict(self) -> dict:
        """
        Build the kernel-specific dictionary.

        Override this method in subclasses to add integrator-specific parameters.
        The base implementation handles common Monte Carlo integrator parameters.
        """
        result = {
            "type": self.kernel_type,
            "rr_depth": self.rr_depth,
        }

        if self.timeout is not None:
            result["timeout"] = self.timeout
        if self.max_depth is not None:
            result["max_depth"] = self.max_depth
        if self.hide_emitters is not None:
            result["hide_emitters"] = self.hide_emitters

        return result

    @property
    def template(self) -> dict:
        # Validation
        if self.stokes and not eradiate.mode().is_polarized:
            raise RuntimeError("stokes should only be set to True in polarized mode.")

        # Build the kernel dict (children can override _build_kernel_dict)
        result = self._build_kernel_dict()

        # Apply wrapping layers
        if self.moment:
            result = {"type": "moment", "nested": result}

        # Important: the 'stokes' integrator has to come last because it needs
        # sensor information
        if self.stokes:
            result = {
                "type": "stokes",
                "integrator": result,
                "meridian_align": self.meridian_align,
            }

        return result


@define(eq=False, slots=False)
class PathIntegrator(MonteCarloIntegrator):
    """
    A thin interface to the path tracer kernel plugin [``path``].

    This integrator samples paths using random walks starting from the sensor.
    It supports multiple scattering and does not account for volume
    interactions.
    """

    @property
    def kernel_type(self) -> str:
        return "path"


@define(eq=False, slots=False)
class VolPathIntegrator(MonteCarloIntegrator):
    """
    A thin interface to the volumetric path tracer kernel plugin [``volpath``].

    This integrator samples paths using random walks starting from the sensor.
    It supports multiple scattering and accounts for volume interactions.
    """

    @property
    def kernel_type(self) -> str:
        return "volpath"


@define(eq=False, slots=False)
class EOVolPathIntegrator(MonteCarloIntegrator):
    """
    A thin interface to the EO volumetric path tracer kernel plugin [``eovolpath``].

    This integrator samples paths using random walks starting from the sensor.
    It supports multiple scattering, accounts for volume interactions, and
    implements the DDIS variance reduction method as described by
    :cite:t:`Buras2011EfficientUnbiasedVariance`.
    """

    rr_depth: int = documented(
        attrs.field(
            default=1000,
            converter=int,
            validator=attrs.validators.instance_of(int),
        ),
        doc="Minimum path depth after which the implementation starts applying "
        "the Russian roulette path termination criterion.",
        type="int",
    )

    rr_factor = documented(
        attrs.field(
            default=0.97,
            converter=float,
            validator=attrs.validators.instance_of(float),
        ),
        doc="Specifies the maximum probability to keep a path when Russian "
        "Roulette is evaluated.",
        type="float",
    )

    ddis_threshold = documented(
        attrs.field(
            default=0.1,
            converter=attrs.converters.optional(float),
            validator=attrs.validators.instance_of(float),
        ),
        doc="Specifies the probability to importance sample the phase using the "
        "emitter as incident direction. Set to <0. to deactivate.",
        type="float",
    )

    @property
    def kernel_type(self) -> str:
        return "eovolpath"

    def _build_kernel_dict(self) -> dict:
        result = super()._build_kernel_dict()
        result["ddis_threshold"] = self.ddis_threshold
        result["rr_factor"] = self.rr_factor
        return result


@define(eq=False, slots=False)
class VolPathMISIntegrator(MonteCarloIntegrator):
    """
    A thin interface to the volumetric path tracer kernel plugin [``volpathmis``].

    This plugin implements spectral multiple importance sampling
    :cite:`Miller2019NullscatteringPathIntegral`.
    """

    use_spectral_mis = attrs.field(
        default=None, converter=attrs.converters.optional(bool)
    )

    @property
    def kernel_type(self) -> str:
        return "volpathmis"

    def _build_kernel_dict(self) -> dict:
        result = super()._build_kernel_dict()
        if self.use_spectral_mis is not None:
            result["use_spectral_mis"] = self.use_spectral_mis
        return result


@define(eq=False, slots=False)
class PiecewiseVolPathIntegrator(MonteCarloIntegrator):
    """
    A thin interface to the piecewise volumetric path tracer kernel plugin [``piecewise_volpath``].

    This integrator samples paths using random walks starting from the sensor.
    It supports multiple scattering and accounts for 1D volume interactions.
    """

    @property
    def kernel_type(self) -> str:
        return "piecewise_volpath"
