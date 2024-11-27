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

    rr_depth: int | None = documented(
        attrs.field(default=None, converter=attrs.converters.optional(int)),
        doc="Minimum path depth after which the implementation starts applying "
        "the Russian roulette path termination criterion. If unset, the kernel "
        "default value (5) is used.",
        type="int or None",
        init_type="int, optional",
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

    @property
    def template(self) -> dict:
        result = {"type": self.kernel_type}

        if self.timeout is not None:
            result["timeout"] = self.timeout
        if self.max_depth is not None:
            result["max_depth"] = self.max_depth
        if self.rr_depth is not None:
            result["rr_depth"] = self.rr_depth
        if self.hide_emitters is not None:
            result["hide_emitters"] = self.hide_emitters

        if self.stokes and not eradiate.mode().is_polarized:
            raise RuntimeError("stokes should only be set to True in polarized mode.")

        if self.stokes:
            result = {
                "type": "stokes",
                "integrator": result,
                "meridian_align": self.meridian_align,
            }

        if self.moment:
            result = {"type": "moment", "nested": result}

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

    @property
    def template(self) -> dict:
        result = super().template
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
