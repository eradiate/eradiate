import typing as t

import attrs

from ._core import Integrator
from ..core import KernelDict
from ...attrs import documented, parse_docs
from ...contexts import KernelDictContext


@parse_docs
@attrs.define
class MonteCarloIntegrator(Integrator):
    """
    Base class for integrator elements wrapping kernel classes
    deriving from
    :class:`mitsuba.MonteCarloIntegrator`.

    .. warning:: This class should not be instantiated.
    """

    max_depth: t.Optional[int] = documented(
        attrs.field(default=None, converter=attrs.converters.optional(int)),
        doc="Longest path depth in the generated measure data (where -1 "
        "corresponds to ∞). A value of 1 will display only visible emitters. 2 "
        "computes only direct illumination (no multiple scattering), etc. If "
        "unset, the kernel default value (-1) is used.",
        type="int, optional",
    )

    rr_depth: t.Optional[int] = documented(
        attrs.field(default=None, converter=attrs.converters.optional(int)),
        doc="Minimum path depth after which the implementation starts applying "
        "the Russian roulette path termination criterion. If unset, the kernel "
        "default value (5) is used.",
        type="int, optional",
    )

    hide_emitters: t.Optional[bool] = documented(
        attrs.field(default=None, converter=attrs.converters.optional(bool)),
        doc="Hide directly visible emitters. If unset, the kernel default "
        "value (``false``) is used.",
        type="bool, optional",
    )

    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        result = {self.id: {}}

        if self.max_depth is not None:
            result[self.id]["max_depth"] = self.max_depth
        if self.rr_depth is not None:
            result[self.id]["rr_depth"] = self.rr_depth
        if self.hide_emitters is not None:
            result[self.id]["hide_emitters"] = self.hide_emitters

        return KernelDict(result)


@parse_docs
@attrs.define
class PathIntegrator(MonteCarloIntegrator):
    """
    A thin interface to the `path tracer kernel plugin <https://eradiate-kernel.readthedocs.io/en/latest/generated/plugins.html#path-tracer-path>`_.

    This integrator samples paths using random walks starting from the sensor.
    It supports multiple scattering and does not account for volume
    interactions.
    """

    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        result = super(PathIntegrator, self).kernel_dict(ctx)
        result[self.id]["type"] = "path"
        return result


@parse_docs
@attrs.define
class VolPathIntegrator(MonteCarloIntegrator):
    """
    A thin interface to the volumetric path tracer kernel plugin.

    This integrator samples paths using random walks starting from the sensor.
    It supports multiple scattering and accounts for volume interactions.
    """

    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        result = super(VolPathIntegrator, self).kernel_dict(ctx)
        result[self.id]["type"] = "volpath"
        return result


@parse_docs
@attrs.define
class VolPathMISIntegrator(MonteCarloIntegrator):
    """
    A thin interface to the volumetric path tracer (with spectral multiple
    importance sampling) kernel plugin
    :cite:`Miller2019NullscatteringPathIntegral`.
    """

    use_spectral_mis = attrs.field(
        default=None, converter=attrs.converters.optional(bool)
    )

    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        result = super(VolPathMISIntegrator, self).kernel_dict(ctx)

        result[self.id]["type"] = "volpathmis"
        if self.use_spectral_mis is not None:
            result[self.id]["use_spectral_mis"] = self.use_spectral_mis

        return result
