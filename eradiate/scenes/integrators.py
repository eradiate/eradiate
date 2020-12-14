"""Integration algorithm configuration.

.. admonition:: Registered factory members [:class:`IntegratorFactory`]
   :class: hint

   .. factorytable::
      :factory: IntegratorFactory
"""

import attr

from .core import SceneElement
from ..util.factory import BaseFactory


@attr.s
class Integrator(SceneElement):
    """Abstract base class for all integrator elements.

    See :class:`~eradiate.scenes.core.SceneElement` for undocumented members.
    """
    id = attr.ib(
        default="integrator",
        validator=attr.validators.optional(attr.validators.instance_of(str)),
    )


class IntegratorFactory(BaseFactory):
    """This factory constructs objects whose classes are derived from
    :class:`Integrator`.

    .. admonition:: Registered factory members
       :class: hint

       .. factorytable::
          :factory: IntegratorFactory
    """
    _constructed_type = Integrator
    registry = {}


_monte_carlo_integrator_params_docs = """
``max_depth`` (int):
    Longest path depth in the generated measure data (where -1 corresponds
    to ∞). A value of 1 will display only visible emitters. 2 will lead to
    direct illumination (no multiple scattering), etc. If set to ``None``,
    the kernel default value (-1) will be used. Default: ``None``.

``rr_depth`` (int):
    Minimum path depth after which the implementation will start to use the
    Russian roulette path termination criterion. If set to ``None``,
    the kernel default value (5) will be used. Default: ``None``.

``hide_emitters`` (bool):
    Hide directly visible emitters. If set to ``None``,
    the kernel default value (``false``) will be used. Default: ``None``.
"""


@attr.s
class MonteCarloIntegrator(Integrator):
    """Base class for integrator elements wrapping kernel classes
    deriving from
    :class:`eradiate.kernel.render.MonteCarloIntegrator <mitsuba.render.MonteCarloIntegrator>`.

    .. warning::
       This class should not be instantiated.
    """
    max_depth = attr.ib(
        default=None,
        converter=attr.converters.optional(int)
    )

    rr_depth = attr.ib(
        default=None,
        converter=attr.converters.optional(int)
    )

    hide_emitters = attr.ib(
        default=None,
        converter=attr.converters.optional(bool)
    )

    def kernel_dict(self, ref=True):
        result = {self.id: {}}

        if self.max_depth is not None:
            result[self.id]["max_depth"] = self.max_depth
        if self.rr_depth is not None:
            result[self.id]["rr_depth"] = self.rr_depth
        if self.hide_emitters is not None:
            result[self.id]["hide_emitters"] = self.hide_emitters

        return result


# Add doctrings
MonteCarloIntegrator.__doc__ += f"""
.. rubric:: Constructor arguments / instance attributes

{_monte_carlo_integrator_params_docs}
"""


@IntegratorFactory.register("path")
@attr.s
class PathIntegrator(MonteCarloIntegrator):
    """A thin interface to the `path tracer kernel plugin <https://eradiate-kernel.readthedocs.io/en/latest/generated/plugins.html#path-tracer-path>`_.

    This integrator samples paths using random walks starting from the sensor.
    It supports multiple scattering and does not account for volume
    interactions.
    """

    def kernel_dict(self, ref=True):
        result = super(PathIntegrator, self).kernel_dict(ref)
        result[self.id]["type"] = "path"
        return result


PathIntegrator.__doc__ += f"""
.. rubric:: Constructor arguments / instance attributes

{_monte_carlo_integrator_params_docs}
"""


@IntegratorFactory.register("volpath")
@attr.s
class VolPathIntegrator(MonteCarloIntegrator):
    """A thin interface to the volumetric path tracer kernel plugin.

    This integrator samples paths using random walks starting from the sensor.
    It supports multiple scattering and accounts for volume interactions.
    """

    def kernel_dict(self, ref=True):
        result = super(VolPathIntegrator, self).kernel_dict(ref)
        result[self.id]["type"] = "volpath"
        return result


VolPathIntegrator.__doc__ += f"""
.. rubric:: Constructor arguments / instance attributes

{_monte_carlo_integrator_params_docs}
"""


@IntegratorFactory.register("volpathmis")
@attr.s
class VolPathMISIntegrator(MonteCarloIntegrator):
    """A thin interface to the volumetric path tracer (with spectral multiple
    importance sampling) kernel plugin
    :cite:`Miller2019NullscatteringPathIntegral`.
    """

    use_spectral_mis = attr.ib(
        default=None,
        converter=attr.converters.optional(bool)
    )

    def kernel_dict(self, ref=True):
        result = super(VolPathMISIntegrator, self).kernel_dict(ref)

        result[self.id]["type"] = "volpathmis"
        if self.use_spectral_mis is not None:
            result[self.id]["use_spectral_mis"] = self.use_spectral_mis

        return result


VolPathMISIntegrator.__doc__ += f"""
.. rubric:: Constructor arguments / instance attributes

{_monte_carlo_integrator_params_docs}

Parameter ``hide_emitters`` (bool):
    Hide directly visible emitters. If set to ``None``,
    the kernel default value (``false``) will be used. Default: ``None``.
"""
