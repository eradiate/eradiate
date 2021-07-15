"""
Particle number fraction vertical distributions.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import attr
import numpy as np
import pint
import pinttr
import xarray as xr
from pinttr.util import units_compatible
from scipy.stats import expon, norm

from .._factory import Factory
from ..attrs import documented, parse_docs
from ..units import unit_context_config as ucc
from ..units import unit_registry as ureg
from ..validators import all_positive

particle_distribution_factory = Factory()


@parse_docs
@attr.s
class ParticleDistribution(ABC):
    r"""
    An abstract base class for particle distributions.

    Particle distributions help define particle layers.

    Particle distributions define how particle number fraction vary with
    altitude.
    The particle layer is split into a number of divisions (sub-layers),
    wherein the particle number fraction is evaluated.

    The particle number fraction vertical distribution is normalised so that:

    .. math::
        \sum_i f_i = 1

    where :math:`f_i` is the particle number fraction in the layer division
    :math:`i`.
    """

    @abstractmethod
    def eval_fraction(self, z: pint.Quantity) -> np.ndarray:
        """
        Evaluate the particle number fraction as a function of altitude.

        Parameter ``z`` (:class:`pint.Quantity`):
            Altitude values.

        Return → :class:`~numpy.ndarray`:
            Particle number fraction.
        """
        pass


@particle_distribution_factory.register(type_id="uniform")
@parse_docs
@attr.s
class UniformParticleDistribution(ParticleDistribution):
    r"""
    Uniform particle distribution.

    Particle number fraction values are computed using the uniform probability
    distribution function:

    .. math::
        f(z) = \frac{1}{z_{\rm top} - z_{\rm bottom}}, \quad
        z \in [z_{\rm top}, z_{\rm bottom}]

    where :math:`z_{\rm top}` and :math:`z_{\rm bottom}` are the layer top and bottom
    altitudes, respectively.
    """

    def eval_fraction(self, z: pint.Quantity) -> np.ndarray:
        f = np.ones(len(z))
        return f / f.sum()


@particle_distribution_factory.register(type_id="exponential")
@parse_docs
@attr.s
class ExponentialParticleDistribution(ParticleDistribution):
    r"""
    Exponential particle distribution.

    Particle number fraction values are computed using the exponential
    probability distribution function:

    .. math::
        f(z) = \lambda  \exp \left( -\lambda z \right)

    where :math:`\lambda` is the rate parameter and :math:`z` is the altitude.
    """
    rate = documented(
        pinttr.ib(
            units=ucc.deferred("collision_coefficient"),
            converter=pinttr.converters.to_units(ucc.deferred("collision_coefficient")),
            validator=pinttr.validators.has_compatible_units,
        ),
        doc="Rate parameter of the exponential distribution.\n"
        "\n"
        "Unit-enabled field (default: ucc[length]).",
        type="float",
    )

    def eval_fraction(self, z: pint.Quantity) -> np.ndarray:
        x = z.magnitude
        loc = z.magnitude.min()
        scale = (1.0 / self.rate).to(z.units).magnitude
        f = expon.pdf(x=x, loc=loc, scale=scale)
        return f / f.sum()


@particle_distribution_factory.register(type_id="gaussian")
@parse_docs
@attr.s
class GaussianParticleDistribution(ParticleDistribution):
    r"""
    Gaussian particle distribution.

    Particle number fraction values are computed using the Gaussian probability
    distribution function:

    .. math::
        f(z) = \frac{1}{2 \pi \sigma}
        \exp{\left[
            -\frac{1}{2}
            \left( \frac{z - \mu}{\sigma} \right)^2
        \right]}

    where :math:`\mu` is the mean of the distribution and :math:`\sigma` is
    the standard deviation of the distribution.
    """
    mean = documented(
        pinttr.ib(
            units=ucc.deferred("length"),
            converter=pinttr.converters.to_units(ucc.deferred("length")),
            validator=pinttr.validators.has_compatible_units,
        ),
        doc="Mean (expectation) of the distribution. "
        "If ``None``, set to the middle of the layer.\n"
        "\n"
        "Unit-enabled field (default: ucc[length]).",
        type="float",
    )
    std = documented(
        pinttr.ib(
            units=ucc.deferred("length"),
            converter=pinttr.converters.to_units(ucc.deferred("length")),
            validator=pinttr.validators.has_compatible_units,
        ),
        doc="Standard deviation of the distribution. If ``None``, set to one "
        "sixth of the layer thickness so that half the layer thickness "
        "equals three standard deviations.\n"
        "\n"
        "Unit-enabled field (default: ucc[length]).",
        type="float",
    )

    def eval_fraction(self, z: pint.Quantity) -> np.ndarray:
        x = z.magnitude
        loc = self.mean.to(z.units).magnitude
        scale = self.std.to(z.units).magnitude
        f = norm.pdf(x=x, loc=loc, scale=scale)
        return f / f.sum()


@particle_distribution_factory.register(type_id="array")
@parse_docs
@attr.s
class ArrayParticleDistribution(ParticleDistribution):
    """
    Flexible particle distribution specified either by a particle number
    fraction array or :class:`~xarray.DataArray`.
    """

    values = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(np.array),
            validator=attr.validators.optional(
                [attr.validators.instance_of(np.ndarray), all_positive]
            ),
        ),
        doc="Particle number fraction values on a regular altitude mesh.",
        type="array",
        default="``None``",
    )
    data_array = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(xr.DataArray),
            validator=attr.validators.optional(
                attr.validators.instance_of(xr.DataArray)
            ),
        ),
        doc="Particle distribution data array. Number fraction as a "
        "function of altitude (``z``).\n"
        "Note that number fraction do not need to be normalised.",
        type=":class:`~xarray.DataArray`",
        default="``None``",
    )

    @data_array.validator
    def _validate_data_array(instance, attribute, value):
        if value is not None and not np.all(value.data >= 0.0):
            raise ValueError("'data_array' data must be all positive.")

    @values.validator
    @data_array.validator
    def _validate_values_and_data_array(instance, attribute, value):
        if instance.values is None and instance.data_array is None:
            raise ValueError("You must specify 'values' or 'data_array'.")
        elif instance.values is not None and instance.data_array is not None:
            raise ValueError(
                "You cannot specify both 'values' and 'data_array' simultaneously."
            )

    @data_array.validator
    def _validate_data_array_and_values(instance, attribute, value):
        if value is not None:
            if not "z" in value.coords:
                raise ValueError("Attribute 'data_array' must have a 'z' coordinate")
            else:
                try:
                    units = ureg.Unit(value.z.units)
                    if not units_compatible(units, ureg.Unit("m")):
                        raise ValueError(
                            f"Coordinate 'z' of attribute "
                            f"'data_array' must have units"
                            f"compatible with m^-1 (got {units})."
                        )
                except AttributeError:
                    raise ValueError(
                        "Coordinate 'z' of attribute 'data_array' must have units."
                    )

    method = documented(
        attr.ib(
            default="linear", converter=str, validator=attr.validators.instance_of(str)
        ),
        doc="Method to interpolate the data along the altitude. \n"
        "This parameter is passed to :meth:`xarray.DataArray.interp`.",
        type="str",
        default='``"linear"``',
    )

    def eval_fraction(self, z: pint.Quantity) -> np.ndarray:
        if self.values is not None:
            data_array = xr.DataArray(
                data=self.values,
                coords={
                    "z": (
                        "z",
                        np.linspace(
                            start=z.to("m").magnitude.min(),
                            stop=z.to("m").magnitude.max(),
                            num=len(self.values),
                        ),
                        {"units": "m"},
                    )
                },
                dims=["z"],
            )
        else:
            data_array = self.data_array

        x = z.to(data_array.z.units).magnitude
        f = data_array.interp(
            coords={"z": x}, method=self.method, kwargs=dict(fill_value=0.0)
        )
        return f.values / f.values.sum()  # Normalise fractions
