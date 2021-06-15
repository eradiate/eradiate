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

from .._attrs import documented, parse_docs
from .._factory import BaseFactory
from ..units import to_quantity
from ..units import unit_context_config as ucc
from ..units import unit_registry as ureg


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
    bottom = documented(
        pinttr.ib(
            units=ucc.deferred("length"),
            converter=pinttr.converters.to_units(ucc.deferred("length")),
            validator=pinttr.validators.has_compatible_units,
        ),
        doc="Layer bottom altitude.\n\nUnit-enabled field (default: ucc[length]).",
        type="float",
    )
    top = documented(
        pinttr.ib(
            units=ucc.deferred("length"),
            converter=pinttr.converters.to_units(ucc.deferred("length")),
            validator=pinttr.validators.has_compatible_units,
        ),
        doc="Layer top altitude.\n\nUnit-enabled field (default: ucc[length]).",
        type="float",
    )

    @bottom.validator
    @top.validator
    def _bottom_and_top_validator(instance, attribute, value):
        if instance.bottom >= instance.top:
            raise ValueError("bottom altitude must be lower than top altitude")

    @classmethod
    def from_dict(cls, d: dict) -> ParticleDistribution:
        """
        Initialise a :class:`ParticleDistribution` from a dictionary.
        """
        return cls(**d)

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


class ParticleDistributionFactory(BaseFactory):
    """
    This factory constructs objects whose classes are derived from
    :class:`ParticleDistribution`.

    .. admonition:: Registered factory members
       :class: hint

       .. factorytable::
          :factory: ParticleDistributionFactory
    """

    _constructed_type = ParticleDistribution
    registry = {}


@ParticleDistributionFactory.register("uniform")
@parse_docs
@attr.s
class Uniform(ParticleDistribution):
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
        if np.any(z < self.bottom) or np.any(z > self.top):
            raise ValueError(
                f"Altitude values do not lie between layer "
                f"bottom ({self.bottom}) and top ({self.top}) "
                f"altitudes. Got {z}."
            )
        else:
            x = z.magnitude
            f = np.ones(len(x))
            return f / f.sum()


@ParticleDistributionFactory.register("exponential")
@parse_docs
@attr.s
class Exponential(ParticleDistribution):
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
            default=None,
            converter=attr.converters.optional(
                pinttr.converters.to_units(ucc.deferred("collision_coefficient"))
            ),
            validator=attr.validators.optional(pinttr.validators.has_compatible_units),
        ),
        doc="Rate parameter of the exponential distribution. If ``None``, "
        "set to the inverse of the layer thickness.\n"
        "\n"
        "Unit-enabled field (default: ucc[length]).",
        type="float",
        default="``None``",
    )

    def __attrs_post_init__(self):
        if self.rate is None:
            self.rate = 1.0 / (self.top - self.bottom)

    def eval_fraction(self, z: pint.Quantity) -> np.ndarray:
        if (self.bottom <= z).all() and (z <= self.top).all():
            x = z.magnitude
            loc = self.bottom.to(z.units).magnitude
            scale = (1.0 / self.rate).to(z.units).magnitude
            f = expon.pdf(x=x, loc=loc, scale=scale)
            return f / f.sum()
        else:
            raise ValueError(
                f"Altitude values do not lie between layer "
                f"bottom ({self.bottom}) and top ({self.top}) "
                f"altitudes. Got {z}."
            )


@ParticleDistributionFactory.register("gaussian")
@parse_docs
@attr.s
class Gaussian(ParticleDistribution):
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
            default=None,
            converter=attr.converters.optional(
                pinttr.converters.to_units(ucc.deferred("length"))
            ),
            validator=attr.validators.optional(pinttr.validators.has_compatible_units),
        ),
        doc="Mean (expectation) of the distribution. "
        "If ``None``, set to the middle of the layer.\n"
        "\n"
        "Unit-enabled field (default: ucc[length]).",
        type="float",
        default="``None``",
    )
    std = documented(
        pinttr.ib(
            units=ucc.deferred("length"),
            default=None,
            converter=attr.converters.optional(
                pinttr.converters.to_units(ucc.deferred("length"))
            ),
            validator=attr.validators.optional(pinttr.validators.has_compatible_units),
        ),
        doc="Standard deviation of the distribution. If ``None``, set to one "
        "sixth of the layer thickness so that half the layer thickness "
        "equals three standard deviations.\n"
        "\n"
        "Unit-enabled field (default: ucc[length]).",
        type="float",
        default="``None``",
    )

    def __attrs_post_init__(self):
        if self.mean is None:
            self.mean = (self.bottom + self.top) / 2.0
        if self.std is None:
            self.std = (self.top - self.bottom) / 6.0

    def eval_fraction(self, z: pint.Quantity) -> np.ndarray:
        if (self.bottom <= z).all() and (z <= self.top).all():
            x = z.magnitude
            loc = self.mean.to(z.units).magnitude
            scale = self.std.to(z.units).magnitude
            f = norm.pdf(x=x, loc=loc, scale=scale)
            return f / f.sum()
        else:
            raise ValueError(
                f"Altitude values do not lie between layer "
                f"bottom ({self.bottom}) and top ({self.top}) "
                f"altitudes. Got {z}."
            )


@ParticleDistributionFactory.register("array")
@parse_docs
@attr.s
class Array(ParticleDistribution):
    """
    Flexible particle distribution specified either by a particle number
    fraction array or :class:`~xarray.DataArray`.
    """

    values = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(np.array),
            validator=attr.validators.optional(attr.validators.instance_of(np.ndarray)),
        ),
        doc="Particle number fraction values on a regular altitude mesh starting "
        "from the layer bottom and stopping at the layer top altitude.",
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
        doc="Particle vertical distribution data array. Number fraction as a "
        "function of altitude (``z``).",
        type=":class:`xarray.DataArray`",
        default="``None``",
    )

    @values.validator
    @data_array.validator
    def _validate_values_and_data_array(instance, attribute, value):
        if instance.values is None and instance.data_array is None:
            raise ValueError("You must specify 'values' or 'data_array'.")
        elif instance.values is not None and instance.data_array is not None:
            raise ValueError(
                "You cannot specify both 'values' and " "'data_array' simultaneously."
            )

    @data_array.validator
    def _validate_data_array(instance, attribute, value):
        if value is not None:
            if not "z" in value.coords:
                raise ValueError("Attribute 'data_array' must have a 'z' " "coordinate")
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
                        "Coordinate 'z' of attribute 'data_array' " "must have units."
                    )
            min_z = to_quantity(instance.data_array.z.min(keep_attrs=True))
            if min_z < instance.bottom:
                raise ValueError(
                    f"Minimum altitude value in data_array "
                    f"({min_z}) is smaller than bottom altitude "
                    f"({instance.bottom})."
                )

            max_z = to_quantity(instance.data_array.z.max(keep_attrs=True))
            if max_z > instance.top:
                raise ValueError(
                    f"Minimum altitude value in data_array "
                    f"({max_z}) is smaller than top altitude"
                    f"({instance.top})."
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

    def __attrs_post_init__(self):
        self.update()

    def update(self):
        if self.values is not None and self.data_array is None:
            self.data_array = xr.DataArray(
                data=self.values,
                coords={
                    "z": (
                        "z",
                        np.linspace(
                            start=self.bottom.to("m").magnitude,
                            stop=self.top.to("m").magnitude,
                            num=len(self.values),
                        ),
                        {"units": "m"},
                    )
                },
                dims=["z"],
            )

    def eval_fraction(self, z: pint.Quantity) -> np.ndarray:
        x = z.to(self.data_array.z.units).magnitude
        f = self.data_array.interp(
            coords={"z": x}, method=self.method, kwargs=dict(fill_value=0.0)
        )
        return f.values / f.values.sum()
