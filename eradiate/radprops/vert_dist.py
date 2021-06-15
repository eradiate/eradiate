"""
Particle vertical distributions.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

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
class VerticalDistribution(ABC):
    r"""
    An abstract base class for particles vertical distributions.

    Vertical distributions help define particle layers.

    The particle layer is split into a number of divisions (sub-layers),
    wherein the particles fraction is evaluated.

    The vertical distribution is normalised so that:

    .. math::
        \sum_i f_i = 1

    where :math:`f_i` is the particles fraction in the layer division :math:`i`.
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

    def __attrs_post_init__(self):
        if self.bottom >= self.top:
            raise ValueError("bottom altitude must be lower than top altitude")

    @classmethod
    def from_dict(cls, d: dict) -> VerticalDistribution:
        """
        Initialise a :class:`VerticalDistribution` from a dictionary.
        """
        return cls(**d)

    @property
    @abstractmethod
    def fractions(self) -> Callable:
        """
        Returns a callable that evaluates the particles fractions in the
        layer, given an array of altitude values.
        """
        pass

    @staticmethod
    def _normalise(x: np.ndarray) -> np.ndarray:
        """
        Scale the values so that their sum is 1.

        Parameter ``x`` (array):
            Values to normalise.

        Returns → array:
            Normalised array.
        """
        _norm = np.sum(x)
        if _norm > 0.0:
            return x / _norm
        else:
            raise ValueError(f"Cannot normalise fractions because the norm is " f"0.")


class VerticalDistributionFactory(BaseFactory):
    """
    This factory constructs objects whose classes are derived from
    :class:`VerticalDistribution`.

    .. admonition:: Registered factory members
       :class: hint

       .. factorytable::
          :factory: VerticalDistributionFactory
    """

    _constructed_type = VerticalDistribution
    registry = {}


@VerticalDistributionFactory.register("uniform")
@parse_docs
@attr.s
class Uniform(VerticalDistribution):
    r"""
    Uniform vertical distribution.

    The uniform probability distribution function is:

    .. math::
        f(z) = \frac{1}{z_{\rm top} - z_{\rm bottom}}, \quad
        z \in [z_{\rm top}, z_{\rm bottom}]

    where :math:`z_{\rm top}` and :math:`z_{\rm bottom}` are the layer top and bottom
    altitudes, respectively.
    """

    @property
    def fractions(self) -> Callable:
        """Return a callable that evaluates the particle fractions."""

        def eval(z: pint.Quantity) -> np.ndarray:
            """
            Evaluate the particle fractions.

            Parameter ``z`` (:class:`pint.Quantity`):
                Altitude values.

            Return → array:
                Particles fractions.
            """
            if (self.bottom <= z).all() and (z <= self.top).all():
                x = z.magnitude
                return self._normalise(np.ones(len(x)))
            else:
                raise ValueError(
                    f"Altitude values do not lie between layer "
                    f"bottom ({self.bottom}) and top ({self.top}) "
                    f"altitudes. Got {z}."
                )

        return eval


@VerticalDistributionFactory.register("exponential")
@parse_docs
@attr.s
class Exponential(VerticalDistribution):
    r"""
    Exponential vertical distribution.

    The exponential probability distribution function is:

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

    @property
    def fractions(self) -> Callable:
        """Return a callable that evaluates the particle fractions."""

        def eval(z: pint.Quantity) -> np.ndarray:
            if (self.bottom <= z).all() and (z <= self.top).all():
                x = z.magnitude
                loc = self.bottom.to(z.units).magnitude
                scale = (1.0 / self.rate).to(z.units).magnitude
                f = expon.pdf(x=x, loc=loc, scale=scale)
                return self._normalise(f)
            else:
                raise ValueError(
                    f"Altitude values do not lie between layer "
                    f"bottom ({self.bottom}) and top ({self.top}) "
                    f"altitudes. Got {z}."
                )

        return eval


@VerticalDistributionFactory.register("gaussian")
@parse_docs
@attr.s
class Gaussian(VerticalDistribution):
    r"""
    Gaussian vertical distribution.

    The Gaussian probability distribution function is:

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

    @property
    def fractions(self) -> Callable:
        """Return a callable that evaluates the particle fractions."""

        def eval(z: pint.Quantity) -> np.ndarray:
            if (self.bottom <= z).all() and (z <= self.top).all():
                x = z.magnitude
                loc = self.mean.to(z.units).magnitude
                scale = self.std.to(z.units).magnitude
                f = norm.pdf(x=x, loc=loc, scale=scale)
                return self._normalise(f)
            else:
                raise ValueError(
                    f"Altitude values do not lie between layer "
                    f"bottom ({self.bottom}) and top ({self.top}) "
                    f"altitudes. Got {z}."
                )

        return eval


@VerticalDistributionFactory.register("array")
@parse_docs
@attr.s
class Array(VerticalDistribution):
    """
    Flexible vertical distribution specified either by an array of values,
    or by a :class:`~xarray.DataArray` object.
    """

    values = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(np.array),
            validator=attr.validators.optional(attr.validators.instance_of(np.ndarray)),
        ),
        doc="Particles fractions values on a regular altitude mesh starting "
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
        doc="Particles vertical distribution data array. Fraction as a function"
        " of altitude (``z``).",
        type=":class:`xarray.DataArray`",
        default="``None``",
    )

    @data_array.validator
    def _validate_data_array(self, attribute, value):
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
        if self.values is None and self.data_array is None:
            raise ValueError("You must specify 'values' or 'data_array'.")
        elif self.values is not None and self.data_array is not None:
            raise ValueError(
                "You cannot specify both 'values' and " "'data_array' simultaneously."
            )
        elif self.data_array is None:
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
        elif self.data_array is not None:
            min_z = to_quantity(self.data_array.z.min(keep_attrs=True))
            if min_z < self.bottom:
                raise ValueError(
                    f"Minimum altitude value in data_array "
                    f"({min_z}) is smaller than bottom altitude "
                    f"({self.bottom})."
                )

            max_z = to_quantity(self.data_array.z.max(keep_attrs=True))
            if max_z > self.top:
                raise ValueError(
                    f"Minimum altitude value in data_array "
                    f"({max_z}) is smaller than top altitude"
                    f"({self.top})."
                )

    @property
    def fractions(self) -> Callable:
        """Return a callable that evaluates the particle fractions."""

        def eval(z: pint.Quantity) -> np.ndarray:
            x = z.to(self.data_array.z.units).magnitude
            f = self.data_array.interp(
                coords={"z": x}, method=self.method, kwargs=dict(fill_value=0.0)
            )
            return self._normalise(f.values)

        return eval
