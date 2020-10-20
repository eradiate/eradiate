"""Radiative property profile definitions."""

from abc import ABC, abstractmethod

import attr
import numpy as np

from ....util.attrs import attrib, attrib_units, unit_enabled
from ....util.factory import BaseFactory
from ....util.units import config_default_units as cdu, ureg


@unit_enabled
@attr.s
class RadProfile(ABC):
    """An abstract base class for radiative property profiles. Classes deriving
    from this one must implement methods which return the albedo and collision
    coefficients as Pint-wrapped Numpy arrays.

    .. seealso:: :class:`.RadProfileFactory`

    """

    @classmethod
    def from_dict(cls, d):
        """Initialise a :class:`RadPropsProfile` from a dictionary."""
        return cls(**d)

    @abstractmethod
    def albedo(self):
        """Return albedo.

        Returns → :class:`pint.Quantity`:
            Profile albedo.
        """
        pass

    @abstractmethod
    def sigma_t(self):
        """Return extinction coefficient.

        Returns → :class:`pint.Quantity`:
            Profile extinction coefficient.
        """
        pass

    @abstractmethod
    def sigma_a(self):
        """Return absorption coefficient.

        Returns → :class:`pint.Quantity`:
            Profile absorption coefficient.
        """
        pass

    @abstractmethod
    def sigma_s(self):
        """Return scattering coefficient.

        Returns → :class:`pint.Quantity`:
            Profile scattering coefficient.
        """
        pass

    def __attrs_post_init__(self):
        self._strip_units()


class RadProfileFactory(BaseFactory):
    """This factory constructs objects whose classes are derived from
    :class:`RadProfile`.
    """
    # TODO: add a table with factory key-class associations
    _constructed_type = RadProfile
    registry = {}


@RadProfileFactory.register(name="array")
@attr.s
class ArrayRadProfile(RadProfile):
    """A flexible radiative property profile whose albedo and extinction
    coefficient are specified as numpy arrays. Both constructor arguments must
    have the same shape.

    .. rubric:: Constructor arguments / instance attributes

    ``albedo_values`` (array):
        An array specifying albedo values.
        **Required, no default**.

        Unit-enabled field (dimensionless).

    ``sigma_t_values`` (array):
        An array specifying extinction coefficient values.
        **Required, no default**.

        Unit-enabled field (default: cdu[length]^-1).
    """
    albedo_values, albedo_values_units = attrib(
        default=None,
        converter=np.array,
        units_compatible=ureg.dimensionless,
        units_default=cdu.get("dimensionless")
    )

    sigma_t_values, sigma_t_values_units = attrib(
        default=None,
        converter=np.array,
        units_compatible=ureg.m ** -1,
        units_default=cdu.get("collision_coefficient")
    )

    def __attrs_post_init__(self):
        super(ArrayRadProfile, self).__attrs_post_init__()
        if self.albedo_values.shape != self.sigma_t_values.shape:
            raise ValueError("'albedo_values' and 'sigma_t_values' must have "
                             "the same length")

    def albedo(self):
        return ureg.Quantity(self.albedo_values, self.albedo_values_units)

    def sigma_t(self):
        return ureg.Quantity(self.sigma_t_values, self.sigma_t_values_units)

    def sigma_a(self):
        return self.sigma_t() * (1. - self.albedo())

    def sigma_s(self):
        return self.sigma_t() * self.albedo()
