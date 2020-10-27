"""Radiative property profile definitions."""

from abc import ABC, abstractmethod

import attr

from ....util.attrs import attrib_quantity, unit_enabled
from ....util.factory import BaseFactory
from ....util.units import config_default_units as cdu
from ....util.units import ureg


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

    @property
    @abstractmethod
    def albedo(self):
        """Return albedo.

        Returns → :class:`pint.Quantity`:
            Profile albedo.
        """
        pass

    @property
    @abstractmethod
    def sigma_t(self):
        """Return extinction coefficient.

        Returns → :class:`pint.Quantity`:
            Profile extinction coefficient.
        """
        pass

    @property
    @abstractmethod
    def sigma_a(self):
        """Return absorption coefficient.

        Returns → :class:`pint.Quantity`:
            Profile absorption coefficient.
        """
        pass

    @property
    @abstractmethod
    def sigma_s(self):
        """Return scattering coefficient.

        Returns → :class:`pint.Quantity`:
            Profile scattering coefficient.
        """
        pass


class RadProfileFactory(BaseFactory):
    """This factory constructs objects whose classes are derived from
    :class:`RadProfile`.

    .. admonition:: Registered factory members
       :class: hint

       .. factorytable::
          :factory: RadProfileFactory
    """
    # TODO: add to docs a table with factory key-class associations
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

    albedo_values = attrib_quantity(
        default=None,
        units_compatible=ureg.dimensionless,
    )

    sigma_t_values = attrib_quantity(
        default=None,
        units_compatible=cdu.generator("collision_coefficient"),
    )

    @albedo_values.validator
    @sigma_t_values.validator
    def _validator_values(instance, attribute, value):
        if instance.albedo_values.shape != instance.sigma_t_values.shape:
            raise ValueError(f"while setting {attribute.name}: "
                             f"'albedo_values' and 'sigma_t_values' must have "
                             f"the same length")

    @property
    def albedo(self):
        return self.albedo_values

    @property
    def sigma_t(self):
        return self.sigma_t_values

    @property
    def sigma_a(self):
        return self.sigma_t * (1. - self.albedo)

    @property
    def sigma_s(self):
        return self.sigma_t * self.albedo
