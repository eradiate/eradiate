"""Radiative property profile definitions."""

from abc import ABC, abstractmethod

import attr
import numpy as np

from . import sigma_s_air
from .... import data
from ....util.attrs import attrib_quantity, unit_enabled, validator_is_positive
from ....util.factory import BaseFactory
from ....util.units import config_default_units as cdu
from ....util.units import ureg


@unit_enabled
@attr.s
class RadProfile(ABC):
    """An abstract base class for radiative property profiles. Classes deriving
    from this one must implement methods which return the albedo and collision
    coefficients as Pint-wrapped Numpy arrays.

    .. seealso::

       :class:`.RadProfileFactory`

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


@RadProfileFactory.register(name="us76_approx")
@unit_enabled
@attr.s
class US76ApproxRadProfile(RadProfile):
    """A US76-approximation radiative profile.

    .. note::

       Instantiating this class requires to download the absorption datasets
       for the ``us76_u86_4`` gas mixture and place them in
       ``$ERADIATE_DIR/resources/data/spectra/absorption/us76_u86_4/fullrange/``
       (create the directory if necessary).

    The radiative properties are computed based upon the so-called US76
    atmospheric vertical profile.
    The scattering coefficient is computed with
    :func:`sigma_s_air<eradiate.scenes.atmosphere.radiative_properties.rayleigh.sigma_s_air>`
    using the total number density from the US76 atmospheric vertical
    profile.
    The absorption coefficient is computed in two steps. First, the
    absorption cross section is computed by interpolating the
    absorption cross section datasets for the ``us76_u86_4`` mixture at the
    wavelength specified in the eradiate mode and at the pressure values
    corresponding to the US76 atmospheric vertical profile. The second step
    consists in multiplying these cross sections by the total number density
    values from the US76 atmospheric vertical profile, in the corresponding
    atmospheric layers.

    .. warning::

       This ``us76_u86_4`` gas mixture does not completely match the gas
       mixture defined in the U.S. Standard Atmosphere 1976 model (see
       :cite:`NASA1976USStandardAtmosphere`). The U.S. Standard Atmosphere 1976
       model (abbreviated *US76 model* in the following) divides the atmosphere
       into two altitude regions: below the altitude of 86 kilometers, the
       model assumes air is a well-mixed dry gas with the following constituent
       species: N2, O2, Ar, CO2, Ne, He, Kr, Xe, CH4, H2. **Note that H2O is
       absent from this list of gas species.** Above 86 kilometers
       of altitude (altitude at which the air number density is smaller than
       the air number density at the surface by a factor larger than 1e5)
       the remaining gas species are N2, O2, Ar, He, O and H and are not well-
       mixed anymore.

       The ``us76_u86_4`` mixture is identical to the gas
       mixture in the US76 model under 86 kilometers of altitude and as far as
       N2, O2, CO2 and CH4 are concerned. The reason for which the other gas
       species from the US76 model under 86 kilometers (Ar, Ne, He, Kr, Xe, H2)
       are not included in this mixture is because they are absent from the
       `HITRAN <https://hitran.org/>`_ spectroscopic database, which means
       that the absorption cross sections cannot be computed for this species.
       The reason for which, we restrict to the first altitude region of the
       US76 model in the definition of the ``us76_u86_4`` mixture is that it
       is simpler to produce an absorption dataset for a well-mixed gas than
       for a non well-mixed gas.

       The approximation of using a gas mixture defined based on the first
       altitude region of the US76 model may be justified by the fact that
       the air number density is at least 1e5 smaller above 86 kilometers
       than the air number density at the surface. Unless the volume fractions
       of the important absorbing species are significantly different, this
       means that the absorption coefficient will also be at least 1e5 smaller
       above 86 kilometers of altitude.
       The approximation of restricting to only 4 gas species of the 10 species
       present in the US76 model for altitudes below 86 km, although forced,
       has consequences that are unknown to us. In other words, we ignore the
       conditions of validity of this approximation. Simply put, it was the
       best approximation we could find up until now.

       All that justifies the naming of the present class. We acknowledge
       the limitations of this *US76-approximation radiative profile* and
       are working on a more flexible solution. In the meantime, we wanted
       to be as transparent (unintentional wordplay) as possible regarding the
       data and methods that are used here.

    .. rubric:: Constructor arguments / instance attributes

    ``n_layers`` (int):
        Number of atmospheric layers.

        Default: 50

    ``height`` (float):
        Atmosphere's height.

        Default: 100 km

        Unit-enabled field (default: cdu[length]).

    ``dataset`` (str):
        Dataset identifier.

        Default: ``"us76_u86_4-fullrange"``

        .. warning::

            This attribute serves as debugging tool. Do not modify it unless
            you know what you are doing.
    """
    # TODO: refactor attributes declaration
    n_layers = attr.ib(
        default=50,
        converter=int,
        validator=validator_is_positive
    )

    height = attrib_quantity(
        default=ureg.Quantity(100., ureg.km),
        validator=validator_is_positive,
        units_compatible=cdu.generator("length"),
        units_add_converter=True,
    )

    _sigma_s_values = attrib_quantity(
        default=None,
        units_compatible=cdu.generator("collision_coefficient"),
        init=False,
        repr=False,
    )

    _sigma_a_values = attrib_quantity(
        default=None,
        units_compatible=cdu.generator("collision_coefficient"),
        init=False,
        repr=False,
    )

    dataset = attr.ib(
        default="us76_u86_4-fullrange",
        validator=attr.validators.in_({"us76_u86_4-fullrange", "test"}),
    )

    def __attrs_post_init__(self):
        self.update()

    def update(self):
        """Update internal variables. An update is required to recompute the
        internal state when any of the ``n_layers`` or ``height`` attributes
        is modified.
        """
        from eradiate import mode
        from ..thermophysics import us76

        # Compute total number density and pressure values
        altitude_mesh = np.linspace(
            start=0.,
            stop=self.height,
            num=self.n_layers + 1
        )
        profile = us76.make_profile(altitude_mesh)
        n_tot = ureg.Quantity(profile.n_tot.values, profile.n_tot.units)
        p = ureg.Quantity(profile.p.values, profile.p.units)

        wavelength = mode.wavelength

        # Compute scattering coefficient
        self._sigma_s_values = sigma_s_air(
            wavelength=wavelength,
            number_density=n_tot,
        )

        # Compute absorption coefficient
        ds = data.open(category="absorption_spectrum", id="us76_u86_4-fullrange")

        # interpolate dataset in wavenumber
        wavenumber = (1 / wavelength).to("cm^-1")
        xsw = ds.xs.interp(w=wavenumber.magnitude)

        # interpolate dataset in pressure
        xsp = xsw.interp(
            p=p.magnitude,
            kwargs=dict(fill_value=0.)  # this is required to handle the
            # pressure values that are smaller than 0.101325 Pa (the pressure
            # point with the smallest value in the absorption datasets) in the
            # US76 profile. These small pressure values occur above the
            # altitude of 93 km. Considering that the air number density at
            # these altitudes is small than the air number density at the
            # surface by a factor larger than 1e5, we assume that the
            # corresponding absorption coefficient is negligible compared to
            # 0.01 km^-1.
        )

        # attach units
        xs = ureg.Quantity(xsp.values, xsp.units)

        # absorption coefficient
        self._sigma_a_values = (n_tot * xs).to("km^-1")

    @property
    def albedo(self):
        return (self.sigma_s / self.sigma_t).to(ureg.dimensionless)

    @property
    def sigma_a(self):
        return self._sigma_a_values

    @property
    def sigma_s(self):
        return self._sigma_s_values

    @property
    def sigma_t(self):
        return self.sigma_a + self.sigma_s
