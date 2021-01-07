""" Unit system-related components. """

__all__ = [
    "ureg",
    "compatible",
    "ensure_units",
    "PhysicalQuantity",
    "DefaultUnits",
    "config_default_units",
    "kernel_default_units"
]

import enum
from contextlib import contextmanager
from copy import deepcopy
from functools import lru_cache

import pint

#: Unit registry common to all Eradiate components. All units used in Eradiate
#: must be created using this registry.
ureg = pint.UnitRegistry()


def compatible(unit1, unit2):
    """Check if two units are compatible. Accounts for angle units.

    Parameter ``unit1`` (:class:`pint.Unit`):
        First unit to check for compatibility.

    Parameter ``unit2`` (:class:`pint.Unit`):
        Second unit to check for compatibility.

    Returns → bool
        ``True`` if ``unit1`` and ``unit2`` have the same dimensionality,
        ``False`` otherwise.
    """
    return (1. * unit1 / unit2).unitless


def ensure_units(value, default_units, convert=False, ureg=ureg):
    """Ensure that a value is wrapped in a Pint quantity container.

    Parameter ``value``:
        Value to ensure the wrapping of.

    Parameter ``default_units`` (callable or :class:`pint.Units` or str):
        Units to use to initialise the :class:`pint.Quantity` if value is not
        a :class:`pint.Quantity`. A callable can be passed; in this case,
        the applied units will be ``default_units()``.

    Parameter ``convert`` (bool):
        If ``True``, ``value`` will also be converted to ``default_units`` if
        it is a :class:`pint.Quantity`.

    Parameter ``ureg`` (:class:`pint.UnitRegistry`):
        Unit registry used to create the :class:`pint.Quantity` container.
        By default, Eradiate's unit registry is used.

    Returns → :class:`pint.Quantity`:
        Converted ``value``.
    """
    if callable(default_units):
        default_units = default_units()

    if isinstance(value, pint.Quantity):
        if convert:
            return value.to(default_units)
        else:
            return value
    else:
        return ureg.Quantity(value, default_units)


class PhysicalQuantity(enum.Enum):
    """An enumeration defining physical quantities known to Eradiate."""
    ALBEDO = enum.auto()
    ANGLE = enum.auto()
    COLLISION_COEFFICIENT = enum.auto()
    DIMENSIONLESS = enum.auto()
    IRRADIANCE = enum.auto()
    LENGTH = enum.auto()
    MASS = enum.auto()
    RADIANCE = enum.auto()
    REFLECTANCE = enum.auto()
    SPEED = enum.auto()
    TIME = enum.auto()
    TRANSMITTANCE = enum.auto()
    WAVELENGTH = enum.auto()

    @classmethod
    @lru_cache(maxsize=32)
    def spectrum(cls):
        """Return a tuple containing a subset of :class:`PhysicalQuantity`
        members suitable for :class:`.Spectrum` initialisation. This function
        caches its results for improved efficiency.
        """
        return (
            cls.ALBEDO,
            cls.COLLISION_COEFFICIENT,
            cls.DIMENSIONLESS,
            cls.IRRADIANCE,
            cls.RADIANCE,
            cls.REFLECTANCE,
            cls.TRANSMITTANCE
        )

    @classmethod
    def from_str(cls, s):
        """Get a member from a string.

        Parameter ``s`` (str):
            String to convert to :class:`.PhysicalQuantity`. ``s`` will first be
            converted to upper case.

        Returns → :class:`.PhysicalQuantity`:
            Retrieved enum member.
        """
        return cls[s.upper()]

    @classmethod
    def from_any(cls, value):
        """Get a member from a many types.

        Parameter ``value`` (:class:`.PhysicalQuantity` or str or int):
            Value to convert to :class:`.PhysicalQuantity`.

        Returns → :class:`.PhysicalQuantity`:
            Retrieved enum member.

        Raises → TypeError:
            If ``value`` is of unsupported type.
        """
        if isinstance(value, cls):
            return value
        elif isinstance(value, str):
            return cls.from_str(value)
        elif isinstance(value, int):
            return cls(value)
        else:
            raise TypeError(str(value))


class DefaultUnits:
    """An interface to flexibly access a set of units.

    This class holds an internal dictionary of callables which should return
    :class:`pint.Unit` instances. It also provides a convenient interface to
    evaluate these callables and return the corresponding unit objects (see
    :meth:`get` and :meth:`units`).

    In addition, the :meth:`update` method allows advanced users to modify the
    contents of the unit map without too much boilerplate. An :meth:`override`
    context manager finally allows for temporary modification of the object's
    unit map and restores it afterwards.

    .. warning::

       The unit registry used to define units must be consistent. Eradiate uses
       a unique unit register :data:`eradiate.util.units.ureg` for that
       purpose.

    .. rubric:: Constructor arguments / instance attributes

    ``units`` (dict or None):
        This parameter is used to initialise the unit map using :meth:`update`.
        If therefore allows for regular Pint unit definitions. Keys should be
        convertible to :class:`.PhysicalQuantity` (see
        :meth:`.PhysicalQuantity.from_any` for accepted types).

        .. admonition:: Example

           .. code:: python

              # Easy way
              du = DefaultUnits({"length": "m", "time": "s"})
              # Directly using Pint objects
              du = DefaultUnits({"length": ureg.m, "time": ureg.s})
              # Directly using PhysicalQuantity enum members
              du = DefaultUnits({PhysicalQuantity.LENGTH: ureg.m, PhysicalQuantity.TIME: ureg.s})

        If ``units`` is set to ``None``, a default set of units is used. Some
        units are defined so as to dynamically update when the units from which
        they derive is updated. The following code snippet describes in detail
        the defaults.

        .. code:: python

           if units is None:
               units = {
                   # We allow for dimensionless quantities
                   PhysicalQuantity.DIMENSIONLESS: lambda: ureg.dimensionless,
                   # Basic quantities must be named after their SI name
                   # https://en.wikipedia.org/wiki/International_System_of_Units
                   PhysicalQuantity.LENGTH: lambda: ureg.m,
                   PhysicalQuantity.TIME: lambda: ureg.s,
                   PhysicalQuantity.MASS: lambda: ureg.kg,
                   # Derived quantity names are more flexible
                   PhysicalQuantity.ALBEDO: lambda: ureg.dimensionless,
                   PhysicalQuantity.ANGLE: lambda: ureg.deg,
                   PhysicalQuantity.REFLECTANCE: lambda: ureg.dimensionless,
                   PhysicalQuantity.TRANSMITTANCE: lambda: ureg.dimensionless,
                   PhysicalQuantity.WAVELENGTH: lambda: ureg.nm,
                   # The following quantities will update automatically based on their parent units
                   PhysicalQuantity.COLLISION_COEFFICIENT: lambda: self.get("length") ** -1,
                   PhysicalQuantity.IRRADIANCE: lambda: ureg.watt / self.get("length") ** 2 / self.get("wavelength"),
                   PhysicalQuantity.RADIANCE: lambda: ureg.watt / self.get("length") ** 2 / ureg.steradian / self.get("wavelength"),
               }
    """

    def __init__(self, units=None):
        self._units = dict()

        if units is None:
            units = {
                # We allow for dimensionless quantities
                PhysicalQuantity.DIMENSIONLESS: lambda: ureg.dimensionless,
                # Basic quantities must be named after their SI name
                # https://en.wikipedia.org/wiki/International_System_of_Units
                PhysicalQuantity.LENGTH: lambda: ureg.m,
                PhysicalQuantity.TIME: lambda: ureg.s,
                PhysicalQuantity.MASS: lambda: ureg.kg,
                # Derived quantity names are more flexible
                PhysicalQuantity.ALBEDO: lambda: ureg.dimensionless,
                PhysicalQuantity.ANGLE: lambda: ureg.deg,
                PhysicalQuantity.REFLECTANCE: lambda: ureg.dimensionless,
                PhysicalQuantity.TRANSMITTANCE: lambda: ureg.dimensionless,
                PhysicalQuantity.WAVELENGTH: lambda: ureg.nm,
                # The following quantities will update automatically based on their parent units
                PhysicalQuantity.COLLISION_COEFFICIENT: lambda: self.get("length") ** -1,
                PhysicalQuantity.IRRADIANCE: lambda: ureg.watt / self.get("length") ** 2 / self.get(
                    "wavelength"),
                PhysicalQuantity.RADIANCE: lambda: ureg.watt / self.get(
                    "length") ** 2 / ureg.steradian / self.get("wavelength"),
            }

        self.update(units)

    def update(self, d):
        """Update the internal unit map with the contents of ``d``.

        .. admonition:: Example

           .. code:: python

              # Create an empty unit set
              du = DefaultUnits({})
              # Add units with the simplified syntax
              du.update({"length": "m", "time": "s"})
              # Add units defined using a callable
              du.update({"speed": lambda: du.get("length") / du.get("time")})

        .. warning::

           Custom callables are very powerful because they can be used to
           define dynamically updating units; however they are also more
           difficult to debug than the ones constructed automatically from
           strings and unit objects.

        Parameter ``d`` (dict)
            Dictionary used to update the unit map. Dictionary keys can be:

            * :class:`.PhysicalQuantity` enum members;
            * strings: if a string is used, it will first be converted to an
              :class:`.PhysicalQuantity` enum member;
            * integers: if an integer is used, it will first be converted to an
              :class:`.PhysicalQuantity` enum member.

            Dictionary values can be:

            * callables: if a callable is found, it is directly used to update
              the unit map;
            * :class:`pint.Unit` objects;
            * strings used to initialise a :class:`pint.Unit` object.

            The latter two types of objects are then used to generate callables
            inserted into the unit map.
        """

        to_update = dict()

        for key, value in d.items():
            if not isinstance(key, PhysicalQuantity):
                key = PhysicalQuantity.from_any(key)

            if isinstance(value, (str, ureg.Unit)):
                to_update[key] = lambda value=value: ureg.Unit(
                    value)  # https://stackoverflow.com/questions/11087047/deferred-evaluation-with-lambda-in-python
            elif callable(value):
                to_update[key] = value
            else:
                raise ValueError(
                    f"Items must be either strings, Pint units or callables; "
                    f"found: {key}: {type(value)}"
                )

        self._units.update(to_update)

    def units(self):
        """Evaluate all callables in the internal unit map.

        Returns → dict:
            A dictionary with the same keys as the internal unit map and the
            evaluation of the unit map's callables as values.
        """
        return {key: value() for key, value in self._units.items()}

    def get(self, key):
        """Get default units for a given quantity.

        Parameter ``key`` (:class:`PhysicalQuantity` or str):
            Quantity to get the units of.

        Returns → :class:`pint.Unit`:
            Requested unit.

        Raises → TypeError:
            The requested ``key`` was of unhandled type.

        Raises → KeyError:
            The requested ``key`` could not be found in the unit map.
        """
        try:
            return self._units[PhysicalQuantity.from_any(key)]()
        except KeyError:
            raise

    def get_str(self, key):
        """Get default units for a given quantity and return it as a string.

        Returns → (:class:`PhysicalQuantity` or str):
            Quantity to get the units of.

        Raises → ``KeyError``:
            The requested ``key`` could not be found in the unit map.
        """
        try:
            return str(self.get(key))
        except KeyError:
            raise

    def generator(self, quantity):
        """Return a callable which, when called, returns the current unit of
        ``quantity``.

        Parameter ``quantity`` (str)
            Quantity to create a unit generator for.

        Returns → callable:
            Created unit generator.
        """
        return lambda: self.get(quantity)

    @contextmanager
    def override(self, d):
        """A context manager which lets the user temporarily alter default units
        for their application.

        .. admonition:: Example

           .. code:: python

               from eradiate.util.units import ureg
               defaults = DefaultUnits()

               print(defaults.get("length"))  # We get m
               print(defaults.get("radiance")) # We get W/m^2/sr/nm

               with defaults.override({"length": "km"}):
                   print(defaults.get("length"))  # We get km
                   print(defaults.get("radiance")) # We get W/km^2/sr/nm

               print(defaults.get("length"))  # We get m
               print(defaults.get("radiance")) # We get W/m^2/sr/nm

               # We add a custom dynamic unit
               defaults.update({
                   "speed": lambda: defaults.get("length") / defaults.get("time")
               })
               print(defaults.get("speed"))  # We get m/s

               with defaults.override({"speed": "km"}):
                   print(defaults.get("speed"))  # We get km/s

               print(defaults.get("speed"))  # We get m/s

        Parameter ``d`` (dict)
            Dictionary of units to override the defaults with.

            Overrides are specified using the same input as for the
            :meth:`update` method. It follows that:

            * custom callables can be used;
            * dynamic units will be updated automatically if their parents are
              updated.
        """
        units_old = deepcopy(self._units)
        self.update(d)

        try:
            yield
        finally:
            self._units = units_old


#: Default unit set used when interpreting config dictionaries
config_default_units = DefaultUnits()

#: Default unit set used when building kernel dictionaries
kernel_default_units = DefaultUnits()
