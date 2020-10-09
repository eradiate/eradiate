""" Unit system-related components. """

from contextlib import contextmanager
from copy import deepcopy

import pint

#: Unit registry common to all Eradiate components.
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

    Parameter ``units`` (dict or None):
        This parameter is used to initialise the unit map using :meth:`update`.
        If therefore allows for regular Pint unit definitions.

        .. admonition:: Example

           .. code:: python

              # Easy way
              du = DefaultUnits({"length": "m", "time": "s"})
              # Directly using Pint objects
              du = DefaultUnits({"length": ureg.m, "time": ureg.s})

        If ``units`` is set to ``None``, a default set of units is used. Some
        units are defined so as to dynamically update when the units from which
        they derive is updated. The following code snippet describes in detail
        the defaults.

        .. code:: python

           if units is None:
               units = {
                   # Basic quantities must be named after their SI name
                   # https://en.wikipedia.org/wiki/International_System_of_Units
                   "length": lambda: ureg.m,
                   "time": lambda: ureg.s,
                   "mass": lambda: ureg.kg,
                   # Derived quantity names are more flexible
                   "wavelength": lambda: ureg.nm,
                   "angle": lambda: ureg.deg,
                   # The following quantities will update automatically based on their parent units
                   "irradiance": lambda: ureg.watt / self.get("length") ** 2 / self.get("wavelength"),
                   "radiance": lambda: ureg.watt / self.get("length") ** 2 / ureg.steradian / self.get("wavelength"),
               }
    """

    def __init__(self, units=None):
        self._units = dict()

        if units is None:
            units = {
                # We allow for dimensionless quantities
                "dimensionless": lambda: ureg.dimensionless,
                # Basic quantities must be named after their SI name
                # https://en.wikipedia.org/wiki/International_System_of_Units
                "length": lambda: ureg.m,
                "time": lambda: ureg.s,
                "mass": lambda: ureg.kg,
                # Derived quantity names are more flexible
                "wavelength": lambda: ureg.nm,
                "angle": lambda: ureg.deg,
                "reflectance": lambda: ureg.dimensionless,
                # The following quantities will update automatically based on their parent units
                "irradiance": lambda: ureg.watt / self.get("length") ** 2 / self.get("wavelength"),
                "radiance": lambda: ureg.watt / self.get("length") ** 2 / ureg.steradian / self.get("wavelength"),
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
            Dictionary used to update the unit map. Dictionary values can be:

            * callables: if a callable is found, it is directly used to update
              the unit map;
            * :class:`pint.Unit` objects;
            * strings used to initialise a :class:`pint.Unit` object.

            The latter two types of objects are then used to generate callables
            inserted into the unit map.

        """

        to_update = dict()

        for key, value in d.items():
            if isinstance(value, (str, ureg.Unit)):
                to_update[key] = lambda value=value: ureg.Unit(
                    value)  # https://stackoverflow.com/questions/11087047/deferred-evaluation-with-lambda-in-python
            elif callable(value):
                to_update[key] = value
            else:
                raise ValueError(
                    f"Items must be either strings or callables; "
                    f"found: {key}: {type(value)}"
                )

        self._units.update(to_update)

    def units(self):
        """Evaluate all callables in the internal unit map.

        Returns → dict
            A dictionary with the same keys as the internal unit map and the
            evaluation of the unit map's callables as values.
        """
        return {key: value() for key, value in self._units.items()}

    def get(self, key):
        """Get default unit for a given quantity.

        Parameter ``key`` (str)
            Quantity to get the unit of.

        Returns → :class:`pint.Unit`
            Requested unit.

        Raises → ``KeyError``
            The requested ``key`` could not be found in the unit map.
        """
        try:
            return self._units[key]()
        except KeyError:
            raise

    def get_str(self, key):
        """Get default unit for a given quantity and return it as a string.

        Returns → str
            Requested unit as a string.

        Raises → ``KeyError``
            The requested ``key`` could not be found in the unit map.
        """
        try:
            return str(self.get(key))
        except KeyError:
            raise

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
            Dictionary of units to add to overwrite the defaults with.

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
