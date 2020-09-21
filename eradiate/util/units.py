""" Unit system-related components. """

from contextlib import contextmanager
from copy import deepcopy
import pint

ureg = pint.UnitRegistry()  #: Unit registry common to all Eradiate components.


class DefaultUnits:

    def __init__(self):
        self.units = {
            "length": lambda: ureg.m,
            "time": lambda: ureg.s,
            "mass": lambda: ureg.kg,
            "wavelength": lambda: ureg.nm,
            "angle": lambda: ureg.deg,
            "irradiance": lambda: ureg.watt / self.units.get("length")() ** 2 / self.units.get("wavelength")(),
            "radiance": lambda: ureg.watt / self.units.get("length")() ** 2 / ureg.steradian / self.units.get("wavelength")(),
        }

    @contextmanager
    def override(self, d):
        """The override context manager lets users alter default units for their application in a
        dynamic manner.

        Parameter ``d`` (dictionary)
            Dictionary of units to add to overwrite the defaults with.

            Overrides can be specified either as a string, defining a unit directly, or as a lambda
            returning a pint unit object, which can reference other units in the default units
            registry

        .. admonition:: Example

            .. code:: python

                ureg = pint.UnitRegistry()
                defaults = DefaultUnits()

                overrides = {
                    "length": "km",
                    "radiance": lambda: ureg.watt / defaults.units.get("length") ** 2 / ureg.steradian / defaults.units.get("wavelength")
                }

                print(defaults.units.get("radiance")())
                with defaults.override(overrides):
                    print(defaults.units.get("radiance")())

            The first print statement will yield a unit of :code:`W/m**2/steradian/nm`,
            while the second will yield a unit of :code:`W/km**2/steradian/nm`.

        """
        to_update = dict()
        for key, value in d.items():
            if isinstance(value, str):
                to_update[key] = lambda: ureg.Unit(value)
            elif callable(value):
                to_update[key] = value
            else:
                raise ValueError(f"Only strings and lambdas are handled by this context manager."
                                 f"Found: {type(value)}")

        units_old = deepcopy(self.units)
        self.units.update(to_update)

        try:
            yield self.units
        finally:
            self.units = units_old
