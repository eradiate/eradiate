""" Unit system-related components. """

import enum
from functools import lru_cache

import pint
import pinttr

#: Unit registry common to all Eradiate components. All units used in Eradiate
#: must be created using this registry.
unit_registry = pint.UnitRegistry()

unit_registry.define("dobson_unit = 2.687e20 * meter^-2 = du = dobson = dobson_units")


class PhysicalQuantity(enum.Enum):
    """An enumeration defining physical quantities known to Eradiate."""

    ALBEDO = "albedo"
    ANGLE = "angle"
    COLLISION_COEFFICIENT = "collision_coefficient"
    DIMENSIONLESS = "dimensionless"
    IRRADIANCE = "irradiance"
    LENGTH = "length"
    MASS = "mass"
    RADIANCE = "radiance"
    REFLECTANCE = "reflectance"
    SPEED = "speed"
    TIME = "time"
    TRANSMITTANCE = "transmittance"
    WAVELENGTH = "wavelength"

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
            cls.TRANSMITTANCE,
        )


def _make_unit_context():
    uctx = pinttr.UnitContext(
        interpret_str=True, ureg=unit_registry, key_converter=PhysicalQuantity
    )

    # fmt: off
    for key, value in {
        # We allow for dimensionless quantities
        PhysicalQuantity.DIMENSIONLESS: pinttr.UnitGenerator(unit_registry.dimensionless),
        # Basic quantities must be named after their SI name
        # https://en.wikipedia.org/wiki/International_System_of_Units
        PhysicalQuantity.LENGTH: pinttr.UnitGenerator(unit_registry.m),
        PhysicalQuantity.TIME: pinttr.UnitGenerator(unit_registry.s),
        PhysicalQuantity.MASS: pinttr.UnitGenerator(unit_registry.kg),
        # Derived quantity names are more flexible
        PhysicalQuantity.ALBEDO: pinttr.UnitGenerator(unit_registry.dimensionless),
        PhysicalQuantity.ANGLE: pinttr.UnitGenerator(unit_registry.deg),
        PhysicalQuantity.REFLECTANCE: pinttr.UnitGenerator(unit_registry.dimensionless),
        PhysicalQuantity.TRANSMITTANCE: pinttr.UnitGenerator(unit_registry.dimensionless),
        PhysicalQuantity.WAVELENGTH: pinttr.UnitGenerator(unit_registry.nm),
    }.items():
        uctx.register(key, value)
    # fmt: on

    # The following quantities will update automatically based on their parent units
    uctx.register(
        PhysicalQuantity.COLLISION_COEFFICIENT,
        pinttr.UnitGenerator(lambda: uctx.get(PhysicalQuantity.LENGTH) ** -1),
    )
    uctx.register(
        PhysicalQuantity.IRRADIANCE,
        pinttr.UnitGenerator(
            lambda: unit_registry.watt
            / uctx.get(PhysicalQuantity.LENGTH) ** 2
            / uctx.get(PhysicalQuantity.WAVELENGTH),
        ),
    )
    uctx.register(
        PhysicalQuantity.RADIANCE,
        pinttr.UnitGenerator(
            lambda: unit_registry.watt
            / uctx.get(PhysicalQuantity.LENGTH) ** 2
            / unit_registry.steradian
            / uctx.get(PhysicalQuantity.WAVELENGTH),
        ),
    )

    return uctx


#: Unit context used when interpreting config dictionaries
unit_context_config = _make_unit_context()

#: Unit context used when building kernel dictionaries
unit_context_kernel = _make_unit_context()


def to_quantity(variable: xarray.Variable) -> pint.Quantity:
    """
    Converts a :class:`~xarray.Variable` to a :class:`~pint.Quantity`.

    Parameter ``variable`` (:class:`~xarray.Variable`):
        Xarray variable. For example, a :class:`~xarray.Dataset`'s data
        variable or coordinate, or a :class:`~xarray.DataArray` or a
        :class:`~xarray.DataArray`'s coordinate.

        .. note::
            The variable's ``attrs`` must include a ``units`` key.

    Returns → :class:`pint.Quantity`:
        The corresponding quantity.

    Raises → ValueError:
        If the variable's ``attrs`` does not include a ``units`` key.
    """
    try:
        units = variable.units
    except KeyError:
        raise ValueError(f"{variable} has no units.")
    else:
        return unit_registry.Quantity(variable.values, units)
