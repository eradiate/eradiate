import pytest

from eradiate.util.units import DefaultUnits, compatible, config_default_units, ensure_units, ureg


def test_compatible():
    assert compatible(ureg.m, ureg.km)
    assert not compatible(ureg.Unit("W/m^2/sr/nm"), ureg.Unit("W/m^2/nm"))


def test_ensure_units():
    assert ensure_units(100, "km") == ureg.Quantity(100, "km")
    assert ensure_units(ureg.Quantity(100, "m"), "km") == ureg.Quantity(100, "m")
    assert ensure_units(ureg.Quantity(100, "m"), "km", convert=True) == ureg.Quantity(0.1, "km")

    units = config_default_units.generator("length")
    with config_default_units.override({"length": "m"}):
        assert ensure_units(100, units) == ureg.Quantity(100, "m")
    with config_default_units.override({"length": "km"}):
        assert ensure_units(100, units) == ureg.Quantity(100, "km")


def test_default_units():
    du = DefaultUnits()

    # Test getter
    assert du.get("length") == ureg.m
    assert du.get("irradiance") == ureg.Unit("W/m^2/nm")
    with pytest.raises(KeyError):
        du.get("distance")

    # Test string getter
    assert du.get_str("length") == "meter"
    assert du.get_str("irradiance") == "watt / meter ** 2 / nanometer"

    # Test evaluation method
    assert du.units() == {
        "albedo": ureg.dimensionless,
        "angle": ureg.deg,
        "collision_coefficient": ureg.Unit("m^-1"),
        "dimensionless": ureg.dimensionless,
        "irradiance": ureg.Unit("W/m^2/nm"),
        "length": ureg.m,
        "mass": ureg.kg,
        "radiance": ureg.Unit("W/m^2/nm/sr"),
        "reflectance": ureg.dimensionless,
        "time": ureg.s,
        "wavelength": ureg.nm,
    }

    # Test updater (not part of the public API so we don't have a lot of safeguards there)
    du.update({"length": "km"})
    assert du.get("length") == ureg.km  # Check the updated item
    assert du.get("irradiance") == ureg.Unit("W/km^2/nm")  # Check a dynamically evaluated item
    du.update({"speed": lambda: du.get("length") / du.get("time")})  # Use a more sophisticated callable
    assert du.get("speed") == ureg.km / ureg.s

    # Test override
    with du.override({"length": "m"}):
        assert du.get("length") == ureg.m  # Basic value
        assert du.get("speed") == ureg.m / ureg.s  # Dynamically evaluated, custom
    with du.override({"time": "ms"}):
        assert du.get("speed") == ureg.km / ureg.ms  # Dynamically evaluated, custom
    with du.override({"length": "km", "wavelength": "m"}):  # Override several units
        assert du.get("irradiance") == ureg.Unit("W/km^2/m")  # Dynamically evaluated, default

    # Test generator
    with du.override({"length": "m"}):
        unit = du.generator("length")
        assert unit() == ureg.m
    with du.override({"length": "km"}):
        assert unit() == ureg.km

    # We define a set of units from scratch and add a procedural override
    du = DefaultUnits({"length": "m", "time": "s"})
    with du.override({"speed": lambda: du.get("length") / du.get("time"),
                      "length": "mile", "time": "hour"}):
        assert du.get("speed") == ureg.mile / ureg.hour
