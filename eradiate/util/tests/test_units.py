import pytest

from eradiate.util.units import DefaultUnits, ureg


def test_default_units():
    du = DefaultUnits()

    # Test getter
    assert du.get("length") == ureg.m
    assert du.get("irradiance") == ureg.Unit("W/m^2/nm")
    with pytest.raises(KeyError):
        du.get("distance")

    # Test evaluation method
    assert du.units() == {
        "length": ureg.m,
        "time": ureg.s,
        "mass": ureg.kg,
        "wavelength": ureg.nm,
        "angle": ureg.deg,
        "irradiance": ureg.Quantity("W/m^2/nm"),
        "radiance": ureg.Quantity("W/m^2/nm/sr"),
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

    # We define a set of units from scratch and add a procedural override
    du = DefaultUnits({"length": "m", "time": "s"})
    with du.override({"speed": lambda: du.get("length") / du.get("time"),
                      "length": "mile", "time": "hour"}):
        assert du.get("speed") == ureg.mile / ureg.hour
