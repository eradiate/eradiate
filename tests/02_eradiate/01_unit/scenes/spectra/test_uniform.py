import mitsuba as mi
import numpy as np
import pint
import pinttr
import pytest

from eradiate import unit_context_config as ucc
from eradiate import unit_context_kernel as uck
from eradiate import unit_registry as ureg
from eradiate.contexts import KernelDictContext
from eradiate.scenes.spectra import UniformSpectrum, spectrum_factory
from eradiate.units import PhysicalQuantity


def test_uniform(modes_all):
    # Instantiate with value only
    s = UniformSpectrum(value=1.0)
    assert s.quantity is PhysicalQuantity.DIMENSIONLESS
    assert s.value == 1.0 * ureg.dimensionless

    # Instantiate with integer value
    s = UniformSpectrum(value=1)
    assert s.quantity is PhysicalQuantity.DIMENSIONLESS
    assert s.value == 1.0 * ureg.dimensionless
    assert isinstance(s.value.magnitude, float)

    # Instantiate with value and quantity
    s = UniformSpectrum(value=1.0, quantity=PhysicalQuantity.COLLISION_COEFFICIENT)
    assert s.value == 1.0 * ureg.m**-1
    s = UniformSpectrum(value=1.0, quantity="collision_coefficient")
    assert s.quantity == PhysicalQuantity.COLLISION_COEFFICIENT
    assert s.value == 1.0 * ureg.m**-1

    # Instantiate with unsupported quantity
    with pytest.raises(ValueError):
        UniformSpectrum(value=1.0, quantity="speed")

    # Instantiate with all arguments
    s = UniformSpectrum(quantity="collision_coefficient", value=1.0)
    assert s.value == ureg.Quantity(1.0, "m^-1")

    # Raise if units and quantity are inconsistent
    with pytest.raises(pinttr.exceptions.UnitsError):
        UniformSpectrum(quantity="collision_coefficient", value=ureg.Quantity(1.0, ""))

    # Instantiate from factory using dict
    s = spectrum_factory.convert(
        {
            "type": "uniform",
            "quantity": "radiance",
            "value": 1.0,
            "value_units": "W/km^2/sr/nm",
        }
    )

    # Produced kernel dict is valid
    ctx = KernelDictContext()
    assert isinstance(s.kernel_dict(ctx).load(), mi.Texture)

    # Unit scaling is properly applied
    with ucc.override({"radiance": "W/m^2/sr/nm"}):
        s = UniformSpectrum(quantity="radiance", value=1.0)
    with uck.override({"radiance": "kW/m^2/sr/nm"}):
        ctx = KernelDictContext()
        d = s.kernel_dict(ctx)
        assert np.allclose(d["spectrum"]["value"], 1e-3)


@pytest.mark.parametrize(
    "quantity, value, w, expected",
    [
        ("dimensionless", 1.0, 550.0, 1.0),
        ("dimensionless", 1.0, [500.0, 600.0] * ureg.nm, 1.0),
        ("collision_coefficient", 1.0, 550.0, 1.0 * ureg.m**-1),
        ("collision_coefficient", 1.0, [500.0, 600.0] * ureg.nm, 1.0 * ureg.m**-1),
    ],
)
def test_uniform_eval_mono(mode_mono, quantity, value, w, expected):
    # No quantity, unitless value
    eval = UniformSpectrum(quantity=quantity, value=value).eval_mono(w)
    assert np.all(expected == eval)
    assert isinstance(eval, pint.Quantity)


def test_integral(mode_mono):
    s = UniformSpectrum(value=0.5)
    assert s.integral(300.0, 400.0) == 50.0 * ureg.nm

    s = UniformSpectrum(quantity="collision_coefficient", value=0.5)
    assert s.integral(300.0, 400.0) == 50.0 * ureg("nm / m")
