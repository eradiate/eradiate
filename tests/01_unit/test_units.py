import numpy as np
import pytest
from pinttrs.exceptions import UnitsError
from pinttrs.util import ensure_units

from eradiate.units import interpret_quantities, magnitude_as, symbol
from eradiate.units import unit_context_config as ucc
from eradiate.units import unit_registry as ureg


def test_symbol():
    # Convert strings
    assert symbol("metre") == "m"
    assert symbol("dimensionless") == ""
    assert symbol("") == ""

    # Convert units
    assert symbol(ureg.dimensionless) == ""
    assert symbol(ureg.m) == "m"


def test_interpret_quantities():
    # If we pass no quantity map, behaviour is the same as pinttrs.interpret_units()
    assert interpret_quantities({"wmin": 550.0}, {}, ucc) == {"wmin": 550.0}
    assert interpret_quantities({"wmin": 550.0, "wmin_units": "nm"}, {}, ucc) == {
        "wmin": 550.0 * ureg.nm
    }

    # If we pass a quantity map, unitless fields are applied units
    assert interpret_quantities({"wmin": 550.0}, {"wmin": "wavelength"}, ucc) == {
        "wmin": 550.0 * ureg.nm
    }

    # Unit context override propagates to default units
    with ucc.override(wavelength="micron"):
        assert interpret_quantities({"wmin": 550.0}, {"wmin": "wavelength"}, ucc) == {
            "wmin": 550.0 * ureg.micron
        }

    # Incompatible units raise
    with pytest.raises(UnitsError):
        assert interpret_quantities(
            {"wmin": 550.0, "wmin_units": "s"}, {"wmin": "wavelength"}, ucc
        )

    # Existing units are not modified...
    assert (
        interpret_quantities({"wmin": 0.55 * ureg.micron}, {"wmin": "wavelength"}, ucc)[
            "wmin"
        ].magnitude
        == 0.55
    )

    # ... unless force mode is used
    assert (
        interpret_quantities(
            {"wmin": 0.55 * ureg.micron}, {"wmin": "wavelength"}, ucc, force=True
        )["wmin"].magnitude
        == 550.0
    )


@pytest.mark.parametrize(
    "value",
    [
        ureg.Quantity(1.0, "m"),
        ureg.Quantity([1.0, 1.0], "m"),
        1.0,
        [1.0, 1.0],
        np.array([1.0, 1.0]),
    ],
    ids=["quantity_float", "quantity_array", "float", "list", "array"],
)
def test_magnitude_as(value):
    assert np.all(
        magnitude_as(value, "km", default_units="m")
        == ensure_units(value, default_units=ureg.Unit("m")).m_as("km")
    )
