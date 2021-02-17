"""Root module testing."""
import pytest
from attr.exceptions import FrozenInstanceError

from eradiate import unit_registry as ureg


def test_modes():
    import eradiate

    # Check that switching to double precision mode works
    eradiate.set_mode("mono_double")
    # We expect that the kernel variant is appropriately selected
    assert eradiate.kernel.variant() == "scalar_mono_double"

    # Check that switching to mono mode works
    eradiate.set_mode("mono")
    # We expect that the kernel variant is appropriately selected
    assert eradiate.kernel.variant() == "scalar_mono"
    # We check for defaults
    assert eradiate.mode().wavelength == ureg.Quantity(550., ureg.nm)

    # Check for unit conversion
    eradiate.set_mode("mono", wavelength=300.)
    assert eradiate.mode().wavelength == ureg.Quantity(300., ureg.nm)

    # Check that mode instances are frozen
    with pytest.raises(FrozenInstanceError):
        eradiate.mode().wavelength = 100
