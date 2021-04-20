"""Root module testing."""
import pytest
from attr.exceptions import FrozenInstanceError

from eradiate import unit_registry as ureg


def test_modes():
    import eradiate
    import mitsuba

    # Check that switching to double precision mode works
    eradiate.set_mode("mono_double")
    # We expect that the kernel variant is appropriately selected
    assert mitsuba.variant() == "scalar_mono_double"
    assert eradiate.mode().is_monochromatic()
    assert eradiate.mode().is_double_precision()

    # Check that switching to mono mode works
    eradiate.set_mode("mono")
    # We expect that the kernel variant is appropriately selected
    assert mitsuba.variant() == "scalar_mono"
    assert eradiate.mode().is_monochromatic()
    assert eradiate.mode().is_single_precision()
