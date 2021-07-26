"""Root module testing."""

import eradiate
from eradiate._mode import ModeFlags


def test_mode_mono(mode_mono):
    mode = eradiate.mode()
    assert mode.kernel_variant == "scalar_mono"


def test_mode_mono_double(mode_mono_double):
    mode = eradiate.mode()
    assert mode.kernel_variant == "scalar_mono_double"


def test_modes():
    import mitsuba

    # Check that switching to double precision mode works
    eradiate.set_mode("mono_double")
    # We expect that the kernel variant is appropriately selected
    assert mitsuba.variant() == "scalar_mono_double"

    # Check that switching to mono mode works
    eradiate.set_mode("mono")
    # We expect that the kernel variant is appropriately selected
    assert mitsuba.variant() == "scalar_mono"


def test_mode_flags():
    # Check flags for mono mode
    eradiate.set_mode("mono")
    assert eradiate.mode().has_flags(ModeFlags.ERT_MONO)
    assert not eradiate.mode().has_flags(ModeFlags.MTS_DOUBLE)

    # Check flags for mono_double mode
    eradiate.set_mode("mono_double")
    assert eradiate.mode().has_flags(ModeFlags.ERT_MONO)
    assert eradiate.mode().has_flags(ModeFlags.MTS_DOUBLE)

    # Check if conversion of string to flags works as intended
    eradiate.set_mode("mono_double")
    assert eradiate.mode().has_flags("any_double")
