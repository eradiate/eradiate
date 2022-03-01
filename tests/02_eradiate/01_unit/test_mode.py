"""Root module testing."""
import pytest

import eradiate
from eradiate import ModeFlags, supported_mode, unsupported_mode
from eradiate.exceptions import UnsupportedModeError


def test_mode_mono(mode_mono):
    mode = eradiate.mode()
    assert mode.kernel_variant == "scalar_mono_double"


def test_mode_mono_single(mode_mono_single):
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
    eradiate.set_mode("mono_single")
    # We expect that the kernel variant is appropriately selected
    assert mitsuba.variant() == "scalar_mono"


def test_mode_flags():
    # Check flags for mono single mode
    eradiate.set_mode("mono_single")
    assert eradiate.mode().has_flags(ModeFlags.ERT_MONO)
    assert eradiate.mode().has_flags(ModeFlags.MI_SINGLE)

    # Check flags for mono_double mode
    eradiate.set_mode("mono_double")
    assert eradiate.mode().has_flags(ModeFlags.ERT_MONO)
    assert eradiate.mode().has_flags(ModeFlags.MI_DOUBLE)

    # Check if conversion of string to flags works as intended
    eradiate.set_mode("mono_double")
    assert eradiate.mode().has_flags("any_double")


def test_supported_mode():
    for mode in ["mono_single", "mono_double", "mono"]:
        eradiate.set_mode(mode)

        with pytest.raises(UnsupportedModeError):
            supported_mode("ANY_CKD")

        supported_mode("ANY_MONO")

    for mode in ["ckd_single", "ckd_double", "ckd"]:
        eradiate.set_mode(mode)

        with pytest.raises(UnsupportedModeError):
            supported_mode("ANY_MONO")

        supported_mode("ANY_CKD")

    for mode in ["mono", "ckd"]:
        eradiate.set_mode(mode)

        with pytest.raises(UnsupportedModeError):
            supported_mode("ANY_SINGLE")

        supported_mode("ANY_DOUBLE")

    for mode in ["mono_double", "ckd_double"]:
        eradiate.set_mode(mode)

        with pytest.raises(UnsupportedModeError):
            supported_mode("ANY_SINGLE")

        supported_mode("ANY_DOUBLE")


def test_unsupported_mode():
    for mode in ["mono_single", "mono_double", "mono"]:
        eradiate.set_mode(mode)

        with pytest.raises(UnsupportedModeError):
            unsupported_mode("ANY_MONO")

        unsupported_mode("ANY_CKD")

    for mode in ["ckd_single", "ckd_double", "ckd"]:
        eradiate.set_mode(mode)

        with pytest.raises(UnsupportedModeError):
            unsupported_mode("ANY_CKD")

        unsupported_mode("ANY_MONO")

    for mode in ["mono_double", "ckd_double", "mono", "ckd"]:
        eradiate.set_mode(mode)

        with pytest.raises(UnsupportedModeError):
            unsupported_mode("ANY_DOUBLE")

        unsupported_mode("ANY_SINGLE")

    for mode in ["mono_single", "ckd_single"]:
        eradiate.set_mode(mode)

        with pytest.raises(UnsupportedModeError):
            unsupported_mode("ANY_SINGLE")

        unsupported_mode("ANY_DOUBLE")
