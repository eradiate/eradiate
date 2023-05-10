"""Root module testing."""
import pytest

import eradiate
from eradiate import supported_mode, unsupported_mode
from eradiate.exceptions import UnsupportedModeError


def test_mode_mono(mode_mono):
    mode = eradiate.mode()
    assert mode.mi_variant == "scalar_mono_double"


def test_mode_mono_single(mode_mono_single):
    mode = eradiate.mode()
    assert mode.mi_variant == "scalar_mono"


def test_mode_mono_double(mode_mono_double):
    mode = eradiate.mode()
    assert mode.mi_variant == "scalar_mono_double"


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


def test_mode_check():
    # Check flags for mono single mode
    eradiate.set_mode("mono_single")
    assert eradiate.mode().check(spectral_mode="mono", mi_double_precision=False)

    # Check flags for mono_double mode
    eradiate.set_mode("mono_double")
    assert eradiate.mode().check(spectral_mode="mono", mi_double_precision=True)


def test_supported_mode():
    for mode in ["mono_single", "mono_double", "mono"]:
        eradiate.set_mode(mode)

        with pytest.raises(UnsupportedModeError):
            supported_mode(spectral_mode="ckd")

        supported_mode(spectral_mode="mono")

    for mode in ["ckd_single", "ckd_double", "ckd"]:
        eradiate.set_mode(mode)

        with pytest.raises(UnsupportedModeError):
            supported_mode(spectral_mode="mono")

        supported_mode(spectral_mode="ckd")

    for mode in ["mono", "ckd"]:
        eradiate.set_mode(mode)

        with pytest.raises(UnsupportedModeError):
            supported_mode(mi_double_precision=False)

        supported_mode(mi_double_precision=True)

    for mode in ["mono_double", "ckd_double"]:
        eradiate.set_mode(mode)

        with pytest.raises(UnsupportedModeError):
            supported_mode(mi_double_precision=False)

        supported_mode(mi_double_precision=True)


def test_unsupported_mode():
    for mode in ["mono_single", "mono_double", "mono"]:
        eradiate.set_mode(mode)

        with pytest.raises(UnsupportedModeError):
            unsupported_mode(spectral_mode="mono")

        unsupported_mode(spectral_mode="ckd")

    for mode in ["ckd_single", "ckd_double", "ckd"]:
        eradiate.set_mode(mode)

        with pytest.raises(UnsupportedModeError):
            unsupported_mode(spectral_mode="ckd")

        unsupported_mode(spectral_mode="mono")

    for mode in ["mono_double", "ckd_double", "mono", "ckd"]:
        eradiate.set_mode(mode)

        with pytest.raises(UnsupportedModeError):
            unsupported_mode(mi_double_precision=True)

        unsupported_mode(mi_double_precision=False)

    for mode in ["mono_single", "ckd_single"]:
        eradiate.set_mode(mode)

        with pytest.raises(UnsupportedModeError):
            unsupported_mode(mi_double_precision=False)

        unsupported_mode(mi_double_precision=True)
