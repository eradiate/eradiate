"""Root module testing."""

import pytest

import eradiate
from eradiate import ModeFlag, supported_mode, unsupported_mode
from eradiate.exceptions import UnsetModeError, UnsupportedModeError


def test_modes_single(modes_all_single):
    mode = eradiate.get_mode()
    assert mode.mi_double_precision is ModeFlag.MI_DOUBLE_PRECISION_NO
    assert mode.is_single_precision is True
    assert "_double" not in mode.mi_variant


def test_modes_double(modes_all_double):
    mode = eradiate.get_mode()
    assert mode.mi_double_precision is ModeFlag.MI_DOUBLE_PRECISION_YES
    assert mode.is_double_precision is True
    assert "_double" in mode.mi_variant


def test_modes_mono(modes_all_mono):
    mode = eradiate.get_mode()
    assert mode.spectral_mode is ModeFlag.SPECTRAL_MODE_MONO
    assert mode.is_mono is True
    assert "_mono" in mode.mi_variant


def test_modes_ckd(modes_all_ckd):
    mode = eradiate.get_mode()
    assert mode.spectral_mode is ModeFlag.SPECTRAL_MODE_CKD
    assert mode.is_ckd is True
    assert mode.is_mono is False
    assert "_mono" in mode.mi_variant


def test_modes_polarized(modes_all_polarized):
    mode = eradiate.get_mode()
    assert mode.mi_polarized is ModeFlag.MI_POLARIZED_YES
    assert mode.is_polarized is True
    assert "_polarized" in mode.mi_variant


def test_mode_none():
    eradiate.set_mode("none")

    with pytest.raises(UnsetModeError):
        eradiate.get_mode(raise_exc=True)

    assert eradiate.get_mode(raise_exc=False) is None


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
    assert eradiate.get_mode().check(spectral_mode="mono", mi_double_precision=False)

    # Check flags for mono_double mode
    eradiate.set_mode("mono_double")
    assert eradiate.get_mode().check(spectral_mode="mono", mi_double_precision=True)


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
