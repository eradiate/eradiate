import pytest

from eradiate.config import ProgressLevel, settings
from eradiate.frame import AzimuthConvention


def test_progress_level_conversion():
    # Conversion from string is supported
    assert ProgressLevel.convert("kernel") is ProgressLevel.KERNEL
    with pytest.raises(KeyError):
        ProgressLevel.convert("foo")

    # Conversion of integer is supported
    assert ProgressLevel.convert(2) is ProgressLevel.KERNEL

    # AzimuthConvention instances pass through
    assert ProgressLevel.convert(ProgressLevel.KERNEL) is ProgressLevel.KERNEL

    # Other types raise
    with pytest.raises(TypeError):
        ProgressLevel.convert(1.0)


def test_settings():
    """
    This test contains a few checks on settings.
    """
    assert isinstance(settings.azimuth_convention, AzimuthConvention)
    assert isinstance(settings.progress, ProgressLevel)
