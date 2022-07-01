import numpy as np
import pinttr
import pytest

from eradiate import unit_registry as ureg
from eradiate.contexts import KernelDictContext
from eradiate.scenes.illumination import DirectionalIllumination


def test_directional(mode_mono):
    # We need a default spectral config
    ctx = KernelDictContext()

    # Constructor
    d = DirectionalIllumination()
    assert d.kernel_dict(ctx).load() is not None

    # Check if a more detailed spec is valid
    d = DirectionalIllumination(irradiance={"type": "uniform", "value": 1.0})
    assert d.kernel_dict(ctx).load() is not None

    # Check if solar irradiance spectrum can be used
    d = DirectionalIllumination(irradiance={"type": "solar_irradiance"})
    assert d.kernel_dict(ctx).load() is not None

    # Check if specification from a float works
    d = DirectionalIllumination(irradiance=1.0)
    assert d.kernel_dict(ctx).load() is not None

    # Check if specification from a constant works
    d = DirectionalIllumination(irradiance=ureg.Quantity(1.0, "W/m^2/nm"))
    assert d.kernel_dict(ctx).load() is not None
    with pytest.raises(pinttr.exceptions.UnitsError):  # Wrong units
        DirectionalIllumination(irradiance=ureg.Quantity(1.0, "W/m^2/sr/nm"))


COS_PI_4 = np.sqrt(2) / 2


@pytest.mark.parametrize(
    "azimuth_convention, expected",
    [
        ("east_right", [0, -COS_PI_4, -COS_PI_4]),
        ("east_left", [0, COS_PI_4, -COS_PI_4]),
        ("north_right", [COS_PI_4, 0, -COS_PI_4]),
        ("north_left", [-COS_PI_4, 0, -COS_PI_4]),
        ("west_right", [0, COS_PI_4, -COS_PI_4]),
        ("west_left", [0, -COS_PI_4, -COS_PI_4]),
        ("south_right", [-COS_PI_4, 0, -COS_PI_4]),
        ("south_left", [COS_PI_4, 0, -COS_PI_4]),
    ],
)
def test_directional_azimuth_convention(mode_mono, azimuth_convention, expected):
    illumination = DirectionalIllumination(
        zenith=45 * ureg.deg,
        azimuth=90 * ureg.deg,
        azimuth_convention=azimuth_convention,
    )
    assert np.allclose(illumination.direction, expected), illumination.direction
