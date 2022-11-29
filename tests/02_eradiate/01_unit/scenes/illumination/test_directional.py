import mitsuba as mi
import numpy as np
import pytest

from eradiate import unit_registry as ureg
from eradiate.contexts import KernelDictContext
from eradiate.scenes.core import traverse
from eradiate.scenes.illumination import DirectionalIllumination
from eradiate.scenes.spectra import SolarIrradianceSpectrum, UniformSpectrum


@pytest.mark.parametrize(
    "kwargs, expected_irradiance_type",
    [({}, SolarIrradianceSpectrum), ({"irradiance": 1.0}, UniformSpectrum)],
    ids=["noargs", "from_scalar"],
)
def test_directional_construct(modes_all, kwargs, expected_irradiance_type):
    # Construction without argument succeeds
    d = DirectionalIllumination(**kwargs)
    assert d
    assert isinstance(d.irradiance, expected_irradiance_type)


def test_directional_kernel_dict(modes_all_double):
    # The associated kernel dict is correctly formed and can be loaded
    d = DirectionalIllumination()
    template, _ = traverse(d)
    ctx = KernelDictContext()
    kernel_dict = template.render(ctx)
    assert isinstance(mi.load_dict(kernel_dict), mi.Emitter)


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
