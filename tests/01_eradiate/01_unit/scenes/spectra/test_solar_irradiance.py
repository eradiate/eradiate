import mitsuba as mi
import numpy as np
import pytest

import eradiate
from eradiate import unit_registry as ureg
from eradiate.exceptions import DataError
from eradiate.scenes.spectra import SolarIrradianceSpectrum
from eradiate.spectral.index import SpectralIndex
from eradiate.test_tools.types import check_scene_element
from eradiate.units import PhysicalQuantity


@pytest.mark.parametrize(
    "tested, expected",
    [
        ({}, SolarIrradianceSpectrum()),
        ({"dataset": "doesnt_exist"}, DataError),
    ],
    ids=[
        "no_args",
        "dataset_doesnt_exist",
    ],
)
def test_solar_irradiance_construct(modes_all, tested, expected):
    if isinstance(expected, SolarIrradianceSpectrum):
        s = SolarIrradianceSpectrum(**tested)
        assert s.quantity is PhysicalQuantity.IRRADIANCE

    elif issubclass(expected, Exception):
        with pytest.raises(expected):
            SolarIrradianceSpectrum(**tested)

    else:
        raise RuntimeError


@pytest.mark.parametrize(
    "tested",
    [
        {},
        {"scale": 2.0},
        {"dataset": "solid_2017-mean"},
    ],
    ids=[
        "no_args",
        "scale",
        "solid_spectrum",
    ],
)
def test_solar_irradiance_kernel_dict(mode_mono, tested):
    s = SolarIrradianceSpectrum(**tested)
    check_scene_element(s, mi.Texture)


def test_solar_irradiance_eval(modes_all_double):
    # Irradiance is correctly interpolated in mono mode
    s = SolarIrradianceSpectrum(dataset="thuillier_2003")

    si = SpectralIndex.new(w=550.0 * ureg.nm)
    expected = 1.87938 * ureg.W / ureg.m**2 / ureg.nm  # computed manually

    assert np.allclose(s.eval(si), expected)

    # Eval raises out of the supported spectral range
    if eradiate.mode().is_mono:
        with pytest.raises(ValueError):
            s.eval(SpectralIndex.new(w=1.0 * ureg.km))

    elif eradiate.mode().is_ckd:
        pass

    else:
        assert False


def test_solar_irradiance_scale(mode_mono):
    s = SolarIrradianceSpectrum(dataset="thuillier_2003")

    # We can scale the spectrum using a float
    s_scaled_float = SolarIrradianceSpectrum(dataset="thuillier_2003", scale=10.0)
    assert (
        s_scaled_float.eval_mono(550.0 * ureg.nm) == s.eval_mono(550.0 * ureg.nm) * 10.0
    )


def test_solar_irradiance_datetime(mode_mono):
    s = SolarIrradianceSpectrum(dataset="thuillier_2003")

    # We can also use a datetime to scale the spectrum
    s_scaled_datetime = SolarIrradianceSpectrum(
        dataset="thuillier_2003", datetime="2021-11-18"
    )
    assert np.isclose(
        s_scaled_datetime.eval_mono(550.0 * ureg.nm),
        s.eval_mono(550.0 * ureg.nm) / 0.98854537**2,
    )
