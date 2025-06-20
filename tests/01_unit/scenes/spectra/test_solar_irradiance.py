import mitsuba as mi
import numpy as np
import pytest

import eradiate
from eradiate import unit_registry as ureg
from eradiate.converters import load_dataset
from eradiate.exceptions import DataError
from eradiate.scenes.spectra import SolarIrradianceSpectrum
from eradiate.spectral.index import SpectralIndex
from eradiate.test_tools.types import check_scene_element
from eradiate.units import PhysicalQuantity


@pytest.mark.parametrize(
    "tested, expected",
    [
        ({}, "SolarIrradianceSpectrum"),
        ({"dataset": "doesnt_exist"}, DataError),
        ({"dataset": "thuillier_2003"}, "SolarIrradianceSpectrum"),
        ({"dataset": "solar_irradiance/thuillier_2003.nc"}, "SolarIrradianceSpectrum"),
        (
            {"dataset": load_dataset("solar_irradiance/thuillier_2003.nc")},
            "SolarIrradianceSpectrum",
        ),
    ],
    ids=[
        "no_args",
        "dataset_doesnt_exist",
        "dataset_keyword",
        "dataset_path",
        "dataset_xarray",
    ],
)
def test_solar_irradiance_construct(mode_mono, tested, expected):
    if expected == "SolarIrradianceSpectrum":
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
        {"dataset": "thuillier_2003"},
    ],
    ids=[
        "no_args",
        "scale",
        "thuillier_spectrum",
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
