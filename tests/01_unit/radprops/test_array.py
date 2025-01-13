import numpy as np
import pytest
import xarray as xr

import eradiate
from eradiate import unit_registry as ureg
from eradiate.radprops import ArrayRadProfile


@pytest.fixture
def test_data():
    data = np.tile(np.linspace(0, 9, 10).reshape(1, -1), (2, 1))
    w = [545, 565]  # in nanometer
    z = np.linspace(50, 950, 10)  # in meters
    return xr.DataArray(
        data=data,
        dims=["w", "z"],
        coords={
            "w": ("w", w, {"description": "wavelength", "units": "nm"}),
            "z": ("z", z, {"description": "altitude", "units": "m"}),
        },
        attrs={"description": "debug scattering coefficient", "units": "1/m"},
    )


def test_array(modes_all_mono, test_data):
    zgrid = eradiate.scenes.geometry.ZGrid(np.linspace(0, 1000, 11))
    si = eradiate.spectral.SpectralIndex.new(w=550 * ureg.nm)

    array_radprofile = ArrayRadProfile(
        has_absorption=True,
        has_scattering=True,
        sigma_a=test_data,
        sigma_s=test_data,
        interpolation_method="nearest",
    )

    sigma_s = array_radprofile.eval_sigma_s(si, zgrid)
    sigma_a = array_radprofile.eval_sigma_s(si, zgrid)

    assert np.all(sigma_a.m_as("1/m") == test_data.isel(w=0).values)
    assert np.all(sigma_s.m_as("1/m") == test_data.isel(w=0).values)


def test_resample_array(modes_all_mono, test_data):
    zgrid = eradiate.scenes.geometry.ZGrid(np.linspace(0, 1000, 21))
    si = eradiate.spectral.SpectralIndex.new(w=550 * ureg.nm)

    array_radprofile = ArrayRadProfile(
        has_absorption=True,
        has_scattering=True,
        sigma_a=test_data,
        sigma_s=test_data,
        interpolation_method="nearest",
        interpolation_kwargs={
            "fill_value": tuple(test_data.isel(w=0).values[[0, -1]]),
        },
    )

    sigma_s = array_radprofile.eval_sigma_s(si, zgrid)
    sigma_a = array_radprofile.eval_sigma_s(si, zgrid)

    gt = np.repeat(test_data.isel(w=0).values, 2)
    assert np.all(sigma_a.m_as("1/m") == gt)
    assert np.all(sigma_s.m_as("1/m") == gt)


def test_array_zeros(modes_all_mono, test_data):
    zgrid = eradiate.scenes.geometry.ZGrid(np.linspace(0, 1000, 11))
    si = eradiate.spectral.SpectralIndex.new(w=550 * ureg.nm)

    array_sigma_a = ArrayRadProfile(
        has_absorption=True,
        has_scattering=False,
        sigma_a=test_data,
        sigma_s=None,
        interpolation_method="nearest",
    )

    sigma_s = array_sigma_a.eval_sigma_s(si, zgrid)
    sigma_a = array_sigma_a.eval_sigma_a(si, zgrid)

    assert np.all(sigma_a.m_as("1/m") == test_data.isel(w=0).values)
    assert np.all(sigma_s.m_as("1/m") == np.zeros(10))

    array_sigma_s = ArrayRadProfile(
        has_absorption=False,
        has_scattering=True,
        sigma_a=None,
        sigma_s=test_data,
        interpolation_method="nearest",
    )

    sigma_s = array_sigma_s.eval_sigma_s(si, zgrid)
    sigma_a = array_sigma_s.eval_sigma_a(si, zgrid)

    assert np.all(sigma_a.m_as("1/m") == np.zeros(10))
    assert np.all(sigma_s.m_as("1/m") == test_data.isel(w=0).values)
