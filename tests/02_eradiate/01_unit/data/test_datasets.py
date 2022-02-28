import pytest

from eradiate.data import data_store


@pytest.mark.parametrize(
    "path",
    [
        "chemistry/molecular_masses.nc",
        "spectra/solar_irradiance/blackbody_sun.nc",
        "spectra/solar_irradiance/thuillier_2003.nc",
        "spectra/solar_irradiance/meftah_2017.nc",
        "spectra/solar_irradiance/solid_2017_mean.nc",
        "spectra/solar_irradiance/whi_2008_time_period_1.nc",
        "thermoprops/afgl_1986-tropical.nc",
        "thermoprops/afgl_1986-midlatitude_summer.nc",
        "thermoprops/afgl_1986-midlatitude_winter.nc",
        "thermoprops/afgl_1986-subarctic_summer.nc",
        "thermoprops/afgl_1986-subarctic_winter.nc",
        "thermoprops/afgl_1986-us_standard.nc",
    ],
)
def test_required_data_sets(path):
    # Data sets required for testing are found
    assert data_store.fetch(path)
