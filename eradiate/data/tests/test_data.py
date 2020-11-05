import pytest

import eradiate.data as data


def test_open():
    # Check that loading directly using  explicit path works
    ds = data.open(path="spectra/solar_irradiance/blackbody_sun.nc")

    # Check that loading registered dataset works
    ds = data.open("solar_irradiance_spectrum", "blackbody_sun")

    # Check that unknown category raises
    with pytest.raises(ValueError):
        ds = data.open("doesnt_exist", "blackbody_sun")

    # Check that unknown ID raises
    with pytest.raises(ValueError):
        ds = data.open("solar_irradiance_spectrum", "doesnt_exist")


def test_required_data_sets():
    # Check that data sets required for testing are found
    assert all(data.find("solar_irradiance_spectrum").values())
    assert data.find("absorption_spectrum")["test"]
