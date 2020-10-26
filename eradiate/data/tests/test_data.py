import pytest

import eradiate.data as data


def test_load():
    # Check that loading directly using  explicit path works
    ds = data.load(path="spectra/blackbody_sun.nc")

    # Check that loading registered dataset works
    ds = data.load("solar_irradiance_spectrum", "blackbody_sun")

    # Check that unknown category raises
    with pytest.raises(ValueError):
        ds = data.load("doesnt_exist", "blackbody_sun")

    # Check that unknown ID raises
    with pytest.raises(ValueError):
        ds = data.load("solar_irradiance_spectrum", "doesnt_exist")
