import pytest

import eradiate.data as data


def test_open():
    # Loading directly using  explicit path works
    ds = data.open(path="spectra/solar_irradiance/blackbody_sun.nc")

    # Loading registered dataset works
    cat = "solar_irradiance_spectrum"
    for dataset_id in data.registered(cat):
        if dataset_id != "solid_2017":  # this dataset is not available in eradiate-data
            data.open(category=cat, id=dataset_id)

    # All spectral response function data sets can be opened
    paths = data.spectral_response_function._SpectralResponseFunctionGetter.PATHS

    for data_set_id in paths:
        data.open(category="spectral_response_function", id=data_set_id)

    # Unknown category raises
    with pytest.raises(ValueError):
        data.open("doesnt_exist", "blackbody_sun")

    # Unknown ID raises
    with pytest.raises(ValueError):
        data.open("solar_irradiance_spectrum", "doesnt_exist")


def test_required_data_sets():
    # Data sets required for testing are found
    assert data.find(category="chemistry")["molecular_masses"]
    assert data.find(category="solar_irradiance_spectrum")["blackbody_sun"]
    assert data.find(category="solar_irradiance_spectrum")["thuillier_2003"]
    assert data.find(category="solar_irradiance_spectrum")["meftah_2017"]
    assert data.find(category="solar_irradiance_spectrum")["solid_2017_mean"]
    assert data.find(category="solar_irradiance_spectrum")["whi_2008"]
    assert data.find(category="thermoprops_profiles")["afgl1986-tropical"]
    assert data.find(category="thermoprops_profiles")["afgl1986-midlatitude_summer"]
    assert data.find(category="thermoprops_profiles")["afgl1986-midlatitude_winter"]
    assert data.find(category="thermoprops_profiles")["afgl1986-subarctic_summer"]
    assert data.find(category="thermoprops_profiles")["afgl1986-subarctic_winter"]
    assert data.find(category="thermoprops_profiles")["afgl1986-us_standard"]
