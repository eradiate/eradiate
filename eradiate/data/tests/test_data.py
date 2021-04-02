import pytest

import eradiate.data as data


def test_open():
    # Check that loading directly using  explicit path works
    ds = data.open(path="spectra/solar_irradiance/blackbody_sun.nc")

    # Check that loading registered dataset works
    cat = "solar_irradiance_spectrum"
    for dataset_id in data.registered(cat):
        if dataset_id != "solid_2017":  # this dataset is not available in eradiate-data
            data.open(category=cat, id=dataset_id)

    # All spectral response function data sets can be opened
    paths = data.spectral_response_function._SpectralResponseFunctionGetter.PATHS

    for data_set_id in paths:
        data.open(category="spectral_response_function",
                  id=data_set_id)

    # Check that unknown category raises
    with pytest.raises(ValueError):
        ds = data.open("doesnt_exist", "blackbody_sun")

    # Check that unknown ID raises
    with pytest.raises(ValueError):
        ds = data.open("solar_irradiance_spectrum", "doesnt_exist")


def test_required_data_sets():
    # Check that data sets required for testing are found
    assert data.find("solar_irradiance_spectrum")["thuillier_2003"]
    assert data.find("solar_irradiance_spectrum")["blackbody_sun"]
    assert data.find("absorption_spectrum")["test"]
