from pathlib import Path

import eradiate.data as data


def test_get():
    # We expect the data to load successfully from the hard drive
    ds = data.get("spectra/blackbody_sun.nc")
    ds = data.get("spectra/thuillier_2003.nc")
    ds = data.get("spectra/whi_2008_time_period_1.nc")
    ds = data.get("spectra/whi_2008_time_period_2.nc")
    ds = data.get("spectra/whi_2008_time_period_3.nc")

    # We check that using the solar irradiance data map also works
    ds = data.get(data.SOLAR_IRRADIANCE_SPECTRA["blackbody_sun"])
    ds = data.get(data.SOLAR_IRRADIANCE_SPECTRA["thuillier_2003"])
    ds = data.get(data.SOLAR_IRRADIANCE_SPECTRA["whi_2008"])
    ds = data.get(data.SOLAR_IRRADIANCE_SPECTRA["whi_2008_1"])
    ds = data.get(data.SOLAR_IRRADIANCE_SPECTRA["whi_2008_2"])
    ds = data.get(data.SOLAR_IRRADIANCE_SPECTRA["whi_2008_3"])


def test_data():
    # Test if data sets advertised as supported exist
    for key, fname_relative in data.SOLAR_IRRADIANCE_SPECTRA.items():
        fname = Path(data.presolver.resolve(fname_relative))
        assert fname.is_file()
