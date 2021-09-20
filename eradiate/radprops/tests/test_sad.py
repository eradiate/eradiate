import os

import numpy as np
import xarray as xr

from eradiate import unit_registry as ureg
from eradiate.radprops.sad import (
    compute_absorption_cross_section,
    compute_absorption_cross_section_helper,
    ensure_array,
    make_absorption_cross_section_data_array,
    name_hitran_query,
    query_hitran,
    working_directory,
)


def test_name_hitran_query():
    """
    Returns a str.
    """
    assert isinstance(
        name_hitran_query(
            molecules=["H2O", "CO2"],
            wavenumber_min=ureg.Quantity(10000, "cm^-1"),
            wavenumber_max=ureg.Quantity(11000, "cm^-1"),
        ),
        str,
    )


def test_working_directory(tmpdir):
    """
    Changes working directory and returns to previous on exit.
    """
    prev_dir = os.getcwd()
    with working_directory(path=tmpdir):
        assert os.getcwd() == tmpdir
    assert os.getcwd() == prev_dir


def test_query_hitran(tmpdir):
    """
    Returned table matches a key in HAPI tables local (dynamic) database.
    """
    import hapi

    table = query_hitran(
        molecules=["CO2"],
        wavenumber_min=ureg.Quantity(8000, "cm^-1"),
        wavenumber_max=ureg.Quantity(8010, "cm^-1"),
        path=tmpdir,
    )

    assert table in hapi.getTableList()


def test_ensure_array():
    """
    Float-, list- and numpy.ndarray- quantities are converted to numpy.ndarray
    quantities and their units are left unchanged.
    """
    x = ureg.Quantity(5, "s")
    assert isinstance(ensure_array(x).magnitude, np.ndarray)
    assert ensure_array(x).units == x.units

    y = ureg.Quantity([5, 6], "N")
    assert isinstance(ensure_array(y).magnitude, np.ndarray)
    assert ensure_array(y).units == y.units

    z = ureg.Quantity(np.linspace(0, 1), "m")
    assert isinstance(ensure_array(z).magnitude, np.ndarray)
    assert ensure_array(z).units == z.units


def test_make_absorption_cross_section_data_array():
    """
    Returns a xarray.DataArray.
    """
    wavenumber = ureg.Quantity(np.linspace(4000, 5000))
    absorption_cross_section = ureg.Quantity(np.random.rand(len(wavenumber)), "m^2")
    pressure = ureg.Quantity(0.8, "atm")
    temperature = ureg.Quantity(280.15, "kelvin")

    da = make_absorption_cross_section_data_array(
        wavenumber=wavenumber,
        absorption_cross_section=absorption_cross_section,
        molecule="CH4",
        pressure=pressure,
        temperature=temperature,
    )

    assert isinstance(da, xr.DataArray)


def test_compute_absorption_cross_section_helper():
    """
    Returns a xarray.DataArray.
    """
    molecule = "CO2"
    wavenumber_min = ureg.Quantity(5000, "cm^-1")
    wavenumber_max = ureg.Quantity(5010, "cm^-1")

    da = compute_absorption_cross_section_helper(
        source_table=query_hitran(
            molecules=[molecule],
            wavenumber_min=wavenumber_min,
            wavenumber_max=wavenumber_max,
        ),
        molecule=molecule,
        wavenumber_min=wavenumber_min,
        wavenumber_max=wavenumber_max,
    )

    assert isinstance(da, xr.DataArray)


def test_compute_absorption_cross_section():
    molecule = "CO2"
    wavenumber_min = ureg.Quantity(5000, "cm^-1")
    wavenumber_max = ureg.Quantity(5010, "cm^-1")

    da = compute_absorption_cross_section(
        molecule=molecule,
        wavenumber_min=wavenumber_min,
        wavenumber_max=wavenumber_max,
    )

    assert isinstance(da, xr.DataArray)
