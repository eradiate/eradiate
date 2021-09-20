"""
Wrapper around hapi (https://github.com/hitranonline/hapi) to compute
monochromatic absorption cross section spectra for molecules.
"""
import contextlib
import os
import pathlib
import tempfile
import typing as t

import hapi
import numpy as np
import pint
import xarray as xr

from ..units import unit_registry as ureg

# TODO: parse https://hitran.org/docs/iso-meta/ to automatically update isotopologues and corresponding abundances
ISOTOPOLOGUE_IDS = dict(
    H2O=[1, 2, 3, 4, 5, 6, 129],
    CO2=[7, 8, 9, 10, 11, 12, 13, 14, 121, 15, 120, 122],
    O3=[16, 17, 18, 19, 20],
    N2O=[21, 22, 23, 24, 25],
    CO=[26, 27, 28, 29, 30, 31],
    CH4=[32, 33, 34, 35],
    O2=[36, 37, 38],
    NO=[39, 40, 41],
    SO2=[42, 43, 137, 138],
    NO2=[44, 130],
)


@contextlib.contextmanager
def working_directory(path: str) -> t.Iterator[None]:
    """
    Changes working directory and returns to previous on exit.
    """
    prev_cwd = pathlib.Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def name_hitran_query(
    molecules: t.List[str], wavenumber_min: pint.Quantity, wavenumber_max: pint.Quantity
) -> str:
    """
    Name a HITRAN query.
    """
    return (
        "_".join(molecules)
        + "-"
        + "_".join(
            [str(int(w.m_as("cm^-1"))) for w in [wavenumber_min, wavenumber_max]]
        )
    )


def query_hitran(
    molecules: t.List[str],
    wavenumber_min: pint.Quantity,
    wavenumber_max: pint.Quantity,
    path: t.Optional[str] = None,
) -> str:
    """
    Query HITRAN for given molecules and wavenumber range.

    Fetched data is written in a directory specified by the parameter ``path``.
    For each molecule, all isotopologues are selected.
    """
    # TODO: create one table per molecule?

    if path is None:
        path = tempfile.TemporaryDirectory().name

    isotopologue_ids = []
    for molecule in molecules:
        isotopologue_ids.extend(ISOTOPOLOGUE_IDS[molecule])

    table_name = name_hitran_query(
        molecules=molecules,
        wavenumber_min=wavenumber_min,
        wavenumber_max=wavenumber_max,
    )

    with working_directory(path):
        hapi.fetch_by_ids(
            TableName=table_name,
            iso_id_list=isotopologue_ids,
            numin=wavenumber_min.m_as("cm^-1"),
            numax=wavenumber_max.m_as("cm^-1"),
        )

    return table_name


def ensure_array(x: pint.Quantity) -> pint.Quantity:
    if not isinstance(x.magnitude, np.ndarray):
        return ureg.Quantity(np.atleast_1d(x.magnitude), x.units)
    else:
        return x


def make_absorption_cross_section_data_array(
    wavenumber: pint.Quantity,
    absorption_cross_section: pint.Quantity,
    molecule: str,
    pressure: pint.Quantity,
    temperature: pint.Quantity,
) -> xr.DataArray:
    """
    Make an absorption cross section spectrum data array.
    """
    pressure = ensure_array(pressure)
    temperature = ensure_array(temperature)
    return xr.DataArray(
        absorption_cross_section.magnitude.reshape(
            wavenumber.size, pressure.size, temperature.size
        ),
        dims=["w", "p", "t"],
        coords={
            "w": (
                "w",
                wavenumber.magnitude,
                dict(
                    standard_name="radiation_wavenumber",
                    long_name="wavenumber",
                    units=wavenumber.units,
                ),
            ),
            "p": (
                "p",
                pressure.magnitude,
                dict(
                    standard_name="air_pressure",
                    long_name="pressure",
                    units=pressure.units,
                ),
            ),
            "t": (
                "t",
                temperature.magnitude,
                dict(
                    standard_name="air_temperature",
                    long_name="temperature",
                    units=temperature.units,
                ),
            ),
        },
        attrs=dict(
            name=f"{molecule}_absorption_cross_section_spectrum",
            standard_name="absorption_cross_section",
            long_name="absorption cross section",
            units=absorption_cross_section.units,
            molecule=molecule,
        ),
    )


def compute_absorption_cross_section_helper(
    source_table: str,
    molecule: str,
    wavenumber_min: pint.Quantity,
    wavenumber_max: pint.Quantity,
    wavenumber_step: pint.Quantity = ureg.Quantity(0.01, "cm^-1"),
    truncation_distance_in_hwhm: int = 50,
    pressure: pint.Quantity = ureg.Quantity(101325, "Pa"),
    temperature: pint.Quantity = ureg.Quantity(296.0, "K"),
) -> xr.DataArray:
    """
    Compute the absorption cross section as a function of wavenumber for a
    given molecule, pressure and temperature (single value).
    """
    # It seems that absorptionCoefficient_Voigt will use the natural relative
    # abundances when mixing each molecule' isotopologues, so we do not
    # need to specify these relative abundances.
    nu, coef = hapi.absorptionCoefficient_Voigt(
        SourceTables=source_table,
        WavenumberRange=(wavenumber_min.m_as("cm^-1"), wavenumber_max.m_as("cm^-1")),
        WavenumberStep=wavenumber_step.m_as("cm^-1"),
        WavenumberWingHW=truncation_distance_in_hwhm,
        Environment=dict(p=pressure.m_as("atm"), T=temperature.m_as("kelvin")),
        GammaL="gamma_air",
        IntensityThreshold=0.0,
        HITRAN_units=True,  # that means the returned quantity is an absorption cross section with values in units of cm^2
    )

    return make_absorption_cross_section_data_array(
        wavenumber=ureg.Quantity(nu, "cm^-1"),
        absorption_cross_section=ureg.Quantity(coef, "cm^2"),
        molecule=molecule,
        pressure=pressure,
        temperature=temperature,
    )


def compute_absorption_cross_section(
    molecule: str,
    wavenumber_min: pint.Quantity,
    wavenumber_max: pint.Quantity,
    wavenumber_step: pint.Quantity = ureg.Quantity(0.01, "cm^-1"),
    truncation_distance_in_hwhm: int = 50,
    pressure: pint.Quantity = ureg.Quantity(101325.0, "Pa"),
    temperature: pint.Quantity = ureg.Quantity(296.0, "K"),
    hitran_data_dir: t.Optional[str] = None,
) -> xr.DataArray:
    """
    Compute the absorption cross section as a function of wavenumber, pressure
    and temperature for a given molecule.

    If ``pressure`` and ``temperature`` are arrays, we loop over the cartesian
    product of the two arrays.
    """
    # fetch spectroscopic parameters from HITRAN
    try:
        source_table = query_hitran(
            molecules=[molecule],
            wavenumber_min=wavenumber_min,
            wavenumber_max=wavenumber_max,
            path=hitran_data_dir,
        )
        data_sets_grid = []
        for p in ensure_array(pressure):
            inner_grid = []
            for t in ensure_array(temperature):
                _da = compute_absorption_cross_section_helper(
                    source_table=source_table,
                    molecule=molecule,
                    wavenumber_min=wavenumber_min,
                    wavenumber_max=wavenumber_max,
                    wavenumber_step=wavenumber_step,
                    truncation_distance_in_hwhm=truncation_distance_in_hwhm,
                    pressure=p,
                    temperature=t,
                )
                inner_grid.append(_da)
            data_sets_grid.append(inner_grid)

        da = xr.combine_nested(data_sets_grid, concat_dim=["p", "t"])

        # update attrs for concatenation dimensions (i.e. 'p' and 't')
        da.p.attrs.update(
            dict(
                standard_name="air_pressure", long_name="pressure", units=pressure.units
            )
        )
        da.t.attrs.update(
            standard_name="air_temperature",
            long_name="temperature",
            units=temperature.units,
        )

        # update general attributes
        da.attrs.update(_da.attrs)

        return da

    except Exception:
        # assume that molecule does not absorb in specified wavenumber range
        p = ensure_array(pressure)
        t = ensure_array(temperature)
        return make_absorption_cross_section_data_array(
            wavenumber=ureg.Quantity(
                np.array([wavenumber_max.m_as("cm^-1"), wavenumber_max.m_as("cm^-1")]),
                "cm^-1",
            ),
            absorption_cross_section=ureg.Quantity(
                np.zeros((2, len(p), len(t))), "m^2"
            ),
            molecule=molecule,
            pressure=p,
            temperature=t,
        )
