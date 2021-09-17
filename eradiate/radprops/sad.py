"""
HAPI wrapper.
"""

import hapi
import pint
import xarray as xr

# TODO: parse https://hitran.org/docs/iso-meta/ to automatically update isotopologues and corresponding abundances
ISOTOPOLOGUES = dict(
    H2O=[1, 2, 3, 4, 5, 6, 129],
    CO2=[7, 8, 9, 10, 11, 12, 13, 14, 121, 15, 120, 122],
    O3=[16, 17, 18, 19, 20],
)


def fetch_table(
    molecule: str, wavelength_min: pint.Quantity, wavelength_max: pint.Quantity
) -> None:
    hapi.fetch_by_ids(
        TableName="auto",
        iso_id_list=ISOTOPOLOGUES[molecule],
        numin=(1 / wavelength_max).m_as("cm^-1"),
        numax=(1 / wavelength_min).m_as("cm^-1"),
    )


def compute_absorption_cross_section(
    molecule: str, wavelength_min: pint.Quantity, wavelength_max: pint.Quantity
) -> xr.DataArray:
    fetch_table(
        molecule=molecule, wavelength_min=wavelength_min, wavelength_max=wavelength_max
    )
    pass
