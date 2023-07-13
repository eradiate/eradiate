from __future__ import annotations

from pathlib import Path

import pandas as pd
import pint

import eradiate

from ..exceptions import DataError


def locate_absorption_data(
    codename: str, mode: str, wavelength_range: pint.Quantity
) -> list[Path]:
    """Locate absorption data for a given wavelength range.

    Returns
    -------
    list of pathlib.Path
        List of paths to absorption datasets.

    Raises
    ------
    ValueError
        If no data is available for the requested wavelength range.

    Notes
    -----
    Datasets that have not already been downloaded will be downloaded from the
    remote data store.
    """
    base_path = f"spectra/absorption/{mode}/{codename}"
    try:
        index_path = eradiate.data.data_store.fetch(f"{base_path}/index.csv")
        df = pd.read_csv(index_path)

        # select all files whose associated wavelength range overlaps with the
        # requested wavelength range
        wl_min_units, wl_max_units = None, None
        wl_min_column = None
        wl_max_column = None
        for c in df.columns:
            if c.startswith("wl_min"):
                wl_min_column = c
                # units are enclosed in []
                wl_min_units = c.split("[")[1].split("]")[0]

            if c.startswith("wl_max"):
                wl_max_column = c
                # units are enclosed in []
                wl_max_units = c.split("[")[1].split("]")[0]

        if wl_min_units is None or wl_max_units is None:
            raise DataError(
                "index file must contain at least one column whose name starts "
                "with 'wl_min' and one column whose name starts with 'wl_max'"
            )

        requested_min = wavelength_range.min().m_as(wl_min_units)
        requested_max = wavelength_range.max().m_as(wl_max_units)

        data_wl_min = df[wl_min_column].min()
        data_wl_max = df[wl_max_column].max()

        if requested_min < data_wl_min or requested_max > data_wl_max:
            raise ValueError(
                f"requested wavelength range {wavelength_range} is outside "
                f"the range of the available data ({data_wl_min} {wl_min_units}"
                f" - {data_wl_max} {wl_max_units})"
            )

        filenames = (
            df["filename"]
            .where(
                (df[wl_max_column] >= requested_min)
                & (df[wl_min_column] <= requested_max)
            )
            .dropna()
            .values.tolist()
        )

        if len(filenames) == 0:
            raise ValueError(f"no files found for wavelength range {wavelength_range}")

        return [
            eradiate.data.data_store.fetch(f"{base_path}/{filename}")
            for filename in filenames
        ]

    except DataError:
        # assume no index file is required because there is only one file
        return [eradiate.data.data_store.fetch(f"{base_path}/{codename}.nc")]
