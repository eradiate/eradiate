"""
Custom solar irradiance spectrum
================================
"""

# %%
# This tutorial illustrates how to add your own solar irradiance spectrum
# dataset from a data file that includes wavelength and solar spectral
# irradiance values, and use it within Eradiate.

# %%
# Create the dataset object
# -------------------------
#
# Say your custom solar irradiance spectrum data is saved in a
# `comma-separated values file <https://en.wikipedia.org/wiki/Comma-separated_values>`_
# called ``my_data.csv`` with wavelength values in the first column and solar
# spectral irradiance values in the second column.
# You would like to be able to use it within Eradiate.
# For that, you need to convert this CSV file into a NetCDF file with the right
# format for Eradiate.
# Here is how that can be achieved.
# First, we read our data into a :class:`pandas.DataFrame`:

import pandas as pd

df = pd.read_csv("02_custom_solar_irradiance_data.csv", header=1, names=["w", "ssi"])

# %%
# Next, we create a :class:`xarray.Dataset` object with the values of wavelength
# and solar spectral irradiance that we have just read.
# We create the data variable ``ssi`` (for *solar spectral irradiance*) with
# the dimension ``w`` (for *wavelength*) and the required metadata (including
# units).
# The dataset must have two coordinates: ``w`` and ``t`` (for *time*) with
# corresponding metadata.
# Our data does not include the time dimension so we just set the time
# coordinate to an empty array with 0 dimension.
# Finally, set the attributes (``attrs``) of our dataset, including a nice
# title!
# Refer to the `CF-1.8 convention document <https://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#description-of-file-contents>`_
# for the meaning of these attributes.

import datetime
import numpy as np
import xarray as xr

ds = xr.Dataset(
    data_vars={
        "ssi": ("w", df.ssi.values, {
            "standard_name": "solar_irradiance_per_unit_wavelength",
            "long_name": "solar spectral irradiance",
            "units": "W/m^2/nm"})
    },
    coords={
        "w": ("w", df.w.values, {
            "standard_name": "wavelength",
            "long_name": "wavelength",
            "units": "nm"}),
        "t": ("t", np.empty(0), {
            "standard_name": "time",
            "long_name": "time"})
    },
    attrs={
        "title": "My awesome dataset",
        "convention": "CF-1.8",
        "history": f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - data set creation - path/to/my_script.py",
        "source": "My custom observation data",
        "references": "My article, doi:10.1000/xyz123"
    }
)
ds

# %%
# Validate the dataset's metadata
# -------------------------------
#
# Before going any further, we must validate the metadata of the dataset we've
# created. The metadata of Eradiate's solar irradiance spectrum datasets must
# follow a specification defined by the ``ssi_dataset_spec`` variable:

from eradiate.xarray.metadata import DatasetSpec, VarSpec

# Define solar irradiance spectra dataset specifications
ssi_dataset_spec = DatasetSpec(
    title="Untitled solar irradiance spectrum",
    convention="CF-1.8",
    source="Unknown",
    history="Unknown",
    references="Unknown",
    var_specs={
        "ssi": VarSpec(
            standard_name="solar_irradiance_per_unit_wavelength",
            units="W/m^2/nm",
            long_name="solar spectral irradiance"
        )
    },
    coord_specs="solar_irradiance_spectrum"
)

# %%
# We can validate our dataset's metadata by running:

ds.ert.validate_metadata(ssi_dataset_spec)

# %%
# .. note:: Loading any Eradiate submodule will automatically set up Eradiate's
#    xarray accessors.

# %%
# Normalisation
# ^^^^^^^^^^^^^
#
# A lazier way to define the dataset is to omit the ``standard_name`` and
# ``long_name`` metadata fields and *normalise* the dataset's metadata, which
# will add the missing fields:

ds = xr.Dataset(
    data_vars={
        "ssi": ("w", df.ssi.values, {"units": "W/m^2/nm"})},
    coords={
        "w": ("w", df.w.values, {"units": "nm"}),
        "t": ("t", np.empty(0))
    },
    attrs={
        "title": "My awesome dataset",
        "convention": "CF-1.8",
        "history": f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - data set creation - this_toolchain version 0.1",
        "source": "Observation data from this instrument",
        "references": "My article, doi:10.1000/xyz123"
    }
)
ds.ert.normalize_metadata(ssi_dataset_spec)
ds

# %%
# This is not recommended but we could even have omitted the ``units`` field
# because our ``ssi`` and ``w`` have the same units as the default units in
# ``ssi_dataset_spec``:

ssi_dataset_spec.var_specs["ssi"].schema["units"]

# %%
ssi_dataset_spec.coord_specs["w"].schema["units"]

# %%
# Optional attributes
# ^^^^^^^^^^^^^^^^^^^
#
# If your data comes from observation, you may want to indicate the observation
# start date and end date in the dataset attributes.
# This information is useful to indicate in what range of dates the dataset can
# be considered as an accurate representation of the actual solar irradiance
# spectrum.
# Use the ``obs_start`` and ``obs_end`` attributes to indicate those dates.
# If applicable, use the ``url`` attributes to indicate the url where the raw data
# can be downloaded.
# Use the ``comment`` attribute to add miscellaneous information, *e.g.* some
# processing that you performed on the raw data.
# Finally, you can create any other attribute that you wish, provided its name
# does not conflict with an existing attribute.

ds.attrs["obs_start"] = str(datetime.date(1992, 3, 24))
ds.attrs["obs_end"] = str(datetime.date(1992, 4, 2))
ds.attrs["url"] = (
    f"https://this.is.where.the.data.can.be.downloaded "
    f"(last accessed on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})"
)
ds.attrs["comment"] = "the original data was re-binned in larger 2nm-wide wavelength bins."
ds.attrs["_my_attribute"] = "Other info"
ds

# %%
# Save the dataset to a netcdf file
# ---------------------------------
#
# We are not done yet!
# To register your dataset in the list of available datasets of Eradiate, you
# must first save the dataset to a NetCDF file.
# It is recommended to save the dataset in
# ``$ERADIATE_DIR/resources/data/spectra/solar_irradiance``:

import os
ds.to_netcdf(os.path.join(
    os.environ["ERADIATE_DIR"],
    "resources/data/spectra/solar_irradiance",
    "my_awesome_dataset.nc"
))

# %%
# If you list the files in that folder, you should see your newly added NetCDF
# file next to the Eradiate's predefined solar irradiance spectrum datasets:

os.listdir(os.path.join(os.environ["ERADIATE_DIR"], "resources/data/spectra/solar_irradiance"))

# %%
# Use your own solar irradiance spectrum dataset
# ----------------------------------------------

from eradiate.data.solar_irradiance_spectra import _SolarIrradianceGetter
_SolarIrradianceGetter.PATHS["my_awesome_dataset"] = "spectra/solar_irradiance/my_awesome_dataset.nc"

# %%
# Now, you are able to use your own solar irradiance spectrum within Eradiate.
# The following code illustrates how to define a directional illumination scene
# element based on the custom solar irradiance spectrum:

from eradiate.scenes.illumination import DirectionalIllumination
from eradiate.scenes.spectra import SolarIrradianceSpectrum
DirectionalIllumination(irradiance=SolarIrradianceSpectrum(dataset="my_awesome_dataset"))
