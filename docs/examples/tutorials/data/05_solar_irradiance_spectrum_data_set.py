"""
Solar irradiance spectrum data sets
===================================
"""

# %%
# .. admonition:: What you will learn
#
#   In this tutorial, you will learn:
#
#   * how to list available :ref:`solar irradiance spectrum data sets<sec-user_guide-data-solar_irradiance_spectrum_data_sets>`
#   * how to visualise the data.

# %%
# Let us start by importing the
# `matplotlib <https://matplotlib.org/>`_
# and eradiate packages.
# We import matplotlib in order to later produce the plots, and we import
# eradiate to list and open the available data sets.

import matplotlib.pyplot as plt
import eradiate

# %%
# List available solar irradiance spectrum datasets
# -------------------------------------------------
# Find the list of available datasets by running:

eradiate.data.registered(category="solar_irradiance_spectrum")


# %%
# Visualise a solar irradiance spectrum
# -------------------------------------
# Let us choose the ``thuillier_2003`` data set.
# Open the data set using the :func:`eradiate.data.open` method.
# Solar irradiance spectra data sets belong to the category
# ``solar_irradiance_spectrum``.

ds = eradiate.data.open(category="solar_irradiance_spectrum", id="thuillier_2003")

# %%
# You can visualise the solar irradiance spectrum corresponding to this dataset
# with a simple call to the ``plot()`` method of the ``ssi`` (solar spectral irradiance) data variable:

ds.ssi.plot(linewidth=0.3)
plt.show()

# %%
# Visualise a solar irradiance spectrum with a time dimension
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# If the solar irradiance spectrum dataset includes a non-empty ``t`` (time)
# coordinate, you must **select** the date for which you want to visualise the
# spectrum:

path = eradiate.path_resolver.resolve("tests/spectra/solar_irradiance/solid_2017.nc")
ds = eradiate.data.open(path=path)
ds.ssi.sel(t="2014-12-31").plot(linewidth=0.3)
plt.show()

# %%
# .. tip::
#   Find out whether a solar irradiance spectrum includes a non-empty time
#   coordinate by printing the content of ``solar_spectrum.dims``; in the
#   dictionary that is printed, the value for the ``t`` key indicates the number
#   of time stamps. If the number is 0, the time coordinate is empty. If the
#   number is larger than zero, the dataset supports selection by time stamps.
