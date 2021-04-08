"""
Visualise a atmosphere thermophysical properties data set
=========================================================
"""

# %%
# This tutorial illustrates how to visualise an atmosphere thermophysical
# properties data set.
# In order to make the plots, we import the matplotlib library:

import matplotlib.pyplot as plt

# %%
# In this example, we will use the atmosphere thermophysical properties data set
# corresponding to the U.S. Standard atmosphere 1976 model.
# In order to create the data set, we import the eradiate package:

import eradiate

# %%
# First, we make the atmosphere thermophysical properties data set:

ds = eradiate.thermoprops.us76.make_profile()

# %%
# Then, we can visualise the data set by calling the
# :meth:`~xarray.DataArray.plot` method on the data variable that we want to
# plot.
# For example, plot the air pressure (data variable ``p``) with respect to
# altitude (coordinate ``z_layer``) using:

ds.p.plot(y="z_layer")
plt.xscale("log")
plt.show()

# %%
# Plot the air number density with respect to altitude:

ds.n.plot(y="z_layer")
plt.xscale("log")
plt.show()

# %%
# Plot the air temperature with respect to altitude:

ds.t.plot(y="z_layer")
plt.show()

# %%
# Plot the mixing ratios with respect to the altitude:

ds.mr.plot(y="z_layer", hue="species")
plt.xscale("log")
plt.show()
