"""
Heterogeneous atmospheres
=========================
"""

# %%
# This tutorial illustrates how to create a heterogeneous atmosphere.

import eradiate
eradiate.set_mode("mono", wavelength=550.)

from eradiate.scenes.atmosphere import HeterogeneousAtmosphere

# %%
# The heterogeneous atmospheres objects are defined by their underlying
# radiative properties profile.
# In this tutorial, we use an approximated radiative properties profile
# corresponding to the US76 thermophysical properties profile:

from eradiate.radprops import US76ApproxRadProfile

# %%
# Get started
# --------------------
# Create a US76-like atmosphere with default parameters:

atmosphere = HeterogeneousAtmosphere(
    profile=US76ApproxRadProfile()
)

# %%
# Set the atmosphere's dimensions
# -------------------------------
# Set the atmosphere's height using the ``height`` attribute of the radiative
# properties profile object:

from eradiate import unit_registry as ureg

atmosphere = HeterogeneousAtmosphere(
    profile=US76ApproxRadProfile(
        height=ureg.Quantity(120, "km")
    )
)

# %%
# The atmosphere's height automatically inherits the height of the radiative
# properties profile object.
# In the example above, the atmosphere is 120 km high.
#
# Set the atmosphere's width using the ``width`` attribute:

atmosphere = HeterogeneousAtmosphere(
    profile=US76ApproxRadProfile(
        height=ureg.Quantity(120, "km")
    ),
    width=ureg.Quantity(2000, "km")
)

# %%
# The atmosphere now has the dimensions: 2000 x 2000 x 120 km.
#
# Inspect the atmosphere's dimensions
# -----------------------------------
# Check the atmosphere's dimensions using:

print("height: ", atmosphere.height.to("km"))
print("width: ", atmosphere.width.to("km"))

# %%
# Control the number of atmospheric layers
# ----------------------------------------
# You can control the division of the atmosphere into homogeneous horizontal
# layers, by setting the ``n_layers`` attribute of the radiative properties
# profile:

atmosphere = HeterogeneousAtmosphere(
    profile=US76ApproxRadProfile(
        height=ureg.Quantity(120, "km"),
        n_layers=120,
    ),
    width=ureg.Quantity(2000, "km")
)

# %%
# The atmosphere is now divided into 120 1-km-thick layers.
#
# Inspect the atmosphere's radiative properties
# ---------------------------------------------
# You can fetch the atmospheric radiative properties into a data set:

ds = atmosphere.profile.to_dataset()

# %%
# You can use this data set to inspect the atmospheric radiative properties.
# For example, observe how the extinction coefficient (``sigma_t``) varies with
# altitude using:

import matplotlib.pyplot as plt

plt.yscale("log")
ds.sigma_t.plot()
