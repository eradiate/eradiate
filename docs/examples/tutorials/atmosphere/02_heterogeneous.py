"""
Heterogeneous atmospheres
=========================
"""

# %%
# This tutorial illustrates how to create a heterogeneous atmosphere.
# We begin by setting the eradiate operational mode to the monochromatic mode
# and set the wavelength to 550 nm:

import eradiate
eradiate.set_mode("mono", wavelength=550.)

# %%
# This is required because heterogeneous atmospheres are objects whose internal
# state depend on the wavelength.
# We import matplotlib.pyplot because we will produce a plot later:

import matplotlib.pyplot as plt

# %%
# We also import the eradiate's unit registry as it will let us set the physical
# quantities parameters of the heterogeneous atmosphere in a convenient way:

from eradiate import unit_registry as ureg

# %%
# The heterogeneous atmospheres objects are defined by their underlying
# radiative properties profile.
# In this tutorial, we use
# :class:`~eradiate.radprops.rad_profile.US76ApproxRadProfile`,
# an approximated radiative properties profile
# corresponding to the
# :mod:`~eradiate.thermoprops.us76`
# atmosphere thermophysical model.

# %%
# Get started
# --------------------
# Create the heterogeneous atmosphere with default parameters (except
# ``profile``, that is required):

atmosphere = eradiate.scenes.atmosphere.HeterogeneousAtmosphere(
    profile=eradiate.radprops.US76ApproxRadProfile()
)

# %%
# The default
# :class:`~eradiate.radprops.rad_profile.US76ApproxRadProfile`
# consists of a 100-km high profile with 50 equally sized -- *i.e.* 2-km
# thick -- layers.
# In each of these layers, the albedo and the extinction coefficient are
# automatically computed in the appropriate pressure and temperature conditions
# corresponding to the :mod:`~eradiate.thermoprops.us76` atmosphere
# thermophysical model, and at the current wavelength.
# You can customise the radiative properties profile (see below).

# %%
# Set the atmosphere's dimensions
# -------------------------------
# Set the atmosphere's height using the ``height`` attribute of the radiative
# properties profile object:

atmosphere = eradiate.scenes.atmosphere.HeterogeneousAtmosphere(
    profile=eradiate.radprops.US76ApproxRadProfile(
        height=ureg.Quantity(120, "km")
    )
)

# %%
# The atmosphere's height automatically inherits the height of the underlying
# radiative properties profile object.
# In the example above, the atmosphere is 120 km high.
#
# Set the atmosphere's width using the ``width`` attribute:

atmosphere = eradiate.scenes.atmosphere.HeterogeneousAtmosphere(
    profile=eradiate.radprops.US76ApproxRadProfile(
        height=ureg.Quantity(120, "km")
    ),
    width=ureg.Quantity(500, "km")
)

# %%
# The atmosphere now has the dimensions: 500 x 500 x 120 km.
#
# .. note::
#    By default, the width of the heterogeneous atmosphere is set to 1000 km.
#    This width guarantees the absence of edge effects in simulations where the
#    sensor is a radiance meter array placed at the top of the atmosphere and
#    looking down with a zenith angle varying from 0 to 75°.
#    Above 75°, the measured values start to be influenced by the fact that the
#    horizontal size of the atmosphere is finite.
#    For accurate results above 75°, consider increasing the atmosphere width,
#    by setting the ``width`` attribute to a larger value.
#
# Inspect the atmosphere's dimensions
# -----------------------------------
# Check the atmosphere's dimensions using:

print(atmosphere.height.to("km"))

# %%

print(atmosphere.width.to("km"))

# %%
# Control the number of atmospheric layers
# ----------------------------------------
# You can control the division of the atmosphere into homogeneous horizontal
# layers, by setting the ``n_layers`` attribute of the radiative properties
# profile:

atmosphere = eradiate.scenes.atmosphere.HeterogeneousAtmosphere(
    profile=eradiate.radprops.US76ApproxRadProfile(
        height=ureg.Quantity(120, "km"),
        n_layers=120,
    )
)

# %%
# The atmosphere is now divided into 120 1-km-thick layers.
#
# .. note::
#    The atmosphere layers are of equal-thickness.
#
# Inspect the atmosphere's radiative properties
# ---------------------------------------------
# You can fetch the atmospheric radiative properties into a data set:

ds = atmosphere.profile.to_dataset()
ds

# %%
# You can use this data set to inspect the atmospheric radiative properties.
# For example, observe how the extinction coefficient (``sigma_t``) varies with
# altitude using:

ds.sigma_t.plot(yscale="log")
plt.show()
