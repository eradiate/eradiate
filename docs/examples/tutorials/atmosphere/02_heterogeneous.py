"""
Heterogeneous atmospheres
=========================
"""

# %%
# This tutorial illustrates how to create a heterogeneous atmosphere.
#
# We begin by setting the eradiate operational mode to the monochromatic mode
# and set the wavelength to 550 nm:

import eradiate

eradiate.set_mode("mono")
spectral_ctx = eradiate.contexts.SpectralContext.new(wavelength=550.0)

# %%
# This is required because heterogeneous atmospheres are objects whose internal
# state depend on the wavelength.
# We import ``matplotlib.pyplot`` because we will produce a plot later:

import matplotlib.pyplot as plt

# %%
# We also import eradiate's unit registry as it will let us set the physical
# quantities parameters of the heterogeneous atmosphere in a convenient way:

from eradiate import unit_registry as ureg


# %%
# Get started
# --------------------
# Create a heterogeneous atmosphere with:

atmosphere = eradiate.scenes.atmosphere.HeterogeneousAtmosphere()

# %%
# The default constructor uses a default radiative properties profile
# that extends from 0 to 100 km with 100 layers -- 1 km thick each.
# This default radiative properties profile corresponds to a purely scattering
# atmosphere (i.e., no absorption) with common scattering coefficient values
# for a *U.S. Standard Atmosphere, 1976* thermophysical properties profile and
# a 550 nm wavelength value.

# %%
# Set the radiative properties profile
# ------------------------------------
# You can set the atmosphere radiative properties profile.
# Let us use the
# :class:`~eradiate.radprops.rad_profile.US76ApproxRadProfile`,
# an approximated radiative properties profile corresponding to the
# :mod:`~eradiate.thermoprops.us76` atmosphere thermophysical model.

atmosphere = eradiate.scenes.atmosphere.HeterogeneousAtmosphere(
    profile=eradiate.radprops.US76ApproxRadProfile()
)

# %%
# The default :class:`~eradiate.radprops.rad_profile.US76ApproxRadProfile`
# consists of a 100-km high profile with 50 equally sized -- *i.e.* 2-km
# thick -- layers.
# In each of these layers, the albedo and the extinction coefficient are
# automatically computed in the appropriate pressure and temperature conditions
# corresponding to the :mod:`~eradiate.thermoprops.us76` atmosphere
# thermophysical model.
#
# Refer to the :mod:`~eradiate.radprops` module documentation for a list of
# available radiative properties profiles.

# %%
# Set the atmosphere's level altitude mesh
# ----------------------------------------
# Set the atmosphere's level altitude mesh using the ``levels`` attribute of
# the radiative properties profile object:

import numpy as np

atmosphere = eradiate.scenes.atmosphere.HeterogeneousAtmosphere(
    profile=eradiate.radprops.US76ApproxRadProfile(
        levels=ureg.Quantity(np.linspace(0, 120, 61), "km")
    )
)

# %%
# This level altitude mesh defines 60 layers, each 2 km-thick.
# The atmosphere's vertical extension automatically inherits that of the
# underlying radiative properties profile object so that, in the example above,
# the atmosphere also extends from 0 to 120 km.
# Use the ``levels`` parameter to control the atmosphere vertical extension as
# well as the number of atmospheric layers and their thicknesses.
#
# For example, you can define a completely arbitrary level altitude mesh with
# four layers of varying thicknesses:

atmosphere = eradiate.scenes.atmosphere.HeterogeneousAtmosphere(
    profile=eradiate.radprops.US76ApproxRadProfile(
        levels=ureg.Quantity(np.array([0.0, 4.0, 12.0, 64.0, 100.0]), "km")
    )
)

# %%
# Set the atmosphere's width
# --------------------------
# Set the atmosphere's width using the ``width`` attribute:

atmosphere = eradiate.scenes.atmosphere.HeterogeneousAtmosphere(
    profile=eradiate.radprops.US76ApproxRadProfile(
        levels=ureg.Quantity(np.linspace(0, 120, 61), "km")
    ),
    width=ureg.Quantity(500, "km"),
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
#    For accurate results above 75°, consider increasing the atmosphere width.
#
# Inspect the atmosphere's dimensions
# -----------------------------------
# Check the atmosphere's dimensions using:

print(atmosphere.height.to("km"))

# %%

print(atmosphere.width.to("km"))


# %%
# Inspect the atmosphere's radiative properties
# ---------------------------------------------
# You can fetch the atmospheric radiative properties profile into a data set:

spectral_ctx = eradiate.contexts.SpectralContext.new()
ds = atmosphere.profile.to_dataset(spectral_ctx)
ds

# %%
# You can use this data set to inspect the atmospheric radiative properties.
# For example, observe how the extinction coefficient (``sigma_t``) varies with
# altitude using:

ds.sigma_t.plot(yscale="log")
plt.show()
