"""
Homogeneous atmospheres
=======================
"""

# %%
# This tutorial illustrates how to create a homogeneous atmosphere.

import eradiate
eradiate.set_mode("mono", wavelength=550.)

from eradiate.scenes.atmosphere import HomogeneousAtmosphere

# %%
# Get started
# -----------
# Create a homogeneous atmosphere with default parameters:

atmosphere = HomogeneousAtmosphere()

# %%
# Set the atmosphere's dimensions
# -------------------------------
# Set the atmosphere's height and width using the ``toa_altitude`` and
# ``width`` attributes, respectively:

from eradiate import unit_registry as ureg

atmosphere = HomogeneousAtmosphere(
    toa_altitude = ureg.Quantity(120, "km"),
    width = ureg.Quantity(1000, "km")
)

# %%
# The atmosphere now has the dimensions 1000 x 1000 x 120 km.

# %%
# Set the atmosphere's radiative properties
# -----------------------------------------
# Set the atmosphere's scattering and absorption coefficients using the
# ``sigma_s`` and ``sigma_a`` attributes, respectively:

atmosphere = HomogeneousAtmosphere(
    sigma_s = ureg.Quantity(1e-3, "km^-1"),
    sigma_a = ureg.Quantity(1e-5, "km^-1")
)

# %%
# The atmosphere now is characterised by
# :math:`k_{\mathrm{s}} = 10^{-3} \, \mathrm{km}^{-1}` and
# :math:`k_{\mathrm{a}} = 10^{-5} \, \mathrm{km}^{-1}`:
