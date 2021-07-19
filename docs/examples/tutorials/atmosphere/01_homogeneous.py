"""
Homogeneous atmospheres
=======================
"""

# %%
# This tutorial illustrates how to create a homogeneous atmosphere.
# We begin by setting the eradiate operational mode to the monochromatic mode
# and set the wavelength to 550 nm:

import eradiate

eradiate.set_mode("mono")

# %%
# Get started
# -----------
# Create a homogeneous atmosphere with default parameters:

atmosphere = eradiate.scenes.atmosphere.HomogeneousAtmosphere()

# %%
# By default, the homogeneous atmosphere is 100 km high and has an absorption
# coefficient (``sigma_a``) set to zero.
# The value of ``sigma_s`` is computed by
# :func:`compute_sigma_s_air() <eradiate.radprops.rayleigh.compute_sigma_s_air>`.
# In the above example, the corresponding value is approximately
# :math:`1.15 \, 10^{-2} \, \mathrm{km}^{-1}`.
#
# .. note::
#    When the atmosphere's width is not specified,
#    :class:`~eradiate.scenes.atmosphere.HomogeneousAtmosphere` automatically
#    determines the width so that the atmosphere is optically thick in the
#    horizontal direction.

# %%
# Set the atmosphere's dimensions
# -------------------------------
# Set the atmosphere's height and width using the ``top`` and ``width``
# attributes, respectively:

from eradiate import unit_registry as ureg

atmosphere = eradiate.scenes.atmosphere.HomogeneousAtmosphere(
    top=ureg.Quantity(120, "km"), width=ureg.Quantity(500, "km")
)

# %%
# The atmosphere above now has the dimensions 500 x 500 x 120 km.

# %%
# Set the atmosphere's radiative properties
# -----------------------------------------
# Set the atmosphere's scattering and absorption coefficients using the
# ``sigma_s`` and ``sigma_a`` attributes, respectively:

atmosphere = eradiate.scenes.atmosphere.HomogeneousAtmosphere(
    sigma_s=ureg.Quantity(1e-3, "km^-1"), sigma_a=ureg.Quantity(1e-5, "km^-1")
)

# %%
# The above atmosphere is now characterised by
# :math:`k_{\mathrm{s}} = 10^{-3} \, \mathrm{km}^{-1}` and
# :math:`k_{\mathrm{a}} = 10^{-5} \, \mathrm{km}^{-1}`:
