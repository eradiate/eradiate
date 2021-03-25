"""
1D-simulations on heterogeneous atmospheres
===========================================
"""

# %%
# This tutorial illustrates how to perform simulations on heterogeneous
# atmospheres using :class:`.OneDimSolverApp`.
#
# Setup
# -----
#
# Let us work in monochromatic mode at the wavelength of 579 nm.
# In order to focus on the atmosphere properties, we do not specify the
# ``illumination`` and ``surface`` fields of the application configuration.
# In other words, we use the default surface (lambertian surface with
# 50%-reflectance) and illumination (directional illumination with 0° - zenith
# and azimuth angles).
#
# For the heterogeneous atmosphere, we use the ``us76_approx`` radiative
# properties profile, with default parameters (see the
# :ref:`sec-atmosphere-heterogeneous` user guide page for more info).
#
# We will compute the top-of-atmosphere bi-directional reflectance factor in
# the principal plane so we use the ``distant`` measure and use a
# one-dimensional film resolution (corresponding dimension maps the zenith
# angle coordinate).
# We set the ``spp`` parameter value so that the results display reduced noise:

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from eradiate.solvers.onedim import OneDimSolverApp

config = {
    "mode": {
        "type": "mono_double",
        "wavelength": 579.
    },
    "atmosphere": {
        "type": "heterogeneous",
        "profile": {
            "type": "us76_approx"
        }
    },
    "measures": [{
        "type": "distant",
        "id": "toa_plane",
        "film_resolution": [90,1],
        "spp": 65536
    }]
}
app = OneDimSolverApp.from_dict(config)

# %%
# Run
# ---
# Let us run the application:

app.run()

# %%
# For future uses, we define a function that plots the results:


def visualise(results):
    plt.xticks(np.arange(-90, 91, step=15))
    results.brf.plot(x="vza", marker="o", markersize=2, linewidth=0.5)


# %%
# Let us visualise our results:

results_579 = app.results["toa_plane"]
visualise(results_579)

# %%
# We observe that the top-of-atmosphere bi-directional reflectance value
# cluster around 0.5, which is consistent with the value of the surface's
# reflectance.
# Indeed, the atmosphere we have created is almost purely scattering:

print(app.scene.atmosphere.profile.albedo)

# %%
# At higher zenith angles, we observe that the scene is less reflective, which
# makes sense since the atmosphere medium gets more optically thick in average
# when viewed from a higher zenith angle.
# For absolute zenith angle values larger than 75 degrees, we observe a sharp
# decrease in the BRF values, which correspond to edge effects due to the
# finite horizontal extent of the atmosphere.
#
# Run in the infrared
# -------------------
# Let us change the wavelength of our simulation to 1281 nm.
# Atmospheric absorption generally gets stronger in the infrared range of the
# spectrum, so we expect the atmosphere to be less transparent at this
# wavelength.

config = {
    "mode": {
        "type": "mono_double",
        "wavelength": 1281.
    },
    "atmosphere": {
        "type": "heterogeneous",
        "profile": {
            "type": "us76_approx"
        }
    },
    "measures": [{
        "type": "distant",
        "id": "toa_plane",
        "film_resolution": [90, 1],
        "spp": 65536
    }]
}
app = OneDimSolverApp.from_dict(config)

# %%
# Let us confirm our intuition:

print(app.scene.atmosphere.profile.albedo)

# %%
# Let us run the infrared simulation and compare our results with the
# 579nm results:

app.run()

# %%

results_1281 = app.results["toa_plane"]
ds = xr.merge([results_579, results_1281])
ds.brf.plot(x="vza", hue="wavelength", marker="o", markersize=2, linewidth=0.5)

# %%
# The TOA-BRF values are now around 0.3-0.4 instead of 0.5, because the
# atmosphere is more absorbing.
