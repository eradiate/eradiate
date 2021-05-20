"""
Polar plot in angular coordinates
=================================
"""

# %%
# This tutorial shows how to plot reflectance data in polar
# coordinates.

# %%
# We start with a basic 1D simulation with a default
# atmospheric profile, a RPV surface (which will
# backscatter) and illumination at 45° zenith and azimuth.
# We have a distant measure which records radiance
# leaving the scene and is used to compute
# top-of-atmosphere reflectance.

import eradiate

eradiate.set_mode("mono_double")

app = eradiate.solvers.onedim.OneDimSolverApp(
    scene={
        "atmosphere": {"type": "heterogeneous"},
        "illumination": {
            "type": "directional",
            "zenith": 45.0,
            "azimuth": 45.0,
        },
        "surface": {"type": "rpv"},
        "measures": [
            {
                "type": "distant",
                "film_resolution": (64, 64),
                "spp": 1000,
            }
        ],
    }
)

app.run()

# %%
# Our simulation is complete. We can now visualise our
# output data against film coordinates:

import matplotlib.pyplot as plt

app.results["measure"].brf.squeeze().plot()
plt.show()

# %%
# Reflectance data is usually plotted against viewing
# angle coordinates. Our film coordinate data set
# cannot be easily plotted using this framework: we first
# have to convert our data array to viewing angle
# coordinates. Eradiate provides a support function to
# automate this process:

import numpy as np

# Define angular grid
thetas = np.deg2rad(np.arange(0.0, 85.01, 5.0))
phis = np.deg2rad(np.arange(0.0, 360.01, 5.0))

# Interpolate data
brf_angular = eradiate.xarray.interp.film_to_angular(
    app.results["measure"].brf, thetas, phis
)

# %%
# The resulting data array uses a structured angle grid as
# coordinates:

brf_angular

# %%
# Plotting our data in polar coordinates is then  straightforward:

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "polar"})
thetav, phiv = np.meshgrid(brf_angular.theta, brf_angular.phi)
plt.pcolormesh(phiv, thetav, brf_angular, shading="nearest")
yticks = np.deg2rad(np.arange(30.0, 90.01, 30.0))
ax.set_yticks(yticks)
ax.set_yticklabels([f"{np.rad2deg(y):1.0f}°" for y in yticks])
ax.grid()
plt.colorbar(label="TOA BRF")
plt.show()

# %%
# We can see here our backscattering hotspot at (45°, 45°), as we were expecting.

# sphinx_gallery_thumbnail_number = 2
