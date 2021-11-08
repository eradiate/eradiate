"""
Adjacency effects for padded canopies
=====================================
"""

# %%
# This tutorial/study documents the sensitivity of radiance computations to
# the finite size of a three dimensional canopy.
#
# Introduction
# ------------
# Since Eradiate does not include a truly periodic integrator at this time, we approximate
# periodicity for explicit canopies by cloning the canopy and padding the scene to a specified level.
# Due to memory constraints, the padded area is about an order of magnitude smaller than the
# width specified by the atmosphere. A rayleigh atmosphere in Eradiate is about 1000km wide in
# each direction, the padded canopy however is only 50km wide.
# The remaining surface area around the canopy is filled with a surface of lambertian reflectance.
# The purpose of this study is to assess the effect of the surrounding surface's reflectance on the
# recorded outgoing radiance and consequently the computed BRF. This study is performed at a
# wavelength of 400nm, to maximise the influence of the rayleigh scattering atmosphere.
#
# We begin by setting the eradiate operational mode to monochromatic double precision
# mode.

import eradiate

eradiate.set_mode("mono_double")

# %%
# Next we import some additional modules

import matplotlib.pyplot as plt
import numpy as np

# %%
# Setup
# -----
# Import the necessary components from Eradiate

from eradiate.experiments import Rami4ATMExperiment
import eradiate.scenes as sc
from eradiate.units import unit_registry as ureg

# %%
# Atmosphere
# ^^^^^^^^^^
# This study utilizes an atmosphere that only exhibits rayleigh scattering with no
# absorption and no aerosol layers. We use a classmethod constructor that yields an
# atmosphere implementing the AFGL 1986 standard atmosphere profile.
#
# Surface
# ^^^^^^^
# Furthermore the scene for this study will consist in a :class:`eradiate.scenes.surface._centralpatch`
# surface, with a lambertian reflectance typical for a canopy below it, while the reflectance of
# the surrounding area will be varied between 0.0 and 1.0.
#
# Illumination
# ^^^^^^^^^^^^
# The scene is illuminated by a :class:`eradiate.scenes.illumination._directional` with a zenith
# angle of 30 degrees.
#
# Measure
# ^^^^^^^
# We record the outgoing radiance (and compute the BRF) in 90 directions between :math:`\pm 90\deg`,
# targetting the original instance of the canopy only.
#
# Canopy
# ^^^^^^
# Finally the canopy in the center of the scene is a homogeneous discrete canopy, filling a cuboid
# volume spanning 25m on each side horizontall and being 2m high. The canopy LAI is 3.0 and the leaf radius
# is 0.1m. Leaf reflectance and transmittance are set to typical values for the observed spectral region.
#
# We set the number of samples per pixel and the size of the padded canopy in kilometers
spp = 1e4
patch_size = 50  # in km

# %%
# Now we assemble the scene, running the same simulation for different reflectance values
# in the lambertian surface surrounding the canopy.

results = []
for outer_refl in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    exp = Rami4ATMExperiment(
        surface=sc.surface.CentralPatchSurface(
            central_patch=sc.surface.LambertianSurface(
                reflectance=0.15, width=patch_size * ureg.km
            ),
            background_surface=sc.surface.LambertianSurface(reflectance=outer_refl),
        ),
        measures=sc.measure.MultiDistantMeasure.from_viewing_angles(
            azimuths=0,
            zeniths=np.arange(-89, 89, 2),
            spp=spp,
            spectral_cfg={"wavelengths": [400.0]},
            target={
                "type": "rectangle",
                "xmin": -25 * ureg.m,
                "xmax": 25 * ureg.m,
                "ymin": -25 * ureg.m,
                "ymax": 25 * ureg.m,
                "z": 2 * ureg.m,
            },
        ),
        illumination=sc.illumination.DirectionalIllumination(
            zenith=30 * ureg.deg, azimuth=0 * ureg.deg, irradiance=5.0
        ),
        atmosphere=sc.atmosphere.MolecularAtmosphere.ussa1976(has_absorption=False),
        canopy=sc.biosphere.DiscreteCanopy.homogeneous(
            lai=3.0,
            leaf_radius=0.1 * ureg.m,
            l_horizontal=25.0 * ureg.m,
            l_vertical=2.0 * ureg.m,
            leaf_transmittance=0.005,
            leaf_reflectance=0.05,
            padding=int(patch_size * 1000 / (50)),
        ),
    )
    exp.run()
    results.append(np.squeeze(exp.results["measure"]["brdf"]))

# %%
# To visualize the effect of the surrounding surface, we compute the relative difference for
# all results, using the case of r=0.0 as a reference

reldiff02 = abs(np.array(results[1]) - np.array(results[0])) / np.array(results[0])
reldiff04 = abs(np.array(results[2]) - np.array(results[0])) / np.array(results[0])
reldiff06 = abs(np.array(results[3]) - np.array(results[0])) / np.array(results[0])
reldiff08 = abs(np.array(results[4]) - np.array(results[0])) / np.array(results[0])
reldiff10 = abs(np.array(results[5]) - np.array(results[0])) / np.array(results[0])

# %%
# Plotting all results reveals the significant influence the surrounding surface has on the
# retrieved BRF value

fig, axes = plt.subplots(1, 1, figsize=(9, 6), dpi=120)
axes.plot(
    np.squeeze(exp.results["measure"]["vza"]),
    reldiff02,
    label="bg_refl = 0.2",
    marker=".",
)
axes.plot(
    np.squeeze(exp.results["measure"]["vza"]),
    reldiff04,
    label="bg_refl = 0.4",
    marker=".",
)
axes.plot(
    np.squeeze(exp.results["measure"]["vza"]),
    reldiff06,
    label="bg_refl = 0.6",
    marker=".",
)
axes.plot(
    np.squeeze(exp.results["measure"]["vza"]),
    reldiff08,
    label="bg_refl = 0.8",
    marker=".",
)
axes.plot(
    np.squeeze(exp.results["measure"]["vza"]),
    reldiff10,
    label="bg_refl = 1.0",
    marker=".",
)
axes.set_xlabel("VZA in degrees - principal plane")
axes.set_ylabel("abs rel diff in TOA reflectance to rho=0")
axes.set_title(
    f"Rami4ATM scene - heterogeneous atm\n400 nm - {patch_size}km canopy - SZA 30° - {spp} SPP"
)
axes.legend()
