"""
RAMI solver application
=======================

This tutorial gives a practical overview of the RAMI solver application class
:class:`.RamiSolverApp`. This application simulates radiative transfer on scenes
typically used in the RAMI benchmarking exercise.

.. warning:: This example demonstrates features under development.
"""

# %%
# Instantiation and configuration
# -------------------------------
#
# We will use the Python API. We start by selecting an operational mode. We
# select the single-precision monochromatic mode and set the wavelength to 550
# nm:

import eradiate

eradiate.set_mode("mono")

# %%
# We also assign an alias to the unit registry:

ureg = eradiate.unit_registry

# %%
# Now, we can define our scene. For this example, we will use the
# `RAMI 3 homogeneous discrete canopy scene`_,
# which can be created using the :class:`.DiscreteCanopy` class. We use typical
# values (in the red optical domain) to set it up:
#
# .. _RAMI 3 homogeneous discrete canopy scene: https://rami-benchmark.jrc.ec.europa.eu/_www/phase/phase_exp.php?strTag=level3&strNext=meas&strPhase=RAMI3&strTagValue=HOM_SOL_DIS

canopy = eradiate.scenes.biosphere.DiscreteCanopy.homogeneous(
    lai=3.0,
    leaf_radius=0.1 * ureg.m,
    l_vertical=2.0 * ureg.m,
    l_horizontal=30.0 * ureg.m,
    leaf_reflectance=0.0546,
    leaf_transmittance=0.0149,
)
canopy

# %%
# We will then create a Lambertian surface with reflectance 0.127. We do not
# care about the size of this surface: the solver will automatically match it
# with the size of the canopy.

surface = eradiate.scenes.surface.LambertianSurface(reflectance=0.127)
surface

# %%
# Next up is the illumination. We use the directional illumination with the
# default solar irradiance spectrum and select arbitrary illumination angles:

illumination = eradiate.scenes.illumination.DirectionalIllumination(
    zenith=30.0, azimuth=45.0
)
illumination

# %%
# The last part we need to add to our scene is a measure. This solver currently
# supports only a distant measure, which records the radiance leaving the scene
# over the entire hemisphere or in a plane (we will see below how to control
# this behavior).
#
# All measures in Eradiate use an underlying kernel component called *sensor*.
# All sensors record their results to a *film*, which consists of a set of
# pixels arranged on a Cartesian grid, and :class:`.DistantReflectanceMeasure`
# features a ``film_resolution`` parameter which controls the number of pixels
# on the film. It should be a 2-element sequence of integers, the first one
# being the width of the film, and the second being the height. We will
# configure our measure to produce a global view of the scene's bidirectional
# reflectance factor over the whole hemisphere and will set the film to a coarse
# resolution of 32 × 32.
#
# When the simulation starts, the Monte Carlo algorithms traces a number rays
# per film pixel controlled by the ``spp`` (for *samples per pixel*) parameter.
# The higher the number of SPP, the lower the variance in results, and the
# higher the computational time. We will set ``spp`` to a rather low value of
# 1000 to quickly get a global view of our BRF.
#
# One last important thing to know about measures is that they must have a
# unique identifier: it will be used to reference the results. We will assign
# the identifier `toa_brf` to our measure.
#
# The :class:`.DistantReflectanceMeasure` class has other parameters, but we do
# not need to modify them for now.

measure = eradiate.scenes.measure.DistantReflectanceMeasure(
    id="toa_brf",
    film_resolution=(32, 32),
    spp=1000,
    spectral_cfg={"wavelengths": [550.0]},
)
measure

# %%
# We are now ready to instantiate our scene. We just have to assemble the
# elements we have created so far:

scene = eradiate.solvers.rami.RamiScene(
    canopy=canopy,
    surface=surface,
    illumination=illumination,
    measures=measure,
)

# %%
# We can now create a solver application object with our scene:

app = eradiate.solvers.rami.RamiSolverApp(scene)

# %%
# Running the simulation
# ----------------------
#
# Once our application object is initialised, we can start the simulation by
# calling the :meth:`.RamiSolverApp.run` method. Progress is displayed during
# computation.

app.run()

# %%
# This method call also includes result post-processing. In addition to
# computing the radiance leaving the scene, the solver computes derived
# quantities such among which the scene's BRF. Results are stored in the
# ``app.results`` dictionary, whose keys are measure identifiers. Each item is
# a xarray :class:`~xarray.Dataset`, which holds one data variable per computed
# quantity:

app.results["toa_brf"]

# %%
# This :class:`~xarray.Dataset` has 5 dimension coordinates:
#
# * ``sza``, ``saa``: Sun angles, which characterize the illumination;
# * ``x``, ``y``: film pixel coordinates, which can be mapped to the viewing
#   angles;
# * ``wavelength``: the wavelength at which we just ran the simulation.
#
# In addition, a pair of non-dimension coordinates (``vza``, ``vaa``) is
# defined: they map pixel coordinates to the corresponding viewing angles.
#
# Data variables are all associated to the 5 dimension coordinates, except for
# the irradiance, which is a property of the scene and is not a function of the
# viewing configuration.
#
# Plotting results
# ----------------
#
# We can now view our results. Since we have here a xarray
# :class:`~xarray.Dataset`, we can use xarray's built-in visualisation
# capabilities:

import matplotlib.pyplot as plt

ds = app.results["toa_brf"]
ds.brf.squeeze().plot.imshow()
plt.show()

# %%
# .. note:: We plot those data against *film coordinates*. If you are interested
#    in plotting those data against viewing angle coordinates, please refer to
#    :ref:`sphx_glr_examples_generated_tutorials_data_01_polar_plot.py`.

# %%
# Note that we used the :meth:`~xarray.DataArray.squeeze` method to drop all
# scalar dimensions: the :meth:`~xarray.plot.imshow` method is indeed meant to
# be used on 2D data arrays. On this image, we can see that we indeed have a
# hotspot in the direction of the illumination, as we would expect. This is good
# news! However, the results are quite noisy.
#
# More samples in the principal plane
# -----------------------------------
#
# We are now going to run a new simulation with a different measure
# configuration. We will still use the :class:`.DistantReflectanceMeasure`
# element, but we will set its film height to 1 pixel. This will trigger a
# special behaviour where the underlying kernel sensor will sample directions
# only on a plane defined by the measure's ``orientation`` parameter. We will set
# ``orientation`` to the same value as the Sun azimuth angle: this will make our
# sensor record radiance in the principal plane.
#
# Since we will be taking samples for much fewer pixels, we can increase the
# SPP at a reasonable computational cost. Let's increase it up to 100000 to
# reduce that noise.

measure = eradiate.scenes.measure.DistantReflectanceMeasure(
    id="toa_brf", film_resolution=(32, 1), spp=100000, orientation=45.0
)

# %%
# We can now redefine a scene and run our computation:

scene = eradiate.solvers.rami.RamiScene(
    canopy=canopy,
    surface=surface,
    illumination=illumination,
    measures=measure,
)
app = eradiate.solvers.rami.RamiSolverApp(scene)
app.run()

# %%
# Let's inspect our results:

ds = app.results["toa_brf"]
ds

# %%
# We still have the same data variables and coordinates; however, if we take a
# look at coordinate values, we can see that the only dimension coordinate with
# non-unit length is `x`: our data is one-dimensional, which is what we expect
# from a computation in the principal plane!

da = ds.brf.squeeze()
da

# %%
# We can now plot our data, again using xarray's plotting facilities. The
# mapping of the viewing zenith angle is such that the direction pointed at by
# the measure's ``orientation`` parameter, which we set to be the backscattering
# orientation, is located to the right of the plot, where the VZA takes positive
# values.

da.plot(x="vza", marker=".", linestyle="--")
plt.show()
