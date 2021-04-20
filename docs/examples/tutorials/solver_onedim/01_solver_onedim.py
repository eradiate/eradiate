"""
One-dimensional solver application
==================================
"""

# %%
#
# This tutorial introduces a classical workflow with one-dimensional scenes
# using the :class:`.OneDimSolverApp` class. It simulates radiative transfer in
# a one-dimensional scene with an atmosphere and a plane surface.
#
# Configuring the application
# ---------------------------
#
# Prior to any action, let's import the Eradiate top module and select an
# operational mode. We will run a monochromatic simulation at 577 nm.

import eradiate
eradiate.set_mode("mono_double")

# %%
# The first thing we have to do is configure the solver application
# which will run our simulations.
# :class:`.OneDimSolverApp` can be instantiated in a variety of ways. While we
# will go with the more classical way of working with it, we will see at the
# end of this tutorial that there are alternatives which you might feel more
# comfortable with.
#
# Let's first start by configuring our scene. :class:`.OneDimSolverApp`
# encapsulates an instance of :class:`.OneDimScene`, which itself holds a
# number of attributes.
#
# Surface
# """""""
#
# The first element of our scene which we will configure is the plane surface
# representing the ground. Surface classes are available from the
# :ref:`eradiate.scenes.surface <sec-reference-scenes-surface>` module and we
# can use any of our choosing. We will use a RPV surface [:class:`.RPVSurface`]
# with default parameters for this example:

surface = eradiate.scenes.surface.RPVSurface()
surface

# %%
# Atmosphere
# """"""""""
#
# The next scene element we need to include is an atmosphere. For this example,
# we will configure a basic heterogeneous atmosphere. Eradiate's atmosphere
# models are available from the
# :ref:`eradiate.scenes.atmosphere <sec-reference-scenes-atmosphere>` module.
# For simplicity, we will use a homogeneous atmosphere
# [:class:`.HomogeneousAtmosphere`]. Although of limited utility, this example
# allows us to focus on our workflow.
#
# .. seealso::
#
#    For a detailed introduction to atmosphere models, please refer to:
#
#    * user guide sections (modelling details):
#
#      * :ref:`sec-atmosphere-homogeneous`
#      * :ref:`sec-atmosphere-heterogeneous`
#
#    * tutorials (usage):
#
#       * :ref:`sphx_glr_examples_generated_tutorials_atmosphere_01_homogeneous.py`
#       * :ref:`sphx_glr_examples_generated_tutorials_atmosphere_02_heterogeneous.py`.
#
# We will configure our atmosphere as follows:
#
# * top-of-atmosphere altitude: 120 km
# * only Rayleigh scattering (no absorption), with scattering coefficient
#   :math:`\sigma_s = 10^{-4} m^{-1}`

# We must use Eradiate's unit registry to specify units
from eradiate import unit_registry as ureg

atmosphere = eradiate.scenes.atmosphere.HomogeneousAtmosphere(
    toa_altitude=120.0 * ureg.km,
    sigma_s=1e-4 * ureg.m ** -1,
)
atmosphere

# %%
# Illumination
# """"""""""""
#
# Next up is the illumination of our scene. We will use a directional
# illumination model [:class:`.DirectionalIllumination`], which consists of an
# infinitely distant collimated light source. This model is configured by
# the zenith and azimuth angles; we can also specify the illumination spectrum,
# which we will set to the default solar irradiance spectrum
# [:class:`.SolarIrradianceSpectrum`]:

illumination = eradiate.scenes.illumination.DirectionalIllumination(
    zenith=30.0 * ureg.deg,
    azimuth=0.0 * ureg.deg,
    irradiance=eradiate.scenes.spectra.SolarIrradianceSpectrum()
)
illumination

# %%
# Measure
# """""""
#
# The final piece of our scene is the observation. We are interested quantities
# derived from the top-of-atmosphere leaving radiance, *e.g.* the
# top-of-atmosphere bidirectional reflectance factor (BRF). Eradiate simulates
# this using the distant measure [:class:`.DistantMeasure`] which, in practice,
# records the radiance leaving the scene. By default, this measure records the
# radiance in the entire hemisphere, which is mapped to its rectangular *film*.
#
# By default, :class:`.DistantMeasure` records the leaving radiance in the
# entire hemisphere. Although this is of limited utility for a detailed
# analysis, it is interesting to examine roughly results and orient more
# detailed computations. We will therefore run a quick simulation with a
# low-resolution (32×32) film and a rather low number of samples per pixels
# (10000). We expect the result to be noisy, but we will see later how to run
# a quick simulation with more samples and less noise.
#
# In order to identify our results more easily afterwards, we will assign a
# ``toa_hsphere`` identifier (for "top-of-atmosphere, hemisphere") to our
# measure.

measure = eradiate.scenes.measure.DistantMeasure(
    id="toa_hsphere",
    film_resolution=(32, 32),
    spp=10000,
    spectral_cfg={"wavelength": 577.0},
)
measure

# %%
# Scene
# """""
#
# We have create all our scene elements: it's now time to assemble them together
# in a scene. We will, for that purpose, use the :class:`.OneDimScene` class,
# which is the scene class required by :class:`.OneDimSolverApp`:

scene = eradiate.solvers.onedim.OneDimScene(
    surface=surface,
    atmosphere=atmosphere,
    illumination=illumination,
    measures=measure,
)

# %%
# .. note:: If you look into detail, you will notice that Eradiate selected a
#    default Monte Carlo integration algorithm for us (see the
#    ``OneDimScene.integrator`` data member). We highly recommend that you do
#    not change that: the default integrator is optimised for the type of scene
#    we are dealing with.

# %%
# Application
# """""""""""
#
# Our final action consists in initialising an application to simulate radiative
# transfer on our freshly created scene. This is done by instantiating the
# :class:`.OneDimSolverApp` class. However, it is important, first, to select
# the operational mode in which we will run Eradiate.

app = eradiate.solvers.onedim.OneDimSolverApp(scene)

# %%
# We are now ready to run our simulation!

# %%
# Running the simulation
# ----------------------
#
# Once our application object is initialised, we can start the simulation by
# calling the :meth:`.OneDimSolverApp.run` method. Progress is displayed during
# computation. Since the configuration we are using is intended to producing
# quick and rough results, we expect a low computational time.

app.run()

# %%
# The results are stored in the ``results`` attribute of our application object.
# These results are stored as labeled multidimensional
# arrays (:class:`xarray.Dataset`) that allow for easy postprocessing, including
# exporting the results data to the NetCDF format.
#
# ``results`` is a dictionary which maps measure identifiers to the associated
# data set:

from pprint import pprint
pprint(app.results)

# %%
# In that case, we have a single measure ``toa_hsphere`` for which we can easily
# display the data set:

ds = app.results["toa_hsphere"]
ds

# %%
# We can see that not only the TOA leaving radiance is saved to this array
# (the ``lo``  variable): it also contains the incoming irradiance
# on the scene (``irradiance``), as well as the  post-processed TOA BRDF and TOA
# BRF.
#

# %%
# Visualising the results
# -----------------------
#
# Using xarray's plotting facilities, we can very easily visualise the TOA BRF:

import matplotlib.pyplot as plt

brf = ds.brf
brf.squeeze().plot()
plt.show()

# %%
# We can see that we have a "hot spot" in the back scattering direction (a
# distinctive feature of the RPV BRDF). If we reduce the amount of scattering in
# our atmosphere, the hot spot becomes sharper (see the bounds of the colour
# map):

scene = eradiate.solvers.onedim.OneDimScene(
    surface=surface,
    atmosphere=eradiate.scenes.atmosphere.HomogeneousAtmosphere(
        toa_altitude=120.0 * ureg.km,
        sigma_s=1e-6 * ureg.m ** -1,
    ),
    illumination=illumination,
    measures=eradiate.scenes.measure.DistantMeasure(
        id="toa_hsphere",
        film_resolution=(32, 32),
        spp=10000,
    ),
)
app = eradiate.solvers.onedim.OneDimSolverApp(scene)
app.run()
app.results["toa_hsphere"].brf.squeeze().plot()
plt.show()

# %%
# Refining the simulation
# -----------------------
#
# While the hemispherical views we have created so far are convenient for
# basic checks, they are fairly noisy and we usually prefer visualising results
# in the principal plane. For that purpose, we can configure our
# :class:`.DistantMeasure` by assigning 1 to the film height. Since we are
# significantly reducing the number of pixels, we can increase the number of
# samples we will take while (more or less) preserving our computational time.

scene = eradiate.solvers.onedim.OneDimScene(
    surface=surface,
    atmosphere=eradiate.scenes.atmosphere.HomogeneousAtmosphere(
        toa_altitude=120.0 * ureg.km,
        sigma_s=1e-4 * ureg.m ** -1,
    ),
    illumination=illumination,
    measures=eradiate.scenes.measure.DistantMeasure(
        id="toa_pplane",
        film_resolution=(32, 1),
        spp=1000000,
        spectral_cfg={"wavelength": 577.0},
    ),
)
app = eradiate.solvers.onedim.OneDimSolverApp(scene)
app.run()

# %%
# For plotting, we specify the viewing zenith angle (VZA) as our x coordinate.
# Note that this works only because the solver takes care of mapping the part of
# the principal plane located opposite the orientation of our measure to
# positive angle values; in our case, we end up with the positive half-plane
# corresponding to the back scattering direction.

app.results["toa_pplane"].brf.squeeze().plot(x="vza")
plt.show()

# %%
# Alternative object configuration methods
# ----------------------------------------
#
# So far, we have been using Eradiate's Python API exclusively. However, all
# scene elements and solver applications can also be configured using
# dictionaries. While it requires a little more knowledge of the API (your IDE
# will not assist you in writing configuration dictionaries), it also saves some
# importing. For instance, the scene used for the previous simulation can also
# be configured as follows:

scene = eradiate.solvers.onedim.OneDimScene(
    surface={
        "type": "rpv",
    },
    atmosphere={
        "type": "homogeneous",
        "toa_altitude": 120.0 * ureg.km,
        "sigma_s": 1e-4 * ureg.m ** -1,
    },
    illumination={
        "type": "directional",
        "zenith": 30.0 * ureg.deg,
        "azimuth": 0.0 * ureg.deg,
        "irradiance": {"type": "solar_irradiance"},
    },
    measures={
        "type": "distant",
        "id": "toa_hsphere",
        "film_resolution": (32, 1),
        "spp": 1000000,
        "spectral_cfg": {"wavelength": 577.0},
    },
)

# %%
# We can take this up one level and configure the whole scene with a single
# dictionary:

scene = eradiate.solvers.onedim.OneDimScene.from_dict({
    "surface": {
        "type": "rpv",
    },
    "atmosphere": {
        "type": "homogeneous",
        "toa_altitude": 120.0 * ureg.km,
        "sigma_s": 1e-4 * ureg.m ** -1,
    },
    "illumination": {
        "type": "directional",
        "zenith": 30.0 * ureg.deg,
        "azimuth": 0.0 * ureg.deg,
        "irradiance": {"type": "solar_irradiance"},
    },
    "measures": {
        "type": "distant",
        "id": "toa_hsphere",
        "film_resolution": (32, 1),
        "spp": 1000000,
        "spectral_cfg": {"wavelength": 577.0},
    },
})

# %%
# Finally, we can configure the entire application with a dictionary. This app
# configuration dictionary is basically the previous scene configuration, to
# which we add a mode configuration section used to force mode setup upon
# instantiation of the application object:

config = {
    "mode": "mono_double",
    "surface": {
        "type": "rpv",
    },
    "atmosphere": {
        "type": "homogeneous",
        "toa_altitude": 120.0 * ureg.km,
        "sigma_s": 1e-4 * ureg.m ** -1,
    },
    "illumination": {
        "type": "directional",
        "zenith": 30.0 * ureg.deg,
        "azimuth": 0.0 * ureg.deg,
        "irradiance": {"type": "solar_irradiance"},
    },
    "measures": {
        "type": "distant",
        "id": "toa_hsphere",
        "film_resolution": (32, 1),
        "spp": 1000000,
        "spectral_cfg": {"wavelength": 577.0},
    },
}
app = eradiate.solvers.onedim.OneDimSolverApp.from_dict(config)
config

# %%
# Ultimately, we can transform this dictionary into a YAML file. The only issue
# here is that we used Pint units for some of the entries; Eradiate covers this
# case and will automatically attach to a scalar field the units defined a
# corresponding field with the ``_units``.
#
# .. note:: While we use YAML in this example, nothing prevents the use of
#    another language to generate your dictionaries.
#
# Let's load the following file
# [:download:`01_solver_onedim_config.yml </examples/tutorials/solver_onedim/01_solver_onedim_config.yml>`]:
#
# .. include:: /examples/tutorials/solver_onedim/01_solver_onedim_config.yml
#    :literal:

import ruamel.yaml as yaml

with open("01_solver_onedim_config.yml", 'r') as f:
    yaml_config = yaml.safe_load(f)
yaml_config

# %%
app = eradiate.solvers.onedim.OneDimSolverApp.from_dict(yaml_config)
