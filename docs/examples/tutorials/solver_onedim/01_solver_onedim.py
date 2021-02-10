"""
One-dimensional solver application
==================================
"""

# %%
#
# This tutorial gives a practical overview of the one-dimensional solver
# application class :class:`.OneDimSolverApp`. This application simulates
# radiative transfer in a one-dimensional scene with an atmosphere.
#
# .. note:: This tutorial uses supplementary material:
#
#    * :download:`01_solver_onedim_config.yml </examples/tutorials/solver_onedim/01_solver_onedim_config.yml>`
#
# Instantiation and configuration
# -------------------------------
#
# We start by importing the :class:`.OneDimSolverApp` class:

from eradiate.solvers.onedim.app import OneDimSolverApp

# %%
# All Eradiate applications are configured with dictionaries. The following
# dictionary configures a :class:`.OneDimSolverApp` instance for a monochromatic
# simulation with a RPV surface with default parameters.

config = {
    "mode": {
        "type": "mono_double",
        "wavelength": 577.
    },
    "surface": {
        "type": "rpv"
    },
    "atmosphere": {
        "type": "homogeneous",
        "toa_altitude": 120.,
        "toa_altitude_units": "km",
        "sigma_s": 1.e-4
    },
    "illumination": {
        "type": "directional",
        "zenith": 30.,
        "azimuth": 0.
    },
    "measures": [{
        "type": "toa_hsphere",
        "spp": 32000,
        "zenith_res": 5.,
        "azimuth_res": 5.
    }]
}
app = OneDimSolverApp.from_dict(config)

# %%
# The content of each section of this configuration dictionary is presented in
# the :ref:`sec-user_guide-onedim_solver_app` guide.
#
# Configure using a YAML file
# ---------------------------
#
# The configuration dictionary can be loaded from a YAML. This is actually what
# the ``ertonedim`` CLI to :class:`.OneDimSolverApp` does. We can load the same
# configuration as before from the ``config.yml`` file using the ruamel library
# as follows:

import ruamel.yaml as yaml

with open("01_solver_onedim_config.yml", 'r') as f:
    yaml_config = yaml.safe_load(f)

app = OneDimSolverApp.from_dict(yaml_config)

# %%
# Running the simulation
# ----------------------
#
# Once our application object is initialised, we can start the simulation by
# calling the :meth:`.OneDimSolverApp.run` method. Progress is displayed during
# computation.

app.run()

# %%
# The application collects results in the ``results`` attribute of our
# application object. These results are stored as labeled multidimensional
# arrays (:class:`xarray.Dataset`) that allow for easy postprocessing, including
# exporting the results data to the NetCDF format.
#
# ``results`` is a dictionary which maps measure identifiers to the associated
# data set:

from pprint import pprint
pprint(app.results)

# %%
# In that case, we have a single measure ``toa_lo_hsphere`` for which we can
# easily display the data set:

ds = app.results["toa_hsphere"]
ds

# %%
# We can see that not only the TOA leaving radiance is saved to this array
# (the ``toa_lo_hsphere```  variable): it also contains the incoming irradiance
# on the scene (``irradiance``), as well as the  post-processed TOA BRDF and TOA
# BRF.
#
# Like any :class:`.Dataset`, this one can be sliced. We can for instance
# extract a 1D view for a particular azimuth angle:

ds.brf.sel(vaa=90.)

# %%
# Visualising the results
# -----------------------
#
# Using Eradiate's plotting helper, accessible with the xarray accessor ``ert``,
# we can very easily visualise the TOA BRF on a polar plot:

import matplotlib.pyplot as plt
import eradiate.util.plot as ertplt

brf = ds.brf
brf.squeeze().ert.plot_pcolormesh_polar()
ertplt.remove_xylabels()
plt.show()

# %%
# It would also be interesting to visualise a slice of this data set in the
# principal plane. For that purpose, Eradiate provides a convience function
# which slices a hemispherical data set to extract a view in the principal plane
# :func:`~eradiate.util.xarray.pplane`. This function produces a new data set
# which can then be plotted as any other xarray data set and we will access it
# using Eradiate's xarray accessor:

pplane_data = ds.brf.ert.extract_pplane()
pplane_data

# %%

pplane_data.plot()
plt.show()
