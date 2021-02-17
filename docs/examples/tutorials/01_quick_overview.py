"""
Quick overview
==============
"""

# %%
# This tutorial gives a very quick introduction to Eradiate's main features. We
# assume here that you have successfully installed the package (see
# :ref:`sec-getting_started-install`). In this guide, we will see:
#
# * how to run a simulation using the one-dimensional solver command-line
#   interface;
# * how to run the same simulation using Eradiate's Python API.
#
# .. note:: This tutorial uses supplementary material:
#
#    * :download:`01_quick_overview_config.yml </examples/tutorials/01_quick_overview_config.yml>`

# %%
# Using the command-line
# ----------------------
# 
# A first entry point to Eradiate is the command-line interface to its solvers. 
# They are directly accessible from a terminal, provided that environment 
# variables are set correctly. 
# 
# For convenience, we will define a command to execute shell commands and 
# display their output.

import subprocess
def shell_command(args):
    print(subprocess.run(args.split(), capture_output=True).stdout.decode("utf-8"))

# %%
# For this example, we will use the ``ertonedim``` 
# application, which works on a one-dimensional scene consisting of a flat 
# surface and an atmosphere. We can first call it with its ``--help`` flag to 
# display the help text:

shell_command("ertonedim --help")

# %%
# This application is configured with a file which uses the 
# `YAML format <https://yaml.org/>`_. A sample configuration file is given 
# with this tutorial. We can visualise it:

shell_command("cat 01_quick_overview_config.yml")

# %%
# The file format is described in the :ref:`sec-user_guide-onedim_solver_app` 
# guide.
#
# In addition, ``ertonedim`` requires the user to specify output and plot 
# filename prefixes. These can be absolute or relative paths.

shell_command("ertonedim 01_quick_overview_config.yml ertonedim ertonedim")

# %%
# The application saves plots and results to the instructed location. 
#
# * Results are saved in the netCDF format and can be further processed using 
#   your tools of choice. One netCDF file is created for each measure specified 
#   in the configuration file.
# * One plot is generated for each measure specified in the configuration file. 
#   Note that Eradiate's applications do not allow for plot customisation; 
#   should that be done, the Python API should be preferred.

# %%
# Using the Python API
# --------------------
# 
# Eradiate also provides total access to its features through a complete and 
# documented API (see :ref:`sec-reference`). The ``ertonedim``
# command-line tool is a thin wrapper around the :class:`.OneDimSolverApp`
# class. We can easily reproduce the previous computation using it.
# 
# We start by loading our YAML configuration file into a dictionary:

import ruamel.yaml as yaml
with open("01_quick_overview_config.yml") as f:
    config = yaml.safe_load(f)
config

# %%
# We can see here that the contents of the configuration file are directly 
# translated into a configuration dictionary. This dictionary can be used as 
# the argument of the :class:`.OneDimSolverApp` constructor:

from eradiate.solvers.onedim.app import OneDimSolverApp
solver = OneDimSolverApp.from_dict(config)

# %%
# Launching the simulation is then as simple as calling the :meth:`.OneDimSolverApp.run` method:

solver.run()

# %%
# Unlike the command-line interface, a call to :meth:`~.OneDimSolverApp.run` 
# without argument will not create any result or plot files. Instead, results 
# are saved in the solver object's ``results`` dictionary. In this case, we 
# have a single measure, so ``solver.results`` contains a single element under 
# the key ``"toa_hsphere"`` which corresponds to the only measure defined in 
# the configuration file. This element is a :class:`xarray.Dataset`:

ds = solver.results["toa_hsphere"]
ds

# %%
# The usual xarray writing facilities can be used to save the results to a 
# netCDF file:

ds.to_netcdf("ertonedim.nc")

# %%
# Results can then easily be plotted using Eradiate's matplotlib/xarray-based 
# plotting facilities:

import matplotlib.pyplot as plt
import eradiate.plot as ertplt

fig = plt.gcf()
ds.brf.squeeze().ert.plot_pcolormesh_polar()
ertplt.remove_xylabels()
plt.show()

# %% 
# The figure can be saved using the usual matplotlib pattern:

fig.savefig("ertonedim_toa_brf.png")
plt.close()

# %% 
# From there, any post-processing or customised plotting can be applied.
#
# We finish this sequence with a quick cleanup:

import glob, os
for fname in  glob.glob("ertonedim*.png") + glob.glob("ertonedim*.nc"):
    os.remove(fname)
