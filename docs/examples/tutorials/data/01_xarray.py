"""
xarray module tutorial
======================
"""

# %%
# .. warning:: Outdated content, update required

# %%
# In this tutorial, we will learn how to use Eradiate's xarray extensions. 
# In addition to this introduction, it is recommended to refer to the 
# :mod:`technical documentation <eradiate.util.xarray>`. We assume here that you
# are familiar with xarray's 
# `basics concepts <http://xarray.pydata.org/en/stable/quick-overview.html>`_ 
# and `terminology <http://xarray.pydata.org/en/stable/terminology.html>`_.
#
# Our approach to data handling
# -----------------------------
#
# Eradiate makes extensive use of the xarray library to propose a neat data
# workflow. Thanks to xarray, we have access to a convenient and efficient data
# handling framework with built-in advanced data arithmetics, convenient data
# browsing and selection facilities, a plotting framework, metadata handling and
# many I/O formats.
# 
# Eradiate extends xarray with the recommended
# `accessor method <http://xarray.pydata.org/en/stable/internals.html#extending-xarray>`_.
# Its :class:`data array <.EradiateDataArrayAccessor>` and
# :class:`dataset <.EradiateDatasetAccessor>` accessors are automatically
# created upon importing any Eradiate module or submodule and respectively
# accessed using the ``DataArray.ert`` and ``Dataset.ert`` members.
# These accessors wrap other facilities used for various purposes, *e.g.* data
# creation, slicing, metadata validation and normalisation, plotting, etc.
#
# Data array slicing and visualisation
# ------------------------------------
#
# In order to demonstrate these facilities, let's first start by creating some
# data. We'll run a simple 1D simulation with a Rayleigh atmosphere and an
# illumination zenith angle of 30°. We keep the default atmosphere and surface
# (Lambertian) and increase the number of samples per pixel to a high value:

from eradiate.solvers.onedim import OneDimSolverApp

app = OneDimSolverApp.from_dict({
    "mode": {
        "type": "mono",
    },
    "atmosphere": {
        "type": "heterogeneous",
        "profile": {"type": "us76_approx"},
        "width": 1e5,
        "width_units": "km"
    },
    "surface": {
        "type": "rpv"
    },
    "illumination": {
        "type": "directional",
        "zenith": 30.,
        "zenith_units": "deg"
    },
    "measures": [{
        "type": "distant",
        "spp": int(1e4)
    }]
})
app.run()

# %%
# We can now access the only measure data we have requested:

sample_data = next(iter(app.results.values()))
sample_data

# %%
# This dataset has 5 coordinate variables and 4 data variables. As we can see
# from the visualisation, we can see that the dataset, data variables and
# coordinate variables are documented. We can very easily visualise the
# top-of-atmosphere BRF over the hemisphere using the data array accessor:

import matplotlib.pyplot as plt
import eradiate.plot as ertplt

brf = sample_data.brf

brf.squeeze().ert.plot_pcolormesh_polar() # note that we must first eliminate the excess dimensions with squeeze()
ertplt.remove_xylabels() # remove X and Y axis labels
plt.show()

# %%
# Some people will argue that this view is not very comprehensive and that
# they'd prefer to use a principal plane view for their analysis. We can easily
# extract a principal plane view from our TOA BRF hemispherical data and plot it
# using the regular xarray plotting tools:

brf.ert.extract_pplane().plot()
plt.show()

# %%
# But we can also extract a plane view for any azimuth value:

# Here we must specify the azimuth and zenith coordinates used to extract this plane view
brf.ert.extract_plane(phi=90., theta_dim="vza", phi_dim="vaa").plot()
plt.show()

# %%
# Advanced: Faceting test
# -----------------------
#
# The following example shows how the :meth:`.plot_pcolormesh_polar` method
# allows for `faceting <http://xarray.pydata.org/en/stable/plotting.html#faceting>`_
# just like xarray's ``pcolormesh()`` method:

# Faceting test
import numpy as np
import xarray as xr
import pint

ureg = pint.UnitRegistry()

res_r = 0.1
rs = np.arange(0, 1, res_r)
res_theta = 20
thetas = np.arange(0, 360, res_theta)

scales = np.array([1, 2, -2], dtype=float)
offsets = np.array([0, -1, 1], dtype=float)

data_base = np.random.random((len(rs), len(thetas)))
data = np.zeros(data_base.shape + scales.shape + offsets.shape)

for i_scale, scale in enumerate(scales):
    for i_offset, offset in enumerate(offsets):
        data[:, :, i_scale, i_offset] = data_base * scale + offset

my_da = xr.DataArray(
    data,
    coords=(rs, thetas, scales, offsets),
    dims=("r", "theta", "scale", "offset"),
)
my_da.attrs["long_name"] = "some quantity"
my_da.attrs["units"] = "dimensionless"
my_da.theta.attrs["units"] = "deg"
my_da.theta.attrs["long_name"] = "azimuth angle"

# Create facet view plots
g = my_da.sel(offset=slice(-1, 1)).ert.plot_pcolormesh_polar(col="scale", row="offset")
# Remove axis labels for all plots
ertplt.remove_xylabels(g)
# Remove axis ticks from all plots except lower left one
ertplt.remove_xyticks(ertplt.get_axes_from_facet_grid(g, exclude="lower_left"))
plt.show()
plt.close()

# %%
# Advanced: Metadata specification, validation and normalisation
# --------------------------------------------------------------
#
# Eradiate ships with a metadata specification, validation and normalisation
# framework. It relies on xarray's
# `metadata system <http://xarray.pydata.org/en/stable/faq.html#what-is-your-approach-to-metadata>`_
# and is based on a small subset of the metadata defined by the
# `CF convention 1.8 <http://www.cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html>`_.
# Eradiate provides facilities to validate the metadata of xarray data objects.
#
# The first type of data whose metadata can be checked are coordinate variables.
# Required metadata fields are ``standard_name`` and ``long_name``. Defining the
# ``units`` field is generally recommended, but it can be unset for fields which
# technically do not have units such as string-based label fields. The metadata
# specification of coordinate variables is stored by
# :class:`.CoordSpec` objects. The following piece of code defines the
# specification for an outgoing zenith angle coordinate variable:

from eradiate.xarray.metadata import CoordSpec, CoordSpecRegistry, VarSpec

theta_o_spec = CoordSpec("outgoing_zenith_angle", "deg", "outgoing zenith angle")

# %%
# For convenience, Eradiate provides :class:`.CoordSpec` collections usually
# used to parametrise data arrays and/or datasets. They are registered as a
# class member of the :class:`.CoordSpecRegistry` class. We can check the list
# of registered coordinate specification sets:

list(CoordSpecRegistry.registry_collections.keys())

# %%
# We can then call the :meth:`.CoordSpecRegistry.get_collection` method to
# retrieve the one we are interested in:

coord_specs_angular_obs = CoordSpecRegistry.get_collection("angular_observation")
coord_specs_angular_obs

# %%
# The ``angular_observation`` collection specifies the five coordinates used to
# describe satellite observations.
#
# By themselves, :class:`.CoordSpec` objects are not of much use.
# :class:`.CoordSpec` and all other classes deriving from :class:`.DataSpec` are
# meant to be used for metadata validation using Eradiate's metadata validation
# and normalisation components. Let's apply this to construct a data array meant
# to store BRF data. For that purpose, we'll use a :class:`VarSpec` object. Our
# variable will be parametrised by angular observation coordinates (see
# :ref`_sec-user_guide-data_guide-working_angular_data` for more information
# about angular data) and the :class:`.VarSpec` constructor allows to pass
# coordinate specification collection identifiers:

var_spec = VarSpec(
    standard_name="toa_brf",
    long_name="TOA bidirectional reflectance factor",
    units="",  # BRF values are dimensionless
    coord_specs="angular_observation"
)
var_spec.coord_specs

# %%
# Now, we will create a data array using the :func:`.make_dataarray` function.
# Note that this function behaves in a way similar to xarray's
# :class:`~xarray.DataArray`. In order to apply our metadata, we will also pass
# ``var_spec`` to this function:

from eradiate.xarray.make import make_dataarray

make_dataarray(
    np.random.random((1, 1, 5, 5, 1)),
    coords=(
        ("sza", [30.]),
        ("saa", [0.]),
        ("vza", [0., 15., 30., 45., 60.]),
        ("vaa", [0., 45., 90., 135., 180.]),
        ("wavelength", [550.])
    ),
    var_spec=var_spec
)

# %%
# As we can see, :func:`.make_dataarray` applied the metadata we specified using
# :class:`.VarSpec`. It is interesting to note that :class:`.VarSpec` can be
# used to specify only part of the metadata: if its ``coord_specs`` attribute is
# set to ``None``, then no operation will be performed on  coordinate variable
# metadata. If its other attributes are unset, no variable metadata will be
# created. And obviously, no metadata modification will be made at all if all
# members are unset!
#
# .. note:: Under the hood, :func:`.make_dataarray` uses the accessor method
#    :meth:`DataArray.ert.normalize_metadata <eradiate.util.xarray.EradiateDataArrayAccessor.normalize_metadata>`.
#    A similar approach can be used for datasets with a :class:`DatasetSpec` and
#    the :meth:`Dataset.ert.normalize_metadata <eradiate.util.xarray.EradiateDatasetAccessor.normalize_metadata>`
#    accessor method.
