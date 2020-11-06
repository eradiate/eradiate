.. _sec-user_guide-post-processing:

Data post-processing and visualization
======================================

.. warning::

   This documentation page was written in Eradiate's early development phase.
   The data handling documentation has been partly updated but a tutorial for
   the usage of the xarray accessor is still missing. Until this tutorial is
   written and covers all the features presented here, this page will be kept.

The :code:`eradiate.util.view` module provides utilities for viewing and plotting data.

EradiateAccessor
------------------

To facilitate data viewing and plotting, Eradiate provides a custom accessor to
:code:`xarray.DataArray` objects. This accessor is enabled by importing :code:`eradiate.util.view`
and decorates the :code:`xarray.DataArray` class.

The specialized methods are available through the added attribute :code:`ert` on all
DataArray objects. See the examples below for usage of the accessor.

DataArray data format specification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
DataArrays holding computational results must conform to the format given in this section
to allow proper functioning of all methods provided by the accessor.
They can be specified with two different naming conventions:

- :code:`local` or internal convention:

    - :code:`theta_i`: Sun/illumination zenith angle in degrees
    - :code:`phi_i`: Sun/illumination azimuth angle in degrees
    - :code:`theta_o`: Viewing zenith angle in degrees
    - :code:`phi_o`: Viewing azimuth angle in degrees
    - :code:`array.attrs["angle_convention"]="local"`
- :code:`eo_scene` or earth observation convention:

    - :code:`sza`: Sun/illumination zenith angle in degrees
    - :code:`saa`: Sun/illumination azimuth angle in degrees
    - :code:`vza`: Viewing zenith angle in degrees
    - :code:`vaa`: Viewing azimuth angle in degrees
    - :code:`array.attrs["angle_convention"]="eo_scene"`

Additionally, all angular dimensions must contain at least an attribute describing their
unit as either of :code:`["degrees", "deg", "radians", "rad"]`

Further dimensions are possible but plotting facilities put constraints on the
dimensionality of plottable data. This is detailed in the respective sections.

:code:`sel` method wrapper
^^^^^^^^^^^^^^^^^^^^^^^^^^
The :code:`EradiateAccessor` provides a wrapper for the :code:`xarray.DataArra.sel` method
that allows access to DataArrays defined in the :code:`eo_scene` convention with the
attribute names of the internal convention. DataArrays containing computational results
should always be manipulated using the wrapped method.

:code:`plot` method
^^^^^^^^^^^^^^^^^^^^^
The :code:`plot` method provides shorthand access to two kinds of data visualization
automating some tedious configuration steps.

Data containing only one angular dimension (theta_o/vza) will be plotted in a linear plot,
for example creating a planar plot, if the data was processed with the :code:`plane` or
:code:`pplane` functions detailed below.

Data containing two angular dimensions (theta_o/vza and phi_o/vaa) will be plotted in
a polar plot, depicting the zenith angle as the radial dimension and the azimuth angle
as the angular dimension of the projection.

The :code:`plot` method will automatically try to determine the plot it should create, by
counting the number of angular dimensions. If the automatic detection fails, the data is
forwarded to xarray's internal plotting facilities, which provide limited capabilities.

Automatic plot choice will only succeed if no dimensions beyond the necessary direcions are
present in the array. For example a dependency on wavelength must be collapsed by selecting
a wavelength value from the original data.

The plotting can be configured with the following parameters:

- mode: chooses the plotting method for polar plots. This can be either of
  `contourf <https://matplotlib.org/3.2.0/api/_as_gen/matplotlib.pyplot.contourf.html>`_ or
  `pcolormesh <https://matplotlib.org/3.2.0/api/_as_gen/matplotlib.axes.Axes.pcolormesh.html>`_.
  For linear plots this parameter has no effect and can be omitted
- ax: An `Axes <https://matplotlib.org/3.2.0/api/axes_api.html>`_ object can be provided
  and the method will fill it with the plotting data. This is useful if the view is used
  in a custom plotting script, e.g. when creating composite plots of mupltiple results.
- ylim: A tuple holding the upper and lower limit for the plot data-axis of the plot.

Convenience functions
---------------------

Plane
^^^^^
The :code:`plane()` method creates a new DataArray object, holding linearized data
suitable for plotting as a linear plot with the :code:`EradiateAccessor`.
It operates on hemispherical data, that means, a user has to reduce the angular dimensionality
of their data to two, by selecting values for the incident light direction, leaving the
outgoing light directions.

The method accepts two parameters:

- hemispherical data to be plotted
- angle in degrees, orienting the plane

Principal plane
^^^^^^^^^^^^^^^
The :code:`pplane()` method creates a planar view holding linearized data similiar to
the :code:`plane()` method but automatically sets the azimuth angles of incoming and
outgoing directions to the same value.

This method operates on bi-hemispherical data and accepts three parameters:

- bi-hemispherical data
- angle in degrees, setting the zenith angle of incoming radiation
- angle in degrees, setting the azimuth angle of incoming and outgoing radiation

Examples
--------

Two short examples to introduce the usage of the view and its plotting facilities

.. code-block:: python

    import xarray as xr
    import numpy as np

    from eradiate.util import view

    # create a dummy DataArray for demonstration purposes
    vza = np.linspace(0, 90, 10)
    vaa = np.linspace(0, 360, 40)
    sza = np.linspace(0, 90, 5)
    saa = [0]
    wavelength = [450, 500, 550, 600]

    data = np.random.rand((len(sza), len(saa), len(vza), len(vaa), len(wavelength)))
    arr = xr.DataArray(data,
                       dims=["theta_i", "phi_i", "theta_o", "phi_o", "wavelength"],
                       coords=[sza, saa, vza, vaa, wavelength])

    arr.attrs["angle_convention"] = "local
    arr.theta_i.attrs["unit"] = "deg"
    arr.phi_i.attrs["unit"] = "deg"
    arr.theta_o.attrs["unit"] = "deg"
    arr.phi_o.attrs["unit"] = "deg"

    bhdata = arr.ert.sel(wavelength=550)
    pplane = view.pplane(bhdata, theta=30, phi=90)

    # request the polar plot for the hemispherical data, setting the limits for the data
    # axis to 0 and 0.1
    pplane.ert.plot(ylim=(0, 0.1))


.. code-block:: python

    import xarray as xr
    import numpy as np

    from eradiate.util import view

    # create a dummy DataArray for demonstration purposes
    vza = np.linspace(0, 90, 10)
    vaa = np.linspace(0, 360, 40)
    sza = np.linspace(0, 90, 5)
    saa = [0]
    wavelength = [450, 500, 550, 600]

    data = np.random.rand((len(sza), len(saa), len(vza), len(vaa), len(wavelength)))
    arr = xr.DataArray(data,
                       dims=["theta_i", "phi_i", "theta_o", "phi_o", "wavelength"],
                       coords=[sza, saa, vza, vaa, wavelength])

    arr.attrs["angle_convention"] = "local
    arr.theta_i.attrs["unit"] = "deg"
    arr.phi_i.attrs["unit"] = "deg"
    arr.theta_o.attrs["unit"] = "deg"
    arr.phi_o.attrs["unit"] = "deg"

    # create hemispherical data by selecting values for the incoming radiation and
    # the wavelength of interest
    hdata = arr.ert.sel(theta_i=0, phi_i=0, wavelength=550)
    plane = view.plane(hdata, phi=90)

    # request the line plot for the planar data, setting the limits for the data
    # axis to 0 and 0.1
    plane.ert.plot(ylim=(0, 0.1))