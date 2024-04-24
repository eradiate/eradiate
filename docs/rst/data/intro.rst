.. _sec-data-intro:

Introduction
============

Input data plays a role equally important to the radiative transfer equation
integration algorithm when it comes to simulating radiative transfer. Data
handling and procurement in Eradiate follows the following principles:

* Eradiate reads standard data formats that are understood by the main data
  processing libraries of the scientific Python ecosystem. Most data are
  supplied in the `NetCDF format <https://www.unidata.ucar.edu/software/netcdf/>`_
  and loaded as `xarray <https://xarray.dev/>`_ datasets. Xarray provides a
  comprehensive, robust and convenient interface to read, write, manipulate and
  visualize NetCDF data.

* Eradiate ships data to use as "sensible defaults", similar to most radiative
  transfer models. We try to make shipped data transparent and take advantage of
  modern documentation facilities to make it comprehensive.

* Eradiate lets users swap their input with data sourced by themselves. If you
  want to use your own surface reflection spectra, aerosol single-scattering
  properties or molecular absorption data, it is possible thanks to the
  documented data formats.

* Eradiate provides an interface to facilitate the delivery of shipped data to
  users. Downloading a dataset is usually not more complicated than issuing a
  single command line in a terminal.

Data store configuration
------------------------

Eradiate ships data managed by its global data store. This data store aggregates
multiple data sources that can point to different locations (local or online)
and implement different shipment behaviours (download with or without integrity
checks). The following configuration items drive the behaviour of Eradiate's
data store:

Development mode
    This behaviour is controlled by the
    :data:`ERADIATE_SOURCE_DIR <eradiate.config.SOURCE_DIR>` environment
    variable.
    In development mode, parts of the data is shipped in a Git submodule.
    Otherwise, these data are downloaded upon access request.

Offline mode
    This behaviour is controlled by the
    :ref:`offline setting<sec-user_guide-config-default>`.
    In offline mode, all download requests made to the data store are denied.
    This mode is safer if you want to deliver the data yourself or operate with
    a bandwidth-limited or unstable connection.

Download directory
    Upon download, Eradiate stores data in a directory defined by the
    :ref:`download\_dir setting<sec-user_guide-config-default>`. The default
    location, if this setting is not overridden by the user, depends on whether
    Eradiate is operating in development mode or not.

Downloading data
----------------

Data download is done using the ``eradiate data fetch`` command
(see :ref:`sec-reference_cli`). The most common way to download data is to
reference a file list when calling ``eradiate data fetch``. Known file lists
are displayed using the ``--list`` option:

.. code-block:: console

   $ eradiate data fetch --list
   Known file lists:
     all
     minimal
     komodo
     gecko
     monotropa
     mycena
     panellus

A specific file list can then be downloaded by simply requesting it, *e.g.*:

.. code-block:: console

   $ eradiate data fetch komodo
   Fetching 'spectra/absorption/mono/komodo/komodo.nc'
   ✓ found
   [/home/username/src/eradiate/.eradiate_downloads/stable/spectra/absorpti
   on/mono/komodo/komodo.nc]
   Fetching 'spectra/absorption/mono/komodo/metadata.json'
   ✓ found
   [/home/username/src/eradiate/.eradiate_downloads/stable/spectra/absorpti
   on/mono/komodo/metadata.json]

Accessing data (advanced users and developers)
----------------------------------------------

Every file managed by the global data store can be accessed using the
:func:`eradiate.data.open_dataset` function:

.. code-block:: python

   >>> import eradiate
   >>> ds = eradiate.data.open_dataset("spectra/solar_irradiance/thuillier_2003.nc")

This function behaves similarly to :func:`xarray.open_dataset`. The
:func:`eradiate.data.load_dataset` also allows eager data loading. File access
will, if necessary, trigger data download and caching.

.. warning::

   The data module does not support concurrent download requests from multiple
   processes running Eradiate. This means that in such cases, two processes
   requesting the same resource using *e.g.* :func:`eradiate.data.load_dataset`
   might trigger two downloads that will overwrite each other, resulting in
   unpredictable (but surely incorrect) behaviour.

   If your use case requires running Eradiate from multiple processes, we
   strongly recommend **downloading all required data in advance** using the
   ``eradiate data fetch`` command (see `Downloading data`_).

.. seealso::

   :mod:`eradiate.data`: complete data module reference.

.. toctree::
   :hidden:

   data_format_details
