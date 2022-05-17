.. _sec-user_guide-data-intro:

Introduction
============

Eradiate ships, processes and produces data. This guide presents:

* the rationale underlying data models used in Eradiate;
* the components used to manipulate data shipped with Eradiate.

Formats
-------

Most data sets used and produced by Eradiate are stored in the
`NetCDF format <https://www.unidata.ucar.edu/software/netcdf/>`_. Eradiate
interacts with these data using the `xarray <https://xarray.pydata.org/>`_
library, whose data model is based on NetCDF. Xarray provides a comprehensive,
robust and convenient interface to read, write, manipulate and visualise NetCDF
data.

Accessing shipped data
----------------------

Eradiate ships with a series of data sets managed its global data store.

.. code-block:: python

   from eradiate.data import data_store

This global data store aggregates multiple subordinated data stores based on
the size and maturity level of data files they manage.
List a data store' registered data sets by reading its ``registry`` property,
*i.e.*:

.. code-block:: python

   list(data_store.stores["small_files"].registry)

for small data files and:

.. code-block:: python

   list(data_store.stores["large_files_stable"].registry)

for large data files.

To open a specific data set, use :func:`eradiate.data.open_dataset`:

.. code-block:: python

   import eradiate
   ds = eradiate.data.open_dataset("spectra/solar_irradiance/thuillier_2003.nc")

To load a data set into memory, use :func:`eradiate.data.load_dataset`:

.. code-block:: python

   ds = eradiate.data.load_dataset("spectra/solar_irradiance/thuillier_2003.nc")

.. warning::

   The data module does not support concurrent download requests from multiple
   processes running Eradiate. This means that in such cases, two processes
   requesting the same resource using *e.g.* :func:`eradiate.data.load_dataset`
   may both trigger two downloads overwriting each other, resulting in
   unpredictable (but surely incorrect) behaviour.

   If your use case requires running Eradiate from multiple processes, we
   strongly advise that you **download all data in advance** using the
   ``eradiate data fetch`` command (see :ref:`sec-reference_cli`).

.. _sec-user_guide-data_guide-working_angular_data:

Working with angular data
-------------------------

Eradiate notably manipulates and produces what we refer to as *angular data*,
which represent variables dependent on one or more directional parameters.
Typical examples are BRDFs
(:math:`f_\mathrm{r} (\theta_\mathrm{i}, \varphi_\mathrm{i}, \theta_\mathrm{o}, \varphi_\mathrm{o})`)
or top-of-atmosphere BRFs
(:math:`\mathit{BRF}_\mathrm{TOA} (\theta_\mathrm{sun}, \varphi_\mathrm{sun}, \theta_\mathrm{view}, \varphi_\mathrm{view})`):
a xarray data array representing them has at least one angular dimension (and
corresponding coordinates). Eradiate has specific functionality to deal more
easily with this sort of data.

Angular dependencies and coordinate variable names
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Angular variable naming in Earth observation and radiative transfer modelling
may sometimes clash or be confusing. Eradiate clearly distinguishes between two
types of angular dependencies for its variables:

* Physical properties such as BRDFs and phase functions have intrinsic
  bidirectional dependencies which are referred to as *incoming* and *outgoing*
  directions. Data sets representing such quantities use  coordinate variables
  ``phi_i``, ``theta_i`` for the incoming direction's azimuth and zenith angles,
  and ``phi_o``, ``theta_o`` for their outgoing counterparts.

* Observations are usually parametrised by *illumination* (or *solar*) and
  *viewing* (or *sensor*) directions. For data sets representing such results,
  Eradiate uses coordinate variables ``sza``, ``saa`` for
  *solar zenith/azimuth angle* and ``vza``, ``vaa`` for
  *viewing zenith/azimuth angle*. A typical example of such variable is
  the top-of-atmosphere bidirectional reflectance factor (TOA BRF).

Under specific circumstances, one can directly convert an observation dataset to
a physical property dataset. This, for instance, applies to top-of-atmosphere
BRF data, but also any BRF computed or measured in a vacuum. In such cases,
incoming/outgoing directions can be directly converted to
illumination/viewing directions. **But in general, this does not work.**

Angular data set types
^^^^^^^^^^^^^^^^^^^^^^

While one should clearly distinguish intrinsic and observation angular
dependencies for correct physical interpretation of radiative data, both share
an asymmetry between 'incoming' and 'outgoing' directions. Eradiate uses
similar semantics to handle both angular data types, and the table below clarifies
the nomenclature for the two types:

.. list-table::
   :header-rows: 1

   * - Type
     - Incoming
     - Outgoing
   * - Intrinsic
     - :math:`\varphi_\mathrm{i}`, :math:`\theta_\mathrm{i}`
     - :math:`\varphi_\mathrm{o}`, :math:`\theta_\mathrm{o}`
   * - Observation
     - :math:`\varphi_\mathrm{s}`, :math:`\theta_\mathrm{s}`
     - :math:`\varphi_\mathrm{v}`, :math:`\theta_\mathrm{v}`

Eradiate's xarray containers do not explicitly keep track of the angular data
set type. However, when relevant, coordinate naming is used to determine whether
an angular data set is of intrinsic or observation type.

Angular data sets with a pair of angular dimensions :math:`(\theta, \varphi)`
are called *hemispherical*. If they have two pairs of angular dimensions
(incoming and outgoing), they are then called *bi-hemispherical*.

Measure data formats
--------------------

Most measures in Earth observation radiative transfer modelling have angular
dependencies. However, Eradiate uses storage data structures inherited from
computer graphics technology and measure results are usually mapped against
*film coordinates* :math:`(x, y) \in [0, 1]^2`. When those data represent
hemispherical quantities, a mapping transformation associate angles to film
coordinates. For convenience, Eradiate ships helpers to convert data from film
coordinates to angular coordinates.
