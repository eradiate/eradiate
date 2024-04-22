Data format details
===================

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
