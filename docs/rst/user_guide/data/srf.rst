.. _sec-user_guide-data-srf:

Spectral response function
==========================

A spectral response function data set provide the spectral response of a
given instrument on a specific platform and in a specific spectral band.

Data sets access
----------------

All spectral response function data sets required by Eradiate are available
within Eradiate using :meth:`eradiate.data.open`.

.. _sec-user_guide-data-srf-naming_convention:

Identifiers format
^^^^^^^^^^^^^^^^^^

Identifiers for spectral response function are built according to the format
``platform-instrument-band.nc`` where:

* ``platform`` identifies the platform (e.g. satellite's name)
* ``instrument`` identifies the instrument onboard the platform,
* ``band`` specifies the spectral band of the instrument that is being
  characterised.

For example the spectral response function data set of the SLSTR instrument
onboard Sentinel-3B and in the spectral band number 5 has the identifier
``sentinel_3b-slstr-5``.

Structure
---------

Spectral response function data sets include two data variables:

* the instrument's spectral response function (``srf``)
* the uncertainties on the ``srf`` data (``srf_u``)

and one
`dimension coordinate <http://xarray.pydata.org/en/stable/data-structures.html#coordinates>`_:

* the wavelength (``w``)

Both data variables (``srf`` and ``srf_u``) are tabulated with respect to
wavelength.

The following additional data set attributes are provided:

* ``platform``: platform identifier (e.g. ``sentinel_3b``)
* ``instrument``: instrument identifier (e.g. ``slstr``)
* ``band``: spectral band number (e.g. ``5``)

Visualise the data
------------------

Refer to the
:ref:`dedicated tutorial <sphx_glr_examples_generated_tutorials_data_03_visualise_srf_data_set.py>`.
