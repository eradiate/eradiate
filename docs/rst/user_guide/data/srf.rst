.. _sec-user_guide-data-srf:

Spectral response function
==========================

A spectral response function data set provide the spectral response of a
given instrument on a specific platform and in a specific spectral band.

Data access
-----------

All spectral response function data sets required by Eradiate are 
managed the data store (see :ref:`sec-user_guide-data-intro` for details).

.. _sec-user_guide-data-srf-naming_convention:

Identifier format
^^^^^^^^^^^^^^^^^

Identifiers for spectral response function are built according to the format
``{platform}-{instrument}-{band}.nc`` where:

* ``platform`` identifies the platform (*e.g.* satellite's name),
* ``instrument`` identifies the instrument onboard the platform,
* ``band`` specifies the spectral band of the instrument that is being
  characterised.

For example, the spectral response function data set of the SLSTR instrument
onboard Sentinel-3B and in the spectral band number 5 has the identifier
``sentinel_3b-slstr-5``.

Structure
---------

Spectral response function data sets include two data variables:

* the instrument's spectral response function (``srf``),
* the uncertainties on the ``srf`` data (``srf_u``),

and one :term:`dimension coordinate`:

* the wavelength (``w``).

Both data variables (``srf`` and ``srf_u``) are tabulated against the
wavelength.

The following additional data set attributes are provided:

* ``platform``: platform identifier (e.g. ``sentinel_3b``),
* ``instrument``: instrument identifier (e.g. ``slstr``),
* ``band``: spectral band number (e.g. ``5``).
