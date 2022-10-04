Particle radiative properties
=============================

A particle radiative property data set provides collision coefficients and
scattering phase matrix data for the given particle type.

Data access
-----------
Particle radiative propery data sets required by Eradiate are
managed the data store (see :ref:`sec-user_guide-data-intro` for details).

Structure
---------

The data set must include the following data variables:

* ``sigma_t`` (``w``): volume extinction coefficient ``[length^-1]``
* ``albedo`` (``w``): single-scattering albedo ``[dimensionless]``
* ``phase`` (``w``, ``mu``, ``i``, ``j``): scattering phase matrix ``[steradian^-1]``

(dimensions in brackets) and the following
:term:`dimension coordinates <dimension coordinate>`:

* ``w``: radiation wavelength ``[length]``
* ``mu``: scattering angle cosine ``[dimensionless]``
* ``i``: scattering phase matrix row index (integer)
* ``j``: scattering phase matrix column index (integer)
