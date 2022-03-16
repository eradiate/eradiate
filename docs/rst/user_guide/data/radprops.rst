.. _sec-user_guide-data-radprops:

Atmosphere radiative properties
===============================

An atmosphere radiative properties data set provide the collision (absorption,
extinction, scattering) coefficient and albedo values within the atmosphere.

Data sets access
----------------

Atmosphere radiative properties data sets are created by
:meth:`.RadProfile.eval_dataset` method.

Structure
---------

Atmospheric radiative properties data sets include four data variables:

* the absorption coefficient (``sigma_a``)
* the extinction coefficient (``sigma_t``)
* the scattering coefficient (``sigma_s``)
* the albedo (``albedo``)

two
`dimension coordinates <https://xarray.pydata.org/en/stable/user-guide/data-structures.html#coordinates>`_:

* the wavelength (``w``)
* the layer altitude (``z_layer``)

and one
`non-dimension coordinates <https://xarray.pydata.org/en/stable/user-guide/data-structures.html#coordinates>`_:

* the level altitude (``z_level``).

All data variables are tabulated with respect to ``w`` and ``z_layer``.
