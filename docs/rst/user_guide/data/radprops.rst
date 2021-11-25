.. _sec-user_guide-data-radprops:

Atmosphere radiative properties
===============================

.. warning:: This content is outdated.

An atmosphere radiative properties data set provide the collision (absorption,
extinction, scattering) coefficient and albedo values within the atmosphere.

Data sets access
----------------

Atmosphere radiative properties data sets are created by
:class:`~eradiate.radprops.rad_profile.RadProfile` objects'
:meth:`~eradiate.radprops.rad_profile.RadProfile.to_dataset` method.

Structure
---------

Atmospheric radiative properties data sets include four data variables:

* the absorption coefficient (``sigma_a``)
* the extinction coefficient (``sigma_t``)
* the scattering coefficient (``sigma_s``)
* the albedo (``albedo``)

two
`dimension coordinates <http://xarray.pydata.org/en/stable/data-structures.html#coordinates>`_:

* the wavelength (``w``)
* the layer altitude (``z_layer``)

and one
`non-dimension coordinates <http://xarray.pydata.org/en/stable/data-structures.html#coordinates>`_:

* the level altitude (``z_level``).

All data variables are tabulated with respect to ``w`` and ``z_layer``.

Visualise the data
------------------
Refer to
:ref:`this tutorial <sphx_glr_examples_generated_tutorials_atmosphere_02_heterogeneous.py>`
for an example of use.
