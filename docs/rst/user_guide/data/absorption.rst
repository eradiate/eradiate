.. _sec-user_guide-data-absorption:

Monochromatic absorption datasets
=================================

Monochromatic absorption data sets provide the volume absorption
coefficient of a gas mixture for a range of molecule volume fraction,
pressure, temperature and wavelength values.

Data access
-----------

Distributed data sets are uploaded to
`Eradiate FTP server<http://eradiate.eu/data/>`_

Structure
---------

Absorption coefficient datasets include one data variable:

* volume absorption coefficient (``k``)

and multiple :term:`dimension coordinates <dimension coordinate>`:

* wavelength (``w``),
* pressure (``p``),
* temperature (``t``),
* volume fraction (``x_m``)


where ``m`` refers to the chemical formula of the molecule.
For example, the water vapor volume fraction is denoted ``x_H2O``.
