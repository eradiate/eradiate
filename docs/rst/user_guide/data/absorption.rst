.. _sec-user_guide-data-absorption:

Absorption cross section data sets
==================================

Absorption cross section data sets provide the monochromatic absorption cross
section spectrum of a given absorbing species at specific pressure and
temperature conditions.


Data sets access
----------------
All data sets are hosted on
`https://eradiate.eu/data <https://eradiate.eu/data>`_.

Structure
---------

Absorption cross section data sets include one data variable:

* monochromatic absorption cross section (``xs``)

one optional data variable:

* mixing ratios (``mr``)

three
`dimension coordinates <http://xarray.pydata.org/en/stable/data-structures.html#coordinates>`_:

* wavenumber (``w``)
* pressure (``p``)
* temperature (``t``)

(temperature may be a
`non-dimension coordinates <http://xarray.pydata.org/en/stable/data-structures.html#coordinates>`_)

and one
`optional dimension coordinate <http://xarray.pydata.org/en/stable/data-structures.html#coordinates>`_:

* molecules (``m``)

Data variable ``xs`` is tabulated with respect to ``w``, ``p`` and ``t``
(optional).

Data variable ``mr``, when present, is tabulated with respect to ``m``.
Data variable ``mr`` should be included in data sets where the absorber is a
mixture of different absorbing molecule.
