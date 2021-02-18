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

one
`dimension coordinate <http://xarray.pydata.org/en/stable/data-structures.html#coordinates>`_:

* wavenumber (``w``)

one
`optional dimension coordinate <http://xarray.pydata.org/en/stable/data-structures.html#coordinates>`_:

* molecules (``m``)

two
`dimension coordinates <http://xarray.pydata.org/en/stable/data-structures.html#coordinates>`_
or
`non-dimension coordinates <http://xarray.pydata.org/en/stable/data-structures.html#coordinates>`_:

* pressure (``p``)
* temperature (``t``)


Data variable ``xs`` is tabulated with respect to ``w``, ``p`` and optionally
``t``.

.. note::
   The two coordinates ``p`` and ``t`` allow to combine
   multiple absorption cross section data sets into a data set where
   pressure and temperature are actual dimension coordinates.
   To prevent from combining data sets along a specific dimension, make the
   corresponding coordinate a non-dimension coordinate.
   For example, rename the temperature coordinate from ``t`` to ``tc`` to
   prevent data sets from being combined along the temperature dimension.

Data variable ``mr``, when present, is tabulated with respect to ``m``.
Data variable ``mr`` should be included in data sets where the absorber is a
mixture of different absorbing molecule.
