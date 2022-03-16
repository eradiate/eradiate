.. _sec-user_guide-data-thermoprops:

Atmosphere thermophysical properties
====================================

An atmosphere thermophysical properties data set provide the spatial variation
of air pressure, air temperature, air number density and individual species
volume mixing ratios.

Data sets access
----------------

All atmosphere thermophysical properties data sets required by Eradiate are
are managed by Eradiate's global data store.
Refer to the :ref:`sec-user_guide-data-intro` page for further details.

Identifiers format
^^^^^^^^^^^^^^^^^^

Identifiers for thermophysical properties data sets
are constructed based on the format ``author_year-title`` where:

* ``author`` specifies the author of the data set,
* ``year`` stands for the year in which the data set was published,
* ``title`` is the data set title.

Structure
---------

Atmosphere thermophysical properties data sets include four data variables:

* air pressure (``p``)
* air temperature (``t``)
* air number density (``n``)
* individual species volume mixing ratios (``mr``)

two
`dimension coordinates <https://xarray.pydata.org/en/stable/user-guide/data-structures.html#coordinates>`_:

* layer altitude (``z_layer``)
* individual species (``species``)

and one
`non-dimension coordinates <https://xarray.pydata.org/en/stable/user-guide/data-structures.html#coordinates>`_:

* level altitude (``z_level``)

Data variables ``p``, ``t`` and ``n`` are tabulated with respect to ``z_layer``.
Data variable ``mr`` is tabulated with respect to ``species`` and ``z_layer``.
