.. _sec-user_guide-data-thermoprops:

Atmosphere: thermophysical properties
=====================================

An atmospheric thermophysical property data set provides the spatial variation
of air pressure, air temperature, air number density and individual species
volume mixing ratios.

Data access
-----------

All atmospheric thermophysical property data sets sets required by Eradiate are
managed the data store (see :ref:`sec-user_guide-data-intro` for details).

Identifier format
^^^^^^^^^^^^^^^^^

Identifiers for thermophysical property data sets are constructed based on the
format ``{author}_{year}-{title}`` where:

* ``author`` specifies the author of the data set,
* ``year`` stands for the year in which the data set was published,
* ``title`` is the data set title.

Structure
---------

Atmospheric thermophysical property data sets include four data variables:

* air pressure (``p``),
* air temperature (``t``),
* air number density (``n``),
* individual species volume mixing ratios (``mr``),

two :term:`dimension coordinates <dimension coordinate>`:

* layer altitude (``z_layer``),
* individual species (``species``),

and one :term:`non-dimension coordinate`:

* level altitude (``z_level``).

Data variables ``p``, ``t`` and ``n`` are tabulated against ``z_layer``.
Data variable ``mr`` is tabulated against ``species`` and ``z_layer``.
