.. _sec-user_guide-data-thermoprops:

Atmosphere thermophysical properties data sets
==============================================

An atmosphere thermophysical properties data set provide the spatial variation
of air pressure, air temperature, air number density and individual species
mixing ratios.

Data sets access
----------------

All data sets are hosted on
`https://eradiate.eu/data <https://eradiate.eu/data>`_.
Download the AFGL (1986) data sets
`here <https://eradiate.eu/data/afgl1986.zip>`_.

Structure
---------

Atmosphere thermophysical properties data sets include four data variables:

* air pressure (``p``)
* air temperature (``t``)
* air number density (``n``)
* individual species mixing ratios (``mr``)

two
`dimension coordinates <http://xarray.pydata.org/en/stable/data-structures.html#coordinates>`_:

* layer altitude (``z_layer``)
* individual species (``species``)

and one
`non-dimension coordinates <http://xarray.pydata.org/en/stable/data-structures.html#coordinates>`_:

* level altitude (``z_level``)

Data variables ``p``, ``t`` and ``n`` are tabulated with respect to ``z_layer``.
Data variable ``mr`` is tabulated with respect to ``species`` and ``z_layer``.

Visualise the data
------------------

Refer to the
:ref:`dedicated tutorial <sphx_glr_examples_generated_tutorials_data_04_visualise_thermoprops_data_set.py>`.