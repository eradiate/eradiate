.. _sec-user_guide-data-thermoprops:

Atmosphere thermophysical properties
====================================

.. warning:: This content is outdated.

An atmosphere thermophysical properties data set provide the spatial variation
of air pressure, air temperature, air number density and individual species
mixing ratios.

Data sets access
----------------

All atmosphere thermophysical properties data sets required by Eradiate are
available within Eradiate using :meth:`eradiate.data.open`.

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
