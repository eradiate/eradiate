.. _sec-data-srf:

Spectral response functions
===========================

A spectral response function dataset provides tabulated values of the spectral
response of a radiometric instrument.
Eradiate's built-in particle SRF datasets are managed by the data store
(see :ref:`sec-data-intro` for details).

Format
------

* **Format** ``xarray.Dataset`` (in-memory), NetCDF (storage)
* **Dimensions**

  * ``w``: radiation wavelength

* **Coordinates** (all dimension coordinates; when relevant, ``units`` are
  required and specified in the units metadata field)

  * ``w`` float [length]

* **Data variables** (when relevant, units are required and specified in the
  units metadata field)

  * ``srf`` (``w``): spectral response function [dimensionless]
  * ``srf_u`` (``w``): uncertainty on the ``srf`` data variable [dimensionless]

.. dropdown:: Full validation schema

   .. literalinclude:: /resources/data_schemas/srf_dataset_v1.yml

Naming convention
^^^^^^^^^^^^^^^^^

SRF data files are usually named ``{platform}-{instrument}-{band}.nc`` where:

* ``platform`` identifies the platform (*e.g.* satellite's name);
* ``instrument`` identifies the instrument onboard the platform;
* ``band`` specifies the spectral band.

For example, the spectral response function data set of the SLSTR instrument
onboard Sentinel-3B and in the spectral band 5 has the identifier
``sentinel_3b-slstr-5``.

Dataset index
-------------

{% for entry in instruments %}
``{{entry.name}}``
{{'^' * ((entry.name|length) + 4)}}

.. dropdown:: Datastore paths

    {% for path in entry.paths %}
    * ``{{path}}``
    {% endfor %}

.. image:: /fig/srf/{{entry.name}}.png
{% endfor %}
