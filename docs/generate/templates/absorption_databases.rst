.. _sec-data-molecular_absorption:

Atmosphere: Molecular absorption
================================

Molecular absorption databases tabulate the volume absorption coefficient of a
gas mixture against the spectral coordinates, the volume fraction of the mixture
components, air pressure and air temperature.
Eradiate's built-in molecular absorption datasets are managed by the data store
(see :ref:`sec-data-intro` for details).

Format (monochromatic)
----------------------

* **Format** ``xarray.Dataset`` (in-memory), NetCDF (storage)
* **Dimensions**

  * ``w``: radiation wavelength
  * ``p``: air pressure
  * ``t``: air temperature
  * ``x_M``, where ``M`` is the molecule formula, *e.g.* ``x_H2O``: gas mixture mole fractions

* **Coordinates** (all dimension coordinates; when relevant, ``units`` are
  required and specified in the units metadata field)

  * ``w`` float [length]
  * ``p`` float [pressure]
  * ``t`` float [temperature]
  * ``x_M`` float [dimensionless]

* **Data variables** (when relevant, units are required and  specified in the
  units metadata field)

  * ``sigma_a`` (``w``, ``p``, ``t``, ``x_M``): volume absorption coefficient [length^-1]

Format (CKD)
------------

* **Format** ``xarray.Dataset`` (in-memory), NetCDF (storage)
* **Dimensions**

  * ``w``: bin central wavelength
  * ``wbv``: bin bound (lower or upper)
  * ``g``: cumulative probability of the absorption coefficient distribution
  * ``ng``: number of quadrature points
  * ``p``: air pressure
  * ``t``: air temperature
  * ``x_M``, where ``M`` is the molecule formula, *e.g.* ``x_H2O``: gas mixture mole fractions

* **Coordinates** (all dimension coordinates; when relevant, ``units`` are
  required and specified in the units metadata field)

  * ``w`` float [length]
  * ``wbv`` str
  * ``g`` float [dimensionless]
  * ``ng`` int [dimensionless]
  * ``p`` float [pressure]
  * ``t`` float [temperature]
  * ``x_M`` float [dimensionless]

* **Data variables** (when relevant, units are required and  specified in the
  units metadata field)

  * ``sigma_a`` (``w``, ``p``, ``t``, ``x_M``): volume absorption coefficient [length^-1]
  * ``wbounds`` (``w``, ``wbv``): bin bound values [length]
  * ``error`` (``w``, ``ng``): relative error on transmittance when using the quadrature, optional [dimensionless]

Database index
--------------

{% for entry in absorption_databases.mono %}
``{{entry.keyword}}`` (mono)
{{'^' * ((entry.keyword|length) + 11)}}

.. dropdown:: Data path

    ``{{entry.path}}``

Spectral sampling: {{entry.spectral_sampling}}

.. image:: /fig/absorption_databases/{{entry.keyword}}.png

{% endfor %}

{% for entry in absorption_databases.ckd %}
``{{entry.keyword}}`` (CKD)
{{'^' * ((entry.keyword|length) + 10)}}

.. dropdown:: Data path

    ``{{entry.path}}``

Spectral bin size: {{entry.spectral_sampling}}

.. image:: /fig/absorption_databases/{{entry.keyword}}.png

{% endfor %}
