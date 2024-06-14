.. _sec-data-molecular_absorption:

Atmosphere: Molecular absorption
================================

Molecular absorption databases tabulate the volume absorption coefficient of a
gas mixture against the spectral coordinates, the volume fraction of the mixture
components, air pressure and air temperature.
Eradiate's built-in molecular absorption datasets are managed by the data store
(see :ref:`sec-data-intro` for details).

Format (CKD)
------------

* **Format** ``xarray.Dataset`` (in-memory), NetCDF (storage)
* **Dimensions**

  * ``w``: radiation wavelength
  * ``g``: cumulative probability of the absorption coefficient distribution
  * ``p``: air pressure
  * ``t``: air temperature
  * ``x_M``, where ``M`` is the molecule formula, *e.g.* ``x_H2O``: gas mixture mole fractions

* **Coordinates** (all dimension coordinates; when relevant, ``units`` are
  required and specified in the units metadata field)

  * ``w`` float [length]
  * ``g``: cumulative probability of the absorption coefficient distribution
  * ``p`` float [pressure]
  * ``t`` float [temperature]
  * ``x_M`` float [dimensionless]

* **Data variables** (when relevant, units are required and  specified in the
  units metadata field)

  * ``sigma_a`` (``w``, ``p``, ``t``, ``x_M``): volume absorption coefficient [length^-1]

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

Database index
--------------


``monotropa``
^^^^^^^^^^^^^

Datastore path: ``spectra/absorption/ckd/monotropa``

Spectral sampling: 100 cm⁻¹

.. image:: /fig/absorption_databases/monotropa.png

``mycena``
^^^^^^^^^^

Datastore path: ``spectra/absorption/ckd/mycena``

Spectral sampling: 10 nm

.. image:: /fig/absorption_databases/mycena.png

``panellus``
^^^^^^^^^^^^

Datastore path: ``spectra/absorption/ckd/panellus``

Spectral sampling: 1 nm

.. image:: /fig/absorption_databases/panellus.png
