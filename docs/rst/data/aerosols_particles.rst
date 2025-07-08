Aerosol / particle single-scattering radiative properties
=========================================================

A particle radiative single-scattering property dataset provides collision
coefficients and scattering phase matrix data for a given particle type.
Eradiate's built-in particle radiative property datasets are managed by the
data store (see :ref:`sec-data-intro` for details).

Format
------

* **Format** ``xarray.Dataset`` (in-memory), NetCDF (storage)
* **Dimensions**

  * ``w``: radiation wavelength
  * ``mu``: scattering angle cosine
  * ``i``: scattering phase matrix row index
  * ``j``: scattering phase matrix column index

* **Coordinates** (all dimension coordinates; when relevant, ``units`` are
  required and specified in the units metadata field)

  * ``w`` float [length]
  * ``mu`` float [dimensionless]
  * ``i``, ``j`` int

* **Data variables** (when relevant, units are required and  specified in the
  units metadata field)

  * ``sigma_t`` (``w``): volume extinction coefficient [length^-1]
  * ``albedo`` (``w``): single-scattering albedo [dimensionless]
  * ``phase`` (``w``, ``mu``, ``i``, ``j``): scattering phase matrix
    [steradian^-1]

* **Conventions**

  * Phase matrix components use C-style indexing (from 0).

.. dropdown:: Full validation schema

   .. literalinclude:: /resources/data_schemas/particle_dataset_v1.yml

Dataset index
-------------

``govaerts_2021-continental-extrapolated``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. dropdown:: Data path

    ``aerosol/govaerts_2021-continental-extrapolated.nc``

.. image:: /fig/particle_radprops/govaerts_2021-continental-extrapolated.png

``govaerts_2021-desert-extrapolated``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. dropdown:: Data path

    ``aerosol/govaerts_2021-desert-extrapolated.nc``

.. image:: /fig/particle_radprops/govaerts_2021-desert-extrapolated.png

``sixsv-biomass_burning``
^^^^^^^^^^^^^^^^^^^^^^^^^

.. dropdown:: Data path

    ``aerosol/sixsv-biomass_burning.nc``

.. image:: /fig/particle_radprops/sixsv-biomass_burning.png

``sixsv-continental``
^^^^^^^^^^^^^^^^^^^^^

.. dropdown:: Data path

    ``aerosol/sixsv-continental.nc``

.. image:: /fig/particle_radprops/sixsv-continental.png

``sixsv-desert``
^^^^^^^^^^^^^^^^

.. dropdown:: Data path

    ``aerosol/sixsv-desert.nc``

.. image:: /fig/particle_radprops/sixsv-desert.png

``sixsv-maritime``
^^^^^^^^^^^^^^^^^^

.. dropdown:: Data path

    ``aerosol/sixsv-maritime.nc``

.. image:: /fig/particle_radprops/sixsv-maritime.png

``sixsv-stratospheric``
^^^^^^^^^^^^^^^^^^^^^^^

.. dropdown:: Data path

    ``aerosol/sixsv-stratospheric.nc``

.. image:: /fig/particle_radprops/sixsv-stratospheric.png

``sixsv-urban``
^^^^^^^^^^^^^^^

.. dropdown:: Data path

    ``aerosol/sixsv-urban.nc``

.. image:: /fig/particle_radprops/sixsv-urban.png

