Aerosol / particle single-scattering radiative properties
=========================================================

A particle radiative single-scattering property dataset provides collision
coefficients and scattering phase matrix data for a given particle type.
Eradiate's built-in particle radiative property datasets are managed by the
data store (see :ref:`sec-data-intro` for details).

Scattering particle properties are provided as xarray Datasets in a format
documented in :ref:`the dedicated page <sec-data-formats-aer_v1>`.

Dataset index
-------------

{% for entry in particle_radprops %}
``{{entry.keyword}}``
{{'^' * ((entry.keyword|length) + 4)}}

{{entry.description or ''}}

.. dropdown:: Data path

    ``{{entry.fname}}``

.. image:: /_images/particle_radprops/{{entry.keyword}}.png

{% endfor %}
