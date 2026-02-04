.. _sec-data-molecular_absorption:

Atmosphere: Molecular absorption
================================

Molecular absorption databases tabulate the volume absorption coefficient of a
gas mixture against the spectral coordinates, the volume fraction of the mixture
components, air pressure and air temperature.

Eradiate's built-in molecular absorption datasets are managed by the data store
(see :ref:`sec-data-intro` for details).

Data format
-----------

.. note::

    Eradiate offloads its absorption data handling to the
    `AxsDB package <https://github.com/eradiate/axsdb/>`__.

The shipped databases are delivered in AxsDB's Ac-v1 format. The full format
documentation is available on the
`dedicated page <https://axsdb.readthedocs.io/en/latest/formats.html#absorption-coefficient-format-v1-ac-v1>`__.

Database index
--------------

{% for entry in absorption_databases.mono %}
``{{entry.keyword}}`` (mono)
{{'^' * ((entry.keyword|length) + 11)}}

.. dropdown:: Data path

    ``{{entry.path}}``

Spectral sampling: {{entry.spectral_sampling}}

.. image:: /_images/absorption_databases/{{entry.keyword}}.png

{% endfor %}

{% for entry in absorption_databases.ckd %}
``{{entry.keyword}}`` (CKD)
{{'^' * ((entry.keyword|length) + 10)}}

.. dropdown:: Data path

    ``{{entry.path}}``

Spectral bin size: {{entry.spectral_sampling}}

.. image:: /_images/absorption_databases/{{entry.keyword}}.png

{% endfor %}
