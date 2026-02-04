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

``gecko`` (mono)
^^^^^^^^^^^^^^^^

.. dropdown:: Data path

    ``absorption_mono/gecko``

Spectral sampling: 0.01 cm⁻¹ in [250, 300] + [600, 3125] nm, 0.1 cm⁻¹ in [300, 600] nm

.. image:: /_images/absorption_databases/gecko.png

``komodo`` (mono)
^^^^^^^^^^^^^^^^^

.. dropdown:: Data path

    ``absorption_mono/komodo``

Spectral sampling: 1 cm⁻¹

.. image:: /_images/absorption_databases/komodo.png


``monotropa`` (CKD)
^^^^^^^^^^^^^^^^^^^

.. dropdown:: Data path

    ``absorption_ckd/monotropa``

Spectral bin size: 100 cm⁻¹

.. image:: /_images/absorption_databases/monotropa.png

``mycena`` (CKD)
^^^^^^^^^^^^^^^^

.. dropdown:: Data path

    ``absorption_ckd/mycena``

Spectral bin size: 10 nm

.. image:: /_images/absorption_databases/mycena.png

``panellus`` (CKD)
^^^^^^^^^^^^^^^^^^

.. dropdown:: Data path

    ``absorption_ckd/panellus``

Spectral bin size: 1 nm

.. image:: /_images/absorption_databases/panellus.png

``tuber`` (CKD)
^^^^^^^^^^^^^^^

.. dropdown:: Data path

    ``absorption_ckd/tuber``

Spectral bin size: 0.1 nm

.. image:: /_images/absorption_databases/tuber.png
