.. _sec-user_guide-molecular_absorption_datasets:

Molecular absorption datasets
=============================

Eradiate ships with several molecular absorption datasets. This guide will provide a list
of available data and include a detailed description of the dataset, its properties and limitations.
Please refer to :ref:`this <sec-user_guide-manual_download>` section for instructions on installing
these datasets.

List of available molecular absorption datasets
-----------------------------------------------

The following list contains all available datasets. The next sections contain a detailed description
for each of these data sets.

.. list-table::
   :widths: 15 30 10
   :header-rows: 1

   * - Dataset name
     - Description
     - Download link
   * - :ref:`spectra_us76_u86_4 <sec-user_guide-molecular_absorption_datasets-spectra_us76_u86_4>`
     - US76 standard atmosphere absorption cross sections below 86km with four molecules (N2, O2, CO2 and CH4)
     - `Link <https://eradiate.eu/data/spectra-us76_u86_4-4000_25711.zip>`_

.. _sec-user_guide-molecular_absorption_datasets-spectra_us76_u86_4:

spectra_us76_u86_4
------------------

Absorption cross section dataset for the ``us76_u86_4`` mixture. The dataset follows the US76 atmospheric profile
:cite:`NASA1976USStandardAtmosphere` up to an altitude of 86 kilometers and contains four molecular species.
The absorption cross section values were computed using `SPECTRA <https://spectra.iao.ru/>`_.

This dataset can be interpolated both in wavenumber and in pressure.
Since the dataset is specific to the US76 atmosphere thermophysical profile, when interpolating along one of the axes,
the other axis' value will match the corresponding value pair in said profile.
In other words, the dataset must be interpolated only once to find the absorption cross section corresponding
to a set of pressure and temperature values.

The dataset is limited to altitudes below 86 kilometers, because for these altitudes, the gasses can be
assumed to be well mixed. Above 86 kilometers the air number density is smaller than a factor of 1e-5 compared
to sea level and the assumption of well mixing breaks down.

The ``us76_u86_4`` mixture is defined as the gas mixture in the US76 atmosphere model for altitudes below
86 kilometers and restricted to four gas species: N2, O2, CO2, CH4.
The volume fractions are the following:

- N2: 0.78084
- O2: 0.209476
- CO2: 0.000314
- CH4: 0.000002

.. admonition:: Note

   The dataset is restricted to these four species, because the others (Ar, Ne, He, Kr, Xe) are not
   available in the `HITRAN <https://hitran.org/>`_ database used by SPECTRA.
   Notably H2O is absent from this list of gas species as well!