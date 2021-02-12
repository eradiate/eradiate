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
     - `Link <https://eradiate.eu/data/spectra-us76_u86_4.zip>`_

.. _sec-user_guide-molecular_absorption_datasets-spectra_us76_u86_4:

spectra-us76_u86_4
------------------

This absorption spectrum is that of the gas mixture ``us76_u86_4`` defined by
the following volume mixing ratio:

- ``N2``: 0.78084
- ``O2``: 0.209476
- ``CO2``: 0.000314
- ``CH4``: 0.000002

This gas mixture corresponds to the gas mixture that is present in the lower
86 km part of the atmosphere in the US76 model
:cite:`NASA1976USStandardAtmosphere`, when the composition is
restricted to 4 gas species, namely **N2**, **O2**, **CO2** and **CH4**.
The dataset is limited to altitudes below 86 kilometers, because for these
altitudes, the gases are assumed to be well mixed in the US76 atmosphere
thermophysical model.

The absorption cross section values were computed using
`SPECTRA <https://spectra.iao.ru/>`_.

The ``spectra-us76_u86_4`` dataset has a wavenumber coordinate and a pressure
coordinate. The table below indicates the characteristics of the spectral and
pressure meshes.

.. list-table::
   :widths: 1 1 1 1
   :header-rows: 1

   * - Coordinate
     - Minimum value
     - Maximum value
     - Step
   * - Wavenumber (cm^-1)
     - 4000
     - 25711
     - 0.00314
   * - Wavelength (nm)
     - ~389
     - 2500
     - (variable)
   * - Pressure (Pa)
     - 0.101325
     - 101325
     - (variable)

The pressure mesh is 64 values geometrically spaced between the minimum and
maximum values.
This dataset can be interpolated both in wavenumber and pressure, but has no
temperature coordinate.
Since the dataset is specific to the US76 atmosphere thermophysical profile,
when interpolating along the pressure axis, the temperature axis' value will
match the corresponding value pair in said profile.
In other words, the dataset must be interpolated only once to find the
absorption cross section corresponding to a set of pressure and temperature
values.
