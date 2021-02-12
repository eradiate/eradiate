.. _sec-user_guide-molecular_absorption_datasets:

Molecular absorption datasets
=============================

Eradiate ships with several molecular absorption datasets. This guide will
provide a list of available data and include a detailed description of the
datasets, their properties and limitations.

List of available molecular absorption datasets
-----------------------------------------------

The list of all available datasets can be found in the documentation for the
:mod:`absorption_spectra <eradiate.data.absorption_spectra>` module.
Click any item in that list to download the corresponding dataset.
Refer to :ref:`this section <sec-user_guide-manual_download>` for instructions
on installing these datasets.

spectra-us76_u86_4
------------------

Absorption cross section dataset for the ``us76_u86_4`` mixture.
The ``us76_u86_4`` mixture is defined as the gas mixture in the US76 atmosphere
model for altitudes below 86 kilometers and restricted to four gas species:
N2, O2, CO2, CH4.
The volume fractions are the following:

- N2: 0.78084
- O2: 0.209476
- CO2: 0.000314
- CH4: 0.000002

.. admonition:: Note

   The dataset is restricted to these four species, because the others
   (Ar, Ne, He, Kr, Xe) are not available in the
   `HITRAN <https://hitran.org/>`_ database used by SPECTRA.
   Notably H2O is absent from this list of gas species as well!

The dataset follows the US76 atmospheric profile
:cite:`NASA1976USStandardAtmosphere` up to an altitude of 86 kilometers and
contains four molecular species. The absorption cross section values were
computed using `SPECTRA <https://spectra.iao.ru/>`_.

This dataset can be interpolated both in wavenumber and in pressure.
Since the dataset is specific to the US76 atmosphere thermophysical profile,
when interpolating along one of the axes, the other axis' value will match the
corresponding value pair in said profile.
In other words, the dataset must be interpolated only once to find the
absorption cross section corresponding to a set of pressure and temperature
values.

The dataset is limited to altitudes below 86 kilometers, because for these
altitudes, the gasses can be assumed to be well mixed. Above 86 kilometers the
air number density is smaller than a factor of 1e-5 compared to sea level and
the assumption of well mixing breaks down.
