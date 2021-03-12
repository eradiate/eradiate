.. _sec-atmosphere-molecular-absorption:

Molecular absorption
====================

There are several ways to compute molecular absorption.
The most general and accurate way is the *monochromatic model*, also known as
the *line-by-line model*.
There are also approximate models, such as the CK model, which are less accurate
but faster.

Monochromatic model
-------------------

In monochromatic mode, the monochromatic model is used.
In the monochromatic model, the monochromatic absorption coefficient
:math:`[L^{-1}]` is computed using the equation:

.. math::

   k_{\mathrm a \lambda} (p, T) = n \, \sigma_{\mathrm a}(\nu, p, T)

where

* :math:`n` is the air number density :math:`[L^{-3}]`,
* :math:`\sigma_{\mathrm a}` is the monochromatic absorption cross section :math:`[L^2]`,
* :math:`\nu = \lambda^{-1}` is the wavenumber :math:`[L^{-1}]`,
* :math:`p` is the pressure :math:`[ML^{-1}T^{-2}]` and
* :math:`T` is the temperature :math:`[\Theta]`.

The monochromatic absorption cross section is computed by interpolating the
absorption cross section data set that corresponds to the absorbing gas species
or mixture on the wavenumber, pressure and temperature axes.

.. note::

   At the moment, molecular absorption is only computed for the so-called
   ``us76_u86_4`` gas mixture and within the
   :mod:`us76 <eradiate.thermoprops.us76>` thermophysical properties profile.

   We are currently working on the functionality to compute absorption by the
   following individual gas species: H2O, CO2, O3, N2O, CO, CH4 and O2.


``us76_u86_4``
~~~~~~~~~~~~~~

The ``us76_u86_4`` gas mixture is defined by the mixing ratios provided in the
table below.
The ``us76_u86_4`` is named in this way because it corresponds to the
gas mixture defined by the ``us76`` atmosphere model
:cite:`NASA1976USStandardAtmosphere` in the region of altitudes below 86 km,
and restricted to the 4 main molecular species, namely N2, O2, CO2 and CH4.

.. list-table::
   :widths: 2 1 1 1 1

   * - Species
     - N2
     - O2
     - CO2
     - CH4
   * - Mixing ratio [%]
     - 78.084
     - 20.9476
     - 0.0314
     - 0.0002

Monochromatic absorption by the ``us76_u86_4`` gas mixture is computed by
interpolating the absorption cross section data set ``spectra-us76_u86_4`` in
wavenumber and pressure.
The absorption cross sections in this data set were computed using the
`SPECTRA <https://spectra.iao.ru>`_
Information System, which uses the HITRAN database (2016-edition)
:cite:`Gordon2016HITRAN2016MolecularSpectroscopic`.
The data set is not interpolated on temperature because the temperature
coordinate is mapped onto the pressure coordinate according to the
pressure-temperature relationship defined by the ``us76`` thermophysical
profile.
In other words, when interpolating the data set in pressure, the
temperature coordinate is automatically set to the appropriate value.
This is why this data set is tied to the ``us76`` thermophysical profile.


The ``spectra-us76_u86_4`` data set covers the wavenumber and pressure ranges
indicated in the table below:

.. list-table::
   :widths: 1 1 1
   :header-rows: 1

   * -
     - Minimum value
     - Maximum value
   * - Wavenumber [cm :math:`^{-1}`]
     - 4000
     - 25711
   * - Wavelength [nm]
     - 390
     - 2500
   * - Pressure [atm]
     - :math:`10^{-6}`
     - 1
   * - Pressure [Pa]
     - 0.101325
     - 101325

The minimum pressure value of :math:`10^{-6}` atm corresponds to a maximum
altitude of approximately 93 km, in the ``us76`` thermophysical profile.
This minimum value is a restriction of SPECTRA.
For higher altitudes, that is to say for lower pressure values, the absorption
cross section value is approximated to 0.

.. note::

   Given that the maximum value of the absorption cross section at 93 km is:

   .. math::

      \max_{\nu} \sigma_{a} = 9.62 \, 10^{-23} \, \mathrm{cm}^2,

   which corresponds to a maximal absorption coefficient value of:

   .. math::

      \max_{\nu} k_{a} = 4.0 \, 10^{-4} \, \mathrm{km}^{-1},

   this approximation seems reasonable.


Approximate models
------------------

.. note::
   The support for approximate models is currently ongoing work.
