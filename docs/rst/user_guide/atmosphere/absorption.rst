.. _sec-atmosphere-molecular-absorption:

Molecular absorption
====================

There are several ways to compute molecular absorption.
The most general and accurate way is the *monochromatic model*, also known as
the *line-by-line model*.
There are also approximate models, such as the *CK model*, which are less
accurate but faster.
:cite:`Taine2014TransfertsThermiquesIntroduction`.

Monochromatic model
-------------------

In monochromatic mode, the monochromatic model is used.
In the monochromatic model, the air monochromatic absorption coefficient,
:math:`k_{\mathrm a \lambda} \, [L^{-1}]`,
due to :math:`N` absorbers, is computed using the equation:

.. math::
   :label: mono_k

   k_{\mathrm a \lambda} (p, T, \vec{x}) = \sum_{i=0}^{N-1} \, x_i \, n \,
   \sigma_{\mathrm{ai}}(\nu, p, T)

where

* :math:`\lambda` is the wavelength :math:`[L]` (subscript indicates spectral
  dependence),
* :math:`p` is the pressure :math:`[ML^{-1}T^{-2}]`,
* :math:`T` is the temperature :math:`[\Theta]`,
* :math:`\vec{x}` is the absorbers volume fraction vector, whose components are
  the individual absorbers volume fractions in air, :math:`x_i` :math:`[/]`,
* :math:`n` is the air number density :math:`[L^{-3}]`,
* :math:`\sigma_{\mathrm {ai}}` is the monochromatic absorption cross section
  :math:`[L^2]`, of the :math:`i` -th absorber and
* :math:`\nu = \lambda^{-1}` is the wavenumber :math:`[L^{-1}]`,

.. note::
   The sum of the absorbers volume fractions is not necessarily 1, because there
   are some non-absorbing species in the air.

.. note::

   Absorbers can be mixtures of absorbers.

In :eq:`mono_k`, the monochromatic absorption cross section values
:math:`\sigma_{\mathrm {ai}}` are computed by interpolating
the absorption cross section data set that corresponds to the absorber
on the wavenumber, pressure and temperature axes.

Wait, what?
^^^^^^^^^^^

Indeed, the absorption cross section data set is interpolated on the
wavenumber axis.
In theory, interpolating the absorption cross section spectrum of a gas
requires the spectrum to have a very high resolution, because the absorption
spectrum typically includes numerous fine features called *lines*.
Absorption lines are characterised by an intensity denoted :math:`S`
(area under the line curve) and a half-width at half-maximum denoted
:math:`\gamma`.

.. image:: fig/line.png
   :align: center

In standard conditions (101325 Pa, 300 K), :math:`\gamma` is around
:math:`10^{-2} \mathrm{cm}^{-1}` for CO2 and H2O
:cite:`Taine2014TransfertsThermiquesIntroduction`,
but changes with pressure and temperature (and also with the absorber) and
generally decreases with increasing altitudes.
In order to resolve each line, the spectrum must be computed at a resolution
better than :math:`\gamma`, e.g.
:math:`3 \, 10^{-3} \mathrm{cm}^{-1}`
if :math:`\gamma = 10^{-2} \mathrm{cm}^{-1}`.
Good and bad spectral resolutions are illustrated below.

.. image:: fig/good_resolution.png
   :align: center

.. image:: fig/bad_resolution.png
   :align: center

.. note::

   At the moment, molecular absorption is only computed for the so-called
   ``us76_u86_4`` absorber and within the
   :mod:`us76 <eradiate.thermoprops.us76>` thermophysical properties profile.

   We are currently working on the functionality to compute absorption by the
   following individual gas species: H2O, CO2, O3, N2O, CO, CH4 and O2.


``us76_u86_4``
^^^^^^^^^^^^^^




Approximate models
------------------

.. note::
   The support for approximate models is currently ongoing work.
