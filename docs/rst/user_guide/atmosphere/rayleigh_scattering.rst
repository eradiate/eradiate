.. _sec-atmosphere-rayleigh-scattering:

Rayleigh scattering
===================

Scattering of light by molecules is well modeled using the Rayleigh
approximation (see :cite:`Liou2002IntroductionAtmosphericRadiation`,
section 3.3.1 for example).

.. _sec-atmosphere-rayleigh-scattering-phase:

Phase function
--------------

We use the phase function :math:`[/]` (units: :math:`\mathrm{str}^{-1}`)
expression provided by :cite:`Hansen1974LightScatteringPlanetary` (eq. 2.14):

.. math::

   p(\theta) = \frac{\Delta}{4 \pi} \left[
                  \frac{3}{4} \left( 1 + \cos^2 \theta \right) \right] +
               \frac{1 - \Delta}{4 \pi}

where

* :math:`\theta` is the scattering angle :math:`[/]` (units:
  :math:`\mathrm{rad}`) and
* :math:`\Delta` :math:`[/]` is given by:

.. math::

   \Delta = \frac{1 - \delta}{1 + \delta / 2}

where
:math:`\delta` is the depolarisation factor :math:`[/]`.

We take :math:`\delta = 0` for air, which gives :math:`\Delta = 1`.

.. warning::

   We use the following normalisation rule for the phase function:

   .. math::

      \int_{0}^{2\pi}\int_{0}^{\pi} p(\theta) \sin\theta \, d\theta \, d\phi = 1

   where :math:`\phi` is the azimuth angle :math:`[/]` (units:
   :math:`\mathrm{rad}`).


Scattering coefficient
----------------------

We use the expression of the scattering coefficient :math:`[L^{-1}]` for a pure
gas provided by :cite:`Eberhard2010CorrectEquationsCommon` (eq. 60), and apply
it to air:

.. math::

   k_{\mathrm s \, \lambda} (n) = \frac{8 \pi^3}{3 \lambda^4} \frac{1}{n}
      \left( \eta_{\lambda}^2(n) - 1 \right)^2 F_{\lambda}

where

* :math:`\lambda` is the wavelength :math:`[L]` (subscript indicates spectral
  dependence),
* :math:`n` is the air number density :math:`[L^{-3}]`,
* :math:`\eta` is the air refractive index :math:`[/]` and
* :math:`F_{\lambda}` is the air King correction factor :math:`[/]`.

.. note::

   Since air is considered as a pure gas, variations of the scattering
   coefficient with composition are neglected.

The spectral dependence of the air refractive index is computed according to
:cite:`Peck1972DispersionAir` (eq. 2), valid in the wavelength range
:math:`[230, 1690]` nm.
In the wavelength range [1690, 2400] nm, we assume that this equation remains
accurate enough, although we have no data to justify that assumption.

The number density dependence is computed using the simple proportionality rule:

.. math::

   \eta(n) = \frac{n}{n_0} \eta_0

where

* :math:`n_0` is the air number density under standard conditions of pressure
  and temperature (101325 Pa, 288.15 K) and
* :math:`\eta_0` is the air refractive index under standard conditions of
  pressure and temperature.

The King correction factor is computed using the data from
:cite:`Bates1984RayleighScatteringAir` (table 1).
The data is interpolated within the [200, 1000] nm range and extrapolated
using the 1000 nm value for wavelength larger than 1000 nm.