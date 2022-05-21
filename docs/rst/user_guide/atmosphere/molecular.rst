.. _sec-molecular-atmosphere:

Molecular atmospheres
=====================

Molecular atmospheres denote the molecular component of heterogeneous
atmospheres.
Molecular atmospheres are characterised by the atmosphere's thermophysical
properties from which are
computed the collision coefficients, and by a phase function.

Thermophysical properties
-------------------------

Molecular atmosphere objects are created from a
:ref:`thermophysical properties data set <sec-user_guide-data-thermoprops>`.

.. _sec-molecular-atmosphere-thermophysical-properties-pre-defined:

Pre-defined
~~~~~~~~~~~

The following atmospheric thermophysical properties models are pre-defined:

* U.S. Standard Atmosphere, 1976 :cite:`NASA1976USStandardAtmosphere`
* AFGL Atmospheric Constituent Profiles (0-120km)
  :cite:`Anderson1986AtmosphericConstituentProfiles` including its six
  `reference atmospheres`:

  * Tropical
  * Midlatitude Summer
  * Midlatitude Winter
  * Subarctic Summer
  * Subarctic Winter
  * U.S. Standard

.. warning::
  
   `U.S. Standard Atmosphere, 1976` and `U.S. Standard` are not the same.

Although they are pre-defined, these thermophysical properties can be customised
by changing the altitude grid as well as the concentration of the different
molecules---currently H₂O, CO₂ and O₃---in the atmosphere.
