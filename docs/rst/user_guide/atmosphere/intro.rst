.. _sec-atmosphere-intro:

Introduction
============

Assumptions
-----------

The air is assumed to behave as an
`ideal gas <https://en.wikipedia.org/wiki/Ideal_gas>`_,
*i.e.* a collection of point particles without interactions between one another.
It is in thermodynamic equilibrium, it is not chemically reacting and it obeys
the ideal gas state equation:

.. math::
   :label: ideal_gas_state

   p = n \, k \, T

where:

* :math:`p` stands for the air pressure :math:`[ML^{-1}T^{-2}]`,
* :math:`n` stands for the air number density :math:`[L^{-3}]`,
* :math:`k` is the
  `Bolzmann constant <https://en.wikipedia.org/wiki/Boltzmann_constant>`_
  :math:`[ML^{2}T^{-2}\Theta^{-1}]` and
* :math:`T` stands for the air temperature :math:`[\Theta]`.

Air can usually be treated as an ideal gas within reasonable tolerance over a
wide parameter range around standard temperature and pressure (273 K, 100 kPa)
and the approximation generally gets better with lower pressure and higher
temperature.

Atmosphere modelling
--------------------

Eradiate represents the atmosphere as a participating medium (the air)
whose envelope is defined by a geometric shape.
The shape is either a cuboid or a spherical shell, depending on the
geometry used.
The participating medium is specified by its radiative properties, *i.e.* the
scattering phase function :math:`p`, single scattering albedo :math:`\varpi`
and volume extinction coefficient :math:`\sigma_{\mathrm{t}}` of its
constituents, for each point :math:`P \,(x, y, z)` in the participating medium:

.. math::
   :label: radprops

   x, y, z \longrightarrow p(x, y, z), \varpi(x, y, z), \sigma_{\mathrm{t}}(x, y, z)

In the one-dimensional approximation, atmospheric radiative properties are invariant
with respect to :math:`x` and :math:`y`, and we can rewrite :eq:`radprops` as:

.. math::
   :label: radprops-onedim

   x, y, z \longrightarrow p(z), \varpi(z), \sigma_{\mathrm{t}}(z)

If the participating medium is uniform, then :eq:`radprops` reduces to:

.. math::

   x, y, z \longrightarrow p_0, \varpi_0, \sigma_{\mathrm{t}, 0}   

where :math:`p_0`, :math:`\varpi_0`, :math:`\sigma_{\mathrm{t}, 0}`  are constants.

The structure of the participating medium is assumed to be isotropic, *i.e.* the
properties of the medium embedded in one cell is invariant to rotation of the
cell.

Atmospheric constituents
^^^^^^^^^^^^^^^^^^^^^^^^

Atmospheric constituents are divided into two broad categories:

* **molecules** : particles in gaseous state, *e.g.*  H₂O, CO₂, O₃, N₂O, CO,
  CH₄, O₂, NO, SO₂, NO₂.
* **particles** : solid or liquid state particles, *e.g.* water droplets, ice
  crystals, dust particles.


Atmosphere types
^^^^^^^^^^^^^^^^

Atmosphere types are defined based on radiative properties uniformness
and nature of atmospheric constituents.

============  ===============================  =================================
Constituents  Uniform radiative properties     Non-uniform radiative properties
============  ===============================  =================================
Molecules     N/A                              :class:`.MolecularAtmosphere`
Particles     N/A                              :class:`.ParticleLayer`
Both          :class:`.HomogeneousAtmosphere`  :class:`.HeterogeneousAtmosphere`
============  ===============================  =================================

Volume interactions
^^^^^^^^^^^^^^^^^^^

The modelling of the interaction of radiation with molecules through scattering
and absorption is built in Eradiate.
By default, air scattering is modeled using the :ref:`sec-atmosphere-rayleigh-scattering` model.
Computation of molecular absorption is described :ref:`here <sec-atmosphere-molecular-absorption>`.

The modelling of the interaction of radiation with particles is not provided.
When working with particles, you must either:

* specify yourself the particle radiative properties,
* use pre-defined particle radiative properties.

.. note::
   
   So far, two pre-defined particle radiative properties data sets are provided.
   We are currently working on adding more data sets.