.. _sec-particle_components:

Particle component
==================

Particle components are 3D non-uniform participating media whose scattering behavior
is characterized by a Mie phase function, stored in a lookup table.
Particle component scene elements are created using :class:`.ParticleComponent` subclasses:

 * :class:`.ParticleOTComponent`
 * :class:`.ParticleExtinctionComponent`
 * :class:`.ParticleStateComponent`
 * :class:`.ParticlePSDComponent`

.. _sec-particle_components-terminology:

Terminology
-----------

.. glossary::

   Amount
     :term:`State variable` with which a radiative coefficient scales
     linearly, factored out of tabulated optical properties, *e.g.* particle
     mass or number concentration, or air number density for Rayleigh
     scattering. A zero amount means the medium is absent.

   Field
     Spatially varying variable.

   Particle profile
     Dataset holding the :term:`fields <field>` of the
     :term:`state variables <state variable>` of a particulate medium
     (:term:`amount` included) on a voxel grid, in the
     :ref:`Ppr <sec-data_formats-ppr_core_v1>` format. Wrapped by
     :class:`.ParticleProfile`.

   Particle properties
     Dataset tabulating per-unit-:term:`amount` optical properties
     (extinction, single-scattering albedo, phase function) against
     wavelength and :term:`state dimensions <state dimension>`, in the
     :ref:`Prt <sec-data_formats-prt_core_v1>` format. Wrapped by
     :class:`.ParticleProperties`.

   State dimension
     Dimension of a lookup table along which a :term:`state variable` is
     tabulated, *e.g.* ``reff``, ``veff`` or ``hum`` in
     :term:`particle properties`.

   State variable
     Quantity describing the local state of a medium, on which its radiative
     properties depend, *e.g.* pressure, temperature, molar fractions,
     particle concentration, effective radius or relative humidity.
