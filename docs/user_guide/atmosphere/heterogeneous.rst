.. _sec-heterogeneous_atmosphere:

Heterogeneous atmospheres
=========================

Eradiate's heterogeneous atmospheres are 1D non-uniform participating media
consisting of both of molecules and particles.
:class:`.HeterogeneousAtmosphere` is a composite atmosphere type holding a list
of components (:class:`.AtmosphereComponent`) constituted of:

* at most one instance of :class:`.MolecularComponent`, representing the
 molecular or clear-sky component of a realistic atmospheres.
* an arbitrary number of instances of :class:`.ParticleOTComponent`, representing the
  cloud/aerosol components of a realistic atmosphere.
