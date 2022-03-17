.. _sec-heterogeneous_atmosphere:

Heterogeneous atmospheres
=========================

Eradiate's heterogeneous atmospheres are 1D non-uniform participating media
consisting of both of molecules and particles.
:class:`.HeterogeneousAtmosphere` is a composite atmosphere type that contains:

* at most one instance of :class:`.MolecularAtmosphere`, representing the
  molecular or clear-sky component of a realistic atmospheres.
* an arbitrary number of instances of :class:`.ParticleLayer`, representing the
  cloud/aerosol components of a realistic atmosphere.
