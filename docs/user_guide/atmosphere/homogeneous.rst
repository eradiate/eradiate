.. _sec-atmosphere-homogeneous:

Homogeneous atmospheres
=======================

The homogeneous atmosphere is a homogeneous participating medium characterised
by spatially invariant extinction coefficient, albedo and phase function.
The :class:`~eradiate.scenes.atmosphere.HomogeneousComponent` class describes
such a medium with user-controlled absorption and scattering coefficients and
phase function. Like other atmospheric components, it is assembled by an
:class:`.Atmosphere` and can be combined with other components.
