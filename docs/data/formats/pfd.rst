Particle field (Pfd)
====================

.. _sec-data_formats-pfd_core_v1:

Pfd-Core v1 [``pfd_core_v1``]
-----------------------------

This format defines a voxellized description of the state variables of a
particulate medium. Each data variable describes a *field*, *i.e.* a spatially
varying quantity, on 3D gridded coordinates. Data variables are not prescribed:
a :class:`.ParticleStateComponent` binds them by name to the state dimensions of
a particle properties dataset and to the amount scaling its extinction.

The grid is described by the ``x_edges``/``y_edges``/``z_edges`` arrays of cell
edges along each axis. The profile is defined on either a Cartesian coordinate
system (``x_edges``/``y_edges`` in length units) or a spherical geocentric
coordinate system (``x_edges``/``y_edges`` in angle units, representing
azimuth/colatitude).

Format
    ``xarray.Dataset`` (in-memory), NetCDF (storage)

Dimensions
    * ``x``: cell index along the x axis
    * ``y``: cell index along the y axis
    * ``z``: cell index along the z (altitude) axis

Data variables
    *When relevant, units are required and specified in the "units" metadata field.*

    * ``x_edges`` float, 1-D [length or angle], size ``len(x) + 1``: grid
      cell edges along the x axis
    * ``y_edges`` float, 1-D [length or angle], size ``len(y) + 1``: grid
      cell edges along the y axis
    * ``z_edges`` float, 1-D [length], size ``len(z) + 1``: grid cell edges
      along the z (altitude) axis
    * At least one ``<name>(z, y, x)`` float [any]: per-voxel field variable.
      The variable used as the amount is zero where no particles are present;
      other variables are then irrelevant and may take any value.

.. _sec-data_formats-pfd_psd_v1:

Pfd-PSD v1 [``pfd_psd_v1``]
---------------------------

Extension of ``pfd_core_v1`` describing a particle field by its particle size
distribution (PSD) parameters and mass concentration. It is the profile
counterpart of :ref:`Prt-PSD v1 <sec-data_formats-prt_psd_v1>` and the format
expected by :class:`.ParticlePSDComponent` with its default bindings.

Data variables
    In addition to the ``pfd_core_v1`` variables:

    * ``reff(z, y, x)`` float [length]: effective radius of the particle
      size distribution
    * ``veff(z, y, x)`` float [—]: effective variance of the particle size
      distribution
    * ``mass_concentration(z, y, x)`` float [mass / volume]: particle mass
      concentration. Zero where no particles are present; ``reff``/``veff``
      are then irrelevant and may take any value.
