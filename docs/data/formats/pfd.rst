Particle field (Pfd)
====================

.. _sec-data-formats-pfd_core_v1:

Pfd-Core v1 [``pfd_core_v1``]
-----------------------------

Dense, per-voxel description of a spatially varying particle field.
Typically used for clouds. Every voxel of the grid carries a value: the
local particle size distribution parameters and the mass concentration used
to scale optical properties looked up from a
:ref:`Prt-Cloud v1 <sec-data-formats-prt_cloud_v1>` dataset. Voxels with no
particles simply carry a mass concentration of zero.

The grid is described by the ``x_levels``/``y_levels``/``z_levels`` arrays of
cell edges along each axis. The profile is defined on either a Cartesian
coordinate system (``x_levels``/``y_levels`` in length units) or a spherical
geocentric coordinate system (``x_levels``/``y_levels`` in angle units,
representing azimuth/colatitude).

Format
    ``xarray.Dataset`` (in-memory), NetCDF (storage)

Dimensions
    * ``x``: cell index along the x axis
    * ``y``: cell index along the y axis
    * ``z``: cell index along the z (altitude) axis

Data variables
    *When relevant, units are required and specified in the "units" metadata field.*

    * ``x_levels`` float, 1-D [length or angle], size ``len(x) + 1``: grid
      cell edges along the x axis
    * ``y_levels`` float, 1-D [length or angle], size ``len(y) + 1``: grid
      cell edges along the y axis
    * ``z_levels`` float, 1-D [length], size ``len(z) + 1``: grid cell edges
      along the z (altitude) axis
    * ``reff(z, y, x)`` float [length]: effective radius of the particle
      size distribution
    * ``veff(z, y, x)`` float [—]: effective variance of the particle size
      distribution
    * ``mass_concentration(z, y, x)`` float [mass / volume]: particle mass
      concentration. Zero where no particles are present; ``reff``/``veff``
      are then irrelevant and may take any value.
